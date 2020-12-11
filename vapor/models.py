# --- Imports ---
import logging
import random
import json
import os
import concurrent.futures as cf
import itertools
import datetime
import io
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from collections.abc import Iterable
import ast

import numpy as np
import pandas as pd
import math
import geopandas as gpd
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from shapely.geometry import Point

import PySAM.Pvsamv1 as pv
# import PySAM.Pvwattsv7 as pv
import PySAM.Windpower as wp
import PySAM.Singleowner as so
import PySAM.Merchantplant as mp
import PySAM.Battery as stbt # see release notes for Version 2.2.0, Dec 2, 2020 ~ SAM 2020.11.29, SSC Version 250 @ https://pypi.org/project/NREL-PySAM/
import PySAM.BatteryTools as bt

import vapor
import vapor.config as config
import vapor.helper as helper

log = logging.getLogger("vapor")

class BuildingLoad():
    """
    Representation of building load, eventually will use comstock.
    """

    def __init__(self):
        self.tmy_df = None

    def load(self):
        self.tmy_df = vapor.load_consumption_profile(shape='datacenter')

    def as_array(self):
        if isinstance(self.tmy_df, type(None)):
            self.load()
        return self.tmy_df['kwh'].values

    def lifetime_as_array(self, n_years):
        years = []
        for y in range(0, n_years):
            load_year = self.as_array()
            years.append(load_year)
            
        # --- concatenate ---
        out = np.concatenate(years).ravel().tolist()
        return out

    def scale(self, annual_load_mwh):
        self.annual_load_kwh = annual_load_mwh * 1000
        scaler = self.tmy_df['kwh'].sum() / (self.annual_load_kwh)
        self.tmy_df['kwh'] = self.tmy_df['kwh'] / scaler

    def fy_bau_co2(self, fy_co2):
        emissions = fy_co2 * self.tmy_df['kwh'].values 
        return emissions.sum()

    def fy_mitigated_co2(self, fy_co2, fy_gen):
        net_load = self.tmy_df['kwh'].values - fy_gen
        clipped_load = np.clip(net_load, a_min=0, a_max=None) #do not credit exports
        emission_profile = fy_co2 * clipped_load
        return emission_profile.sum()
    
    def fy_hourly_offset(self, fy_gen):
        """return fy load after offsetting self consumption."""
        net_load = self.tmy_df['kwh'].values - fy_gen
        clipped_load = np.clip(net_load, a_min=0, a_max=None) #do not credit exports
        return clipped_load.sum()

class Cambium():

    def __init__(self, cambium_df, aggregate_region, region, resource_file,
                analysis_period=None, construction_year=None):
        self.aggregate_region = aggregate_region
        self.region = region
        self.cambium_df = cambium_df
        self.resource_file = resource_file

        # --- Construction Year defaults to next year --- 
        if construction_year == None:
            self.construction_year = datetime.datetime.now().year + 1
        else:
            self.construction_year = construction_year

        # --- Analysis period defaults to config ---
        if analysis_period == None:
            self.analysis_period = config.SYSTEM_LIFETIME
        else:
            self.analysis_period = analysis_period
        
        # --- Retirement year defaults to next year + analysis_period ---
        self.retirement_year = self.construction_year + self.analysis_period

    def _load_tz(self):
        r = pd.read_csv(self.resource_file)
        if 'Local Time Zone' in r.columns:
            tz = int(r.at[0, 'Local Time Zone'])
        else:
            tz = int(5)  # i.e. wind gen is UTC (required by SAM), cambium is EST, add five hours to cambium so that profile is aligned in UTC
        return tz
    
    def clean(self):
        
        # --- subset region ---
        self.cambium_df = self.cambium_df.loc[self.cambium_df[self.aggregate_region] == self.region]
        self.variables = set(self.cambium_df['variable'])

        # --- clean up dt of cambium ---
        self.cambium_df['year'] = self.cambium_df['year'].astype(int)
        self.cambium_df['dayofyear'] = self.cambium_df['timestamp'].dt.dayofyear
        self.cambium_df['leapyear'] = self.cambium_df['timestamp'].dt.is_leap_year
        self.cambium_df.loc[(self.cambium_df['dayofyear'] > 59) & (self.cambium_df['leapyear'] == True), 'dayofyear'] -= 1
        self.cambium_df['hourofyear'] = (self.cambium_df['dayofyear'] - 1) * 24 + self.cambium_df['timestamp'].dt.hour

        # --- Convert cambium to generator TZ ---
        self.cambium_df['timestamp'] + (np.timedelta64(1, 'h') * self._load_tz())
        self.cambium_df.sort_values('timestamp', inplace=True)

        # --- start year should be used for slices ---
        self.start_year = self.construction_year
        if self.start_year < int(self.cambium_df['year'].min()):
            self.start_year = int(self.cambium_df['year'].min())
        self.analysis_period = (self.retirement_year - self.start_year)

        # --- Subset years ---
        self.cambium_df = self.cambium_df.loc[(self.cambium_df['year'] >= self.start_year) &
                                            (self.cambium_df['year'] < self.retirement_year)]

    def calc_cambium_lifetime_rev(self, gen, var, inflation=config.INFLATION, lower_thresh=0.01):
        """saves a temp 2D array with lifetime 'cleared' energy and hourly revenue."""

        assert var in self.variables

        rev = pd.DataFrame(self.cambium_df.loc[(self.cambium_df['variable'] == var), 'value'])

        try:
            assert len(rev) == self.analysis_period * 8760 == len(gen)
        except AssertionError as e: #bug fix for a few missing hours in high_re_cost scenario
            resampled = self.cambium_df.loc[(self.cambium_df['variable'] == var)].set_index('timestamp').resample('H').mean()
            resampled['dayofyear'] = resampled.index.dayofyear
            resampled['leapyear'] = resampled.index.dayofyear
            resampled = resampled.loc[~((resampled.index.month == 2) & (resampled.index.day == 29))]
            resampled['value'] = resampled['value'].interpolate()
            rev = pd.DataFrame(resampled['value'])
            assert len(rev) == self.analysis_period * 8760 == len(gen)

        rev['gen'] = gen
        rev['cleared'] = rev['gen'] / 1000 #kw to MW
        rev.loc[rev['cleared'] < rev['cleared'].max() * lower_thresh, 'cleared'] = 0 # clip minimum output to 1% of maximum output so merchantplant works
        rev = rev[['cleared','value']]

        # --- create array of hourly lifetime inflation ---
        if var in ['cambium_co2_rate_avg','cambium_co2_rate_marg', 'cambium_co2_rate_lrmer']: inflation = None #no inflation on emissions
        if inflation != None:
            assert isinstance(inflation, float)
            annual_inf = (1 + inflation) ** np.arange(self.analysis_period)
            hourly_inf = []
            for y in annual_inf:
                y_list = [y] * 8760
                for i in y_list:
                    hourly_inf.append(i)
            rev['value'] * hourly_inf
        return rev

    def calc_cambium_lifetime_rev_tuple(self, gen, var):
        rev = self.calc_cambium_lifetime_rev(gen, var)
        tup = tuple(map(tuple, rev.values))
        return tup

    def calc_cambium_lifetime_product(self, gen, var):
        rev = self.calc_cambium_lifetime_rev(gen, var)
        product = np.array(rev['cleared'] * rev['value'])
        return product

    def calc_cambium_lifetime_sum(self, gen, var):
        product = self.calc_cambium_lifetime_product(gen, var,)
        return np.sum(product)

class SystemCost():

    def load(self):
        # --- load ATB ---
        atb_cache_path = os.path.join('data', 'ATB', 'cached_atb.pkl')
        if os.path.exists(atb_cache_path):
            atb = pd.read_pickle(atb_cache_path)
        else:
            atb = pd.read_csv(os.path.join('data', 'ATB', 'ATBe.csv'))
            not_batt = atb.loc[
                        (atb['atb_year'] == 2020) & \
                        (atb['core_metric_case'] == 'Market') & \
                        (atb['crpyears'] == 20) & \
                        (atb['technology'].isin(['LandbasedWind', 'UtilityPV'])) & \
                        (atb['scenario'].isin(['Advanced','Conservative','Moderate'])) & \
                        (atb['core_metric_parameter'].isin(['CAPEX', 'Fixed O&M'])) & \
                        (atb['core_metric_variable'] == 2021) #make installation year
                    ]

            batt = atb.loc[
                        (atb['atb_year'] == 2020) & \
                        (atb['core_metric_case'] == 'Market') & \
                        (atb['crpyears'] == 20) & \
                        (atb['technology'].isin(['Battery'])) & \
                        (atb['techdetail'].isin(['2Hr Battery Storage', '4Hr Battery Storage'])) & \
                        (atb['scenario'].isin(['Advanced','Conservative','Moderate'])) & \
                        (atb['core_metric_parameter'].isin(['CAPEX', 'Fixed O&M'])) & \
                        (atb['core_metric_variable'] == 2021)
                    ]

            atb = pd.concat([not_batt, batt], axis='rows')

            atb['scenario'] = atb['scenario'].map({'Low': 'Low', 'Mid': 'Mid', 'Constant': 'High','Advanced':'Low', 'Conservative':'High','Moderate':'Mid'})
            atb['technology'] = atb['technology'].map({'UtilityPV':'pv', 'LandbasedWind':'wind', 'Battery':'battery'})
            atb['duration'] = 0
            atb.loc[atb['techdetail'] == '2Hr Battery Storage', 'duration'] = 2
            atb.loc[atb['techdetail'] == '4Hr Battery Storage', 'duration'] = 4
            atb.to_pickle(atb_cache_path)

        # --- create ATB tables we need ---
        self.capex = atb.loc[atb['core_metric_parameter'] == 'CAPEX'].groupby(['technology', 'scenario', 'duration', 'techdetail'], as_index=False)['value'].mean()
        self.om = atb.loc[atb['core_metric_parameter'] == 'Fixed O&M'].groupby(['technology', 'scenario', 'duration', 'techdetail'], as_index=False)['value'].mean()

        # --- load multipliers ---
        multipliers = pd.read_csv(os.path.join('data', 'ATB', 'reg_cap_cost_mult_default.csv'))
        multipliers.columns = ['region', 'technology', 'value']
        multipliers = multipliers.loc[multipliers['technology'].isin(['upv_1', 'wind-ons_1','BATTERY'])]
        multipliers['technology'] = multipliers['technology'].map({'wind-ons_1': 'wind', 'upv_1': 'pv', 'BATTERY':'battery'})
        self.multipliers = multipliers

    def get_capex(self, region, tech, techdetail, duration=0, scenario=config.COST_SCENARIO):
        """Return locationalized capex cost $/kw for a region"""
        
        capex = self.capex.loc[
                    (self.capex['technology'] == tech) & \
                    (self.capex['duration'] == duration) & \
                    (self.capex['techdetail'] == techdetail) & \
                    (self.capex['scenario'] == scenario),
                    'value'].values[0]

        try: #only works for reeds regions right now
            multiplier = self.multipliers.loc[
                        (self.multipliers['region'] == region) &
                        (self.multipliers['technology'] == tech),
                        'value'].values[0]
        except Exception:
            multiplier = 1 #region not found

        return capex * multiplier

    def get_om(self, region, tech, duration=0, scenario=config.COST_SCENARIO):
        """Return locationalized capex cost $/kw/yr for a region"""

        om = self.om.loc[
                    (self.om['technology'] == tech) & \
                    (self.om['duration'] == duration) & \
                    (self.om['scenario'] == scenario),
                    'value'].values[0]
        
        try: #only works for reeds regions right now
            multiplier = self.multipliers.loc[
                        (self.multipliers['region'] == region) &
                        (self.multipliers['technology'] == tech),
                        'value'].values[0]
        except Exception:
            multiplier = 1

        return om * multiplier


class GenericMerchantPlant():

    def __init__(self, system_config, resource_file, cambium, region, load=None):


        self.system_config = system_config
        self.resource_file = resource_file
        self.cambium = cambium
        self.region = region
        self.load = load
        
        self.storage = False
        if 'BatteryTools' in self.system_config.keys():
            if (self.system_config['BatteryTools']['desired_power'] > 0) & (self.system_config['BatteryTools']['desired_capacity'] > 0):
                self.storage = True

    def size_load(self):
        self.load = BuildingLoad()
        self.load.load()
        self.load.scale(self.generator.Outputs.annual_energy / 1000) #size load to match system aep

    def _size_battery(self):
        if self.storage:
            # --- Force duration to be 0, 2, 4 ---
            desired_duration = self.system_config['BatteryTools']['desired_capacity']
            
            if desired_duration < 1:
                self.battery_duration = 0
                self.storage = False
            elif desired_duration < 2:
                self.battery_duration = 2
            elif desired_duration <= 4:
                self.battery_duration = 4

            self.system_config['BatteryTools']['desired_capacity'] = self.battery_duration * self.system_config['BatteryTools']['desired_power'] #convert capacity expressed in hours to kWh
            self.battery_params = self.battery_sizing_config = bt.battery_model_sizing(
                                                model=self.battery,
                                                **self.system_config['BatteryTools'])
        else:
            self.battery_params = None
    
    def run_battery(self):
        self.battery = stbt.from_existing(self.generator)# Simualtion, Lifetime, BatterySystem, SystemOutput, Load, BatteryCell, Inverter, Losses, batteryDispatch, ElectricityRates, FuelCell, PriceSignal
        self.battery.Lifetime.system_use_lifetime_output = 1 #needed for wind
        self.battery.Lifetime.analysis_period = self.cambium.analysis_period #needed for wind
        self.battery.SystemOutput.gen = self.gen_profile_no_batt #this isn't inherited if tech is wind, and there is no lifetime output for wind, so we need to manually assign this
        self.battery.SystemOutput.capacity_factor = self.generator.Outputs.capacity_factor
        self.battery.SystemOutput.annual_energy = self.generator.Outputs.annual_energy
        self.battery.Load.load = self.load.as_array()
        self.battery.BatterySystem.en_batt = 1
        self.battery.BatterySystem.batt_meter_position = 0  # 0: BTM, 1: FTM
        self.battery.BatterySystem.batt_ac_or_dc = 1  # 0: dc, 1: ac
        self.battery.BatterySystem.batt_dc_ac_efficiency = 96
        self.battery.BatterySystem.batt_dc_dc_efficiency = 98
        self.battery.BatterySystem.batt_ac_dc_efficiency = 98
        self.battery.BatterySystem.batt_current_choice = 1
        self.battery.BatterySystem.batt_replacement_capacity = 50
        self.battery.BatterySystem.batt_replacement_option = 1
        self.battery.BatterySystem.batt_surface_area = (1.586 ** 2) * 6
        self.battery.BatterySystem.batt_mass = 10133.271
        self.battery.BatterySystem.batt_inverter_efficiency_cutoff = 90

        self.battery.BatteryDispatch.batt_dispatch_choice = 2 # target power
        self.battery.BatteryDispatch.batt_target_choice = 1
        self.battery.BatteryDispatch.batt_target_power = np.full(8760, 0) #charge when net load is negative, discharge when positive
        self.battery.BatteryDispatch.batt_target_power_monthly = np.full(12, 0)

        self.battery.BatteryDispatch.batt_dispatch_auto_can_clipcharge = 1
        self.battery.BatteryDispatch.batt_dispatch_auto_can_charge = 1
        self.battery.BatteryDispatch.batt_dispatch_auto_can_gridcharge = 0  # can't grid charge
        self.battery.BatteryCell.batt_chem = 1
        self.battery.BatteryCell.batt_Vnom_default = 3.6
        self.battery.BatteryCell.batt_Qfull = 2.25
        self.battery.BatteryCell.batt_Qfull_flow = 0
        self.battery.BatteryCell.batt_Qexp = 0.04
        self.battery.BatteryCell.batt_Qnom = 2.0
        self.battery.BatteryCell.batt_C_rate = 0.2
        self.battery.BatteryCell.batt_Vfull = 4.1
        self.battery.BatteryCell.batt_Vexp = 4.05
        self.battery.BatteryCell.batt_Vnom = 3.4
        self.battery.BatteryCell.batt_resistance = 0.001
        self.battery.BatteryCell.batt_initial_SOC = 50
        self.battery.BatteryCell.batt_minimum_SOC = 10
        self.battery.BatteryCell.batt_maximum_SOC = 100
        self.battery.BatteryCell.batt_minimum_modetime = 0
        self.battery.BatteryCell.batt_calendar_choice = 0
        self.battery.BatteryCell.batt_lifetime_matrix = [[20, 0, 100], [20, 5000, 80], [20, 10000, 60], [80, 0, 100], [80, 1000, 80], [80, 2000, 60]]
        self.battery.BatteryCell.batt_calendar_lifetime_matrix = [[0, 100], [3650, 80], [7300, 50]]
        self.battery.BatteryCell.batt_calendar_q0 = 1.02
        self.battery.BatteryCell.batt_calendar_a = 0.00266
        self.battery.BatteryCell.batt_calendar_b = -7280
        self.battery.BatteryCell.batt_calendar_c = 930
        self.battery.BatteryCell.batt_voltage_matrix = [[0, 0]]
        self.battery.BatteryCell.cap_vs_temp = [[-10, 60], [0, 80], [25, 100], [40, 100]]
        self.battery.BatteryCell.batt_Cp = 1004
        self.battery.BatteryCell.batt_h_to_ambient = 500
        self.battery.BatteryCell.batt_room_temperature_celsius = np.full(8760, fill_value=20)

        # new --- Thomas 20201211 --- BatteryTools >> battery_model_sizing >> size_battery kep throwing 
        # errors about there not being a 'batt_computed_bank_capacity', this can be calculated based on 
        # supplied values above and using formulas from BatteryTools >> calculate_battery_size >> size_from_strings
        num_series = math.ceil(self.system_config['BatteryTools']['desired_voltage'] / self.battery.BatteryCell.batt_Vnom_default)
        num_strings = math.ceil(self.system_config['BatteryTools']['desired_capacity'] * 1000 / (self.battery.BatteryCell.batt_Qfull * self.battery.BatteryCell.batt_Vnom_default * num_series))
        computed_voltage = self.battery.BatteryCell.batt_Vnom_default * num_series
        self.battery.BatterySystem.batt_computed_bank_capacity = self.battery.BatteryCell.batt_Qfull * computed_voltage * num_strings * 0.001
        del num_series, num_strings, computed_voltage
        # original continued
        self._size_battery()
        self.battery.execute()

        # ts = pd.DataFrame({
        #     'batt_to_load':self.battery.Outputs.batt_to_load[0:8760],
        #     'pv_to_load':self.battery.Outputs.pv_to_load[0:8760],
        #     'grid_to_load':self.battery.Outputs.grid_to_load[0:8760],
        #     'pv_to_grid':self.battery.Outputs.pv_to_grid[0:8760],
        #     'pv_to_batt':self.battery.Outputs.pv_to_batt[0:8760],
        #     'batt_SOC':self.battery.Outputs.batt_SOC[0:8760],
        #     'load':self.load.as_array()
        # })
        # ts['total_RE'] = ts['batt_to_load'] + ts['pv_to_load']
        # ts['pct_RE'] = ts['total_RE'] / ts['load']
        # ts.to_csv('ts.csv')
    
    def run_financial(self):
            # FinancialParamters, SystemCosts, TaxCreditIncentives, Depreciation, PaymentIncentives, Revenue, batterySystem, SystemOutput, UtilityBill, Lifetime, FuelCell, Capacity Payments, GridLimits
        self.financial = mp.from_existing(self.generator)
        self.financial.Lifetime.system_use_lifetime_output = 1
        self.financial.FinancialParameters.analysis_period = self.cambium.analysis_period
        self.financial.FinancialParameters.debt_option = config.DEBT_OPTION
        self.financial.FinancialParameters.debt_percent = config.DEBT_PERCENT * 100
        self.financial.FinancialParameters.inflation_rate = config.INFLATION * 100
        self.financial.FinancialParameters.dscr = config.DSCR
        self.financial.FinancialParameters.real_discount_rate = config.DISCOUNT_RATE * 100
        self.financial.FinancialParameters.term_int_rate = config.INTEREST_RATE * 100
        self.financial.FinancialParameters.term_tenor = config.SYSTEM_LIFETIME
        self.financial.FinancialParameters.insurance_rate = 0
        self.financial.FinancialParameters.federal_tax_rate = [21]
        self.financial.FinancialParameters.state_tax_rate = [7]
        self.financial.FinancialParameters.property_tax_rate = 0
        self.financial.FinancialParameters.prop_tax_cost_assessed_percent = 100
        self.financial.FinancialParameters.prop_tax_assessed_decline = 0
        self.financial.Depreciation.depr_custom_schedule = [0]
        self.financial.SystemCosts.om_fixed = [0]
        self.financial.SystemCosts.om_fixed_escal = 0
        self.financial.SystemCosts.om_production = [0]
        self.financial.SystemCosts.om_production_escal = 0
        self.financial.SystemCosts.om_capacity_escal = 0
        self.financial.SystemCosts.om_fuel_cost = [0]
        self.financial.SystemCosts.om_fuel_cost_escal = 0
        self.financial.SystemCosts.om_replacement_cost_escal = 0
        self.financial.SystemOutput.degradation = [config.DEGRADATION*100]
        self.financial.SystemOutput.system_capacity = self.fitted_capacity
        self.financial.SystemOutput.gen = self.gen_profile
        self.financial.SystemOutput.system_pre_curtailment_kwac = self.gen_profile
        self.financial.SystemOutput.annual_energy_pre_curtailment_ac = self.generator.Outputs.annual_energy

        # --- Consider all generation as revenue at the market price ---
        # We are valuing the output to both the grid (and the developer) as the cambium market price.
        # this values self-consumption at the hourly market clearing price, same as exports.
        # however, battery charging (which can not be from the grid) and discharging is optimized around load
        # using a peak shaving algorithm.
        # this way, we are simulating a system designed to first, provide the maximum self consumption
        # while being valued using real-time costs of energy.
        self.financial.Revenue.mp_enable_energy_market_revenue = 1
        self.financial.Revenue.mp_energy_market_revenue = self.market_profile
        self.financial.Revenue.mp_enable_ancserv1 = 0
        self.financial.Revenue.mp_enable_ancserv2 = 0
        self.financial.Revenue.mp_enable_ancserv3 = 0
        self.financial.Revenue.mp_enable_ancserv4 = 0
        self.financial.Revenue.mp_ancserv1_revenue = [(0, 0) for i in range(len(self.market_profile))]
        self.financial.Revenue.mp_ancserv2_revenue = [(0, 0) for i in range(len(self.market_profile))]
        self.financial.Revenue.mp_ancserv3_revenue = [(0, 0) for i in range(len(self.market_profile))]
        self.financial.Revenue.mp_ancserv4_revenue = [(0, 0) for i in range(len(self.market_profile))]
        self.financial.CapacityPayments.cp_capacity_payment_type = 0
        self.financial.CapacityPayments.cp_capacity_payment_amount = [0]
        self.financial.CapacityPayments.cp_capacity_credit_percent = [0]
        self.financial.CapacityPayments.cp_capacity_payment_esc = 0
        self.financial.CapacityPayments.cp_system_nameplate = self.fitted_capacity
        if self.storage:
            self.financial.CapacityPayments.cp_battery_nameplate = self.battery_sizing_config['power']
        else:
            self.financial.CapacityPayments.cp_battery_nameplate = 0

        # --- Calculate System Costs ---
        cost = SystemCost()
        cost.load()

        if self.tech == 'pv':
            capex_kW = cost.get_capex(region=self.region, tech=self.tech, techdetail='KansasCity') #self.tech set in run_generator()

        elif self.tech == 'wind':
            capex_kW = cost.get_capex(region=self.region, tech=self.tech, techdetail=f"LTRG{self.turbine_class}") #self.tech set in run_generator()

        system_cost = self.fitted_capacity * capex_kW
        cost_per_inverter = 60 * 0.07 * 1000
        marginal_inverter_cost = 1 * cost_per_inverter
        system_cost += marginal_inverter_cost

        # --- Calculate storage system cost ---
        if self.storage:
            battery_capex_kw = cost.get_capex(tech='battery', region=self.region,
                                        duration=self.battery_duration, techdetail=f"{self.battery_duration}Hr Battery Storage")
            battery_om_kw = cost.get_om(tech='battery', region=self.region, duration=self.battery_duration)
            battery_cost = battery_capex_kw * self.battery_sizing_config['power']
            system_cost += battery_cost

            # --- Assign Battery OM ---
            self.financial.SystemCosts.om_capacity1 = [battery_om_kw]

        # --- Cost of hub height deviations ---
        if self.tech == 'wind':
            hub_height_diff = self.generator.Turbine.wind_turbine_hub_ht - 88 #88m default in ATB
            hub_price_diff = (self.fitted_capacity * 6.383) * hub_height_diff
            system_cost += hub_price_diff

        self.financial.SystemCosts.total_installed_cost = system_cost
        self.financial.FinancialParameters.construction_financing_cost = system_cost * 0.009

        # --- Calcualte O&M Costs ---
        system_om_kw = cost.get_om(region=self.region, tech=self.tech)
        self.financial.SystemCosts.om_capacity = [system_om_kw]
        self.financial.execute()

    def execute_all(self):
        self.run_generator()
        if self.load == None:
            self.size_load()
        self.make_gen_profile_no_batt() #8760 * analysis period of generation without battery
        if self.storage: 
            self.run_battery()
        self.make_gen_profile() # 8760 * analysis period, with batt if it exists
        self.make_market_profile()
        self.run_financial()

        if self.storage:
            self.outputs = {**self.financial.Outputs.export(), **self.generator.Outputs.export(), **self.battery.Outputs.export()}
            self.outputs = {k:v for k,v in self.outputs.items() if isinstance(v, (str, float, int))}
            self.outputs['lifetime_gen_profile'] = np.array(self.battery.Outputs.pv_to_load) +\
                                                    np.array(self.battery.Outputs.batt_to_load) +\
                                                    np.array(self.battery.Outputs.pv_to_grid)
        else:
            self.outputs = {**self.financial.Outputs.export(), **self.generator.Outputs.export()}
            self.outputs = {k:v for k,v in self.outputs.items() if isinstance(v, (str, float, int))}
            self.outputs['lifetime_gen_profile'] = np.clip(self.generator.Outputs.gen, a_min=0, a_max=None) #clip parasitic loss to get pv_to_load + batt_to_load + pv_to_grid

class PVMerchantPlant(GenericMerchantPlant):

    def make_gen_profile_no_batt(self):
        self.gen_profile_no_batt = self.generator.Outputs.gen
        assert len(self.gen_profile_no_batt) == 8760 * self.cambium.analysis_period
    
    def make_gen_profile(self):
        """If no battery is sized, this should be the same as gen_profile_no_batt, if battery, includes dispatch."""
        self.gen_profile = self.generator.Outputs.gen
        assert len(self.gen_profile) == 8760 * self.cambium.analysis_period

    def make_market_profile(self):
        self.market_profile = self.cambium.calc_cambium_lifetime_rev_tuple(self.gen_profile, 'cambium_grid_value')
        assert len(self.market_profile) == 8760 * self.cambium.analysis_period

    def _size_system(self):
        # --- Calc number of strings and inverters ---
        desired_system_capacity = self.system_config['SystemDesign']['system_capacity']
        modules_per_string = 12
        kw_per_string = 0.310 * self.generator.SystemDesign.subarray1_modules_per_string
        n_strings = int(desired_system_capacity / kw_per_string)
        self.fitted_capacity = n_strings * kw_per_string  # used later

        baseline_inverter_kw = self.fitted_capacity / 1.2
        desired_inverter_kw = self.fitted_capacity / self.system_config['SystemDesign']['dc_ac_ratio']
        kw_per_inverter = 60
        baseline_inverters = max(1, int(baseline_inverter_kw / kw_per_inverter))
        inverters = max(1, int(desired_inverter_kw / kw_per_inverter))
        self.marginal_inverters = inverters - baseline_inverters  # used for cost later
        self.system_config['SystemDesign']['subarray1_nstrings'] = n_strings
        self.system_config['SystemDesign']['inverter_count'] = inverters

    def run_generator(self):
        self.tech = 'pv' #used for system costs
        self.generator = pv.default('FlatPlatePVSingleOwner')
        self.generator.SolarResource.solar_resource_file = self.resource_file
        self._size_system()
        self.generator.SystemDesign.assign({k: v for k, v in self.system_config['SystemDesign'].items() if k in list(self.generator.SystemDesign.export().keys())})
        self.generator.SystemDesign.system_capacity = self.fitted_capacity
        self.generator.SystemDesign.subarray1_backtrack - 0
        self.generator.Lifetime.analysis_period = self.cambium.analysis_period #get analysis period from cambium, which accounts for retirement year
        self.generator.Lifetime.system_use_lifetime_output = 1
        self.generator.Lifetime.dc_degradation = [config.DEGRADATION*100] 
        self.generator.execute()

class WindMerchantPlant(GenericMerchantPlant):

    def make_gen_profile_no_batt(self):
        """Necessarily does not have battery."""
        assert len(self.generator.Outputs.gen) == 8760
        self.gen_profile_no_batt = self._single_year_to_multi_year_gen(self.generator.Outputs.gen, 0.05)
        assert len(self.gen_profile_no_batt) == 8760 * self.cambium.analysis_period

    def make_gen_profile(self):
        """If no battery is sized, this should be the same as gen_profile_no_batt, if battery, includes dispatch."""
        if self.storage:
            self.gen_profile = self.generator.Outputs.gen
        else:
            self.gen_profile = self._single_year_to_multi_year_gen(self.generator.Outputs.gen, 0.05)
        assert len(self.gen_profile) == 8760 * self.cambium.analysis_period

    def make_market_profile(self):
        self.market_profile = self.cambium.calc_cambium_lifetime_rev_tuple(self.gen_profile, 'cambium_grid_value')
        assert len(self.market_profile) == 8760 * self.cambium.analysis_period

    def _set_num_turbines_in_row(self, n_turbines, rotor_diameter=77, spacing=None, angle_deg=0):
        xcoords = []
        ycoords = []
        row_spacing = max(spacing, rotor_diameter * 3)
        dx = row_spacing * np.cos(np.radians(angle_deg))
        dy = row_spacing * np.sin(np.radians(angle_deg))
        x0 = 0
        y0 = 0
        
        for i in range(n_turbines):
            turb = Point((x0 + i * dx, y0 + i * dy))
            xcoords.append(turb.x)
            ycoords.append(turb.y)

        return xcoords, ycoords

    def _size_system(self):
        # --- calc layout of farm ---
        self.desired_farm_kw = self.system_config['Farm']['system_capacity']
        self.turbine_kw = min(self.desired_farm_kw, 2400) #based on ATB
        self.n_turbines = int(self.desired_farm_kw // self.turbine_kw)
        self.n_turbines = min(300, self.n_turbines) #SAM maximum
        self.fitted_capacity = self.n_turbines * self.turbine_kw
        self.turbine_class = self.system_config['Turbine']['turbine_class']

    def _single_year_to_multi_year_gen(self, gen, degradation):
        assert len(gen) == 8760
        # --- apply degradation --- 
        years = []
        for y in range(0, self.cambium.analysis_period):
            gen_year = np.array(gen)
            gen_year = gen_year * ((1 - degradation) ** y)
            years.append(gen_year)
        
        # --- concatenate ---
        out = np.concatenate(years).ravel().tolist()
        return out

    def run_generator(self):
        self.tech = 'wind'
        self._size_system()
        self.generator = wp.default('WindPowerSingleOwner') #Resource, Turbine, Farm, Losses, Uncertainty, AdjustmentFactors
        self.generator.Resource.wind_resource_filename = self.resource_file
        self.generator.Resource.wind_resource_model_choice = 0

        # --- Wind class power curve ---
        wind_class_dict = {
            1:{'cut_in':9.01, 'cut_out':12.89},
            2:{'cut_in':8.77, 'cut_out':9.01},
            3:{'cut_in':8.57, 'cut_out':8.77},
            4:{'cut_in':8.35, 'cut_out':8.57},
            5:{'cut_in':8.07, 'cut_out':8.35},
            6:{'cut_in':7.62, 'cut_out':8.07},
            7:{'cut_in':7.1, 'cut_out':7.62},
            8:{'cut_in':6.53, 'cut_out':7.1},
            9:{'cut_in':5.9, 'cut_out':6.53},
            10:{'cut_in':1.72, 'cut_out':5.9},
        }

        powercurve_dict = { 
            'turbine_size':2400, #From ATB
            'rotor_diameter':116, #SAM default
            'elevation':0,
            'max_cp':0.45, # SAM default
            'max_tip_speed':116, #Match rotor diameter
            'max_tip_sp_ratio':8, #SAM default
            'cut_in':wind_class_dict[self.turbine_class]['cut_in'],
            'cut_out':25, #not sure how to interpret maximum wind speeds, as they are too low for sensible cutout
            'drive_train':0,
        }
        
        self.generator.Turbine.calculate_powercurve(**powercurve_dict)
        self.generator.Turbine.wind_resource_shear = 0.14
        self.generator.Turbine.wind_turbine_rotor_diameter = 116

        # --- Create dummy farm layout in a row ---
        xcoords, ycoords = self._set_num_turbines_in_row(n_turbines=self.n_turbines, spacing=250)
        self.generator.Farm.wind_farm_xCoordinates = xcoords
        self.generator.Farm.wind_farm_yCoordinates = ycoords
        self.generator.Farm.assign({k: v for k, v in self.system_config['Farm'].items() if k in list(self.generator.Farm.export().keys())})
        self.generator.Farm.wind_farm_wake_model = 0
        self.generator.Farm.wind_resource_turbulence_coeff = 0.1
        self.generator.Farm.system_capacity = self.fitted_capacity
        
        self.generator.Losses.wake_int_loss = 0
        self.generator.Losses.wake_ext_loss = 1.1
        self.generator.Losses.wake_future_loss = 0
        self.generator.Losses.avail_bop_loss = 0.5
        self.generator.Losses.avail_grid_loss = 1.5
        self.generator.Losses.avail_turb_loss = 3.58
        self.generator.Losses.elec_eff_loss = 1.91
        self.generator.Losses.elec_parasitic_loss = 0.1
        self.generator.Losses.env_degrad_loss = config.DEGRADATION * 100
        self.generator.Losses.env_exposure_loss = 0
        self.generator.Losses.env_env_loss = 0.4
        self.generator.Losses.env_icing_loss = 0.21
        self.generator.Losses.ops_env_loss = 1
        self.generator.Losses.ops_grid_loss = 0.84
        self.generator.Losses.ops_load_loss = 0.99
        self.generator.Losses.ops_strategies_loss = 0
        self.generator.Losses.turb_generic_loss = 1.7
        self.generator.Losses.turb_hysteresis_loss = 0.4
        self.generator.Losses.turb_perf_loss = 1.1
        self.generator.Losses.turb_specific_loss = 0.81
        self.generator.Uncertainty.total_uncert = 12.085
        self.generator.execute()
