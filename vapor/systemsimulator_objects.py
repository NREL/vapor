# --- Imports ---
import logging
import random
import json
import os
import pickle
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
import geopandas as gpd
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer

import PySAM.Pvsamv1 as pv
import PySAM.Windpower as wp
import PySAM.Singleowner as so
import PySAM.Merchantplant as mp
# import PySAM.Battery as stbt # see release notes for Version 2.2.0, Dec 2, 2020 ~ SAM 2020.11.29, SSC Version 250 @ https://pypi.org/project/NREL-PySAM/
import PySAM.StandAloneBattery as stbt
import PySAM.BatteryTools as bt

import vapor.config as config
import vapor.helper as helper
from vapor.models import *

log = logging.getLogger("vapor")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~ SIMULATE SYSTEMS ~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class GenericSystemSimulator():
    def __init__(self,
                 tech, param_grid,
                 aggregate_region,
                 region, resource_file,
                 probe_divisors=[1],
                 opt_var=None, cambium_df=None,
                 analysis_period=None, construction_year=None,
                 buildingload=None,
                 goal_type=None, goal_pct=None,
                 workers=config.PROCESS_WORKERS):

        assert tech in ['pv','wind']

        self.param_grid = param_grid
        self.tech = tech
        self.opt_var = opt_var
        self.resource_file = resource_file
        self.region = region
        self.aggregate_region = aggregate_region
        self.probe_divisors = probe_divisors

        self.buildingload=buildingload
        self.goal_type=goal_type
        self.goal_pct=goal_pct

        self.workers = workers

        self.output_verbosity = 0

        if isinstance(cambium_df, pd.DataFrame):
            self.cambium = Cambium(cambium_df, aggregate_region, region, resource_file, analysis_period, construction_year)
        else:
            log.info('WARNING... no cambium dataframe passed to ParametricSystemSimulator, system optimization will only be available for economic metrics.')
            self.cambium = None

        # --- Initialize empty results ---
        self.results_df = None

    def _pysam_pipeline(self, system_config):

        if self.tech == 'pv':
            model = PVMerchantPlant(system_config, self.resource_file, self.cambium, self.region, self.buildingload)

        elif self.tech == 'wind':
            model = WindMerchantPlant(system_config, self.resource_file, self.cambium, self.region, self.buildingload)

        model.execute_all()
        outputs = model.outputs
        return outputs


    def _cambium_pipeline(self, generation):
        """
        Calculate average annual impact through 2050 for cambium variables.
        For annual results, see get_cambium_ts(). #TODO
        """
        
        # --- convert generation from str representation to list ---
        if isinstance(generation, str):
            generation = eval(generation)

        # --- convert generation to numpy array ---
        # NOTE: the commented out code below was causing issues in original values; essentially the 
        # "self.cambium.calc_cambium_lifetime_product(generation, var)" in the for loop calls
        # "self.calc_cambium_lifetime_rev(gen, var)" within models, which _also_ scales the values by 1,000
        # in essence we were double scaling some values
        # generation = np.array(generation) / 1000  # kW to MWh  
        generation = generation[:, np.newaxis] #wide to tall

        outputs = {}

        for var in self.cambium.variables:

            # --- calc lifetime values ---
            lifetime_product = self.cambium.calc_cambium_lifetime_product(generation, var)

            # --- store in dict ---
            outputs[f"{var}_sum"] = lifetime_product.sum()
            outputs[f"{var}_per_kwh"] = lifetime_product.sum() / generation.sum()
            # outputs[f"{var}_annual"] = lifetime_product.sum() / analysis_period

            # --- calc annual sums ---
            year_end = 0
            for y in range(self.cambium.start_year, self.cambium.retirement_year):
                outputs[f"{var}_{y}"] = lifetime_product[year_end:year_end+8760].sum()
                year_end += 8760

        return outputs


    def _base_worker(self, system_config):
        """
        Worker for multiprocessing execution of PySAM parametric analysis.

        Parameters
        ----------
        system_config : dict
            Dictionary of variables, provided by ParametricSystemDesigner().

        Returns
        -------
        output : Dict
            Dictionary containing the 'system_config' along with the desired outputs.
        """
        output = self._pysam_pipeline(system_config)

        # --- Compute average annual cambium values ---
        if self.cambium != None:
            generation = output['lifetime_gen_profile']
            try:
                assert len(generation) == 8760 * self.cambium.analysis_period
            except Exception as e:
                breakpoint()
            cambium_outputs = self._cambium_pipeline(generation)
        else:
            cambium_outputs = {}

        # --- Add all dicts together ---
        output = {**output, **cambium_outputs}
        return output

    
    def _check_if_maximizing(self, specific_opt_var):
        """Check if a particular optimization variable should be optimized as a minimum or maximum."""

        assert isinstance(specific_opt_var, str)

        if specific_opt_var in ['capacity_factor', 'annual_energy', 'kwh_per_kw',
                                'flip_actual_irr', 'project_return_aftertax_irr', 'project_return_aftertax_npv',
                                'cambinum_capacity_value',
                                'cambium_busbar_energy_value',
                                'cambium_enduse_energy_value',
                                'cambium_as_value',
                                'cambium_portfolio_value',
                                'cambium_grid_value']:
            return True
        
        elif specific_opt_var in ['lcoe_real','lcoe_nom',
                                  'marginal_cost_mwh', 'adjusted_installed_cost',
                                  'cambium_co2_rate_avg',
                                  'cambium_co2_rate_marg',
                                  'cambium_co2_rate_lrmer']:#,
                                #   'lifetime_cambium_co2_rate_lrmer']:
            return False
        
        else:
            log.error(f"{specific_opt_var} not in self._check_if_maximizing")
    

class BayesianSimulatorAddon():
    """Add on for Bayesian optimization"""

    def _force_discrete_bayesian_params(self, system_config, discrete_params=config.DISCRETE_PARAMS):
        """Force certain paramters to be discrete during bayesian optimization."""
        for k, v in system_config.items():
            if k in discrete_params:
                system_config[k] = int(v)
        return system_config
    
    def _create_output_metrics(self, output):
        output['lifetime_output_mwh'] = output['lifetime_gen_profile'].sum() / 1000
        output['marginal_cost_mwh'] = output['project_return_aftertax_npv'] / output['lifetime_output_mwh'] * -1

        for cambium_var in self.cambium.variables:
            output[f"lifetime_{cambium_var}"] = self.cambium.calc_cambium_lifetime_sum(gen=output['lifetime_gen_profile'], var=cambium_var)

        output['grid_value_per_mwh'] = output['lifetime_cambium_grid_value'] / output['lifetime_output_mwh']
        output['lifetime_cambium_co2_rate_avg_mwh'] = output['lifetime_cambium_co2_rate_avg'] / output['lifetime_output_mwh']
        output['lifetime_cambium_co2_rate_lrmer_mwh'] = output['lifetime_cambium_co2_rate_lrmer'] / output['lifetime_output_mwh']
        return output

    def _worker_return_score(self, **kwargs):

        assert isinstance(self.opt_var, str)

        # --- convert keyword args provided by bayesian opt into a dict of params ---
        bayes_grid = {**kwargs}

        # --- force discrete params ---
        bayes_grid = self._force_discrete_bayesian_params(bayes_grid)

        # --- convert to nested dict ---
        system_config = self._unflatten_param_grid(bayes_grid)
        
        # --- force some variables dependent on others ---
        # if self.tech == 'pv':
        #     if system_config['SystemDesign']['subarray1_track_mode'] in [1, 2, 3, 4]: #if not fixed tilt
        #         system_config['SystemDesign']['subarray1_tilt'] = 0
        #         system_config['SystemDesign']['subarray1_backtrack'] = 0

        # --- Get output dict ---
        output = self._base_worker(system_config)

        # --- Create new outputs ---
        output = self._create_output_metrics(output)

        # --- Fetch score ---
        if self.opt_var in self.cambium.variables:
            score = output[f"{self.opt_var}_sum"]
        else:
            score = output[self.opt_var]

        # --- Check if goal is met ---
        score = self._check_if_goal_met(score=score, output=output)

        # --- Check how to deal with nans in score ---
        if np.isnan(score):
            score =  -100

        # --- Check if we are maximizing or minimizing ---
        if self._check_if_maximizing(self.opt_var):
            return score
        else:  # return negative if minimizing
            return -1 * score

    def _flatten_param_grid(self):
        constant_items = [] #if value is numeric
        bayes_items = [] #if value is tuple

        for k_module, v_module in self.param_grid.items():
            for k_sub, v_sub in v_module.items():
                if isinstance(v_sub, Iterable): #min max tuple if iterable
                    bayes_items.append((f"{k_module}#{k_sub}", v_sub))
                else:
                    constant_items.append((f"{k_module}#{k_sub}", v_sub))
        
        self.constant_grid = dict(constant_items)
        self.bayes_grid = dict(bayes_items)

    def _unflatten_param_grid(self, bayes_grid):
        
        unflattened_param_grid = {}
        flattened_param_grid = dict(list(bayes_grid.items()) + list(self.constant_grid.items()))

        for k_flat, v_flat in flattened_param_grid.items():
            module, sub = k_flat.split('#')
            
            if module not in unflattened_param_grid.keys():
                unflattened_param_grid[module] = {}
            
            unflattened_param_grid[module][sub] = v_flat
        
        return unflattened_param_grid

    def _nested_param_grid_to_df(self, param_grid):
        "Flatten a nested param dictionary, only keeping bottom level to add to df as columns."
        param_grid_out = {}
        for k_module, v_module in param_grid.items():
            for k_sub, v_sub in v_module.items():
                param_grid_out[k_sub] = v_sub
        return param_grid_out

    def optimize(self):
        
        self.cambium.clean()

        self._flatten_param_grid()
        
        # --- initialize optimizer ---
        bounds_transformer = SequentialDomainReductionTransformer(eta=0.95, gamma_osc=0.95)
        optimizer = BayesianOptimization(
            f=self._worker_return_score,
            pbounds=self.bayes_grid,
            random_state=1,
            verbose=1,
            bounds_transformer=bounds_transformer,
        )
        
        # --- probe largest system config ---
        for divisor in self.probe_divisors:
            if self.tech == 'pv':
                probe_dict = self.bayes_grid.copy()
                #probe_dict['SystemDesign#subarray1_track_mode'] = 1
                probe_dict['SystemDesign#subarray1_azimuth'] = 180
                probe_dict['SystemDesign#subarray1_tilt'] = float(self.resource_file.split('/')[-1].split('_')[1])
                probe_dict['SystemDesign#dc_ac_ratio'] = 1.2
                if 'SystemDesign#system_capacity' in probe_dict.keys():
                    probe_dict['SystemDesign#system_capacity'] = np.max(probe_dict['SystemDesign#system_capacity']) / divisor
                if 'BatteryTools#desired_power' in probe_dict.keys():
                    probe_dict['BatteryTools#desired_power'] = np.max(probe_dict['BatteryTools#desired_power']) / divisor
                    probe_dict['BatteryTools#desired_capacity'] = np.max(probe_dict['BatteryTools#desired_capacity'])
            elif self.tech == 'wind':
                probe_dict = self.bayes_grid.copy()
                probe_dict['Turbine#wind_turbine_hub_ht'] = 100
                probe_dict['Turbine#turbine_class'] = 7
                if 'Farm#system_capacity' in probe_dict.keys():
                    probe_dict['Farm#system_capacity'] = np.max(probe_dict['Farm#system_capacity'])  / divisor
                if 'BatteryTools#desired_power' in probe_dict.keys():
                    probe_dict['BatteryTools#desired_power'] = np.max(probe_dict['BatteryTools#desired_power'])  / divisor
                    probe_dict['BatteryTools#desired_capacity'] = np.max(probe_dict['BatteryTools#desired_capacity'])
            optimizer.probe(params=probe_dict, lazy=False)
        
        # --- run optimizer ---
        optimizer.maximize(
            init_points=config.BAYES_INIT_POINTS,
            n_iter=config.BAYES_ITER,
            acq=config.BAYES_ACQ_FUNC,
            **config.BAYES_KWARGS
        )

        # --- rerun best system with no battery ---
        best_params = optimizer.max['params'] 
        if 'BatteryTools#desired_capacity' in best_params.keys(): #rerun system without battery
            if (best_params['BatteryTools#desired_capacity'] > 0) | (best_params['BatteryTools#desired_power'] > 0):
                best_params['BatteryTools#desired_capacity'] = 0
                best_params['BatteryTools#desired_power'] = 0
                optimizer.probe(params=best_params, lazy=False)

        # --- best score ---
        best_score = optimizer.max['target']  # currently unused
        if self._check_if_maximizing(self.opt_var):
            self.best_score = best_score
        else:
            self.best_score = -1 * best_score

        # --- access best params ---
        self.best_params = optimizer.max['params']

        # --- force discrete params ---
        self.best_params = self._force_discrete_bayesian_params(self.best_params)

        # --- convert to nested dict ---
        self.best_params = self._unflatten_param_grid(self.best_params)

        # --- rerun best params ---
        output = self._base_worker(self.best_params)

        # --- Create new outputs ---
        output = self._create_output_metrics(output)

        # --- flatten param grid for df ---
        df_param_grid = self._nested_param_grid_to_df(self.best_params)

        # --- combine flattened param grid with output results ---
        dict_for_df = {**output, **df_param_grid}

        # --- Convert any iterables in dict to str representations ---
        numpy_converted = []
        list_converted = []
        for k,v in dict_for_df.items():
            if isinstance(v, (str, int, float)):
                continue
            elif isinstance(v, (np.ndarray, np.generic)):
                dict_for_df[k] = str(v)
                numpy_converted.append(k)
            elif isinstance(v, (list, tuple)):
                dict_for_df[k] = str(v)
                list_converted.append(k)

        # --- convert to df ---
        self.best_df = pd.DataFrame(dict_for_df, index=[self.opt_var])

        # --- convert columns back to iterables ---
        for c in self.best_df.columns:
            try:
                if c in numpy_converted:
                    self.best_df[c] = [np.fromstring(i[1:-1], dtype=np.int, sep=' ') for i in list(self.best_df[c])]
                elif c in list_converted:
                    self.best_df[c] = [ast.literal_eval(i) for i in list(self.best_df[c])]
                else:
                    continue
            except Exception as e:
                # log.warning(f'Warning! Error converting {c} back to iterable representation')
                pass

        # --- add entire param grid ---
        self.best_df['system_config'] = [self.best_params]

class FixedCapacityAddon():

    def _check_if_goal_met(self, score, output):
        """Needed for compatability with _worker_return_score."""
        return score #always true

class MeetGoalAddon():

    def _check_if_goal_met(self, score, output):
        """Check if goal is met on an annual basis (such as MWh of generation or CO2 savings) is met."""

        if self.goal_type == 'annual_recs':
            value_without_sys = self.buildingload.annual_load_kwh
            value_with_sys = value_without_sys - output['lifetime_gen_profile'][0:8760].sum()

        elif self.goal_type == 'hourly_energy':
            value_without_sys = self.buildingload.as_array().sum()
            fy_gen_profile = output['lifetime_gen_profile'][0:8760]
            value_with_sys = self.buildingload.fy_hourly_offset(fy_gen_profile)

        elif self.goal_type == 'hourly_co2':
            mask = (self.cambium.cambium_df['variable'] == 'cambium_co2_rate_avg')
            fy_gen_profile = output['lifetime_dispatch_profile'][0:8760]
            fy_co2_profile = self.cambium.cambium_df.loc[mask,'value'][0:8760].values.astype(float) / 1000 #convert from float16 to float64
            value_without_sys = self.buildingload.fy_bau_co2(fy_co2_profile)
            value_with_sys = self.buildingload.fy_mitigated_co2(fy_co2_profile, fy_gen_profile)

        if value_with_sys < 0: #if new co2 value is negative
            pct_fulfilled = ((abs(value_with_sys) / value_without_sys) + 1) * 100
        else:
            pct_fulfilled = (1 - (value_with_sys / value_without_sys)) * 100
        pct_fulfilled = round(pct_fulfilled, 1) #ie 99.999 = 100

        log.info(f"{self.region} -- pct_fulfilled: {pct_fulfilled} score: {score}")
        if pct_fulfilled >= self.goal_pct:
            return score
        else:
            if self._check_if_maximizing(self.opt_var):
                return -1e100
            else:
                return 1e100

        

