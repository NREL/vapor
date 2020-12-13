import pickle
import os
from vapor.models import WindMerchantPlant, PVMerchantPlant, Cambium, BuildingLoad
from vapor import datafetcher, systemdesigner, load_cambium_data, FixedCapacityMerchantPlant
import vapor

def test_wind_no_storage():
    system_config = {'BatteryTools': {'desired_capacity': 0, 'desired_power': 0, 'desired_voltage': 500}, 'Turbine': {'wind_turbine_hub_ht': 80, 'turbine_class':7}, 'Farm': {'system_capacity': 100000}, 'Resource': {'wind_resource_model_choice': 0}, 'Lifetime': {'analysis_period': 25, 'system_use_lifetime_output': 1}, 'PriceSignal': {'mp_enable_energy_market_revenue': 0,
                                                                                                                                                                                                                                                                                                                            'forecast_price_signal_model': 1, 'mp_enable_ancserv1': 0, 'mp_enable_ancserv2': 0, 'mp_enable_ancserv3': 0, 'mp_enable_ancserv4': 0}, 'BatterySystem': {'en_batt': 1, 'batt_meter_position': 1}, 'BatteryDispatch': {'batt_dispatch_auto_can_charge': 1, 'batt_dispatch_auto_can_clipcharge': 1, 'batt_dispatch_auto_can_gridcharge': 1, 'batt_dispatch_choice': 1}}
    resource_file = 'data/PySAM Downloaded Weather Files/-89.578_39.394_wtk_60_2012.srw'

    cambium_df = datafetcher.load_cambium_data(aggregate_region='inter', scenario='StdScen20_MidCase')
    cambium = Cambium(cambium_df, 'inter', 'western', resource_file)
    cambium.clean()

    model = WindMerchantPlant(system_config, resource_file, cambium, 'texas')
    model.execute_all()
    print(f"wind no storage, irr {model.outputs['project_return_aftertax_irr']}, npv {model.outputs['project_return_aftertax_npv']}")


def test_wind_storage():
    system_config = {'BatteryTools': {'desired_capacity': 4, 'desired_power': 1000, 'desired_voltage': 500}, 'Turbine': {'wind_turbine_hub_ht': 100, 'turbine_class':7}, 'Farm': {'system_capacity': 100000}, 'Resource': {'wind_resource_model_choice': 0}, 'Lifetime': {'analysis_period': 25, 'system_use_lifetime_output': 1}, 'PriceSignal': {'mp_enable_energy_market_revenue': 0,
                                                                                                                                                                                                                                                                                                                                  'forecast_price_signal_model': 1, 'mp_enable_ancserv1': 0, 'mp_enable_ancserv2': 0, 'mp_enable_ancserv3': 0, 'mp_enable_ancserv4': 0}, 'BatterySystem': {'en_batt': 1, 'batt_meter_position': 1}, 'BatteryDispatch': {'batt_dispatch_auto_can_charge': 1, 'batt_dispatch_auto_can_clipcharge': 1, 'batt_dispatch_auto_can_gridcharge': 1, 'batt_dispatch_choice': 1}}
    resource_file = 'data/PySAM Downloaded Weather Files/-89.578_39.394_wtk_60_2012.srw'

    cambium_df = datafetcher.load_cambium_data(aggregate_region='inter', scenario='StdScen20_MidCase')
    cambium = Cambium(cambium_df, 'inter', 'western', resource_file)
    cambium.clean()

    model = WindMerchantPlant(system_config, resource_file, cambium, 'western')
    model.execute_all()
    print(f"wind storage, irr {model.outputs['project_return_aftertax_irr']}, npv {model.outputs['project_return_aftertax_npv']}")


def test_pv_no_storage():
    system_config = {'BatteryTools': {'desired_capacity': 0.0, 'desired_power': 0.0, 'desired_voltage': 500}, 'SystemDesign': {'dc_ac_ratio': 1.2, 'subarray1_azimuth': 180.0, 'subarray1_tilt': 0, 'subarray1_track_mode': 1, 'system_capacity': 100000, 'subarray2_enable': 0, 'subarray3_enable': 0, 'subarray4_enable': 0, 'subarray1_nstrings': 26881, 'inverter_count': 105}, 'Lifetime': {'analysis_period': 25, 'system_use_lifetime_output': 1},
                     'PriceSignal': {'mp_enable_energy_market_revenue': 0, 'forecast_price_signal_model': 1, 'mp_enable_ancserv1': 0, 'mp_enable_ancserv2': 0, 'mp_enable_ancserv3': 0, 'mp_enable_ancserv4': 0}, 'BatterySystem': {'en_batt': 1, 'batt_meter_position': 1}, 'BatteryDispatch': {'batt_dispatch_auto_can_charge': 1, 'batt_dispatch_auto_can_clipcharge': 1, 'batt_dispatch_auto_can_gridcharge': 1, 'batt_dispatch_choice': 1}}
    resource_file = 'data/PySAM Downloaded Weather Files/-89.578_39.394_psm3_60_tmy.csv'

    cambium_df = datafetcher.load_cambium_data(aggregate_region='inter', scenario='StdScen20_MidCase')
    cambium = Cambium(cambium_df, 'inter', 'western', resource_file)
    cambium.clean()

    model = PVMerchantPlant(system_config, resource_file, cambium, 'western')
    model.execute_all()
    print(f"PV no storage, irr {model.outputs['project_return_aftertax_irr']}, npv {model.outputs['project_return_aftertax_npv']}")

def test_pv_storage():
    system_config = {'BatteryTools': {'desired_capacity': 4, 'desired_power': 1000, 'desired_voltage': 500}, 'SystemDesign': {'dc_ac_ratio': 1.2, 'subarray1_azimuth': 180.0, 'subarray1_tilt': 0, 'subarray1_track_mode': 1, 'system_capacity': 100000, 'subarray2_enable': 0, 'subarray3_enable': 0, 'subarray4_enable': 0, 'subarray1_nstrings': 26881, 'inverter_count': 105}, 'Lifetime': {'analysis_period': 25, 'system_use_lifetime_output': 1},
                     'PriceSignal': {'mp_enable_energy_market_revenue': 0, 'forecast_price_signal_model': 1, 'mp_enable_ancserv1': 0, 'mp_enable_ancserv2': 0, 'mp_enable_ancserv3': 0, 'mp_enable_ancserv4': 0}, 'BatterySystem': {'en_batt': 1, 'batt_meter_position': 1}, 'BatteryDispatch': {'batt_dispatch_auto_can_charge': 1, 'batt_dispatch_auto_can_clipcharge': 1, 'batt_dispatch_auto_can_gridcharge': 1, 'batt_dispatch_choice': 1}}
    resource_file = 'data/PySAM Downloaded Weather Files/-89.578_39.394_psm3_60_tmy.csv'

    cambium_df = datafetcher.load_cambium_data(aggregate_region='inter', scenario='StdScen20_MidCase')
    cambium = Cambium(cambium_df, 'inter', 'western', resource_file)
    cambium.clean()

    model = PVMerchantPlant(system_config, resource_file, cambium, 'western')
    model.execute_all()

    print(f"PV storage, irr {model.outputs['project_return_aftertax_irr']}, npv {model.outputs['project_return_aftertax_npv']}")

def test_optimize():
    grid = systemdesigner.BayesianSystemDesigner(
                                        tech='pv',
                                        re_capacity_mw=100,
                                        batt_capacity_mw=0,
                                        verbose=False
                                    )
    param_grid = grid.get_param_grid()

    resource_file = 'data/PySAM Downloaded Weather Files/-89.578_39.394_psm3_60_tmy.csv'

    cambium_df = load_cambium_data(aggregate_region='census_reg', scenario='StdScen20_MidCase')

    vapor.config.BAYES_INIT_POINTS=2
    vapor.config.BAYES_ITER=3

    simulator = FixedCapacityMerchantPlant(
        param_grid=param_grid,
        tech='pv',
        aggregate_region='census_reg',
        region='MTN',
        opt_var='marginal_cost_mwh',
        resource_file=resource_file,
        cambium_df = cambium_df
    )

    simulator.simulate()

def test_optimize_lrmer():
    grid = systemdesigner.BayesianSystemDesigner(
                                        tech='pv',
                                        re_capacity_mw=100,
                                        batt_capacity_mw=0,
                                        verbose=False
                                    )
    param_grid = grid.get_param_grid()

    resource_file = 'data/PySAM Downloaded Weather Files/-89.578_39.394_psm3_60_tmy.csv'

    cambium_df = load_cambium_data(aggregate_region='census_reg', scenario='StdScen20_MidCase')

    vapor.config.BAYES_INIT_POINTS=2
    vapor.config.BAYES_ITER=3

    simulator = FixedCapacityMerchantPlant(
        param_grid=param_grid,
        tech='pv',
        aggregate_region='census_reg',
        region='MTN',
        opt_var='cambium_co2_rate_lrmer',
        resource_file=resource_file,
        cambium_df = cambium_df
    )

    simulator.simulate()


# def test_storage_unit():
    
#     system_config = {'BatteryTools': {'desired_capacity': 4, 'desired_power': 1000, 'desired_voltage': 500}, 'SystemDesign': {'dc_ac_ratio': 1.2, 'subarray1_azimuth': 180.0, 'subarray1_tilt': 0, 'subarray1_track_mode': 1, 'system_capacity': 100000, 'subarray2_enable': 0, 'subarray3_enable': 0, 'subarray4_enable': 0, 'subarray1_nstrings': 26881, 'inverter_count': 105}, 'Lifetime': {'analysis_period': 25, 'system_use_lifetime_output': 1},
#                      'PriceSignal': {'mp_enable_energy_market_revenue': 0, 'forecast_price_signal_model': 1, 'mp_enable_ancserv1': 0, 'mp_enable_ancserv2': 0, 'mp_enable_ancserv3': 0, 'mp_enable_ancserv4': 0}, 'BatterySystem': {'en_batt': 1, 'batt_meter_position': 1}, 'BatteryDispatch': {'batt_dispatch_auto_can_charge': 1, 'batt_dispatch_auto_can_clipcharge': 1, 'batt_dispatch_auto_can_gridcharge': 1, 'batt_dispatch_choice': 1}}
#     resource_file = 'data/PySAM Downloaded Weather Files/-89.578_39.394_psm3_60_tmy.csv'

#     cambium_df = datafetcher.load_cambium_data(aggregate_region='inter', scenario='StdScen20_MidCase')
#     cambium = Cambium(cambium_df, 'inter', 'western', resource_file)
#     cambium.clean()

#     model = PVMerchantPlant(system_config, resource_file, cambium, 'western')

#     # from models 'execute_all()' for Generic Plant Class
#     model.run_generator()
#     if model.load == None:
#         model.size_load()
#     model.make_gen_profile_no_batt() #8760 * analysis period of generation without battery
#     if model.storage: 
#         model.run_battery()
#     print(model.storage)
#     # print(f"PV no storage, irr {model.outputs['project_return_aftertax_irr']}, npv {model.outputs['project_return_aftertax_npv']}")
