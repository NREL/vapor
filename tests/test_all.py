import pickle
import os
from vapor.models import WindMerchantPlant, PVMerchantPlant, Cambium, BuildingLoad
from vapor import datafetcher

def test_wind_no_storage():
    system_config = {'BatteryTools': {'desired_capacity': 0, 'desired_power': 0, 'desired_voltage': 500}, 'Turbine': {'wind_turbine_hub_ht': 80, 'turbine_class':7}, 'Farm': {'system_capacity': 100000}, 'Resource': {'wind_resource_model_choice': 0}, 'Lifetime': {'analysis_period': 25, 'system_use_lifetime_output': 1}, 'PriceSignal': {'mp_enable_energy_market_revenue': 0,
                                                                                                                                                                                                                                                                                                                            'forecast_price_signal_model': 1, 'mp_enable_ancserv1': 0, 'mp_enable_ancserv2': 0, 'mp_enable_ancserv3': 0, 'mp_enable_ancserv4': 0}, 'BatterySystem': {'en_batt': 1, 'batt_meter_position': 1}, 'BatteryDispatch': {'batt_dispatch_auto_can_charge': 1, 'batt_dispatch_auto_can_clipcharge': 1, 'batt_dispatch_auto_can_gridcharge': 1, 'batt_dispatch_choice': 1}}
    resource_file = 'data/PySAM Downloaded Weather Files/-89.578_39.394_wtk_60_2012.srw'

    cambium_df = datafetcher.load_cambium_data(aggregate_region='inter', scenario='StdScen19_Mid_Case')
    cambium = Cambium(cambium_df, 'inter', 'western', resource_file)
    cambium.clean()

    model = WindMerchantPlant(system_config, resource_file, cambium, 'texas')
    model.execute_all()
    print(f"wind no storage, irr {model.outputs['project_return_aftertax_irr']}, npv {model.outputs['project_return_aftertax_npv']}")


def test_wind_storage():
    system_config = {'BatteryTools': {'desired_capacity': 4, 'desired_power': 1000, 'desired_voltage': 500}, 'Turbine': {'wind_turbine_hub_ht': 100, 'turbine_class':7}, 'Farm': {'system_capacity': 100000}, 'Resource': {'wind_resource_model_choice': 0}, 'Lifetime': {'analysis_period': 25, 'system_use_lifetime_output': 1}, 'PriceSignal': {'mp_enable_energy_market_revenue': 0,
                                                                                                                                                                                                                                                                                                                                  'forecast_price_signal_model': 1, 'mp_enable_ancserv1': 0, 'mp_enable_ancserv2': 0, 'mp_enable_ancserv3': 0, 'mp_enable_ancserv4': 0}, 'BatterySystem': {'en_batt': 1, 'batt_meter_position': 1}, 'BatteryDispatch': {'batt_dispatch_auto_can_charge': 1, 'batt_dispatch_auto_can_clipcharge': 1, 'batt_dispatch_auto_can_gridcharge': 1, 'batt_dispatch_choice': 1}}
    resource_file = 'data/PySAM Downloaded Weather Files/-89.578_39.394_wtk_60_2012.srw'

    cambium_df = datafetcher.load_cambium_data(aggregate_region='inter', scenario='StdScen19_Mid_Case')
    cambium = Cambium(cambium_df, 'inter', 'western', resource_file)
    cambium.clean()

    model = WindMerchantPlant(
        system_config, resource_file, cambium, 'western')
    model.execute_all()
    print(f"wind storage, irr {model.outputs['project_return_aftertax_irr']}, npv {model.outputs['project_return_aftertax_npv']}")


def test_pv_no_storage():
    system_config = {'BatteryTools': {'desired_capacity': 0.0, 'desired_power': 0.0, 'desired_voltage': 500}, 'SystemDesign': {'dc_ac_ratio': 1.2, 'subarray1_azimuth': 180.0, 'subarray1_tilt': 0, 'subarray1_track_mode': 1, 'system_capacity': 100000, 'subarray2_enable': 0, 'subarray3_enable': 0, 'subarray4_enable': 0, 'subarray1_nstrings': 26881, 'inverter_count': 105}, 'Lifetime': {'analysis_period': 25, 'system_use_lifetime_output': 1},
                     'PriceSignal': {'mp_enable_energy_market_revenue': 0, 'forecast_price_signal_model': 1, 'mp_enable_ancserv1': 0, 'mp_enable_ancserv2': 0, 'mp_enable_ancserv3': 0, 'mp_enable_ancserv4': 0}, 'BatterySystem': {'en_batt': 1, 'batt_meter_position': 1}, 'BatteryDispatch': {'batt_dispatch_auto_can_charge': 1, 'batt_dispatch_auto_can_clipcharge': 1, 'batt_dispatch_auto_can_gridcharge': 1, 'batt_dispatch_choice': 1}}
    resource_file = 'data/PySAM Downloaded Weather Files/-89.578_39.394_psm3_60_tmy.csv'

    cambium_df = datafetcher.load_cambium_data(aggregate_region='inter', scenario='StdScen19_Mid_Case')
    cambium = Cambium(cambium_df, 'inter', 'western', resource_file)
    cambium.clean()

    model = PVMerchantPlant(system_config, resource_file, cambium, 'western')
    model.execute_all()
    print(f"PV no storage, irr {model.outputs['project_return_aftertax_irr']}, npv {model.outputs['project_return_aftertax_npv']}")

def test_pv_storage():
    system_config = {'BatteryTools': {'desired_capacity': 4, 'desired_power': 1000, 'desired_voltage': 500}, 'SystemDesign': {'dc_ac_ratio': 1.2, 'subarray1_azimuth': 180.0, 'subarray1_tilt': 0, 'subarray1_track_mode': 1, 'system_capacity': 100000, 'subarray2_enable': 0, 'subarray3_enable': 0, 'subarray4_enable': 0, 'subarray1_nstrings': 26881, 'inverter_count': 105}, 'Lifetime': {'analysis_period': 25, 'system_use_lifetime_output': 1},
                     'PriceSignal': {'mp_enable_energy_market_revenue': 0, 'forecast_price_signal_model': 1, 'mp_enable_ancserv1': 0, 'mp_enable_ancserv2': 0, 'mp_enable_ancserv3': 0, 'mp_enable_ancserv4': 0}, 'BatterySystem': {'en_batt': 1, 'batt_meter_position': 1}, 'BatteryDispatch': {'batt_dispatch_auto_can_charge': 1, 'batt_dispatch_auto_can_clipcharge': 1, 'batt_dispatch_auto_can_gridcharge': 1, 'batt_dispatch_choice': 1}}
    resource_file = 'data/PySAM Downloaded Weather Files/-89.578_39.394_psm3_60_tmy.csv'

    cambium_df = datafetcher.load_cambium_data(aggregate_region='inter', scenario='StdScen19_Mid_Case')
    cambium = Cambium(cambium_df, 'inter', 'western', resource_file)
    cambium.clean()

    model = PVMerchantPlant(system_config, resource_file, cambium, 'western')
    model.execute_all()

    print(f"PV storage, irr {model.outputs['project_return_aftertax_irr']}, npv {model.outputs['project_return_aftertax_npv']}")
