# --- Imports ---
import logging
import random
import json
import os
import concurrent.futures as cf
import itertools
import io
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from collections.abc import Iterable

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# import PySAM.Pvsamv1 as pv
import PySAM.Pvwattsv7 as pv
import PySAM.Windpower as wp
import PySAM.Singleowner as so

import vapor.config as config

log = logging.getLogger("vapor")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~ DESIGN SYSTEMS FOR PySAM ~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class BayesianSystemDesigner():
    """
    Export Grid of Params with Lower and Upper bounds for Bayesian Optimization

    tech: ['pv', 'wind', 'either']
    re_capacity_mw: (min_system_size, max_system_size) OR int(system_size)
    batt_capacity_mw: (min_system_size, max_system_size) OR int(system_size)
    batt_capacity_mwh: (min_system_size, max_system_size) OR int(system_size)
    """

    def __init__(self, tech,
                re_capacity_mw, #float, or tuple with lower and upper bounds
                batt_capacity_mw=0, #float, or tuple with lower and upper bounds
                batt_duration=[0,2,4], #0, 2, or 4hr
                verbose=True,
                params=None):
        
        if verbose:
            log.info('\n')
            log.info(f'Initializing BayesianSystemDesigner for {tech}')
        
        self.tech = tech

        if isinstance(re_capacity_mw, (float, int)):
            self.re_capacity_kw=re_capacity_mw * 1000  # pysam takes this as kw
        elif isinstance(re_capacity_mw, (tuple, list)):
            self.re_capacity_kw = (re_capacity_mw[0] * 1000, re_capacity_mw[1] * 1000)
        
        if isinstance(batt_capacity_mw, (float, int)):
            self.batt_capacity_kw = batt_capacity_mw * 1000  # pysam takes this as kw
        elif isinstance(batt_capacity_mw, (tuple, list)):
            self.batt_capacity_kw = (batt_capacity_mw[0] * 1000, batt_capacity_mw[1] * 1000)
        else:
            self.batt_capacity_kw = 0

        if isinstance(batt_duration, (float, int)):
            self.batt_duration = batt_duration
        elif isinstance(batt_duration, (tuple, list)):
            self.batt_duration = np.array(batt_duration)

        self.storage = False
        if isinstance(self.batt_capacity_kw, (int, float)):
            if self.batt_capacity_kw > 0:
                self.storage = True
        elif isinstance(self.batt_capacity_kw, (tuple)):
            if max(self.batt_capacity_kw) > 0:
                self.storage = True

        # --- Initiate Generator and assign params if not passed ---
        if tech == 'pv':
            self.gen = pv.default('PVWattsSingleOwner')
            self.default_params = self.gen.SystemDesign.export()

            if params == None:  # assign default grid of solar params
                self.param_grid = {
                    'SystemDesign':{
                        'system_capacity': self.re_capacity_kw,
                        'subarray1_track_mode': 1, #np.array([0, 1, 2, 4]), #1 = fixed
                        'subarray1_tilt': np.arange(0, 90, 10),
                        'subarray1_azimuth': np.arange(80, 280, 10),
                        'dc_ac_ratio': np.arange(0.8, 1.3, 0.1),
                    }
                }

        elif tech == 'wind':
            self.gen = wp.default('WindPowerSingleOwner')
            self.default_params = self.gen.Turbine.export()

            if params == None:  # assign default grid of wind params
                self.param_grid = {
                    'Turbine': {'wind_turbine_hub_ht': np.array([60, 170]), 'turbine_class': np.array([1, 10])},
                    'Farm': {'system_capacity': self.re_capacity_kw},
                }
        
        else: raise NotImplementedError(f'Please write a wrapper to account for the new technology type {tech}')


        # --- Add Battery Params ---
        if self.storage:
            self.param_grid['BatteryTools'] = {'desired_power': self.batt_capacity_kw,
                                               'desired_capacity': self.batt_duration,
                                               'desired_voltage':500}
            self.param_grid['BatterySystem'] = {'en_batt':1,
                                                'batt_meter_position':0}
        else:
            self.param_grid['BatterySystem'] = {'en_batt': 0}


    def _bound_params(self):
        """Take paramters expressed in array form, and convert them into tuples with lower and upper bounds."""

        for k_module, v_module in self.param_grid.items():
            for sub_k, sub_v in v_module.items():
                if isinstance(sub_v, Iterable): #min max tuple if iterable
                    self.param_grid[k_module][sub_k] = (min(sub_v), max(sub_v))
                else:
                    self.param_grid[k_module][sub_k] = sub_v
        
    def get_param_grid(self, load=None):
        self._bound_params()
        return self.param_grid
