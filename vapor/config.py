#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 14:00:39 2020

@author: skoebric
"""

import os

# --- API KEYS ---
NREL_API_KEY = 'ORchxEHd3f2DXVV8QWRJWRKpqzfD1jUqaBUnr845'#'OJOhB72tEAwfSX6HF3AUpL6cEebh24cgdjbK7D1X'
NREL_API_EMAIL = 'thomas.bowen@NREL.gov'#'sam.koebrich@NREL.gov'

# --- SCOPE / FINANCING ---
LAST_YEAR = 2050 #run analysis through this year
SYSTEM_LIFETIME=25
CAMBIUM_SCENARIO = 'StdScen20_MidCase'
COST_SCENARIO = 'Mid' 
INFLATION = 0.025
DEGRADATION = 0.05
RATE_ESCALATION = 0.015
INTEREST_RATE = 0.005
DEBT_OPTION = 0 # 0 for debt percent, 1 for dscr
DEBT_PERCENT = 0.8
DSCR = 1.4
BATT_CLEARED_DERATE = 0.97
DISCOUNT_RATE = 0.064


# --- BAYESIAN OPTIMIZATION ---
DISCRETE_PARAMS = [ #non-continuous parameters that must be evaluated discretely (i.e. array type can't be 1.5)
        #'SystemDesign#subarray1_track_mode',
        'BatteryTools#desired_capacity',
        'Turbine#turbine_class'
        ]

BAYES_INIT_POINTS = 5
BAYES_ITER = 75
BAYES_ACQ_FUNC = 'ucb' #bayesian acquisition function
BAYES_KWARGS = {'kappa':20} #higher kappa (i.e. 10) favors exploration, WITHIN the sequential domain reduction

# --- RESOURCE DATA ---
RESOURCE_YEAR = 'tmy'
RESOURCE_INTERVAL_MIN = 60 #minutes

# --- MULTIPROCESSING ---
PROCESS_WORKERS = 8#int(os.cpu_count() / 2)
THREAD_WORKERS = 8#int(os.cpu_count() / 2)

# --- RESOURCE SAMPLING --- # NEW: THOMAS BOWEN 03182021
SAMPLING_BEST = True # will choose to pick the 'best' resource site in a region rather than the centroid of that region

# --- SPOT CHECKING VALUES --- # NEW: THOMAS BOWEN 05222021
# when these regions are processed by models Wind and PV Generic plants, the market profile and system configs will be saved to be checked in SAM
# see the notebook 'rerunning_results_with_new_resource_selection' for how these areas were selected
REGIONS_SPOT_CHECK_WIND = ['p4', 'p33', 'p56', 'p85', 'texas'] 
REGIONS_SPOT_CHECK_PV = ['p10',  'p13',  'p29', 'p33', 'p91', 'p92', 'p94', 'p129', 'p130', 'p134'] 