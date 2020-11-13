#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 14:00:39 2020

@author: skoebric
"""

import os

# --- API KEYS ---
NREL_API_KEY = 'OJOhB72tEAwfSX6HF3AUpL6cEebh24cgdjbK7D1X'
NREL_API_EMAIL = 'sam.koebrich@NREL.gov'

# --- SCOPE / FINANCING ---
LAST_YEAR = 2050 #run analysis through this year
SYSTEM_LIFETIME=25
CAMBIUM_SCENARIO = 'StdScen19_Mid_Case'
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
PROCESS_WORKERS = int(os.cpu_count() / 2) - 2
THREAD_WORKERS = int(os.cpu_count() / 2) - 2

