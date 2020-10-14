# --- Imports ---
import logging
import random
import json
import time
import os
import concurrent.futures as cf
import itertools
import io
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# import PySAM.Pvsamv1 as pv
import PySAM.Pvwattsv7 as pv
import PySAM.Singleowner as so

# --- Absolute Imports ---
import vapor.datafetcher as datafetcher
import vapor.systemdesigner as systemdesigner
import vapor.systemsimulator as systemsimulator
import vapor.visualizer as visualizer
import vapor.helper as helper
import vapor.config as config
import vapor.models as models

log = logging.getLogger("vapor")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~ 1) Find Best System by Region ~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class RegionalPipeline():
    
    def __init__(self, 
                tech, 
                aggregate_region,
                aggregate_func='per_kwh',
                scenario=config.CAMBIUM_SCENARIO,
                cambium_last_year=config.LAST_YEAR,
                re_capacity_mw=100,
                batt_capacity_mw=25,
                batt_duration=4):
        
        assert tech in ['pv', 'wind', 'either']
        assert aggregate_region in ['pca','pca_res','census_reg','inter','state'] #only these regions supported
        assert isinstance(cambium_last_year, int)

        self.tech = tech
        self.aggregate_region = aggregate_region
        self.last_year = cambium_last_year
        self.re_capacity_mw = re_capacity_mw
        self.batt_capacity_mw = batt_capacity_mw
        self.batt_duration = batt_duration
        self.aggregate_func = aggregate_func
        self.scenario = scenario

        # --- Initialize Attributes ---
        self.param_grid = None
        self.cambium_df = None
        self.resource_file_dict = None
        self.geometry = None
        self.best_systems = None

    def setup(self):

        # --- Get Resource Files for Region ---
        if self.resource_file_dict == None:

            # --- Get lookup of centroids ---
            CentroidsLookup = datafetcher.GetCentroidOfRegions(aggregate_region=self.aggregate_region)
            CentroidsLookup.find_centroids()
            self.centroids_dict = CentroidsLookup.centroids_lookup
            self.geometry = CentroidsLookup.region_shape
            
            fetcher = datafetcher.FetchResourceFiles(tech=self.tech)

            # --- fetch resource data for lat/lon tuples ---
            fetcher.fetch(self.centroids_dict.values()) 

            # --- convert tuples dict to aggregate region dict ---
            tuple_dict = fetcher.resource_file_paths_dict #keys=region, values=centroid resource file path
            self.resource_file_dict = {k:tuple_dict[v] for k,v in self.centroids_dict.items()}

        # --- Load Cambium Data ---
        if not isinstance(self.cambium_df, pd.DataFrame):
            self.cambium_df = datafetcher.load_cambium_data(aggregate_region=self.aggregate_region, scenario=self.scenario)

        # --- Get Parameter Grid for Tech ---
        if self.param_grid == None:
            grid = systemdesigner.BayesianSystemDesigner(tech=self.tech,
                                                         re_capacity_mw=self.re_capacity_mw,
                                                         batt_capacity_mw=self.batt_capacity_mw,
                                                         batt_duration=self.batt_duration)
            self.param_grid = grid.get_param_grid()
    
    def _worker(self, job):

        # --- unpack job ---
        region, resource_fp, opt_var = job
        
        simulator = systemsimulator.FixedCapacityMerchantPlant(
                                            param_grid=self.param_grid,
                                            tech=self.tech,
                                            aggregate_region=self.aggregate_region,
                                            aggregate_func=self.aggregate_func,
                                            region=region,
                                            opt_var=opt_var,
                                            resource_file=resource_fp,
                                            cambium_df=self.cambium_df)
        simulator.simulate()
        best_df = simulator.best_df
        best_df['region'] = region
        best_df['tech'] = self.tech
        best_df['opt_var'] = opt_var
        
        return best_df


    def run(self, opt_vars):
        
        # --- Construct list of jobs ---
        jobs = []
        for region, resource_fp in self.resource_file_dict.items():
            for opt_var in opt_vars:
                jobs.append((region, resource_fp, opt_var))
            
        # --- Run Jobs ---
        if config.PROCESS_WORKERS > 1: 
            log.info(f'....starting simulations for {len(jobs)} jobs with {config.PROCESS_WORKERS} process workers')
            results_list = []
            start = time.time()
            with cf.ProcessPoolExecutor(max_workers=config.PROCESS_WORKERS) as executor:
                checkpoint = max(1, int(len(jobs) * 0.05))

                # --- Submit to worker ---
                futures = [executor.submit(self._worker, job) for job in jobs]
                for f in cf.as_completed(futures):
                    results_list.append(f.result())
                    if len(results_list) % checkpoint == 0:
                        seconds_per_job = (time.time() - start) / len(results_list)
                        eta = (len(jobs) - len(results_list)) * seconds_per_job / 60 / 60
                        log.info(f"........finished optimizing job {len(results_list)} / {len(jobs)} {round(seconds_per_job, 2)} s/j ETA: {eta}")
        else:
            results_list = [self._worker(job) for job in jobs]

        self.best_systems = pd.concat(results_list, axis='rows')


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~ 2) Find Best For Lat/Lons Given Capacity ~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class ExistingPipeline():

    def __init__(self, 
                aggregate_region,
                scenario,
                optimization='Bayesian',
                aggregate_func='per_kwh',
                cambium_last_year=config.LAST_YEAR):
        
        assert optimization in ['Bayesian'] #PARAMETRIC DEPRECATED
        assert aggregate_region in ['pca','pca_res','census_reg','inter','state'] #only these regions supported
        assert isinstance(cambium_last_year, int)

        self.optimization = optimization
        self.last_year = cambium_last_year
        self.aggregate_region = aggregate_region
        self.scenario = scenario

        # --- Initialize Attributes ---
        self.cambium_df = None
        self.geometry = None
        self.best_systems = None

    def setup(self, df):

        assert isinstance(df, pd.DataFrame)
        assert 'longitude' in df.columns
        assert 'latitude' in df.columns
        assert 'tech' in df.columns
        assert 're_capacity_mw' in df.columns
        assert 'batt_capacity_mw' in df.columns

        # --- Load Cambium Data ---
        if not isinstance(self.cambium_df, pd.DataFrame):
            self.cambium_df = datafetcher.load_cambium_data(aggregate_region=self.aggregate_region, scenario=self.scenario)


    def fetch_resource(self, df):

        # --- Get lookup of centroids ---
        CentroidsLookup = datafetcher.CoordsToRegionCentroid(aggregate_region=self.aggregate_region)
        df = CentroidsLookup.match_centroids(df)

        return df
    
    def _worker(self, row, opt_var):
 
        # --- Design System ---
        grid = systemdesigner.BayesianSystemDesigner(
                                                tech=row['tech'],
                                                re_capacity_mw=row['re_capacity_mw'],
                                                batt_capacity_mw=row['batt_capacity_mw'],
                                                verbose=False)
        param_grid = grid.get_param_grid()

        simulator = systemsimulator.FixedCapacityMerchantPlant(
                                            param_grid=param_grid,
                                            tech=row['tech'],
                                            aggregate_region=self.aggregate_region,
                                            aggregate_func='per_kwh',
                                            region=row['region'],
                                            opt_var=opt_var,
                                            resource_file=row['resource_fp'],
                                            cambium_df=self.cambium_df,
                                            initial_probe=False,
                                            construction_year=row['ppa_estimated_signing_year'])

        simulator.simulate()
        best_df = simulator.best_df
        best_df.index = [row.name]
        row_out = pd.DataFrame(row).T
        out_df = pd.concat([row_out, best_df], axis='columns')
        
        return out_df

    def run(self, df, opt_vars):

        # --- Construct list of jobs ---
        for opt_var in opt_vars:
        
            # --- Run Jobs ---
            if config.PROCESS_WORKERS > 1: 
                self.results_list = []
                start = time.time()
                
                with cf.ProcessPoolExecutor(max_workers=config.PROCESS_WORKERS) as executor:
                    checkpoint = max(1, int(len(df) * 0.1))

                    # --- Submit to worker ---
                    futures = [executor.submit(self._worker, row, opt_var) for _, row in df.iterrows()]
                    
                    for f in cf.as_completed(futures):
                        self.results_list.append(f.result())
                        
                        if len(self.results_list) % checkpoint == 0:
                            seconds_per_job = (time.time() - start) / len(self.results_list)
                            eta = (len(df) - len(self.results_list)) * seconds_per_job / 60 / 60
                            log.info(f"........finished optimizing job {len(self.results_list)} / {len(df)} {round(seconds_per_job, 2)} s/j ETA: {eta}")
            else:
                self.results_list = [self._worker(row, opt_var) for _, row in df.iterrows()]

            self.best_systems = pd.concat(self.results_list, axis='rows')


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~ 3) Design System(s) to meet goal ~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class GoalPipeline():

    def __init__(self, 
                tech, 
                aggregate_region,
                aggregate_func='per_kwh',
                scenario=config.CAMBIUM_SCENARIO,
                cambium_last_year=config.LAST_YEAR,
                re_capacity_mw=[0,300],
                batt_capacity_mw=[0,300],
                batt_duration=[2,4],
                annual_load_mwh=50000,
                goal_type='CO2',
                goal_pct=70):
        
        assert tech in ['pv', 'wind', 'either']
        assert aggregate_region in ['pca','pca_res','census_reg','inter','state'] #only these regions supported
        assert isinstance(cambium_last_year, int)
        assert goal_type in ['hourly_energy', 'annual_recs', 'hourly_co2']
        assert isinstance(goal_pct, (float, int))
        assert 0 <= goal_pct <= 100

        self.tech = tech
        self.aggregate_region = aggregate_region
        self.last_year = cambium_last_year
        self.re_capacity_mw = re_capacity_mw
        self.batt_capacity_mw = batt_capacity_mw
        self.batt_duration = batt_duration
        self.aggregate_func = aggregate_func
        self.scenario = scenario
        self.annual_load_mwh = annual_load_mwh
        self.goal_type = goal_type
        self.goal_pct = goal_pct

        # --- Initialize Attributes ---
        self.param_grid = None
        self.cambium_df = None
        self.resource_file_dict = None
        self.geometry = None
        self.best_systems = None

    def setup(self):

        # --- Get Resource Files for Region ---
        if self.resource_file_dict == None:

            # --- Get lookup of centroids ---
            CentroidsLookup = datafetcher.GetCentroidOfRegions(aggregate_region=self.aggregate_region)
            CentroidsLookup.find_centroids()
            self.centroids_dict = CentroidsLookup.centroids_lookup
            self.geometry = CentroidsLookup.region_shape
            
            fetcher = datafetcher.FetchResourceFiles(tech=self.tech)

            # --- fetch resource data for lat/lon tuples ---
            fetcher.fetch(self.centroids_dict.values()) 

            # --- convert tuples dict to aggregate region dict ---
            tuple_dict = fetcher.resource_file_paths_dict #keys=region, values=centroid resource file path
            self.resource_file_dict = {k:tuple_dict[v] for k,v in self.centroids_dict.items()}

        # --- Load Cambium Data ---
        if not isinstance(self.cambium_df, pd.DataFrame):
            self.cambium_df = datafetcher.load_cambium_data(aggregate_region=self.aggregate_region, scenario=self.scenario)

        # --- Get Parameter Grid for Tech ---
        if self.param_grid == None:
            grid = systemdesigner.BayesianSystemDesigner(tech=self.tech,
                                                         re_capacity_mw=self.re_capacity_mw,
                                                         batt_capacity_mw=self.batt_capacity_mw,
                                                         batt_duration=self.batt_duration)
            self.param_grid = grid.get_param_grid()
        
        # --- Load BuildingLoad Profile ---
        self.buildingload = models.BuildingLoad()
        self.buildingload.load()
        self.buildingload.scale(self.annual_load_mwh)
    
    def _worker(self, job):

        # --- unpack job ---
        region, resource_fp, opt_var = job
        
        simulator = systemsimulator.MeetGoalMerchantPlant(
                                            param_grid=self.param_grid,
                                            tech=self.tech,
                                            aggregate_region=self.aggregate_region,
                                            aggregate_func=self.aggregate_func,
                                            region=region,
                                            opt_var=opt_var,
                                            resource_file=resource_fp,
                                            cambium_df=self.cambium_df,
                                            buildingload=self.buildingload,
                                            goal_type=self.goal_type,
                                            goal_pct=self.goal_pct)
        simulator.simulate()
        best_df = simulator.best_df
        best_df['region'] = region
        best_df['tech'] = self.tech
        best_df['opt_var'] = opt_var
        
        return best_df


    def run(self, opt_vars):
        
        # --- Construct list of jobs ---
        jobs = []
        for region, resource_fp in self.resource_file_dict.items():
            for opt_var in opt_vars:
                jobs.append((region, resource_fp, opt_var))
        jobs = reversed(jobs)
        # --- Run Jobs ---
        if config.PROCESS_WORKERS > 1: 
            log.info(f'....starting simulations for {len(jobs)} jobs with {config.PROCESS_WORKERS} process workers')
            results_list = []
            start = time.time()
            with cf.ProcessPoolExecutor(max_workers=config.PROCESS_WORKERS) as executor:
                checkpoint = max(1, int(len(jobs) * 0.05))

                # --- Submit to worker ---
                futures = [executor.submit(self._worker, job) for job in jobs]
                for f in cf.as_completed(futures):
                    results_list.append(f.result())
                    if len(results_list) % checkpoint == 0:
                        seconds_per_job = (time.time() - start) / len(results_list)
                        eta = (len(jobs) - len(results_list)) * seconds_per_job / 60 / 60
                        log.info(f"........finished optimizing job {len(results_list)} / {len(jobs)} {round(seconds_per_job, 2)} s/j ETA: {eta}")
        else:
            results_list = [self._worker(job) for job in jobs]

        self.best_systems = pd.concat(results_list, axis='rows')
