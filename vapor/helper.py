#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 12:52:52 2020

@author: skoebric
"""

import us
import concurrent.futures as cf
from requests.packages.urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import requests
import time
import geocoder
import os
import pickle
import pandas as pd
import numpy as np
from shapely.geometry import Point

import vapor.config as config

import logging
log = logging.getLogger("vapor")
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("geocoder").setLevel(logging.WARNING)

def memory_downcaster(df, cat_cols=[]):
 
    assert isinstance(df, pd.DataFrame) | isinstance(df, pd.Series)
 
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in df.columns:

        if df[col].dtype not in [object, '<M8[ns]']:  # Exclude strings
             
            # make variables for Int, max and min
            IsInt = False
            mx = df[col].max()
            mn = df[col].min()
             
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(df[col]).all(): 
                NAlist.append(col)
                df[col].fillna(mn-1,inplace=True)  
                    
            # test if column can be converted to an integer
            asint = df[col].fillna(0).astype(np.int64)
            result = (df[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True
 
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif mx < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif mx < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)    
             
            # Make float datatypes 32 bit
            else:
                df[col] = df[col].astype(np.float16)
        
        # --- convert categories to save memory ---
        unique_count = df[col].nunique()
        if col in cat_cols:
            df[col] = df[col].astype('category')

    return df


class AddressToLonLatGeocoder():
    def __init__(self, service='google', workers=1):
        self.service = service
        self.workers = workers

    def _requests_retry_session(self, retries=10,
                                backoff_factor=1,
                                status_forcelist=(
                                    429, 500, 502, 504, 'Unknown'),
                                session=None):
        """https://www.peterbe.com/plog/best-practice-with-retries-with-requests"""
        session = session or requests.Session()
        session.verify = False
        retry = Retry(
            total=retries,
            read=retries,
            connect=retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        self.session = session

    def worker(self, job):
        with self.session as session:
            query = geocoder.google(job, key='') #ADD GOOGLE API, or CHANGE to different geocoder provider

        if query.ok:
            lon = query.latlng[1]
            lat = query.latlng[0]
            lon_lat_tuple = (lon, lat)
            return (job, lon_lat_tuple)

        else:
            log.warning(f'....failed on {job}')
            return (job, None)

    def run(self, jobs, retry_nones=False):

        # --- Initialize Session ---
        self._requests_retry_session()

        # --- Look for cache ---
        try:
            with open(os.path.join('data','geocoder','cache.pkl'), 'rb') as handle:
                cache = pickle.load(handle)
        except Exception:
            cache = {}
        
        if retry_nones:
            cache = {k:v for k,v in cache.items() if v != None}

        cached_jobs = {k:v for k,v in cache.items() if k in list(jobs)}
        needed_jobs = [j for j in jobs if j not in cache.keys()]
        log.info(f'....{len(cached_jobs)} jobs found in cache, {len(needed_jobs)} still needed')
        
        results_dict = {}
        if self.workers > 1:
            start = time.time()
            job_count = 0
            checkpoint = int(max(1, (len(needed_jobs) * 0.05)))

            with cf.ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = [executor.submit(self.worker, job) for job in needed_jobs]
                for result in cf.as_completed(futures):
                    self.results_dict[result[0]] = result[1]

                    job_count += 1
                    if job_count % checkpoint == 0:
                        eta = (((time.time() - start) / (job_count))) * (len(needed_jobs) - job_count) / 60
                        log.info(f'........finished geocoding job {job_count}/{len(needed_jobs)}  ETA: {eta} min')
        else:
            for job in needed_jobs:
                results_dict[job] = self.worker(job)[1]

        # --- Combine cached jobs with results ---
        self.results_dict = {**cached_jobs, **results_dict}

        # --- Save cache ---
        with open(os.path.join('data','geocoder','cache.pkl'), 'wb') as handle:
            pickle.dump(self.results_dict, handle)
