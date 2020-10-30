# --- Imports ---
from vapor import config
from vapor import helper

import logging
import random
import json
import os
import concurrent.futures as cf
import itertools
import io
import requests
import copy
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from datetime import datetime as dt

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point

# import PySAM.Pvsamv1 as pv
import PySAM.Pvwattsv7 as pv
import PySAM.Singleowner as so

import vapor.config as config

log = logging.getLogger("vapor")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~ LOAD CAMBIUM DATA ~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def load_cambium_data(aggregate_region,
                      scenario,
                      inflation=config.INFLATION,
                      save_processed=True, cambium_path=None):
    """Turn cambium data into a single long_df."""
    assert scenario in ['StdScen19_High_RE_Cost','StdScen19_Low_RE_Cost','StdScen19_Mid_Case']
    if cambium_path == None: #Deal with possible user passed path
        cambium_path = os.path.join('data','cambium_processed',f"{scenario}_{aggregate_region}.pkl")
    
    if os.path.exists(cambium_path): #Try to read pickle first...
        log.info('\n')
        log.info(f'Loading cambium data from pickle')
        long_df = pd.read_pickle(cambium_path)
    
    else: #Otherwise reprocess...

        log.info('\n')
        log.info(f'Merging cambium data from csvs')
        all_csvs = os.listdir(os.path.join('data','cambium_csvs'))
        all_csvs = [i for i in all_csvs if '.csv' in i]
        scenario_csvs = [i for i in all_csvs if scenario in i] #filter scenario
        scenario_csvs = [i for i in scenario_csvs if '_hourly_' in i] #the right csvs
        scenario_csvs.sort()
        assert len(scenario_csvs) == 134 #if files are broken down by pca
        
        dfs = []
        for c in scenario_csvs:
            pca = c.split('_')[-1].replace('.csv','')
            log.info(f'....Working on {pca}')
            c_df = pd.read_csv(os.path.join('data','cambium_csvs', c))
            
            # --- Rename Columns ---
            rename_cols = {'r':'pca',
                           'energy_cost_busbar':'cambium_busbar_energy_value',
                           'capacity_cost_busbar':'cambium_capacity_value',
                           'policy_cost_busbar':'cambium_policy_value',
                           'co2_rate_avg_gen':'cambium_co2_rate_avg',
                           'ancillary_service_cost_busbar':'cambium_as_value'}
            c_df = c_df.rename(rename_cols, axis='columns')

            # --- Calc total grid value ---
            c_df['cambium_grid_value'] = c_df[['cambium_busbar_energy_value',
                                               'cambium_capacity_value',
                                               'cambium_policy_value']].sum(axis=1)
            
            # --- Subset ---
            keep_cols = ['cambium_busbar_energy_value',
                         'cambium_capacity_value',
                         'cambium_policy_value',
                         'cambium_co2_rate_avg',
                         'cambium_as_value',
                         'cambium_grid_value',
                         'pca','timestamp']
            c_df = c_df[keep_cols]

            # --- Wide to long ---
            c_df['timestamp'] = pd.to_datetime(c_df['timestamp'])
            c_df['year'] = [ts.year for ts in c_df['timestamp']]
            c_df.sort_values('timestamp')
            dfs.append(c_df)
                    
            # --- Make a copy with the subsequent odd year ---
            odd_df = c_df.copy()
            odd_df['year'] = [ts.year + 1 for ts in c_df['timestamp']]
            odd_df['timestamp'] = [ts.replace(year=ts.year+1) for ts in odd_df['timestamp']]
            dfs.append(odd_df)

        scenario_df = pd.concat(dfs, axis='rows')
        del dfs

        # --- Identify any missing years ---
        for y in range(2020,2051):
            valid_year = y
            while valid_year not in scenario_df.year.unique():
                valid_year -= 1
            if valid_year != y:
                valid_df = scenario_df.loc[scenario_df['year'] == valid_year]
                valid_df['year'] = [ts.year + 1 for ts in valid_df['timestamp']]
                valid_df['timestamp'] = [ts.replace(year=ts.year+1) for ts in valid_df['timestamp']]
                scenario_df = pd.concat([scenario_df, valid_df], axis='rows')

        # --- Clear Memory ---
        scenario_df = helper.memory_downcaster(scenario_df)
        
        # --- Map on region_hierarchy ---
        region_hierarchy = pd.read_csv(os.path.join('data','geography','region_hierarchy.csv'))
        scenario_df = scenario_df.merge(region_hierarchy, on='pca', how='left')

        regions_to_pickle = ['pca', 'rto', 'census_reg', 'state', 'inter']
        regions_to_pickle.append(regions_to_pickle.pop(regions_to_pickle.index(aggregate_region))) #move selected region to end of list so we deal with it last and can return
        log.info(f'....saving pickles for {regions_to_pickle}')
        for agg_r in regions_to_pickle:
            log.info(f'........aggregating for {agg_r}')
            grouped = scenario_df.copy()
            regions_to_drop = regions_to_pickle.copy()
            regions_to_drop.remove(agg_r)
            grouped.drop(regions_to_drop, axis='columns', inplace=True)

            # --- Dissolve into agg_r ---
            groupby_cols = ['timestamp', agg_r]
            agg_cols = [c for c in grouped.columns if c not in groupby_cols]
            grouped = grouped.groupby(groupby_cols, as_index=False)[agg_cols].mean()

            # --- Wide to long ---
            long_df = grouped.melt(id_vars=[agg_r, 'year','timestamp'])
            del grouped

            # --- Cut down memory ---
            long_df['variable'] = long_df['variable'].astype('category')
            long_df[agg_r] = long_df[agg_r].astype('category')
            long_df['year'] = long_df['year'].astype('category')
            
            # --- Save to Pickle ---
            if save_processed:
                long_df.to_pickle(os.path.join('data','cambium_processed',f"{scenario}_{agg_r}.pkl"))
        
    return long_df.sort_values('timestamp')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~ FETCH RESOURCE DATA ~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class FetchResourceFiles():
    """
    Download U.S. solar and wind resource data for SAM from NRELs developer network
    https://developer.nrel.gov/
    Inputs
    ------
    tech (str): one of 'wind' or 'pv'
    workers (int): number of threads to use when parellelizing downloads
    resource_year (int): year to grab resources from. 
        can be 'tmy' for solar
    resource_interval_min (int): time interval of resource data
    nrel_api_key (str): NREL developer API key, available here https://developer.nrel.gov/signup/
    nrel_api_email (str): email associated with nrel_api_key
    Methods
    -------
    run():
        fetch resource profiles for an iterable of lat/lons and save to disk. 
        the attribute `.resource_file_paths_dict` offers a dictionary with keys as lat/lon tuples and 
    """

    def __init__(self, tech,
                 nrel_api_key=config.NREL_API_KEY,
                 nrel_api_email=config.NREL_API_EMAIL,
                 workers=config.THREAD_WORKERS,
                 resource_year=config.RESOURCE_YEAR,
                 resource_interval_min=config.RESOURCE_INTERVAL_MIN):

        self.tech = tech
        self.nrel_api_key = nrel_api_key
        self.nrel_api_email = nrel_api_email

        self.resource_year = resource_year
        self.resource_interval_min = resource_interval_min
        self.workers = workers

        # --- Make folder to store resource_files ---
        self.SAM_resource_dir = os.path.join('data', 'PySAM Downloaded Weather Files')
        if not os.path.exists(self.SAM_resource_dir):
            os.mkdir(self.SAM_resource_dir)

        if tech == 'pv':
            self.data_function = self._NSRDB_worker
        elif tech == 'wind':
            self.data_function = self._windtk_worker

            if self.resource_year == 'tmy':  # tmy not available for wind
                self.resource_year = 2012

        else:
            raise NotImplementedError(
                f'Please write a wrapper to fetch data for the new technology type {tech}')

    def _requests_retry_session(self, retries=10,
                                backoff_factor=1,
                                status_forcelist=(429, 500, 502, 504),
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
        return session

    def _csv_to_srw(self, raw_csv):
        # --- grab df ---
        for_df = copy.deepcopy(raw_csv)
        df = pd.read_csv(for_df, skiprows=1)

        # --- resample to 8760 ---
        df['datetime'] = pd.to_datetime(
            df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
        df.set_index('datetime', inplace=True)
        df = df.resample('H').first()

        # --- drop leap days ---
        df = df.loc[~((df.index.month == 2) & (df.index.day == 29))]
        df.reset_index(inplace=True, drop=True)

        # --- grab header data ---
        for_header = copy.deepcopy(raw_csv)
        header = pd.read_csv(for_header, nrows=1, header=None).values

        site_id = header[0, 1]
        site_tz = header[0, 3]
        site_lon = header[0, 7]
        site_lat = header[0, 9]
        site_year = df.iloc[0]['Year']

        # --- create header lines ---
        h1 = np.array([int(site_id), 'city??', 'state??', 'USA', site_year,
                    site_lat, site_lon, 'elevation??', 1, 8760])  # meta info
        h2 = np.array(["WTK .csv converted to .srw for SAM", None, None,
                    None, None, None, None, None, None, None])  # descriptive text
        header = pd.DataFrame(np.vstack([h1, h2]))
        assert header.shape == (2, 10)

        # --- iterate through all available heights where all data is available ---
        heights = []
        for c in df.columns:
            if 'at ' in c:
                height = c.split('at ')[1].split('m')[0]
                if height in heights:
                    pass
                else:
                    heights.append(int(height))

        heights = sorted(list(set(heights)))  # take unique

        height_dfs = []
        for h in heights:
            try:
                h_df = df[[f"air temperature at {h}m (C)",
                        f"air pressure at {h}m (Pa)",
                        f"wind direction at {h}m (deg)",
                        f"wind speed at {h}m (m/s)"
                        ]]

                # convert from Pa to atm
                h_df[f"air pressure at {h}m (Pa)"] = h_df[f"air pressure at {h}m (Pa)"] / 101325

                h_df.loc[-1] = [h, h, h, h]
                h_df.loc[-2] = ['C', 'atm', 'degrees', 'm/s']
                h_df.loc[-3] = ['Temperature', 'Pressure', 'Direction', 'Speed']
                h_df.index = h_df.index + 3
                h_df = h_df.sort_index()
                height_dfs.append(h_df)

            except Exception as e:
                pass

        data_df = pd.concat(height_dfs, axis='columns')
        data_df.columns = range(data_df.shape[1])

        out = pd.concat([header, data_df], axis='rows')
        out.reset_index(drop=True, inplace=True)
        return out

    def _NSRDB_worker(self, job):
        """Query NSRDB to save .csv 8760 of TMY solar data. To be applied on row with a 'lat' and 'long column."""

        # --- unpack job ---
        lon, lat = job

        # --- initialize sesson ---
        retry_session = self._requests_retry_session()

        # --- Intialize File Path ---
        file_path = os.path.join(self.SAM_resource_dir, f"{lon}_{lat}_psm3_{self.resource_interval_min}_{self.resource_year}.csv")

        # --- See if file path already exists ---
        if os.path.exists(file_path):
            return file_path  # file already exists, just return path...

        else:
            log.info(f"Downloading NSRDB file for {lon}_{lat}...")

            # --- Find url for closest point ---
            lookup_base_url = 'https://developer.nrel.gov/api/solar/'
            lookup_query_url = f"nsrdb_data_query.json?api_key={self.nrel_api_key}&wkt=POINT({lon}+{lat})"
            lookup_url = lookup_base_url + lookup_query_url
            lookup_response = retry_session.get(lookup_url)

            if lookup_response.ok:
                lookup_json = lookup_response.json()
                links = lookup_json['outputs'][0]['links']
                year_url_dict = {d['year']: d['link']
                                 for d in links if d['interval'] == self.resource_interval_min}
                year_url = year_url_dict[self.resource_year]
                year_url = year_url.replace('yourapikey', self.nrel_api_key).replace(
                    'youremail', self.nrel_api_email)

                # --- Get year data ---
                year_response = retry_session.get(year_url)
                if year_response.ok:
                    # --- Convert response to string, read as pandas df, write to csv ---
                    csv = io.StringIO(year_response.text)
                    df = pd.read_csv(csv)
                    df.to_csv(file_path, index=False)
                    return file_path
                else:
                    breakpoint()
                    return 'error at year_response'

            else:
                return 'error at lookup_response'

    def _windtk_worker(self, job):

        # --- unpack job ---
        lon, lat = job

        # --- initialize sesson ---
        retry_session = self._requests_retry_session()

        # --- Intialize File Path ---
        file_path = os.path.join(self.SAM_resource_dir, f"{lon}_{lat}_wtk_{self.resource_interval_min}_{self.resource_year}.srw")

        # --- See if file path already exists ---
        if os.path.exists(file_path):
            return file_path  # file already exists, just return path...

        else:
            log.info(f"Downloading wind toolkit file for {lon}_{lat}...")
            
            # --- Find url for closest point ---
            year_base_url = 'https://developer.nrel.gov/api/wind-toolkit/v2/wind/'
            year_query_url = f"wtk-download.csv?api_key={self.nrel_api_key}&wkt=POINT({lon}+{lat})&names={self.resource_year}&utc=true&email={self.nrel_api_email}"
            year_url = year_base_url + year_query_url
            year_response = retry_session.get(year_url)

            if year_response.ok:
                # --- Convert response to string, read as pandas df, write to csv ---
                raw_csv = io.StringIO(year_response.text)
                df = self._csv_to_srw(raw_csv)
                df.to_csv(file_path, index=False, header=False, na_rep='')
                return file_path
            else:
                raise ValueError('error at year_response')

    def fetch(self, points):
        """
        Creates dict with {region:path_to_SAM_resource_file}.
        Input
        -----
        points(iterable): iterable of lon/lat tuples, i.e. Shapely Points
        """
        log.info(f'Beginning data download for {self.tech} using {self.workers} thread workers')

        # --- Initialize Session w/ retries ---
        if self.workers > 1:

            with cf.ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = [executor.submit(self.data_function, job)
                           for job in points]
                results = [f.result() for f in futures]

        else:
            results = []
            for job in points:
                results.append(self.data_function(job))

        self.resource_file_paths = results
        self.resource_file_paths_dict = dict(zip(points, results))

        log.info('....finished data download')

class GetCentroidOfRegions():
    def __init__(self,
                 aggregate_region='pca_res',
                 shapefile_path=None):
        """Get a centroid for each region of a given aggregate_region across the U.S."""

        assert aggregate_region in ['pca', 'pca_res', 'rto', 'census_reg', 'state', 'inter']
        self.aggregate_region = aggregate_region
        self.shapefile_path = shapefile_path


    def _load_shapefile(self):
        """Load shapefiles, if no path is specified, the ReEDS Wind Region file will be used."""
        if self.shapefile_path == None:
            self.shapefile_path = os.path.join('data','geography', 'ReEDS_Resource_Regions.shp')
        assert os.path.isfile(self.shapefile_path)

        gdf = gpd.read_file(self.shapefile_path)
        gdf = gdf.to_crs("EPSG:4326")
        return gdf

    def _aggregate_shapefile(self, gdf):
        gdf = gdf.dissolve(by=self.aggregate_region)
        return gdf

    def _make_centroids_lookup(self, gdf):
        """Returns 'lat' and 'lon' columns for cetroid for each region."""

        gdf['lon'] = gdf.centroid.x.round(3)
        gdf['lat'] = gdf.centroid.y.round(3)
        gdf.reset_index(inplace=True)

        # --- create dict with tuple lookups ---
        gdf['point'] = [(row['lon'], row['lat']) for _, row in gdf.iterrows()]
        self.centroids_lookup = dict(zip(gdf[self.aggregate_region], gdf['point']))

        # --- rename shapefile ---
        self.region_shape = gdf[[self.aggregate_region,'geometry']]
        self.region_shape.columns = ['region','geometry']
    
    def find_centroids(self):
        gdf = self._load_shapefile()
        gdf = self._aggregate_shapefile(gdf)
        self._make_centroids_lookup(gdf)


class CoordsToRegionCentroid(GetCentroidOfRegions):

    def match_centroids(self, df):
        """For each lon/lat pair in a dataframe, get the resource region file path."""

        # --- create centroids lookup ---
        self.find_centroids()

        # --- convert points iterable to geopandas dataframe ---
        df['geometry'] = [Point(row['longitude'], row['latitude']) for _, row in df.iterrows()]
        points_gdf = gpd.GeoDataFrame(df)
        points_gdf.crs = "EPSG:4326"  

        # --- find aggregate region for each point ---
        points_gdf = gpd.sjoin(points_gdf, self.region_shape, how='left', op='within')

        # --- map on resource region point ---
        points_gdf['resource_point'] = points_gdf['region'].map(self.centroids_lookup)

        resource_points_dict = {}
        for tech in points_gdf['tech'].unique():
            gdf_subset = points_gdf.loc[points_gdf['tech'] == tech]
            resource_points_to_fetch = list(set(gdf_subset['resource_point']))
            if np.nan in resource_points_to_fetch:
                resource_points_to_fetch.remove(np.nan)
            fetcher = FetchResourceFiles(tech=tech)
            fetcher.fetch(resource_points_to_fetch)
            points_gdf.loc[points_gdf['tech'] == tech, 'resource_fp'] = points_gdf.loc[points_gdf['tech']
                                                                                       == tech, 'resource_point'].map(fetcher.resource_file_paths_dict)

        # --- drop nulls ---
        points_gdf = points_gdf.dropna(subset=['resource_fp','resource_point'])
        return points_gdf

def load_bnef():
    # --- Load Dataframe ---
    bnef = pd.read_csv(os.path.join('data','bnef','bnef_ppas_may_2020.csv'))

    # --- Rename Column Names ---
    bnef.columns = [c.lower().replace(' ','_')\
                            .replace('&','')\
                            .replace('__','_')\
                            .replace('(','')\
                            .replace(')','') for c in bnef.columns]

    # --- Rename Columns ---
    rename_dict = {
        'capacity_mw':'re_capacity_mw',
        'sector_subsector':'tech'
    }
    bnef = bnef.rename(rename_dict, axis='columns')

    # --- Add Battery columns ---
    bnef['batt_capacity_mw'] = 0
    bnef['batt_capacity_mwh'] = 0

    # --- clean tech ---
    tech_dict = {
        'Wind, Onshore':'wind',
        'Solar, PV':'pv'
    }
    bnef['tech'] = bnef['tech'].map(tech_dict)

    # --- Estimate last year ---
    bnef['term'] = bnef['term'].fillna(15)
    bnef['ppa_estimated_last_year'] = bnef['ppa_estimated_signing_year'] + bnef['term']


    # --- Filter DF ---
    bnef = bnef.loc[bnef['country'] == 'United States']
    bnef = bnef.loc[bnef['tech'].isin(['pv','wind'])]
    bnef = bnef.loc[bnef['ppa_estimated_last_year'] > 2020]
    bnef = bnef.loc[bnef['re_capacity_mw'] > 0]
    bnef = bnef.loc[~bnef['state'].isin(['Various','Hawaii','Alaska'])]

    # --- Map on lon/lat ---
    bnef['geocoder_lookup'] = bnef['country'] + ' ' + bnef['state'] + ' ' + bnef['project_name']
    GeoCoder = helper.AddressToLonLatGeocoder()
    GeoCoder.run(bnef['geocoder_lookup'])
    bnef['lon_lat_tuple'] = bnef['geocoder_lookup'].map(GeoCoder.results_dict)
    bnef['longitude'] = [i[0] if isinstance(i, tuple) else None for i in bnef['lon_lat_tuple']]
    bnef['latitude'] = [i[1] if isinstance(i, tuple) else None for i in bnef['lon_lat_tuple']]
    bnef = bnef.dropna(subset=['longitude','latitude'])
    return bnef

def load_consumption_profile(shape='datacenter'):
    """eventually, connect to comstock and make API call for location/building type."""
    assert shape in ['datacenter','commercial']

    if shape == 'commercial':
        fp = os.path.join('data','load','sample_load.csv')
        load = pd.read_csv(fp)
        
        # --- clean ---
        load = load[load.columns[0:2]]
        load.columns = ['timestamp','kwh']
        load['hour'] = range(0,8760)
        load.drop('timestamp', axis='columns', inplace=True)

    elif shape == 'datacenter':
        load = pd.DataFrame({'hour': range(0,8760),
                            'kwh': np.full(8760, 100)})

    return load
