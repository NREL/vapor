
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
import warnings
warnings.filterwarnings("ignore", category=UserWarning) #geopandas warning

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# import PySAM.Pvsamv1 as pv
import PySAM.Pvwattsv7 as pv
import PySAM.Singleowner as so

# --- Absolute Imports ---
from vapor.datafetcher import *
from vapor.systemdesigner import *
from vapor.systemsimulator_objects import *
from vapor.systemsimulator import *
from vapor.visualizer import *
from vapor.helper import *
from vapor.config import *
from vapor.pipeline import *
from vapor.models import *

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.FileHandler(os.path.join('logs',"vapor.txt")),
        logging.StreamHandler()
    ])

log = logging.getLogger("vapor_main")
log.info("Starting log for vapor...")

# --- Warnings ---
pd.options.mode.chained_assignment = None  # default='warn'
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
