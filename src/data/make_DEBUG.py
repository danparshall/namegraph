# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
#from dotenv import find_dotenv, load_dotenv
import os


# load repo files
import cleanup
import extract

from importlib import reload
reload(cleanup)
reload(extract)

filepath_raw = "../../data/raw/SAMP_1M.tsv"
filepath_raw = "../../data/raw/SAMP_100k.tsv"
folder_interim = '../../data/interim/'


logger = logging.getLogger(__name__)
logger.info('making interim data set from raw data')
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)


rf = cleanup.load_registry(filepath_raw, logger)
rf = cleanup.clean_nombres(rf, folder_interim)

print("Parsing rows")
surnames_extracted = rf.progress_apply(lambda row: extract.parse_fullrow(row), axis=1, result_type='expand')

sf = surnames_extracted[surnames_extracted.sur_padre.map(lambda x: x != "WTF PADRE PROBLEM" and ' ' in x) 
                        & surnames_extracted.sur_madre.map(lambda x: x != '')
                        ]
pcounts = sf.sur_padre.value_counts()
pcounts = pcounts.reset_index()
pcounts.columns = ['obsname', 'n_obs']

filepath_allnames = os.path.join(folder_interim, 'allnames.tsv')
pcounts.to_csv(filepath_allnames, sep='\t', index=False)
