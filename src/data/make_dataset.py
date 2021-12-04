# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
#from dotenv import find_dotenv, load_dotenv




# load repo files
import cleanup
import extract

from importlib import reload
reload(cleanup)
reload(extract)


# %run make_dataset.py "../../data/raw/SAMP_100k.tsv" '../../data/interim/'


@click.command()
@click.argument('filepath_raw', type=click.Path(exists=True))
@click.argument('folder_interim', type=click.Path())
def main(filepath_raw, folder_interim):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../interim).
    """
    logger = logging.getLogger(__name__)
    logger.info('making interim data set from raw data')

    ## STUFF FROM NB 1.0
    print("Loading registry")
    rf = cleanup.load_registry(filepath_raw, logger)

    print("Cleaning registry")
    rf = cleanup.clean_nombres(rf, folder_interim)


    ## BEGIN NB 2.0
    print("Parsing rows to extract surnames")
    surnames_extracted = rf.progress_apply(lambda row: extract.parse_fullrow(row), axis=1, result_type='expand')

    print("Cleaning surnames")
    nf, funky_prenames = extract.clean_surnames(rf, surnames_extracted)


    ## BEGIN NB 3.0



    ## BEGIN 4.0




if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
#    load_dotenv(find_dotenv())

    main()
