# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
#from dotenv import find_dotenv, load_dotenv

# load repo files
import test

from importlib import reload

import sys
sys.path.append("../data")
import cleanup
import extract
import parents
reload(cleanup)
reload(extract)
reload(parents)

# %run test_dataset.py "../../data/testdata/00-test_simpsons.tsv" '../../data/testdata/interim/'

date_cols = ['dt_birth', 'dt_death', 'dt_marriage']

dtypes_namedata = { 'cedula': str, 'nombre': str, 'nombre_spouse': str, 'marital_status': str,
                    'place_birth': str, 'nombre_padre': str, 'nombre_madre': str, 
                    'ced_spouse': str, 'ced_padre': str, 'ced_madre': str,
                    'is_nat': bool, 'is_nat_padre': bool, 'is_nat_madre': bool,
                    }

dtypes_surname = {  'cedula': str, 'sur_padre': str, 'sur_madre': str, 'prenames': str,
                    'has_padre': bool, 'is_plegal': bool, 'has_madre': bool, 'is_mlegal': bool,
                    }

dtypes_cleaned = {  'cedula': str, 'nombre': str, 'prenames': str, 'gender': str,
                    'nombre_padre': str, 'sur_padre': str, 'has_padre': bool, 'is_plegal': bool,
                    'nombre_madre': str, 'sur_madre': str, 'has_madre': bool, 'is_mlegal': bool,
                    'is_funky': bool, 'nlen_padre': int, 'nlen_madre': int, 'n_char_nombre': int,
                    'n_char_prename': int, 'maybe_husb': bool
                    }

dtypes_allnames = { 'obsname': str, 'n_sur': float, 'n_pre': float, 'sratio': float, 'pratio': float
                    }

dtypes_namecounts= {'obsname': str, 'n_sur': float, 'n_pre': float, 'sratio': float,
                    'pratio': float, 'nlen': float, 'is_multimatch': bool
                    }

dtypes_newfreqfile = {  'cedula': str, 'nombre': str, 'sur_padre': str, 'sur_madre': str,
                        'pre1': str, 'pre2': str, 'pre3': str, 'junk': str, 'nlen': int,
                        }
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

    test.test_data(rf, '../../data/testdata/01-simpsons_test_cases.tsv', dtypes_namedata, date_cols)

    ## BEGIN NB 2.0
    print("Parsing rows to extract surnames")
    surnames_extracted = rf.progress_apply(lambda row: extract.parse_fullrow(row), axis=1, result_type='expand')

    test.test_data(surnames_extracted, '../../data/testdata/02-simpsons_test_cases_surname.tsv', dtypes_surname)

    names_cleaned, allnames = extract.allnames_nf_manipulation(
        rf, surnames_extracted)

    test.test_data(names_cleaned, "../../data/testdata/03-simpsons_test_cases_names_cleaned.tsv", dtypes_cleaned)
    test.test_names(allnames, "../../data/testdata/04-simpsons_test_cases_allnames.tsv", dtypes_allnames)

    newfreqfile = extract.freqfile(names_cleaned)

    test.test_data(newfreqfile, "../../data/testdata/05-simpsons_test_cases_newfreqfile.tsv", dtypes_newfreqfile)

    namecounts = extract.namecounts(newfreqfile)
    
    test.test_names(namecounts, "../../data/testdata/06-simpsons_test_cases_namecounts.tsv", dtypes_newfreqfile)
    
    ## BEGIN NB 3.0
    wts_pre, wts_sur = parents.wts(allnames)

    padre = names_cleaned.progress_apply(lambda row: parents.extract_prename_parent(row, 'nombre_padre', wts_pre, wts_sur),
                                         axis=1, result_type='expand')

    madre = names_cleaned.progress_apply(lambda row: parents.extract_prename_parent(row, 'nombre_madre', wts_pre, wts_sur),
                                         axis=1, result_type='expand')
    pos = 0
    for i in [padre, madre, allnames]:
        i.to_csv("../../data/testdata/interim/PADRE"+str(pos)+".tsv", sep='\t', index=False)
        pos += 1


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
