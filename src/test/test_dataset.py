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
import match
import utils
reload(cleanup)
reload(extract)
reload(parents)
reload(match)
reload(utils)

# %run test_dataset.py "../../data/testdata/00-test_cases.tsv" '../../data/testdata/interim/'

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
    # test.test_data(rf, '../../data/testdata/01-test_cases_cleanup.tsv', utils.get_dtypes_reg(), utils.get_date_cols())

    print("RF :", len(rf))
    print(rf.shape)  # head())

    ## BEGIN NB 2.0
    print("Parsing rows to extract surnames")

    surnames_extracted = rf.apply(lambda row: extract.parse_fullrow(row), axis=1, result_type='expand')
    surnames_extracted.to_csv('../../data/testdata/interim/02-surname.tsv', sep='\t', index=False)

    nf, funky_prenames = extract.clean_names(rf, surnames_extracted)

    print("len(NF):", len(nf))
    print(nf.shape)  # .head())
    # initial extraction
    parsed = extract.parse_prenames(nf)
    name_counts = extract.make_allnames(parsed)

    # now do some cleaning
    nf = extract.fix_mixed_presur_names(nf, name_counts)
    nf, rf = extract.fix_husband_honorific(nf, rf, funky_prenames)
    # now re-parse the cleaned data
    parsed = extract.parse_prenames(nf)
    name_counts = extract.make_allnames(parsed)
    allnames = extract.merge_underscore_names(name_counts)

    nf.to_csv('../../data/testdata/interim/03-names_cleaned.tsv', sep='\t', index=False)
    allnames.to_csv('../../data/testdata/interim/04-allnames.tsv', sep='\t', index=False)
    parsed.to_csv('../../data/testdata/interim/05-newfreqfile.tsv', sep='\t', index=False)
    name_counts.to_csv('../../data/testdata/interim/06-namecounts.tsv', sep='\t', index=False)

    # test.test_data(surnames_extracted, '../../data/testdata/02-test_cases_surname.tsv', utils.get_dtypes_surname())
    # test.test_data(nf, "../../data/testdata/03-test_cases_names_cleaned.tsv", utils.get_dtypes_cleaned())
    # test.test_names(allnames, "../../data/testdata/04-test_cases_allnames.tsv", utils.get_dtypes_allnames())
    # test.test_data(parsed, "../../data/testdata/05-test_cases_newfreqfile.tsv", utils.get_dtypes_newfreqfile())
    # test.test_names(name_counts, "../../data/testdata/06-test_cases_namecounts.tsv", utils.get_dtypes_newfreqfile())
    
    # ## BEGIN NB 3.0
    wts_pre, wts_sur = parents.wts(allnames)

    padre = nf.progress_apply(lambda row: parents.extract_prename_parent(row, 'nombre_padre', wts_pre, wts_sur),
                                         axis=1, result_type='expand')

    madre = nf.progress_apply(lambda row: parents.extract_prename_parent(row, 'nombre_madre', wts_pre, wts_sur),
                                         axis=1, result_type='expand')

    padre.to_csv('../../data/testdata/interim/07-padre.tsv', sep='\t', index = False)
    madre.to_csv('../../data/testdata/interim/08-madre.tsv', sep='\t', index=False)

    # test.test_data(padre, "../../data/testdata/07-test_cases_padre.tsv", utils.get_dtypes_padres())
    # test.test_data(madre, "../../data/testdata/08-test_cases_madre.tsv", utils.get_dtypes_padres())

    # ## BEGIN 4.0
    ncleaned_rf = match.merge_ncleaned_rf(nf,rf)

    matched_padres, matched_madres = match.exact_name(ncleaned_rf)
    matched_padres.to_csv('../../data/testdata/interim/09-matched_padres.tsv', sep='\t', index = False)
    matched_madres.to_csv('../../data/testdata/interim/10-matched_madres.tsv', sep='\t', index = False)

    ## BEGIN 5.0
    names = match.create_names(parsed, rf)  
    # Guys that has ced_padre or ced_madre
    ceds_found_madre = match.ceds_found(names, matched_madres, 'ced_madre')
    ceds_found_padre = match.ceds_found(names, matched_padres, 'ced_padre')

    mparsed = match.parsed(madre, ceds_found_madre)
    pparsed = match.parsed(padre, ceds_found_padre)

    file_out_padre = '../../data/testdata/interim/11-MADRES_matched_by_name.tsv'
    file_out_madre = '../../data/testdata/interim/12-PADRES_matched_by_name.tsv'
    match.matched_by_name(mparsed, names, 'F', file_out_madre)
    match.matched_by_name(pparsed, names, 'M', file_out_padre)




if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
#    load_dotenv(find_dotenv())

    main()
