# -*- coding: utf-8 -*-
import click
click.disable_unicode_literals_warning = True
import logging
from pathlib import Path
#from dotenv import find_dotenv, load_dotenv




# load repo files
import cleanup
import extract
import parents
import match
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
    surnames_extracted = rf.apply(
        lambda row: extract.parse_fullrow(row), axis=1, result_type='expand')
    surnames_extracted.to_csv(
        folder_interim + '/02-surname.tsv', sep='\t', index=False)

    nf, funky_prenames = extract.clean_names(rf, surnames_extracted) 
    del surnames_extracted

    print("len(NF):", len(nf))
    print(nf.shape)
    # initial extraction
    parsed = extract.parse_prenames(nf)
    name_counts = extract.make_allnames(parsed)

    # now do some cleaning
    nf = extract.fix_mixed_presur_names(nf, name_counts)
    nf, rf = extract.fix_husband_honorific(nf, rf)
    # now re-parse the cleaned data
    parsed = extract.parse_prenames(nf)
    name_counts = extract.make_allnames(parsed)
    allnames = extract.merge_underscore_names(name_counts)

    nf.to_csv(folder_interim + '/03-names_cleaned.tsv', sep='\t', index=False)
    allnames.to_csv(folder_interim + '/04-allnames.tsv', sep='\t', index=False)
    parsed.to_csv(folder_interim + '/05-newfreqfile.tsv', sep='\t', index=False)
    name_counts.to_csv(folder_interim + '/06-namecounts.tsv', sep='\t', index=False)
    del name_counts

    ## BEGIN NB 3.0
    print("Padre")
    wts_pre, wts_sur = parents.wts(allnames)
    del allnames
    padre = nf.progress_apply(lambda row: 
                    parents.extract_prename_parent(
                        row, 'nombre_padre', wts_pre, wts_sur, funky_prenames),
                    axis=1, result_type='expand')
    print("Madre")
    madre = nf.progress_apply(lambda row: 
                    parents.extract_prename_parent(
                        row, 'nombre_madre', wts_pre, wts_sur, funky_prenames),
                    axis=1, result_type='expand')
    del funky_prenames

    padre.to_csv(folder_interim + '/07-padre.tsv', sep='\t', index = False)
    madre.to_csv(folder_interim + '/08-madre.tsv', sep='\t', index=False)
    
    ## BEGIN 4.0
    ncleaned_rf = match.merge_ncleaned_rf(nf, rf)
    del nf
    print("Matching records with cedulas")
    match_by_cedula_padre = match.match_by_cedula_padre(ncleaned_rf)
    match_by_cedula_madre = match.match_by_cedula_madre(ncleaned_rf)
    match_by_cedula_spouse = match.match_by_cedula_spouse(ncleaned_rf)
    match_by_cedula_padre.to_csv(folder_interim + '/09-match_by_cedula_padres.tsv', 
                                 sep='\t', index=False)
    match_by_cedula_madre.to_csv(folder_interim + '/10-match_by_cedula_madres.tsv',
                                 sep='\t', index=False)
    match_by_cedula_spouse.to_csv(folder_interim + '/15-match_by_cedula_spouse.tsv', 
                                 sep='\t', index=False)
    del match_by_cedula_madre, match_by_cedula_padre, match_by_cedula_spouse

    ceds_found_padre_cedula = match.ceds_found(
        ncleaned_rf, 'ced_padre', only_cedula = True)
    ceds_found_madre_cedula = match.ceds_found(
        ncleaned_rf, 'ced_madre', only_cedula=True)

    print("Matching exact names")
    matched_padres = match.exact_name_padre(
        ncleaned_rf, ceds_found_padre_cedula)
    matched_madres = match.exact_name_madre(
        ncleaned_rf, ceds_found_madre_cedula)
    matched_padres.to_csv(
        folder_interim + '/11-matched_padres.tsv', sep='\t', index=False)
    matched_madres.to_csv(
        folder_interim + '/12-matched_madres.tsv', sep='\t', index=False)
    del ncleaned_rf, ceds_found_padre_cedula, ceds_found_madre_cedula
    ## BEGIN 5.0
    print("Matching partial")
    names = match.create_names(parsed, rf)
    del parsed, rf
    # Guys that has ced_padre or ced_madre
    ceds_found_madre = match.ceds_found(
        names, 'ced_madre', matched_madres)
    ceds_found_padre = match.ceds_found(
        names, 'ced_padre', matched_padres)
    del matched_madres, matched_padres
    mparsed = match.parsed(madre, ceds_found_madre)
    pparsed = match.parsed(padre, ceds_found_padre)
    del padre, madre

    file_out_madre = folder_interim + '/13-MADRES_matched_by_name.tsv'
    file_out_padre = folder_interim + '/14-PADRES_matched_by_name.tsv'
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
