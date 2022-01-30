from collections import Counter
import math
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import re
import datetime as dt
import os
import utils

import unidecode
from fuzzywuzzy import fuzz



# enable progress bar on long operations
from tqdm.auto import tqdm
tqdm.pandas()


# in all cases, we look for a word boundary as the first group, then our funky name as the second
re_von = re.compile(u"(^|\s)(V[AO]N \w{2,})(\s|$)")              # these results are subset of "re_vande"
re_vande = re.compile(u"(^|\s)(V[AO]N DE[RN]? \w{2,})(\s|$)")
re_sant = re.compile(u"(^|\s)(SANT?A? \w{2,})(\s|$)")            # SAN and SANTA (SANTO doesn't form compounds)
re_dela = re.compile(u"(^|\s)(DE L[AO]S? ?[AO]? ?\w{2,})(\s|$)") # these results are subset of "re_laos"
re_laos = re.compile(u"(^|\s)(L[AEO]S? \w{2,})(\s|$)")
re_del  = re.compile(u"(^|\s)(DEL \w{2,})(\s|$)")
re_de   = re.compile(r"(^|\s)(DE \w{2,})(\s|$)")



def regex_compound_names(nombre, compound_names):
    """ Checks a name against several regexes, to see if it's a compound name; 

    NOTE! expands "compound_names" as a side-effect!  <<< I'm a bad person

    Args:
        nombre  : string of citizen's name

    Returns:
        boolean indicating if the name matches one of the known forms for multi-part names
    """
    mdel   = re_del.search(nombre)
    msant  = re_sant.search(nombre)
    mlaos  = re_laos.search(nombre)
    mdela  = re_dela.search(nombre)
    mvon   = re_von.search(nombre)
    mvande = re_vande.search(nombre)
    mde    = re_de.search(nombre)
    
    poss_funks = set()
    
    if mdel:
        poss_funks.add(mdel.group(2))
    if msant:
        poss_funks.add(msant.group(2))
    if mvon:
        # "VAN DE" types are a subset of "VAN" types
        if mvande:
            poss_funks.add(mvande.group(2))
        else:
            poss_funks.add(mvon.group(2))
    if mlaos:
        # "DE LA" type names are a subset of "LA" types
        if mdela:
            poss_funks.add(mdela.group(2))
        else:
            poss_funks.add(mlaos.group(2))
    if mde:
        poss_funks.add(mde.group(2))

    if poss_funks:
        for funk in poss_funks:
            compound_names.add(funk)
        return True
    else:
        return False



def get_compound_names(folder_interim):
    """ Returns list (from longest to shortest) of all known compound names.

    Args:
        folder_interim  : string containing path to the interim data folder; we check for "allnames.tsv" there

    Returns:
        list of all known compound names, whether found algorithmically (thus in "allnames.tsv"), or manually (and thus
        hardcoded here).  The hardcoded names will need to be altered/expanded based on your own registry of citizens.
    """

    compound_names = set()
    filepath = os.path.join(folder_interim, 'allnames.tsv')
    if os.path.exists(filepath):
    # we check if an "allnames" file has already been produced, and apply the regexes if so.
        allnames = pd.read_csv(filepath, sep='\t', keep_default_na=False, na_values=utils.get_nan_values())

        # NOTE: expands the set of compound names as a side effect
        allnames.obsname.map(lambda x: regex_compound_names(x, compound_names))
    else:
    # but even if not, we will proceed; there are plenty of hand-crafted ones to use
        pass

    # found by hand in NB 3.0; not sure if I should do something with them here
    longnames = { "GOMEZ DE LA TORRE", "SOLANO DE LA SALA", "MARQUEZ DE LA PLATA", "ESPINOZA DE LOS MONTEROS",
                 "MARTINEZ DE LA VEGA", 'DE LA TORRE', 'ESPINOSA DE LOS MONTEROS', 'VASQUEZ DE LA BANDERA',
                 'LEON DE LA TORRE', 'QUISHPE DE LA VEGA', 'BENSTLEY DE VAN HOLDEN', 'CALDERON DE LA BARCA',
                 'DE OLIVEIRA E SOUZA', 'DE GENOT DE NIEUKERKEN', 'PI DE LA SERRA', 'ALVAREZ DE LA CAMPA',
                 'MORENO DE LOS RIOS', 'PEREZ DE VILLA AMIL', 'VON ROSSING UND VON HUGO',
                 'SANTA CRUZ DE OVIEDO', 'GOMEZ DE LA MAZA', 'BILBAO DE LA VIEJA', 'PEREZ DE LA BLANCA',
                 'RODRIGUEZ DE LA PARRA', 'DE TONNAC DE VILLENEUVE', 'FLOR DE LAS BASTIDAS', 'MARTINEZ DE LA COTERA',
                 'TORRES GOMEZ DE CADIZ', 'CHAN PIRES DE URBANO', 'PAZ Y MINO', 'PEREZ DE LA BLANCA',
                 'DE L HERMITE', 'DE L ENDROIT', 'PONCE DE LEON', 'NUNEZ DEL ARCO',
                 'BENSTLEY DE VAN HOLDEN', 'BENSTLEY DE VAN HOLDE',  # latter is misspelling of the family name,
                }
    y_names = {'PAZ Y MINO', 'ROMERO Y CORDERO', 'MAS Y RUBI', 'ZACCO Y BARBERO', 'VALLE Y GARCIA',
               'ORTECHO Y ARMAZA', 'MIER Y TERAN', 'HEVIA Y VACA', 'JIJON Y CAAMANO', 'ARAGON Y CASTANO',
               'COSTA Y LAURENT', 'SYLVA Y ANTUNA', 'VANEGAS Y CORTAZAR', 'LLOPIS Y CIRERA', 'GUINOT Y RICO',
               'QUITO Y SACA',
              }
    dedications = {'DE LOS DOLORES', 'DE LAS MERCEDES', 'DE LA ESE', 'DE LA ROSA', 'DE LA PALOMA', 'DE LA TORRE',
                   'DEL NINO JESUS', 'DE SANTA ANA', 'DE SAN JOSE', "DE SAN ROMAN", "DE SAN ANTONIO", 'DE LA ASUNCION',
                    'DE LOS ANGELES', 'DE LOS REYES', 'DE LOS MILAGROS', 'DE LA TRINIDAD', 'DE LA PAZ',
                    'DE LA NUBE', 'DE LA NUVE', 'DE LA CARIDAD', 
                    
                }
    others = {"MONTES DE OCA", "ROMO LEROUX", 'SAENZ DE VITERI', "FERNANDEZ DE CORDOVA", "FLORES DE VALGAZ",
                'DI LUCA', 'VARGAS MACHUCA', 'DIAZ GRANADOS', 'SIG TU', 'MAN GING', 'DE LA FE', "DE BERAS",
                "DE BOYER DE CAMPRIEU", "DEL NINO JESUS", "DE EL CISNE", "DE EL ROCIO", "DE LA FE", 'DE SAN JOSE', 
                'DE SANTA ANA', 'DEL LOS ANGELES',
    }    
    test_data = {"DE LA CUEVA", "RUIZ ESCODA", "DEL BOSQUE", "DE LA MORA", "DEL ARCO", "DEL CORRAL", "DE LA HOZ",
                "DE LA OSSA", "DE LOS RIOS", "DEL ROZAL", "DE LAS CASAS", "DE LA CRUZ", "DE LA RENTA", "DE VIGO",
                "DEL CORAL", "DE LA ROCHE", "DE LA ESPRIELLA", "DE LA GUARDIA", "DEL PINO", 
             }
    compound_names = sorted(compound_names|longnames|y_names|dedications|others, key=len, reverse=True)
    return compound_names



def add_underscores(nombre, compound_names):
    """ Replaces spaces in compound names with the underscore character (making them easier to parse later).

    Multi-part names are common in Romance languages, but they're a pain for our purposes.  In later stages of this
    project, we identify personal and family names by splitting on spaces; multi-part names make that very challenging.
    This function transforms multi-part names into single-token names, making later stages much easier. 
    E.g., "IGLESIAS DE LA CUEVA JULIO JOSE" would become "IGLESIAS DE_LA_CUEVA JULIO JOSE"  (i.e., the maternal surname
    is now a single token, which is much easier to parse).

    Establishing the list of compound names isn't trivial.  Some of them can be done with regular expressions for common
    cases, but many are only found when comparing surnames in later stages of the project (and some were discovered
    manually).  This means that the workflow must be run twice - once for initial cleaning, and then again after the
    "allnames.tsv" file has been produced.


    Args:
        nombre          : string of citizen's name
        compound_names  : list (sorted by decreasing length) of all known compound names

    Returns:
        string containing the original name, but any parts matching a known compound name are single tokens
    """
    for d in compound_names:
        if d in nombre:
            # if we've got a candidate match, confirm with a regex (to be sure there's no end-effects)
            m = re.search("(^|\s)(" + d + ")(\s|$)", nombre)
            if m:
                new = "_".join(m.group(2).split())
                parts = [x.strip() for x in nombre.split(m.group(2))]
                nombre = ' '.join([parts[0], new, parts[1]]).strip()
    return nombre



re_star = re.compile(r'^[\s*+/]+$')
re_starplus = re.compile(r'(^[\s*]+)([\w\s]+)')         # triggers on one or more of "·./+$\(){}[]<>"
re_oddchars = re.compile(r"[,'·./+$<>{}()\[\]\\]+")
re_dash = re.compile(r"\w+\s?-\s?\w+")
re_exes = re.compile(r"^[Xx\s]+$")
re_irish = re.compile(r"DE LA O \w+|.*(^|\s)(O['\s]+\w{2,})(\s|$)")
re_mac = re.compile(r".*(^|\s)(MA?C\s\w{2,})\s")
re_dela_aos = re.compile(r'(^|\s)(DE LA [AOS]{1})(\s|$)')
re_solo_d = re.compile(r"(^|\s)(D['\s]+\w{2,})(\s|$)")  # ~600
re_solo_l = re.compile(r"(^|\s)(L['\s]+\w{2,})(\s|$)")  # ~35
# ~20; sometimes "DE L HERMITE", others "D L ANGELES"
re_del_broken = re.compile(r'(^|\s)(DE L) \w.*')
# ^^ alternatively, could "D L ANGELES" be an abbreviation for "DE LOS ANGELES" ?

def fix_nombre(nombre):
    """ Fixes multiple issues with spacing, special characters, etc.

    Args:
        nombre  : string of citizen name

    Returns:
        cleaned-up version of input string.

    """

    # blank/null return empty string
    if isinstance(nombre, float):
        nombre = ""

    # accents, enyes, etc are ALMOST always used.  But better to ditch them
    nombre = unidecode.unidecode(nombre)

    # The surnames "DE LA A" and "DE LA O" exist, and are a plague.  Fix them now
    # 2020/09/15... so does "DE LA S"
    m_dela = re_dela_aos.search(nombre)
    if m_dela:
        new = '-'.join(m_dela.group(2).split()).strip()
        parts = [x.strip() for x in nombre.split(m_dela.group(2))]
#        nombre = parts[0] + new + parts[1]
        nombre = " ".join([parts[0], new, parts[1]]).strip()

    ## remove apostrophe/space from irish surnames (e.g O'BRIAN ==> OBRIAN)
    ## complicated because "DE LA O BRIAN" could be "O'BRIAN", so I have to play tricks with the grouping
    ## NB - I'm now handling "DE LA O" directly, could make this more like the others
    m = re_irish.match(nombre)
    if m:
        g = m.group(2)
        if g:
            new = "".join(g.split("'"))
            new = "".join(new.split())
            parts = [x.strip() for x in nombre.split(g)]
            nombre = " ".join([parts[0], new, parts[1]]).strip()

    # similarly, fix MAC/MC names
    mac = re_mac.match(nombre)
    if mac:
        g = mac.group(2)
        if g:
            new = "".join(g.split())
            parts = [x.strip() for x in nombre.split(g)]
            nombre = " ".join([parts[0], new, parts[1]]).strip()

    # plenty of D'ARTAN as well
    m_solo_d = re_solo_d.search(nombre)
    if m_solo_d:
        g = m_solo_d.group(2)
        if g:
            new = "".join(g.split())
            parts = [x.strip() for x in nombre.split(g)]
            nombre = " ".join([parts[0], new, parts[1]]).strip()

    # ditch weird characters before proceeding
    nombre = ''.join(re_oddchars.split(nombre))

    if re_star.match(nombre):
        nombre = ""
    if re_exes.match(nombre):
        nombre = ""
    if re_starplus.match(nombre):
        nombre = "**" + re_starplus.match(nombre).group(2) + "**"
    if " - " in nombre:
        nombre = "-".join(nombre.split(" - "))
    if " -" in nombre:
        nombre = "-".join(nombre.split(" -"))
    if "- " in nombre:
        nombre = "-".join(nombre.split("- "))
    return nombre

def load_registry(filepath_raw, logger, N_ROWS=None):

    rf = pd.read_csv(filepath_raw, sep='\t', encoding='utf-8',
                     parse_dates=utils.get_date_cols(), dtype=utils.get_dtypes_reg(),
                     keep_default_na=False, na_values=utils.get_nan_values(),
                     nrows=N_ROWS,
                     )
    # replace NaN with empty string
    text_cols = ['nombre', 'nombre_spouse', 'nombre_padre',
                 'nombre_madre', 'ced_spouse', 'ced_padre', 'ced_madre']
    rf[text_cols] = rf[text_cols].fillna("")
    return rf



def clean_nombres(rf, folder_interim, 
    namecols = ['nombre', 'nombre_spouse', 'nombre_padre', 'nombre_madre']):
    """ Cleans up all of the columns containing names.

    Args:
        rf              : dataframe of citizen info
        folder_interim  : location of interim data; we check for 'allnames.tsv' there
        namecols        : list of columns which the cleaning will be applied to

    Returns:
        original dataframe, but the name columns have been cleaned up
    """

    
#    folder_interim = os.path.split(filepath_raw)[0]
    compound_names = get_compound_names(folder_interim)
    print("# of compound_names:", len(compound_names))

    # cleanup names
    print("Cleaning nombre columns")
    for col in namecols:
        print('\t' + col)
        rf[col] = rf[col].apply(lambda x: fix_nombre(x))
        rf[col] = rf[col].apply(lambda x: add_underscores(x, compound_names))
    return rf

