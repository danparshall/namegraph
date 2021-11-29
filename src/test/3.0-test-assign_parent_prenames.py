"""
Assign parent prenames
"""

# We've already identified the "first surname" (i.e. father's surname) for each parent, via comparison to the citizen. Now we inspect the parent's name more closely, and try to determine which tokens are prenames vs additional surname.
# We take advantage of knowing how often names are used as prenames or surnames with the citizens in general. Basically we ask for each token "if this were a prename (surname), would this name be more likely overall". I'm pretty sure it's a MAXENT kinda thing.

from collections import Counter
import math
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import re
import datetime as dt

import unidecode
from fuzzywuzzy import fuzz

# enable progress bar on long operations
from tqdm.auto import tqdm
tqdm.pandas()

full_run = True
N_ROWS = None  # 1000000
READ_DATE = '20211122'

LOC_RAW = "../../data/testdata/"
LOC_INTERIM = "../../data/testdata/interim/"

TODAY = dt.datetime.now().strftime("%Y%m%d")

TODAY = READ_DATE
TODAY

nan_values = ['-1.#IND', '1.#QNAN', '1.#IND', '-1.#QNAN', '#N/A N/A', '#N/A', 'N/A', 'n/a',  # 'NA' is sometimes name
              '<NA>', '#NA', 'NULL', 'null', 'NaN', '-NaN', 'nan', '-nan', '']


dtypes_reg = {"cedula":   str,
              "nombre":   str,
              "gender":  'category',
              "dt_birth": str,
              "nationality":   'category',
              "nombre_spouse":  str,
              "dt_death":       str,
              "orig_cedula":    str,
              "marital_status": 'category',
              "nombre_padre":	str,
              "nombre_madre":	str,
              "ced_spouse":	str,
              "ced_padre":	str,
              "ced_madre":	str,
              "dt_marriage":	str
              }

usecols_reg = ['cedula', 'nombre', 'gender',
               'nombre_spouse', 'nombre_padre', 'nombre_madre']

# ~80 sec

dtypes_names = {'cedula': str, 'sur_padre': str, 'sur_madre': str, 'prenames': str,
                'has_padre': bool, 'has_madre': bool, 'is_plegal': 'category', 'is_mlegal': 'category'
                }
nf = pd.read_csv(LOC_INTERIM + "names_cleaned_" + READ_DATE + ".tsv", sep='\t',
                 dtype=dtypes_names,
                 keep_default_na=False, na_values=nan_values,
                 nrows=N_ROWS
                 )
print("# NF recs :", len(nf))

if True:
    nf.loc[nf.sur_padre.isnull(), 'sur_padre'] = ""
    nf.loc[nf.sur_madre.isnull(), 'sur_madre'] = ""
    nf.loc[nf.prenames.isnull(), 'prenames'] = ""
    nf['nlen_pre'] = nf.prenames.map(lambda x: len(x.split()))
    nf['is_plegal'] = nf.is_plegal.map(
        lambda x: np.nan if x is np.nan else bool(x))
    nf['is_mlegal'] = nf.is_mlegal.map(
        lambda x: np.nan if x is np.nan else bool(x))


# ~90 sec
nf.drop(['n_char_nombre', 'n_char_prenames', 'nlen_pre'], axis=1, inplace=True)
nf.head()
(nf.has_padre & nf.has_madre & (nf.sur_padre != "") & (nf.sur_madre != "")).sum()
nf.is_plegal.value_counts()

"""
Load prenames from regular data
"""
file_freq = LOC_INTERIM + "NEWFREQFILE_" + READ_DATE + ".tsv"

freq = pd.read_csv(file_freq, sep='\t', dtype=str)
len(freq)


def count_all_names(freq):
    tmp = pd.concat([freq.sur_padre, freq.sur_madre], axis=0).value_counts()
    count_sur = pd.DataFrame({'obsname': tmp.index, 'n_sur': tmp.values})
    tmp = pd.concat([freq.pre1, freq.pre2], axis=0).value_counts()
    count_pre = pd.DataFrame({'obsname': tmp.index, 'n_pre': tmp.values})

    count_names = count_sur.merge(count_pre, on='obsname', how='outer')
    count_names.fillna(0, inplace=True)

    # add null record, so that null names get weight factor of 1
    count_names.loc[count_names.obsname == "", ['n_sur', 'n_pre']] = 0

    count_names['n_sur'] = count_names.n_sur + 0.5
    count_names['n_pre'] = count_names.n_pre + 0.5

    count_names['sratio'] = count_names.n_sur / count_names.n_pre
    count_names['pratio'] = count_names.n_pre / count_names.n_sur

    return count_names

ncounts = count_all_names(freq)
ncounts[:20]
ncounts[ncounts.obsname.map(lambda x: len(x.split()) > 1)][-50:]

def merge_underscore_names(ncounts):
    under_prenames = set(
        ncounts[ncounts.obsname.map(lambda x: "_" in x)].obsname)

    for upre in tqdm(under_prenames):

        u_rec = ncounts[ncounts.obsname == upre].iloc[0]

        norm_pre = ' '.join(upre.split("_"))
        norm_rec = ncounts[ncounts.obsname == norm_pre]
        if len(norm_rec) == 1:
            norm_rec = norm_rec.iloc[0]
            ncounts.loc[ncounts.obsname == norm_pre,
                        'n_sur'] = u_rec.n_sur + norm_rec.n_sur - 0.5
            ncounts.loc[ncounts.obsname == norm_pre,
                        'n_pre'] = u_rec.n_pre + norm_rec.n_pre - 0.5
        elif len(norm_rec) == 0:
            tmp = u_rec.copy(deep=True)
            tmp.obsname = norm_pre
            ncounts = ncounts.append(tmp)

    ncounts = ncounts[~ncounts.obsname.isin(under_prenames)]
    ncounts['sratio'] = ncounts.n_sur/ncounts.n_pre
    ncounts['pratio'] = ncounts.n_pre/ncounts.n_sur

    subspace = ncounts[ncounts.obsname.map(lambda x: " " in x)].copy(deep=True)
    subspace['obsname'] = subspace.obsname.map(lambda x: "_".join(x.split()))
    return pd.concat([ncounts, subspace], axis=0)


allnames = pd.read_csv(LOC_INTERIM + "ALLNAMES_" +
                       READ_DATE + ".tsv", sep='\t')
allnames[allnames.obsname == 'RODRIGUEZ']  # use as prename looks legit

"""
Identify "multinames" for madres/padres
"""

funky_prenames = set()
with open(LOC_INTERIM + "funky_prenames_" + READ_DATE + ".txt") as f:
    for line in f:
        funky_prenames.add(line.strip())

funky_prenames = list(funky_prenames)
funky_prenames.sort(reverse=True, key=len)
print("# funkies :", len(funky_prenames))


def fix_funk(nombre, funks):
    """ The 'funks' list should be sorted in descending length, to prevent substrings from being clobbered.
    
    NB: there's a potential bug in here, bc the list is sorted according to character length, but checks
    here are being done according to number of tokens.  But very unlikely to cause an issue, so ignoring for now
    """
    nlen = len(nombre.split())
    for funk in funks:
        flen = len(funk.split())
        if (nlen > flen):
            if (funk in nombre):
                defunk = '_'.join(funk.split())
                nombre = defunk.join(nombre.split(funk))
                nlen = len(nombre.split())
        else:
            # since the list is sorted, once we have a match that uses all the tokens, just skip ahead
            continue
    return nombre


def parse_pres(pres, pname, row=None):

    if len(pres.split()) == 1:
        pname['pre1'] = pres
    elif len(pres.split()) == 2:
        pname['pre1'], pname['pre2'] = pres.split()
    else:
        pres = fix_funk(pres, funky_prenames).split()
        if len(pres) == 0:
            print(
                "WTF at {0} - PRES: {1} PNAME: {2}".format(row.cedula, pres, pname))
            pass
        else:
            pname['pre1'] = pres[0]
            if len(pres) > 1:
                pname['pre2'] = pres[1]
                if len(pres) > 2:
                    pname['pre3'] = pres[2]
                    if len(pres) > 3:
                        pname['junk'] = ' '.join(pres[4:])
    return pname


def assign_pres(pres, pname):
    pname['pre1'] = pres[0]
    if len(pres) > 1:
        pname['pre2'] = pres[1]
        if len(pres) > 2:
            pname['pre3'] = pres[2]
            if len(pres) > 3:
                pname['junk'] = ' '.join(pres[4:])
    return pname

# this uses dict lookup, rather than dataframe lookup.  Faster by ~100x

wts_pre = dict()
wts_sur = dict()

for _, row in allnames.iterrows():
    wts_pre[row.obsname] = row.pratio
    wts_sur[row.obsname] = row.sratio


def calc_evidence(pres, surs, wts_pre, wts_sur):
    """ Calculate the evidence for a particular configuration of prename/surname allocation. """
    evidence = 1
    for pre in pres:
        try:
            wt_pre = wts_pre[pre]
        except KeyError:
            wt_pre = 1
        evidence = evidence * wt_pre

    for sur in surs:
        try:
            wt_sur = wts_sur[sur]
        except KeyError:
            wt_sur = 1
        evidence = evidence * wt_sur
    return evidence


# 2 minutes to load
# we'll only need these to check names at the start of a string
# these results are subset of "re_beg_vande"
re_beg_von = re.compile(u"^(V[AO]N \w{2,})(\s|$)")
re_beg_vande = re.compile(u"^(V[AO]N DE[RN]? \w{2,})(\s|$)")
# SANTA and SAN (in lieu of SANTO)
re_beg_sant = re.compile(u"^(SANT?A? \w{2,})(\s|$)")
# these results are subset of "re_beg_laos"
re_beg_dela = re.compile(u"^(DE L[AO]S? ?\w{2,})(\s|$)")
re_beg_laos = re.compile(u"^(L[AEO]S? \w{2,})(\s|$)")
re_beg_del = re.compile(u"^(DEL \w{2,})(\s|$)")
re_beg_de = re.compile(r"^(DE \w{2,})(\s|$)")


def get_starting_multimatch(nombre):
    """ Check if the beginning of the string is a multiname. If so, return the multiname."""

    mdela = re_beg_dela.match(nombre)
    if mdela:
        return mdela.group(1)

    mlaos = re_beg_laos.match(nombre)
    if mlaos:
        return mlaos.group(1)

    mde = re_beg_de.match(nombre)
    if mde:
        return mde.group(1)

    mdel = re_beg_del.match(nombre)
    if mdel:
        return mdel.group(1)

    mvande = re_beg_vande.match(nombre)
    if mvande:
        return mvande.group(1)

    mvon = re_beg_von.match(nombre)
    if mvon:
        return mvon.group(1)

    return ""


# we'll only need these to check names at the end of a string
# these results are subset of "re_end_vande"
re_end_von = re.compile(u".*\s(V[AO]N \w{2,})$")
re_end_vande = re.compile(u".*\s(V[AO]N DE[RN]? \w{2,})$")
# SANTA and SAN (in lieu of SANTO)
re_end_sant = re.compile(u".*\s(SANT?A? \w{2,})$")
# these results are subset of "re_end_laos"
re_end_dela = re.compile(u".*\s(DE L[AO]S? ?\w{2,})$")
re_end_laos = re.compile(u".*\s(L[AEO]S? \w{2,})$")
re_end_del = re.compile(u".*\s(DEL \w{2,})$")
re_end_de = re.compile(r".*\s(DE \w{2,})$")


def get_ending_multimatch(nombre):
    """ Check if the end of the string is a multiname. If so, return the multiname."""

    mdela = re_end_dela.match(nombre)
    if mdela:
        return mdela.group(1)

    mlaos = re_end_laos.match(nombre)
    if mlaos:
        return mlaos.group(1)

    mde = re_end_de.match(nombre)
    if mde:
        return mde.group(1)

    mdel = re_end_del.match(nombre)
    if mdel:
        return mdel.group(1)

    mvande = re_end_vande.match(nombre)
    if mvande:
        return mvande.group(1)

    mvon = re_end_von.match(nombre)
    if mvon:
        return mvon.group(1)

    return ""


def best_split_by_evidence(my_pres, row, verbose=False):
    #     As is, this function flags around 1 name per 1000 as having issues.
    flag_split = False
    best_split = None    # setting a default, for weird cases I didn't think of
    evids = []
    for ind in range(len(my_pres)):
        surs = my_pres[:ind]
        pres = my_pres[ind:]
        evids.append(calc_evidence(pres, surs, wts_pre, wts_sur))

    if len(evids) > 0:
        best_split = np.argmax(evids)
        if (best_split > 1):
            flag_split = True
            if verbose:
                print("WARNING at ced {0} - split at {1} implies three surnames : {2}".format(
                    row.cedula, best_split, my_pres))
    else:
        print("WARNING at ced {0} - no evids".format(row.cedula))
        flag_split = True
    return best_split, flag_split


def extract_prename_parent(row, target_col):
    """ 

    """
    target_sur = 'sur_' + target_col.split("_")[1]
    sur1 = row[target_sur]
    pname = {'cedula': row.cedula,
             'sur1': sur1, 'sur2': "", 'pre1': "",
             'pre2': "", 'pre3': "", 'junk': "", 'flag': False}

    # use direct string-count method, to handle "DE LA CRUZ", etc
    n_dup = row[target_col].count(sur1)
    if n_dup > 2:
        print("ERROR COUNTING DUPS :", row)
        # there's also "CHEN CHEN SHU CHEN"

        # workaround for things like "LEON LEON ALEXANDRA LEONOR"
        is_doubled = row[target_col].split().count(sur1) >= 2
    else:
        is_doubled = n_dup == 2
    if is_doubled:
        pname['sur2'] = sur1

    is_pstart = row[target_col].startswith(sur1)
    is_pend = row[target_col].endswith(sur1)

    if is_pend:
        # name is in normal form; sur2 not present, so everything before sur1 is a prename
        # works even if sur1==sur2
        pres = ''.join(row[target_col].split(sur1)).strip()
        pname = parse_pres(pres, pname, row)

        if False:
            if is_doubled:
                pname['sur2'] = sur1
            else:
                pname['sur2'] = ""

    elif not is_pstart:
        # name is in normal form, sur2 follows sur1, everything before sur1 is a prename
        parts = [x.strip() for x in row[target_col].split(sur1, maxsplit=1)]

        if len(parts) > 1:
            pname['sur2'] = parts[1]
            pres = row[target_col].split(sur1)[0]
            pname = parse_pres(pres, pname, row)
        else:
            pname['sur2'] = "WTF MPARSE ERROR"
            pname['flag'] = True

    elif is_pstart:
        # name is in legal form, could be short or long version
        pres = ''.join(row[target_col].split(sur1)).strip()

        if len(pres.split()) == 1:
            # easy case, only thing left must be the prename
            pname['pre1'] = pres

        elif is_doubled:
            # special case when sur1 == sur2, only thing to do is figure out the prenames
            pname = parse_pres(pres, pname, row)

        else:
            # harder case.  Could be "sur1 sur2 pre1 pre2 ..." or "sur1 pre1 pre2 ..."
            pre1 = ""  # init value; might get clobbered

            # first check if the starting chunk is a multipart name
            m_beg = get_starting_multimatch(pres)
            if m_beg:
                # if it _IS_ a multipart name, it has to be a surname (else it would follow the other prenames)
                pname['sur2'] = '_'.join(m_beg.split())
                pres = ''.join(pres.split(m_beg))
                parts = pres.split()

                if len(parts) > 0:
                    pre1 = parts[0]
                    if len(parts) > 1:
                        pres = ''.join(parts[1:]).strip()
                else:
                    # this triggers when the entire chunk matches a multipart format
                    print(
                        "WTF at {0} - THIS SHOULDNT HAPPEN : {1}".format(row, row[target_col]))
                    pname['flag'] = True

                    # as a hack, just assume that the final token is a first name.
                    pname['pre1'] = m_beg.split()[-1]
                    pname['sur2'] = ' '.join(m_beg.split()[:-1])
                    return pname

            # now check if ending is multipart
            m_end = get_ending_multimatch(pres)
            if m_end:
                # if this person DOES have a multipart prename, it will be at the end.  Extract and continue
                pres = ''.join(pres.split(m_end))
                m_end = '_'.join(m_end.split())
                pres = pres + ' ' + m_end

            # everything left is either a singleton surname, or a prename
            if pre1:
                # the 'pre1' is determined in the "m_beg" stage
                my_pres = [pre1] + pres.split()
            else:
                my_pres = pres.split()

            best_split, flag_split = best_split_by_evidence(my_pres, row)
            pname['flag'] = flag_split

            if best_split:
                pres = my_pres[best_split:]
                pname = assign_pres(pres, pname)
                surs = my_pres[:best_split]
                if surs:
                    pname['sur2'] = surs[0]

    return pname

### Surnames only found for either madre/padre. Likely some are errors

surs_madre = set(nf.sur_madre)
len(surs_madre)
surs_padre = set(nf.sur_padre)
len(surs_padre)
surs_both = surs_padre & surs_madre
len(surs_both)
surs_padre - surs_madre
surs_madre - surs_padre

"""
Get padre/madre prenames
"""
if full_run:
    padre_prenames = nf.progress_apply(lambda row: extract_prename_parent(row, 'nombre_padre'), 
                                       axis=1, result_type='expand')
    padre_prenames.to_csv(LOC_INTERIM+"PADRES_" + TODAY + ".tsv", sep='\t', index=False)
else:
    padre_prenames = pd.read_csv(LOC_INTERIM+'PADRES_' + READ_DATE + ".tsv", sep='\t')

# 2h
# del padre_prenames

if full_run:
    madre_prenames = nf[:1000].progress_apply(lambda row: extract_prename_parent(row, 'nombre_madre'), 
                                           axis=1, result_type='expand')
    madre_prenames.to_csv(LOC_INTERIM+"MADRES_" + TODAY + ".tsv", sep='\t', index=False)
else:
    madre_prenames = pd.read_csv(LOC_INTERIM+'MADRES_' + READ_DATE + ".tsv", sep='\t')

# 2h
if 'cedula' not in padre_prenames.columns:
    padre_prenames['cedula'] = nf.cedula.copy()
len(nf)

"""
Test PADRES
"""
new_types = {'cedula': str, 'sur1': str, 'sur2': str, 'pre1': str,
             'pre2': str, 'pre3': str, 'junk': str, 'flag': bool
             }

test_cases_PADRES = pd.read_csv(LOC_RAW + "simpsons_test_cases_padres.tsv",
                                sep='\t', dtype=new_types, keep_default_na=False, na_values=nan_values)
test_cases_PADRES.fillna('',inplace = True)

for i in range(2):
    print(".................................")
print("padres test")

for row in test_cases_PADRES.to_dict(orient='records'):
    try:
        res = padre_prenames[padre_prenames.cedula == 
                                row['cedula']].iloc[0].to_dict()
    except IndexError:
        print(row['cedula'], ": NOT IN THIS SUBFRAME")
        continue
    if res == row:
        print(row['cedula'], ": OK")
    else:
        print(row['cedula'], ": FAILED")
        print(row)
        print(res)

"""
Test MADRES
"""
new_types = {'cedula': str, 'sur1': str, 'sur2': str, 'pre1': str,
             'pre2': str, 'pre3': str, 'junk': str, 'flag': bool
             }

test_cases_MADRES = pd.read_csv(LOC_RAW + "simpsons_test_cases_madres.tsv",
                                  sep='\t', dtype=new_types, keep_default_na=False, na_values=nan_values)
test_cases_MADRES.fillna('',inplace = True)

for i in range(2):
    print(".................................")
print("madres test")

for row in test_cases_MADRES.to_dict(orient='records'):
    try:
        res = madre_prenames[madre_prenames.cedula ==
                             row['cedula']].iloc[0].to_dict()
    except IndexError:
        print(row['cedula'], ": NOT IN THIS SUBFRAME")
        continue
    if res == row:
        print(row['cedula'], ": OK")
    else:
        print(row['cedula'], ": FAILED")
        print(row)
        print(res)



"""
Attempt matching (should do this in another notebook)
"""
nan_values = ['-1.#IND', '1.#QNAN', '1.#IND', '-1.#QNAN', '#N/A N/A', '#N/A', 'N/A', 'n/a',  # 'NA' is sometimes name
              '<NA>', '#NA', 'NULL', 'null', 'NaN', '-NaN', 'nan', '-nan', '']


# read cleaned-up input file
print("LOADING REG DATA FOR : " + READ_DATE)
dtypes_reg = {'cedula': str, 'nombre': str, 'gender': 'category', 'nationality': 'category',
              'orig_cedula': str, 'marital_status': 'category',
              'nombre_spouse': str, 'nombre_padre': str, 'nombre_madre': str,
              'ced_spouse': str, 'ced_padre': str, 'ced_madre': str
              }

usecols = ['cedula', 'dt_birth', 'dt_death', 'dt_marriage', 'nombre_spouse', ]
rf = pd.read_csv(LOC_RAW + "simpsons_test_cases.tsv", sep='\t', dtype=dtypes_reg,
                 parse_dates=['dt_birth', 'dt_death', 'dt_marriage'], usecols=usecols,
                 keep_default_na=False, na_values=nan_values,
                 nrows=N_ROWS
                 )
print("Loaded {0} rows".format(len(rf)))
rf.head()

nf = nf.merge(rf, how='left', on='cedula')
nf['nombre_spouse'] = nf.nombre_spouse.fillna('')
del rf
# padre_bothsur = padre_prenames[(padre_prenames.sur2 != "") & (
#     nf.dt_birth >= dt.datetime(1960, 1, 1))]
# len(padre_bothsur)
# targets = nf[nf.cedula.isin(set(padre_bothsur.cedula))]
# print(targets.head())
# target = targets.iloc[1]
# target_padre = padre_prenames[padre_prenames.cedula == target.cedula].iloc[0]
# target_padre
# padre_prenames.head()
# obv_padres = nf[nf.has_padre & nf.is_plegal & (nf.nlen_padre == 4)][[
#     'cedula', 'nombre_padre']]
# obv_padres

# whoa = nf.merge(obv_padres.rename(
#     columns={'cedula': 'ced_kid', 'nombre_padre': 'nombre'}), on='nombre')
# len(whoa)

nan_values = ['-1.#IND', '1.#QNAN', '1.#IND', '-1.#QNAN', '#N/A N/A', '#N/A', 'N/A', 'n/a',  # 'NA' is sometimes name
              '<NA>', '#NA', 'NULL', 'null', 'NaN', '-NaN', 'nan', '-nan', '']

# read cleaned-up input file
print("LOADING REG DATA FOR : " + READ_DATE)
dtypes_reg = {'cedula': str, 'nombre': str, 'gender': 'category', 'nationality': 'category',
              'orig_cedula': str, 'marital_status': 'category',
              'nombre_spouse': str, 'nombre_padre': str, 'nombre_madre': str,
              'ced_spouse': str, 'ced_padre': str, 'ced_madre': str
              }

usecols = ['cedula', 'ced_padre', ]
rf = pd.read_csv(LOC_RAW + "simpsons_test_cases.tsv", sep='\t', dtype=dtypes_reg,
                 usecols=usecols,
                 keep_default_na=False, na_values=nan_values,
                 nrows=N_ROWS
                 )
print("Loaded {0} rows".format(len(rf)))
# whoa = whoa.merge(rf, on='cedula', how='left', suffixes=('_pred', '_obs'))
# whoa.sample(30)
# nf.head()
# sub_pad = nf[(nf.sur_padre == target_padre.sur1) & (nf.sur_madre == target_padre.sur2)
#              & (nf.gender == 1)
#              & (nf.dt_birth <= dt.datetime(target.dt_birth.year - 13, target.dt_birth.month, target.dt_birth.day))
#              ]
# sub_pad
# sub_mad = nf[(nf.sur_padre == "VERGARA") & (nf.gender == 2)
#              & (nf.dt_birth <= dt.datetime(target.dt_birth.year - 13, target.dt_birth.month, target.dt_birth.day))]
# len(sub_mad)

# whoa_padre = nf[['cedula', 'nombre']].rename({'cedula': 'ced_padre', 'nombre': 'nombre_padre'}, axis=1
#                                              ).merge(nf.loc[(nf.nlen_padre == 4), ['cedula', 'nombre_padre']], on='nombre_padre')
# print("# naive-matched padre :", len(whoa_padre))

# whoa_madre = nf[['cedula', 'nombre']].rename({'cedula': 'ced_madre', 'nombre': 'nombre_madre'}, axis=1
#                                              ).merge(nf.loc[(nf.nlen_padre == 4), ['cedula', 'nombre_madre']], on='nombre_madre')
# print("# naive-matched madre :", len(whoa_padre))
"""
if padre starts with surname, it's in legal form.  Check the evidence for other name chunks
if padre ends with surname, it's in shortened normal form.  

if neither of those, then it's in long normal form.  
So everything after the surname should be grandmother, and everything before will be prenames

"""


def extract_prename(row):
    """ ORiginal version. So. fuckyng. slow (1 hour per million)"""
    sur1 = row.sur_padre
    pname = {'sur1': sur1, 'sur2': "", 'pre1': "",
             'pre2': "", 'pre3': "", 'junk': ""}

    is_pstart = row.nombre_padre.startswith(sur1)
    is_pend = row.nombre_padre.endswith(sur1)

    if is_pend:
        # name is in normal form; sur2 not present, so everything before sur1 is a prename
        pres = ''.join(row.nombre_padre.split(sur1))
        sur2 = ""
        pname = parse_pres(pres, pname)

    elif not is_pstart:
        # name is in normal form, sur2 follows sur1, everything before sur1 is a prename
        parts = row.nombre_padre.split(sur1)

        if len(parts) > 1:
            pname['sur2'] = parts[1]
            pres = row.nombre_padre.split(sur1)[0]
            pname = parse_pres(pres, pname)
        else:
            pname['sur2'] = "WTF PPARSE ERROR"

    elif is_pstart:
        # name is in legal form, could be short or long version
        pres = ''.join(row.nombre_padre.split(sur1))

        if len(pres.split()) == 1:
            # easy case, only thing left must be the prename
            pname['pre1'] = pres

        else:
            # harder case.  Could be "sur1 sur2 pre1 pre2 ..." or "sur1 pre1 pre2 ..."
            pres = fix_funk(pres, funky_prenames).split()

            # check if second token is more likely to be a prename or surname
            name_ratio = ncounts[ncounts.obsname == pres[0]]
            if (len(name_ratio) > 0) and (name_ratio.iloc[0].sratio > 1):
                pname['sur2'] = pres[0]
                pres = pres[1:]

            # everything left is a prename
            pname['pre1'] = pres[0]
            if len(pres) > 1:
                pname['pre2'] = pres[1]
                if len(pres) > 2:
                    pname['pre3'] = pres[2]
                    if len(pres) > 3:
                        pname['junk'] = ' '.join(pres[4:])

    return pname


def extract_prename(row):
    """ Followup.  I suspect the slowdown happens in the 'elif pstart' clause
    """
    sur1 = row.sur_padre
    pname = {'sur1': sur1, 'sur2': "", 'pre1': "",
             'pre2': "", 'pre3': "", 'junk': ""}

    is_pstart = row.nombre_padre.startswith(sur1)
    is_pend = row.nombre_padre.endswith(sur1)

    if is_pend:
        # name is in normal form; sur2 not present, so everything before sur1 is a prename
        pres = ''.join(row.nombre_padre.split(sur1)).strip()
        sur2 = ""
        pname = parse_pres(pres, pname)

    elif not is_pstart:
        # name is in normal form, sur2 follows sur1, everything before sur1 is a prename
        parts = row.nombre_padre.split(sur1)

        if len(parts) > 1:
            pname['sur2'] = parts[1]
            pres = row.nombre_padre.split(sur1)[0]
            pname = parse_pres(pres, pname)
        else:
            pname['sur2'] = "WTF PPARSE ERROR"

    elif is_pstart:
        # name is in legal form, could be short or long version
        pres = ''.join(row.nombre_padre.split(sur1)).strip()

        if len(pres.split()) == 1:
            # easy case, only thing left must be the prename
            pname['pre1'] = pres

        else:
            # harder case.  Could be "sur1 sur2 pre1 pre2 ..." or "sur1 pre1 pre2 ..."

            pre1 = ""  # init value; might get clobbered

            # first check if the starting chunk is a multipart name
            m_beg = get_starting_multimatch(pres)
            if m_beg:
                # if it _IS_ multipart name, it must be surname (else it would be at the end, following prenames)
                pname['sur2'] = m_beg
                pres = ''.join(pres.split(m_beg))
                parts = pres.split()
                if len(parts) > 0:
                    pre1 = parts[0]

                    if len(parts) > 1:
                        pres = ''.join(parts[1:]).strip()

                else:
                    print("WTF THIS SHOULDNT HAPPEN :", row.nombre_padre)

            m_end = get_ending_multimatch(pres)
            if m_end:
                # if this person DOES have a multipart, it will be at the end.  Extract and continue
                pres = ''.join(pres.split(m_end))
                m_end = '_'.join(m_end.split())

            # everything left is a prename
            # the 'pre1' is possibly determined in the "m_beg" stage
            pres = [pre1] + pres.split()
            pname['pre1'] = pres[0]
            if len(pres) > 1:
                pname['pre2'] = pres[1]
                if len(pres) > 2:
                    pname['pre3'] = pres[2]
                    if len(pres) > 3:
                        pname['junk'] = ' '.join(pres[4:])

    return pname


tst = nf[nf.nombre_padre.map(lambda x: "DE LA CRUZ " in x)]


def findit(row):
    return row.nombre_padre.startswith(row.sur_padre)


# tst[tst.apply(lambda row: findit(row), axis=1)]


def extract_parent_prename(row):
    out = {"pres_padre": "", "pres_madre": "",
           "freq_plegal": False, "freq_mlegal": False}

    is_pstart = row.nombre_padre.startswith(row.sur_padre)
    is_pend = row.nombre_padre.endswith(row.sur_padre)
    if is_pstart or is_pend:
        out['pres_padre'] = ''.join(
            row.nombre_padre.split(row.sur_padre)).strip()
        out['freq_plegal'] = is_pstart == row.is_plegal
    else:
        out['pres_padre'] = ' || '.join(row.nombre_padre.split(row.sur_padre))

    is_mstart = row.nombre_madre.startswith(row.sur_madre)
    is_mend = row.nombre_madre.endswith(row.sur_madre)
    if is_mstart or is_mend:
        out['pres_madre'] = ''.join(
            row.nombre_madre.split(row.sur_madre)).strip()
        out['freq_mlegal'] = is_mstart == row.is_mlegal
    else:
        out['pres_madre'] = ' || '.join(row.nombre_madre.split(row.sur_madre))

    return out


def extract_parent_prename(row):
    out = {"pres_padre": "", "pres_madre": "",
           "freq_plegal": False, "freq_mlegal": False}

    is_pstart = row.nombre_padre.startswith(row.sur_padre)
    is_pend = row.nombre_padre.endswith(row.sur_padre)
    if is_pstart or is_pend:
        out['pres_padre'] = ''.join(
            row.nombre_padre.split(row.sur_padre)).strip()
        out['freq_plegal'] = is_pstart == row.is_plegal

    else:
        out['pres_padre'] = ' || '.join(row.nombre_padre.split(row.sur_padre))

    is_mstart = row.nombre_madre.startswith(row.sur_madre)
    is_mend = row.nombre_madre.endswith(row.sur_madre)
    if is_mstart or is_mend:
        out['pres_madre'] = ''.join(
            row.nombre_madre.split(row.sur_madre)).strip()
        out['freq_mlegal'] = is_mstart == row.is_mlegal
    else:
        out['pres_madre'] = ' || '.join(row.nombre_madre.split(row.sur_madre))

    return out


def parse_parents(row, namecounts):
    EVIDENCE_THRESH = 10

    out = {'sur2_padre': "", 'pre1_padre': "", 'pre2_padre': "", 'pre3_padre': "", 'junk_padre': "",
           'sur2_madre': "", 'pre1_madre': "", 'pre2_madre': "", 'pre3_madre': "", 'junk_madre': "",
           'cedula': row.cedula
           }
    try:
        # father
        if row.is_plegal == True:
            tokens = row.nombre_padre.split()
            name_rec = namecounts[namecounts.obsname == tokens[1]]
            if len(name_rec) == 1:
                name_rec = name_rec.iloc[0]

            ppres = ' '.join(tokens[1:])
        else:
            ppres, psur2 = row.nombre_padre.split(row.sur_padre, maxsplit=1)

        pseq = ['pre1_padre', 'pre2_padre', 'pre3_padre', 'junk_padre']
        for ind, pre in enumerate(ppres.split()):
            if ind < 3:
                out[pseq[ind]] = pre
            else:
                out['junk_padre'] = ' '.join(ppres.split()[3:])
        out['sur2_padre'] = psur2

        # mother
        if row.is_mlegal == True:
            tokens = row.nombre_madre.split()
            msur2 = tokens[1]
            mpres = ' '.join(tokens[1:])
        else:
            mpres, msur2 = row.nombre_madre.split(row.sur_madre, maxsplit=1)
        mseq = ['pre1_madre', 'pre2_madre', 'pre3_madre', 'junk_madre']
        for ind, pre in enumerate(mpres.split()):
            if ind < 3:
                out[mseq[ind]] = pre
            else:
                out['junk_madre'] = ' '.join(mpres.split()[3:])
        out['sur2_madre'] = msur2

    except:
        print("\nWTF :\n", row)
    return out

"""
funcs
"""
# in all cases, we look for a word boundary as the first group, then our funky name as the second
# these results are subset of "re_vande"
re_von = re.compile(u"(^|\s)(V[AO]N \w{2,})(\s|$)")
re_vande = re.compile(u"(^|\s)(V[AO]N DE[RN]? \w{2,})(\s|$)")
re_sant = re.compile(u"(^|\s)(SANT?A? \w{2,})(\s|$)")
# these results are subset of "re_laos"
re_dela = re.compile(u"(^|\s)(DE L[AO]S? ?[AO]? ?\w{2,})(\s|$)")
re_laos = re.compile(u"(^|\s)(L[AEO]S? \w{2,})(\s|$)")
re_del = re.compile(u"(^|\s)(DEL \w{2,})(\s|$)")
re_de = re.compile(r"(^|\s)(DE \w{2,})(\s|$)")


def regex_funky(nombre):
    """ This is a little slow (~4mins / million rows), but pretty thorough.  """

    mdel = re_del.search(nombre)
    msant = re_sant.search(nombre)

    mlaos = re_laos.search(nombre)
    mdela = re_dela.search(nombre)

    mvon = re_von.search(nombre)
    mvande = re_vande.search(nombre)

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

    if poss_funks:
        for funk in poss_funks:
            funky_prenames.add(funk)
        return True
    else:
        return False


nf = rf.loc[(rf.sur_padre != "") & rf.has_padre & (rf.sur_madre != "") & rf.has_madre & (rf.prenames != ""),
            ['cedula', 'nombre', 'prenames', 'gender', 'nombre_padre', 'sur_padre', 'has_padre', 'nombre_madre', 'sur_madre', 'has_madre']]
len(nf)

funky_prenames = set()
nf['is_funky'] = nf.prenames.map(regex_funky)
print("# funkies :", len(funky_prenames))
funky_prenames = list(funky_prenames)
funky_prenames.sort(reverse=True, key=len)
len(funky_prenames)

def fix_funk(nombre, funks):
    """ The 'funks' list should be sorted in descending length, to prevent substrings from being clobbered.
    
    NB: there's a potential bug in here, bc the list is sorted according to character length, but checks
    here are being done according to number of tokens.  But very unlikely to cause an issue, so ignoring for now
    """
    nlen = len(nombre.split())
    for funk in funks:
        flen = len(funk.split())
        if (nlen >= flen):
            if (funk in nombre):
                defunk = '_'.join(funk.split())
                nombre = defunk.join(nombre.split(funk))
                nlen = len(nombre.split())
        else:
            # since the list is sorted, once we have a match that uses all the tokens, just skip ahead
            continue
    return nombre


nf.loc[nf.is_funky, 'prenames'] = nf[nf.is_funky].prenames.map(
    lambda x: fix_funk(x, funky_prenames))

def parse_prename(prenames):
    """ The surnames are parsed, but the prenames must """

    out = {'pre1': "", 'pre2': "", 'pre3': "", 'junk': ""}

    mdel = re_del.search(prenames)
    msant = re_sant.search(prenames)

    mlaos = re_laos.search(prenames)
    mdela = re_dela.search(prenames)

    mde = re_de.search(prenames)

    mvon = re_von.search(prenames)
    mvande = re_vande.search(prenames)

    curr_funks = set()
    if mdel:
        curr_funks.add(mdel.group(2))

    if mde:
        curr_funks.add(mde.group(2))
    if msant:
        curr_funks.add(msant.group(2))
    if mvon:
        # "VAN DE" types are a subset of "VAN" types
        if mvande:
            curr_funks.add(mvande.group(2))
        else:
            curr_funks.add(mvon.group(2))
    if mlaos:
        # "DE LA" type names are a subset of "LA" types
        if mdela:
            curr_funks.add(mdela.group(2))
        else:
            curr_funks.add(mlaos.group(2))

    # sort greedily, first by number of tokens, and then by number of characters
    curr_funks = list(curr_funks)
    curr_funks.sort(reverse=True, key=lambda x: (len(x.split()), len(x)))

    for funk in curr_funks:
        if len(funk) < len(prenames.split()):
            continue

        parts = prenames.split(funk)
        sub = "_".join(funk.split())
        if len(parts) == 2:
            if parts[0] == "":
                # match was at beginning of the string
                prenames = " ".join([sub] + parts)
            elif parts[1] == "":
                # match was at end of the string
                prenames = " ".join(parts + [sub])
            else:
                prenames = parts[0] + sub + parts[1]

        # these shouldn't happen
        elif len(parts) > 2:
            print("TOOLONG :", prenames, parts)

    # now assign name pices
    pres = prenames.split()
    if len(pres) >= 1:
        out['pre1'] = pres[0]
    if len(pres) >= 2:
        out['pre2'] = pres[1]
    if len(pres) >= 3:
        out['pre3'] = pres[2]
    if len(pres) >= 4:
        out['junk'] = ' '.join(pres[3:])
    return out


def parse_prename(prenames):
    """ The surnames are parsed, but the prenames must """

    out = {'pre1': "", 'pre2': "", 'pre3': "", 'junk': ""}

    # now assign name pices
    pres = prenames.split()
    if len(pres) >= 1:
        out['pre1'] = pres[0]
    if len(pres) >= 2:
        out['pre2'] = pres[1]
    if len(pres) >= 3:
        out['pre3'] = pres[2]
    if len(pres) >= 4:
        out['junk'] = ' '.join(pres[3:])
    return out


tmp = pd.concat([freq.sur_padre, freq.sur_madre], axis=0).value_counts()
count_sur = pd.DataFrame({'obsname': tmp.index, 'n_sur': tmp.values})
count_sur.sample(10)
tmp = pd.concat([freq.pre1, freq.pre2], axis=0).value_counts()
count_pre = pd.DataFrame({'obsname': tmp.index, 'n_pre': tmp.values})
count_pre.sample(20)
count_names = count_sur.merge(count_pre, on='obsname', how='outer')
count_names.fillna(0, inplace=True)

# add null record, so that null names get weight factor of 1
#count_names = count_names.append({'obsname':np.nan, 'n_sur':0, 'n_pre':0}, ignore_index=True)

count_names.loc[count_names.obsname == "", ['n_sur', 'n_pre']] = 0
count_names.sample(10)
count_names['n_sur'] = count_names.n_sur + 0.5
count_names['n_pre'] = count_names.n_pre + 0.5

count_names['sratio'] = count_names.n_sur / count_names.n_pre
count_names['pratio'] = count_names.n_pre / count_names.n_sur
count_names[count_names.obsname == ""]

### Investigate multinames

def is_name_multimatch(nombre):
    mdel = re_del.search(nombre)
    msant = re_sant.search(nombre)

    mlaos = re_laos.search(nombre)
    mdela = re_dela.search(nombre)

    mde = re_de.search(nombre)

    mvon = re_von.search(nombre)
    mvande = re_vande.search(nombre)

    if mdel or msant or mlaos or mdela or mde or mvon or mvande:
        return True
    else:
        return False

    """
    if mdel:
        curr_funks.add(mdel.group(2))
        
    if mde:
        curr_funks.add(mde.group(2))
    if msant:
        curr_funks.add(msant.group(2))
    if mvon:
        # "VAN DE" types are a subset of "VAN" types
        if mvande:
            curr_funks.add(mvande.group(2))
        else:
            curr_funks.add(mvon.group(2))
    if mlaos:
        # "DE LA" type names are a subset of "LA" types
        if mdela:
            curr_funks.add(mdela.group(2))
        else:
            curr_funks.add(mlaos.group(2))
    """


count_names['nlen'] = count_names.obsname.map(lambda x: len(x.split()))
count_names['is_multimatch'] = count_names.obsname.map(is_name_multimatch)
count_names[(count_names.nlen == 3) & count_names.is_multimatch]
sub = count_names[(count_names.nlen == 2) & ~count_names.is_multimatch]
len(sub)

sub = count_names[(count_names.nlen == 2) & ~count_names.is_multimatch]
len(sub)

def check_name_splitting(obsname):
    tokens = obsname.split()

    try:
        probably_sur = count_names[count_names.obsname ==
                                   tokens[0]].iloc[0].n_sur
    except IndexError:
        probably_sur = 0.5

    if tokens[1] == 'MARIA':
        probably_pre = 1107271.5
    else:
        try:
            probably_pre = count_names[count_names.obsname ==
                                       tokens[1]].iloc[0].n_sur
        except IndexError:
            probably_pre = 0.5

    return probably_sur / probably_pre


def calc_name_ratios(df):
    # count of surnames & prenames
    tmp = pd.concat([df.sur1, df.sur2], axis=0).value_counts()
    count_sur = pd.DataFrame({'obsname': tmp.index, 'n_sur': tmp.values})
    tmp = pd.concat([df.pre1, df.pre2], axis=0).value_counts()
    count_pre = pd.DataFrame({'obsname': tmp.index, 'n_pre': tmp.values})

    # merge to single list of names
    count_names = count_sur.merge(count_pre, on='obsname', how='outer')
    count_names.fillna(0, inplace=True)

    # add null record, so that null names get weight factor of 1
    count_names = count_names.append(
        {'obsname': np.nan, 'n_sur': 0, 'n_pre': 0}, ignore_index=True)

    # laplace prior
    count_names['n_sur'] = count_names.n_sur + 0.5
    count_names['n_pre'] = count_names.n_pre + 0.5

    # evidence for each instance being surname or prename
    count_names['sratio'] = count_names.n_sur / count_names.n_pre
    count_names['pratio'] = count_names.n_pre / count_names.n_sur

    return count_names


def calc_name_evidence(df, count_names):
    df['wt_sur1'] = df.join(count_names[['obsname', 'sratio']].set_index(
        'obsname'), on='sur1')['sratio']
    df['wt_sur2'] = df.join(count_names[['obsname', 'sratio']].set_index(
        'obsname'), on='sur2')['sratio']
    df['wt_pre1'] = df.join(count_names[['obsname', 'pratio']].set_index(
        'obsname'), on='pre1')['pratio']
    df['wt_pre2'] = df.join(count_names[['obsname', 'pratio']].set_index(
        'obsname'), on='pre2')['pratio']
    df['wt_pre3'] = df.join(count_names[['obsname', 'pratio']].set_index(
        'obsname'), on='pre3')['pratio']

    df['evidence'] = df.wt_sur1 * df.wt_sur2 * \
        df.wt_pre1 * df.wt_pre2 * df.wt_pre3
    return df


def alt_schemes_2len(row, threshold=100):
    new = row.copy()

    alt_evidence = 1/row.evidence
    if alt_evidence > threshold:
        new.sur1 = row.pre1
        new.pre1 = row.sur1
        new.wt_sur1 = 1/row.wt_pre1
        new.wt_pre1 = 1/row.wt_sur1
        new.evidence = alt_evidence
    return new


def alt_schemes_3len(row, threshold=100):

    new = row.copy()

    # first check to be sure this isn't a double-name
    if row.sur1 == row.sur2:
        return new

    comp_evids = [row.evidence]

    # s1 p1 p2 (person has 2 prenames; only middle column has wrong weighting)
    comp_evids.append(row.wt_sur1 * (1/row.wt_sur2) * row.wt_pre1)

    # p1 s1 s2 (sequence is wrong; middle col has correct weight)
    comp_evids.append((1/row.wt_sur1) * row.wt_sur2 * (1/row.wt_pre1))

    # p1 p2 s1 (everything is wrong )
    comp_evids.append((1/row.wt_sur1) * (1/row.wt_sur2) * (1/row.wt_pre1))

    # get index of permutation with highest evidence
    best_perm = np.array(comp_evids).argmax()
#    print(comp_evids)

    # check if best alternative is higher than our prior before continuing
    if comp_evids[best_perm] < threshold:
        pass

    elif best_perm == 0:
        # original version really was the best
        pass

    elif best_perm == 1:
        # s1 s2 p1
        # s1 p1 p2 (person has 2 prenames; only middle column has wrong weighting)
        new = row.copy()

        new.sur2 = np.nan
        new.pre1 = row.sur2
        new.pre2 = row.pre1

        new.evidence = comp_evids[best_perm]
        new.wt_sur2 = 1
        new.wt_pre1 = 1/row.wt_sur2
        new.wt_pre2 = 1/row.wt_pre1

    elif best_perm == 2:
        # s1 s2 p1
        # p1 s1 s2 (sequence is wrong; middle col has correct weight)
        new = row.copy()

        new.sur1 = row.sur2
        new.sur2 = row.pre1
        new.pre1 = row.sur1

        new.evidence = comp_evids[best_perm]
        new.wt_sur1 = row.wt_sur2
        new.wt_sur2 = 1/row.wt_pre1
        new.wt_pre1 = 1/row.wt_sur1

    elif best_perm == 3:
        # p1 p2 s1 (everything is wrong )

        new.sur1 = row.pre1
        new.sur2 = np.nan
        new.pre1 = row.sur1
        new.pre2 = row.sur2

        new.evidence = comp_evids[best_perm]
        new.wt_sur1 = 1/row.wt_pre1
        new.wt_sur2 = 1
        new.wt_pre1 = 1/row.wt_sur1
        new.wt_pre2 = 1/row.wt_sur2

    else:
        raise ValueError("How did we even get here?")

    return new


def alt_schemes_4len(row, threshold=100):

    new = row.copy()

    # first check to be sure this isn't a double-name
    if row.sur1 == row.sur2:
        return new

    ### calculate evidence in each possible configuration ###
    comp_evids = [row.evidence]

    # p1 p2 s1 s2 (every col has wrong weighting, so take reciprocal of all of them)
    comp_evids.append((1/row.wt_sur1) * (1/row.wt_sur2) *
                      (1/row.wt_pre1) * (1/row.wt_pre2) * row.wt_pre3)

    # s1 p1 p2 p3 (only second col has wrong weighting)
    comp_evids.append(row.wt_sur1 * (1/row.wt_sur2) *
                      row.wt_pre1 * row.wt_pre2 * row.wt_pre3)

    # p1 p2 p3 s1 (only third col _doesn't have the wrong weighting)
    comp_evids.append((1/row.wt_sur1) * (1/row.wt_sur2) *
                      row.wt_pre1 * (1/row.wt_pre2) * row.wt_pre3)

    # get index of permutation with highest evidence
    best_perm = np.array(comp_evids).argmax()

    if row.pre1 == row.pre2:
        # person has two last names, and reversed sequence. Force this even if it's a weak last name (eg DAMIAN)
        best_perm = 1

    # check if best alternative is higher than our prior before continuing
    if comp_evids[best_perm] < threshold:
        pass
    elif best_perm == 0:
        # original version really was the best
        pass

    elif best_perm == 1:
        # p1 p2 s1 s2 (flipped from normal)
        new.pre1 = row.sur1
        new.pre2 = row.sur2
        new.sur1 = row.pre1
        new.sur2 = row.pre2
        new.pre3 = np.nan
        new.evidence = comp_evids[best_perm]
        new.wt_pre1 = 1/row.wt_sur1
        new.wt_pre2 = 1/row.wt_sur2
        new.wt_sur1 = 1/row.wt_pre1
        new.wt_sur2 = 1/row.wt_pre2
        new.wt_pre3 = 1

    elif best_perm == 2:
        # s1 p1 p2 p3 (person has correct order, but 3 prenames)
        #        new.sur1 = row.sur1  # no change
        new.pre1 = row.sur2
        new.pre2 = row.pre1
        new.pre3 = row.pre2
        new.sur2 = np.nan
        new.evidence = comp_evids[best_perm]
#        new.wt_sur1 = row.wt_sur1
        new.wt_pre1 = 1/row.wt_sur2
        new.wt_pre2 = 1/row.wt_pre1
        new.wt_pre3 = 1/row.wt_pre2
        new.wt_sur2 = 1

    elif best_perm == 3:
        # p1 p2 p3 s1 (person has wrong order and 3 prenames)
        new.pre1 = row.sur1
        new.pre2 = row.sur2
        new.pre3 = row.pre1
        new.sur1 = row.pre2
        new.sur2 = np.nan

        new.evidence = comp_evids[best_perm]
        new.wt_pre1 = row.wt_sur1
        new.wt_pre2 = 1/row.wt_sur2
        new.wt_pre3 = 1/row.wt_pre1
        new.wt_sur1 = 1/row.wt_pre2
        new.wt_sur2 = 1

    else:
        raise ValueError("How did we even get here?")
    return new


def check_name_permutations(row, threshold=None):
    if threshold is None:
        threshold = np.min([1000*row.evidence, 100])

    if row.nlen == 2:
        return alt_schemes_2len(row, threshold)
    elif row.nlen == 3:
        return alt_schemes_3len(row, threshold)
    elif row.nlen == 4:
        return alt_schemes_4len(row, threshold)
    else:
        # don't bother for the hard ones
        return row


def name_ordering(row):
    row.nombre_padre.split()


freq['sur1'] = freq.sur1.astype('category')
freq['sur2'] = freq.sur2.astype('category')
freq['pre1'] = freq.pre1.astype('category')
freq['pre2'] = freq.pre2.astype('category')


def set_nanfree(vals):
    """ Makes a nan-free set from an iterable.
    
    NaN can behave wierd in sets/dicts, because there are actually 16M possible NaN.
    see https://github.com/numpy/numpy/issues/9358
    """
    myset = set()
    for val in vals:
        try:
            if not np.isnan(val):
                myset.add(val)
        except TypeError:
            myset.add(val)
    return myset


def prepare_catcol_merge(df, nf):
    sur1 = set_nanfree(nf.sur1.values)
    sur2 = set_nanfree(nf.sur2.values)
    pre1 = set_nanfree(nf.pre1.values)
    pre2 = set_nanfree(nf.pre2.values)

    df['sur1'] = df.sur1.cat.add_categories(sur1 - set(df.sur1.cat.categories))
    df['sur2'] = df.sur2.cat.add_categories(sur2 - set(df.sur2.cat.categories))
    df['pre1'] = df.pre1.cat.add_categories(pre1 - set(df.pre1.cat.categories))
    df['pre2'] = df.pre2.cat.add_categories(pre2 - set(df.pre2.cat.categories))
    return df


def prepare_catcol_merge(df, nf):
    new_sur1 = set(nf.sur1) - set(df.sur1.cat.categories)
    new_sur2 = set(nf.sur2) - set(df.sur2.cat.categories)
    new_pre1 = set(nf.pre1) - set(df.pre1.cat.categories)
    new_pre2 = set(nf.pre2) - set(df.pre2.cat.categories)

    df['sur1'] = df.sur1.cat.add_categories(new_sur1)
    df['sur2'] = df.sur2.cat.add_categories(new_sur2)
    df['pre1'] = df.pre1.cat.add_categories(new_pre1)
    df['pre2'] = df.pre2.cat.add_categories(new_pre2)
    return df

### Check output of swapping


# So the "parse_fullrow" method works so well, that this should only be used in cases where there weren't parent names to match.

# HOWEVER, it's still great for parsing out first/last names of the parents

# PARENT_PARSE:

# should be done by comparing the extracted surname against the position within the parent appelidos.  Probably ought to do male names first, and identify funky versions.  The extracted surname ought to correspond to position 0 (if in normal form) or to position N-1 (if in alternate form).  In alternate form, the remainder of the name can be safely assigned as a surname.  In standard form, follow the "madre approach" to identify the remainder.  Establish ratio of normal/alternate form.  Establish "mismatch ratio" for when parent forms aren't concordant.

# For the ~9 % of records without both matching parents, do a few other checks.
# 1) if the entirety of the parent name is found within the child(i.e. no parent prename), flag as a "junior-style" name
# 2) for other names, compute a levenshtein similarity, update parent to match child

iters = []
avgs = []
stds = []

wtfs = []
subs = []

run = True
ind = 0
old_avg = 1
while run:
    print('iteration', ind)
    count_names = calc_name_ratios(freq)
    freq = calc_name_evidence(freq, count_names)

    print("{0}, {1:.3e}, {2:.3e}".format(
        ind, freq.evidence.mean(), freq.evidence.std()))
    iters.append(ind)
    avgs.append(freq.evidence.mean())
    stds.append(freq.evidence.std())

    sub = freq[(freq.evidence < 2**ind)].copy(deep=True)
    print("# susp :", len(sub))
    tmp = sub.apply(lambda row: check_name_permutations(row),
                    axis=1, result_type='expand')
    subs.append(sub.copy(deep=True))
    wtfs.append(tmp.copy(deep=True))
    tmp = tmp[tmp.cedula.notnull()]
    tmp.replace(np.nan, "", inplace=True)

    freq = prepare_catcol_merge(freq, tmp)
    freq.loc[freq.cedula.isin(sub.cedula), :] = tmp

    if abs(avgs[-1] - old_avg)/old_avg < 0.01:
        run = False
    if ind >= 10:
        run = False
    old_avg = avgs[-1]
    ind += 1
