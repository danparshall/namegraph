"""
Match parent (partial record)
"""

# Now we get to the hard part - trying to find the parent record even when the full name hasn't been provided.  There are a few tricks here to make this tractable.

# 1 - split the dataframe in half on gender(i.e. only consider guys when looking for fathers)
# 2 - use "categorical" datatype, so we only have to check 30k names, not 18M records
# 3 - batch-process by surname(e.g. do everyone named "LOPEZ" at once
#                              saves on search overhead)
# 4 - exclude candidates who are too young/old to be the parent

# Together, those sped up the matching by a factor of ~100x.  It now runs in roughly a day.  For the padres, was able to identify around a third of the records (i.e. another 2M, bringing the total known links to 14M).  For the madres, I discovered that the prename assignment hadn't worked well, far too many had bad data.  I did a quick check, and unfortunately I think there's some issue with the extraction algorithm for madres ( in NB3).

from fuzzywuzzy import fuzz
from collections import Counter
from tqdm.auto import tqdm
import unidecode
import re
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
import pandas as pd
import math

# enable progress bar on long operations
tqdm.pandas()

full_run = False
N_ROWS = None  # 1000000
READ_DATE = '20200824'
READ_DATE = '20201026'
READ_DATE = '20201111'


LOC_RAW = "../data/raw/"
LOC_INTERIM = "../data/interim/"


MIN_PARENT_AGE = 12  # I truly hope there aren't any parents this young

TODAY = dt.datetime.now().strftime('%Y%m%d')

nan_values = ['-1.#IND', '1.#QNAN', '1.#IND', '-1.#QNAN', '#N/A N/A', '#N/A', 'N/A', 'n/a',  # 'NA' is sometimes name
              '<NA>', '#NA', 'NULL', 'null', 'NaN', '-NaN', 'nan', '-nan', '']


# read cleaned-up input file
print("LOADING REG DATA FOR : " + READ_DATE)
dtypes_reg = {'cedula': str, 'nombre': str, 'gender': 'category', 'nationality': 'category',
              'orig_cedula': str, 'marital_status': 'category',
              'nombre_spouse': str, 'nombre_padre': str, 'nombre_madre': str,
              'ced_spouse': str, 'ced_padre': str, 'ced_madre': str
              }

usecols = ['cedula', 'gender', 'dt_birth', 'dt_death', 'nombre_padre',
           'ced_padre', 'nombre_madre', 'ced_madre', 'nombre_spouse']
cols_reg = usecols[1:]

if True:
    rf = pd.read_csv(LOC_RAW + "REG_NAMES_current.tsv", sep='\t', dtype=dtypes_reg,
                     parse_dates=['dt_birth', 'dt_death'], usecols=usecols,
                     keep_default_na=False, na_values=nan_values,
                     nrows=N_ROWS
                     )

    for col in ['ced_madre', 'ced_padre', 'nombre_spouse']:
        rf[col] = rf[col].fillna('')

    print("Loaded {0} rows".format(len(rf)))
rf['has_pced'] = rf.ced_padre != ''
rf['has_mced'] = rf.ced_madre != ''
rf['yob'] = rf.dt_birth.dt.year
rf.groupby('yob').has_pced.mean()
fig, ax = plt.subplots()

xlim = [1900, 2020]
ax.set(xlabel="year of birth", ylabel="fract. with parent cedula", xlim=xlim)
ax.plot(xlim, [0, 0], 'k--', alpha=0.3)
ax.plot(xlim, [1, 1], 'k--', alpha=0.3)
ax.plot(rf.groupby('yob').has_pced.mean(), 'b-', label='pad. ced.')
ax.plot(rf.groupby('yob').has_mced.mean(), 'r-', label='mad. ced.')
ax.legend()
READ_DATE = '20201111'
rf.head()

dtypes_names = {'cedula': str, 'sur_padre': 'category', 'sur_madre': 'category', 'prenames': str,
                'has_padre': bool, 'has_madre': bool, 'is_plegal': bool, 'is_mlegal': bool,
                'gender': int,
                }

usecols_names = ['cedula', 'nombre', 'prenames', 'gender',
                 'nombre_padre',
                 'sur_padre', 'has_padre', 'is_plegal',
                 'sur_madre', 'has_madre', 'is_mlegal',
                 'is_funky',
                 ]


nf = pd.read_csv("names_parsed_20200721.tsv", sep='\t', dtype=dtypes_names)

# ~90 sec

guys = nf[nf.gender == 1]
gals = nf[nf.gender == 2]

"""
year  pop (millions)   myest
1950  3.47        2.309
1960  4.54        3.699
1970  6.07        5.646
1980  7.99        8.064
1990 10.23       10.81
2000 12.68       13.77 
2010 15.01       16.72
2020 17.6 (est)  18.69

"""

for y in np.arange(1900, 2021, 10):
    y_beg = dt.datetime(y-78, 1, 1)
    y_end = dt.datetime(y, 1, 1)
    print('{0} {1:>8}'.format(
        y, ((rf.dt_birth >= y_beg) & (rf.dt_birth < y_end)).sum()))

"""
Load parsed namedata
"""
dtypes_names = {'cedula': str, 'nombre': str, 'sur_padre': 'category', 'sur_madre': 'category',
                'pre1': 'category', 'pre2': 'category', 'pre3': 'category',
                'junk': str, 'nlen': int
                }
usecols_names = ['cedula', 'sur_padre', 'sur_madre',
                 'pre1', 'pre2', 'pre3', 'junk', 'nlen']

loc_names = LOC_INTERIM + "NEWFREQFILE_" + READ_DATE + ".tsv"
names = pd.read_csv(loc_names, sep='\t',
                    dtype=dtypes_names, usecols=usecols_names)
cols_cat = ['sur_padre', 'sur_madre', 'pre1', 'pre2', 'pre3']

for col in cols_cat:
    names[col].cat.add_categories('', inplace=True)
    names[col].fillna('', inplace=True)

names['junk'].fillna('', inplace=True)
names.head()

names = names.merge(rf, on='cedula', how='left')
names.head()

### Load matched data
pmatch = pd.read_csv(LOC_INTERIM + 'matched_padres_' + READ_DATE + '.tsv', sep='\t', dtype=str,) # usecols=['cedula'])
len(pmatch)

mmatch = pd.read_csv(LOC_INTERIM + 'matched_madres_' + READ_DATE + '.tsv', sep='\t', dtype=str,) # usecols=['cedula'])
len(mmatch)
pmatch.head()
pmatch[pmatch.padre_official.isnull() | pmatch.padre_matched.isnull()]
pmatch.padre_official.isnull().sum()

### Load parsed parent names
#ceds_found_padre = set(nf[nf.cedula.isin(set(pmatch.cedula)) & (nf.ced_padre != "")].cedula)
ceds_found_padre = set(names[names.cedula.isin(
    set(pmatch.cedula)) | (names.ced_padre != '')].cedula)
#ceds_found_padre = set(pmatch.cedula)
len(ceds_found_padre)
ceds_found_madre = set(names[names.cedula.isin(set(mmatch.cedula))
                             | (names.ced_madre != '')].cedula)
len(ceds_found_madre)
(names.ced_padre != '').sum()

dtype_parsed = {'cedula': str,
                'sur1': 'category', 'sur2': 'category',
                'pre1': 'category', 'pre2': 'category', 'pre3': 'category',
                'junk': str}

pparsed = pd.read_csv('../data/interim/PADRES_20201111.tsv',
                      sep='\t', dtype=dtype_parsed)

# fill NaN with empty string
for col in ['sur1', 'sur2', 'pre1', 'pre2', 'pre3']:
    pparsed[col].cat.add_categories('', inplace=True)
pparsed.fillna('', inplace=True)

len(pparsed)
pparsed = pparsed[~pparsed.cedula.isin(
    ceds_found_padre)]   # 12 M recs, 34517 names
len(pparsed)

mparsed = pd.read_csv('../data/interim/MADRES_20201111.tsv', sep='\t',
                      dtype=dtype_parsed)

# fill NaN with empty string
for col in ['sur1', 'sur2', 'pre1', 'pre2', 'pre3']:
    mparsed[col].cat.add_categories('', inplace=True)
mparsed.fillna('', inplace=True)

len(mparsed)
mparsed = mparsed[~mparsed.cedula.isin(
    ceds_found_madre)]   # 12 M recs, 34517 names
len(mparsed)

ITERS_PER_SEC = 40
SECS_PER_DAY = 60*60*24
len(pparsed) / ITERS_PER_SEC / SECS_PER_DAY
nc = pd.read_csv('../data/interim/NAMECOUNTS_20201111.tsv', sep='\t')
len(nc)
pcount = pparsed.sur1.value_counts()
pcount[pcount > 1]
mcount = mparsed.sur1.value_counts()
mcount[mcount > 1][:10]
surm_unique = set(mcount[mcount <= 1].index)
len(surm_unique)
surp_unique = set(pcount[pcount <= 1].index)
len(surp_unique)
surp_multi = set(pcount[pcount > 1].index)
len(surp_multi)
surm_multi = set(mcount[mcount > 1].index)
len(surm_multi)

"""
Match
"""
# 99th % for age is 53.7 for men, 42.8 for women

guys = names[names.gender == '1']
len(guys)
MAX_PADRE_AGE = 53.7
MAX_MADRE_AGE = 42.8
guys.drop(columns=['has_pced', 'has_mced', 'yob'], inplace=True)
guys.head()


def match_padre(par, nf, guys):

    try:
        rec = nf[nf.cedula == par.cedula].iloc[0]
    except IndexError:
        return None

    sub = guys[(guys.sur_padre == par.sur1)]
    sub = sub[(sub.dt_birth < rec.dt_birth -
               dt.timedelta(365.2425*MIN_PARENT_AGE))]
    sub = sub[(sub.dt_birth > rec.dt_birth -
               dt.timedelta(365.2425*MAX_PADRE_AGE))]

    if par.sur2:
        sub = sub[sub.sur_madre == par.sur2]

    if par.pre1:
        sub = sub[sub.prenames.map(lambda x: par.pre1 in x)]
    if len(sub) == 0:
        return None

    if par.pre2:
        sub = sub[sub.prenames.map(lambda x: par.pre2 in x)]
    if len(sub) == 0:
        return None

    if par.pre3:
        sub = sub[sub.prenames.map(lambda x: par.pre3 in x)]
    if len(sub) == 0:
        return None

    elif len(sub) == 1:
        return sub.iloc[0].cedula
    else:
        return "Found {0} options".format(len(sub))


def match_padre_namedata(par, sub):

    if par.sur2:
        sub = sub[sub.sur_madre == par.sur2]

    # if we have 2 prenames, use them in sequence
    if par.pre2:
        sub = sub[(sub.pre1 == par.pre1) & (sub.pre2 == par.pre2)]
    if len(sub) == 0:
        return ''

    # if we only have 1 prename, it might be in either column
    if par.pre1:
        sub = sub[(sub.pre1 == par.pre1) | (sub.pre2 == par.pre1)]

    # check mother's name against candidate's spouse
    if (len(sub) > 1) and par.sur2:
        tmp = sub[sub.nombre_spouse.map(lambda x: par.sur2 in x)]
        if len(tmp) > 0:
            return "MAMAS: " + ';'.join(list(set(tmp.cedula)))

    # return results
    if len(sub) == 0:
        return ''
    elif len(sub) == 1:
        return sub.iloc[0].cedula
    elif len(sub) < 100:
        return ';'.join(list(set(sub.cedula)))
    else:
        return "Found {0} options".format(len(sub))


def match_madre_namedata(par, sub):

    if par.sur2:
        sub = sub[sub.sur_madre == par.sur2]

    # if we have 2 prenames, use them in sequence
    if par.pre2:
        sub = sub[(sub.pre1 == par.pre1) & (sub.pre2 == par.pre2)]
    if len(sub) == 0:
        return ''

    # if we only have 1 prename, it might be in either column
    if par.pre1:
        sub = sub[(sub.pre1 == par.pre1) | (sub.pre2 == par.pre1)]

    # check father's name against candidate's spouse
    if (len(sub) > 1) and par.sur1:
        tmp = sub[sub.nombre_spouse.map(lambda x: par.sur1 in x)]
        if len(tmp) > 0:
            return "PAPAS: " + ';'.join(list(set(tmp.cedula)))

    # return results
    if len(sub) == 0:
        return ''
    elif len(sub) == 1:
        return sub.iloc[0].cedula
    elif len(sub) < 100:
        return ';'.join(list(set(sub.cedula)))
    else:
        return "Found {0} options".format(len(sub))


### NB - check to see if the categoricals persist in the "sub_citizens" frame.  May be worthwhile

file_out = 'MADRES_matched_by_name_' + TODAY + '.tsv'
with open(file_out, 'wt') as f:
    results = []
    past = set()

    for ind, chk_name in tqdm(enumerate(sorted(mcount[mcount > 1].index))):

        if ind % 1000 == 0:
            print("  >>>>>>>>>>>> ITER " + str(ind))

        if pd.isnull(chk_name) or chk_name == '':
            continue

        # copying only takes ~15 mins overhead, and probably makes subsequent searching faster.  Do it.
        sub_citizens = gals[gals.sur_padre == chk_name].copy(deep=True)
        sub_madres = mparsed[mparsed.sur1 == chk_name]

        if len(sub_madres) > 1000:
            # show the progress if there are a lot of names
            print(chk_name, len(sub_madres))
            for par in tqdm(sub_madres.itertuples()):
                if par.cedula in past:
                    break
                out = match_madre_namedata(par, sub_citizens)
                results.append((par.cedula, out))
                past.add(par.cedula)
                f.write(par.cedula + '\t' + out + '\n')
        else:
            for par in sub_madres.itertuples():
                if par.cedula in past:
                    break
                out = match_madre_namedata(par, sub_citizens)
                results.append((par.cedula, out))
                past.add(par.cedula)
                f.write(par.cedula + '\t' + out + '\n')
results[:10]

### NB - check to see if the categoricals persist in the "sub_citizens" frame.  May be worthwhile

file_out = 'PADRES_matched_by_name_' + TODAY + '.tsv'
with open(file_out, 'wt') as f:
    results = []
    past = set()

    for ind, chk_name in tqdm(enumerate(sorted(pcount[pcount > 1].index))):

        if ind % 1000 == 0:
            print("  >>>>>>>>>>>> ITER " + str(ind))

        if pd.isnull(chk_name) or chk_name == '':
            continue

        # copying only takes ~15 mins overhead, and probably makes subsequent searching faster.  Do it.
        sub_citizens = guys[guys.sur_padre == chk_name].copy(deep=True)

        sub_padres = pparsed[pparsed.sur1 == chk_name]

        if len(sub_padres) > 1000:
            # show the progress if there are a lot of names
            print(chk_name, len(sub_padres))
            for par in tqdm(sub_padres.itertuples()):
                if par.cedula in past:
                    break
                out = match_padre_namedata(par, sub_citizens)
                results.append((par.cedula, out))
                past.add(par.cedula)
                f.write(par.cedula + '\t' + out + '\n')
        else:
            for par in sub_padres.itertuples():
                if par.cedula in past:
                    break
                out = match_padre_namedata(par, sub_citizens)
                results.append((par.cedula, out))
                past.add(par.cedula)
                f.write(par.cedula + '\t' + out + '\n')

results = []
past = set()

for ind, chk_name in tqdm(enumerate(set(pcount[pcount > 1].index))):

    if ind % 1000 == 0:
        print("  ITER " + str(ind))

    if pd.isnull(chk_name) or chk_name == '':
        continue

    # copying only takes ~15 mins overhead, and probably makes subsequent searching faster.  Do it.
    sub_citizens = guys[guys.sur_padre == chk_name].copy(deep=True)

    sub_padres = pparsed[pparsed.sur1 == chk_name]

    if len(sub_padres) > 1000:
        # show the progress if there are a lot of names
        print(chk_name, len(sub_padres))
        for par in tqdm(sub_padres.itertuples()):
            if par.cedula in past:
                break
            out = match_padre_namedata(par, sub_citizens)
            results.append((par.cedula, out))
            past.add(par.cedula)
    else:
        for par in sub_padres.itertuples():
            if par.cedula in past:
                break
            out = match_padre_namedata(par, sub_citizens)
            results.append((par.cedula, out))
            past.add(par.cedula)

gals = names[names.gender == '2']
gals.drop(columns=['has_pced', 'has_mced', 'yob'], inplace=True)
len(gals)

df = pd.DataFrame(data=zip(*results))  # , columns=['ced_kid', 'ced_pad'])
df = df.T
df.columns = ['ced_kid', 'ced_pad']
len(df)
df[df.ced_pad.notnull()][-40:]
df[df.ced_pad.map(lambda x: x.startswith("Found"))]  # [df.ced_pad == '']
len(df)
df.fillna('', inplace=True)

top_0 = set(nc[(nc.n_sur > 100000)].obsname)  # eg PAREDES
len(top_0) * 90
top_1 = set(nc[(nc.n_sur < 100000) & (
    nc.n_sur >= 10000)].obsname)  # eg VILLALBA
len(top_1) * 6
top_2 = set(nc[(nc.n_sur < 10000) & (
    nc.n_sur >= 1000)].obsname)   # eg CORDOVEZ
len(top_2)
top_3 = set(nc[(nc.n_sur < 1000) & (nc.n_sur >= 100)].obsname)   # eg CORDOVEZ
len(top_3)
24*60
df = pd.read_csv('partial_padre_matching_20201120.tsv', sep='\t')
len(df)
df.head()
df.ced_pad.isnull().sum()  # 255k (20%, can't find candidate
df.fillna('', inplace=True)
# 893k (68%) have multiple options
df[df.ced_pad.map(lambda x: x.startswith('Found'))]
df[df.ced_pad.map(lambda x: not x.startswith('Found'))
   ]  # 414k (31%) have single match
413/1307


def get_n_cand(res):
    if pd.isnull(res) or res == '':
        return 0
    elif res.startswith('Found'):
        return float(res.split()[1])
    else:
        return 1


df['n_cand'] = df.ced_pad.map(get_n_cand)
df.describe()
df[df.n_cand > 1].describe()

rec = nf[nf.cedula == par.cedula].iloc[0]
rec

sub = guys[(guys.sur_padre == par.sur1)]
sub = sub[(sub.dt_birth < rec.dt_birth - dt.timedelta(365.2425*MIN_PARENT_AGE))]
sub = sub[(sub.dt_birth > rec.dt_birth - dt.timedelta(365.2425*MAX_PADRE_AGE))]
sub
if par.sur2:
    sub = sub[sub.sur_madre == par.sur2]
sub
sub = sub[sub.prenames.map(lambda x: (par.pre1.strip() in x))]
sub
if par.pre2:
    sub = sub[sub.prenames.map(lambda x: par.pre2 in x)]

if par.pre3:
    sub = sub[sub.prenames.map(lambda x: par.pre3 in x)]

print(len(sub))
sub[sub.nombre_spouse == ""]  # [sub.prenames.map(lambda x: "FRANCISCO" in x)]
