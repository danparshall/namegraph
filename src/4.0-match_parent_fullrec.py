"""
Match parent (full recs)
"""

# Here we handle a special case, but one surprisingly common: when the full parent name has been correctly recorded, in legal form.  Or even in the casual form, as long as all 4 name parts are there.  It adds around 2M matches(on top of the 10M that were provided originally).

# Among those where we have ground-truth, the error rate is < 0.2 % .  Worth checking out the plots showing birthdates of parents relative to kids - there are some funky things that need to get fixed here.  But overall, this was high-yield, low effort.

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


LOC_RAW = "../data/raw/"
LOC_INTERIM = "../data/interim/"


MIN_PARENT_AGE = 12  # I truly hope there aren't any parents this young

nan_values = ['-1.#IND', '1.#QNAN', '1.#IND', '-1.#QNAN', '#N/A N/A', '#N/A', 'N/A', 'n/a',  # 'NA' is sometimes name
              '<NA>', '#NA', 'NULL', 'null', 'NaN', '-NaN', 'nan', '-nan', '']

# read cleaned-up input file
print("LOADING REG DATA FOR : " + READ_DATE)
dtypes_reg = {'cedula': str, 'nombre': str, 'gender': 'category', 'nationality': 'category',
              'orig_cedula': str, 'marital_status': 'category',
              'nombre_spouse': str, 'nombre_padre': str, 'nombre_madre': str,
              'ced_spouse': str, 'ced_padre': str, 'ced_madre': str
              }

usecols = ['cedula', 'dt_birth', 'dt_death', 'dt_marriage',
           'nombre_spouse', 'ced_spouse', 'ced_padre', 'ced_madre']
cols_reg = usecols[1:]
rf = pd.read_csv(LOC_RAW + "REG_NAMES_" + READ_DATE + ".tsv", sep='\t', dtype=dtypes_reg,
                 parse_dates=['dt_birth', 'dt_death', 'dt_marriage'], usecols=usecols,
                 keep_default_na=False, na_values=nan_values,
                 nrows=N_ROWS
                 )

for col in ['nombre_spouse', 'ced_spouse', 'ced_madre', 'ced_padre']:
    rf[col] = rf[col].fillna('')

print("Loaded {0} rows".format(len(rf)))

dtypes_names = {'cedula': str, 'sur_padre': str, 'sur_madre': str, 'prenames': str,
                'has_padre': bool, 'has_madre': bool, 'is_plegal': bool, 'is_mlegal': bool
                }

loc_nf = LOC_INTERIM + "names_cleaned_" + READ_DATE + ".tsv"
print("Loading from", loc_nf)
nf = pd.read_csv(loc_nf, sep='\t',
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
nf.is_plegal.sum()
nf.drop(['n_char_nombre', 'n_char_prenames', 'nlen_pre'], axis=1, inplace=True)
nf.head()
(nf.has_padre & nf.has_madre & (nf.sur_padre != "") & (nf.sur_madre != "")).sum()
nf.is_plegal.value_counts()
"""
year  pop (millions)
1950  3.47
1960  4.54
1970  6.07
1980  7.99
1990 10.23
2000 12.68
2010 15.01
2020 17.6 (est)

"""

for y in np.arange(1900, 2021, 10):
    y_beg = dt.datetime(y-78, 1, 1)
    y_end = dt.datetime(y, 1, 1)
    print('{0} {1:>8}'.format(
        y, ((rf.dt_birth >= y_beg) & (rf.dt_birth < y_end)).sum()))

### Load prenames from regular data

file_freq = LOC_INTERIM + "NEWFREQFILE_" + READ_DATE + ".tsv"
#file_freq = LOC_INTERIM + "NEWFREQFILE_20200824.tsv"

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

"""
Attempt matching
"""
for col in cols_reg:
    if col in nf.columns:
        del nf[col]

nf = nf.merge(rf, how='left', on='cedula')
nf.head()

### First try to match the exact names, in cases where we have all 4

obv_padres = nf[nf.has_padre & nf.is_plegal & (nf.nlen_padre == 4)][[
    'cedula', 'nombre_padre']]
obv_padres.rename(columns={'cedula': 'ced_kid',
                  'nombre_padre': 'nombre'}, inplace=True)

obv_madres = nf[nf.has_madre & nf.is_mlegal & (nf.nlen_madre == 4)][[
    'cedula', 'nombre_madre']]
obv_madres.rename(columns={'cedula': 'ced_kid',
                  'nombre_madre': 'nombre'}, inplace=True)
clean_pads = nf[['nombre', 'cedula', 'dt_birth']].merge(
    obv_padres, on='nombre')
clean_pads.rename(columns={'cedula': 'ced_pad', 'ced_kid': 'cedula',
                  'dt_birth': 'dt_birth_padre'}, inplace=True)

clean_mads = nf[['nombre', 'cedula', 'dt_birth']].merge(
    obv_madres, on='nombre')
clean_mads.rename(columns={'cedula': 'ced_mad', 'ced_kid': 'cedula',
                  'dt_birth': 'dt_birth_madre'}, inplace=True)
len(clean_pads)
len(obv_padres)
ceds_nopad = set(obv_padres.ced_kid) - set(clean_pads.cedula)
len(ceds_nopad)
fig, ax = plt.subplots(figsize=(9, 6))

ax.hist(nf[nf.cedula.isin(set(obv_padres.ced_kid))].dt_birth.dt.year,
        bins=np.arange(1920, 2020), color='b', alpha=0.4, label='padre has 4 names')
ax.hist(nf[nf.cedula.isin(ceds_nopad)].dt_birth.dt.year,
        bins=np.arange(1920, 2020), color='r', alpha=0.4, label='padre has 4 names, but not matched')

ax.set(xlabel='year of birth', ylabel='number of citizens')
ax.legend()

fig, ax = plt.subplots(figsize=(9, 6))

ax.hist(nf[~nf.cedula.isin(ceds_nopad)].dt_birth.dt.year,
        bins=np.arange(1920, 2020), color='b', alpha=0.4)
ax.hist(nf[nf.cedula.isin(ceds_nopad)].dt_birth.dt.year,
        bins=np.arange(1920, 2020), color='r', alpha=0.4)

whoa_papa = nf.merge(
    clean_pads[['cedula', 'ced_pad', 'dt_birth_padre']], how='left', on='cedula')
print("# poss padre recs :", len(whoa_papa))

whoa_mama = nf.merge(
    clean_mads[['cedula', 'ced_mad', 'dt_birth_madre']], how='left', on='cedula')
print("# poss madre recs :", len(whoa_mama))

# 2 mins
valid_matched_padres = whoa_papa[~whoa_papa.duplicated(['cedula'], keep=False)
                                 & (whoa_papa.ced_pad.notnull())
                                 & (whoa_papa.dt_birth > whoa_papa.dt_birth_padre + dt.timedelta(365.26 * MIN_PARENT_AGE))
                                 ]
len(valid_matched_padres)
valid_matched_madres = whoa_mama[~whoa_mama.duplicated(['cedula'], keep=False)
                                 & (whoa_mama.ced_mad.notnull())
                                 & (whoa_mama.dt_birth > whoa_mama.dt_birth_madre + dt.timedelta(365.26 * MIN_PARENT_AGE))
                                 ]


whoa_papa[whoa_papa.cedula.duplicated(keep=False)
          & (whoa_papa.dt_birth > whoa_papa.dt_birth_padre + dt.timedelta(365.26 * MIN_PARENT_AGE))
          ].sort_values('cedula')

matched_padres = whoa_papa[~whoa_papa.duplicated(['cedula'], keep=False)
                           & (whoa_papa.ced_pad.notnull())
                           #                          & (whoa_papa.dt_birth > whoa_papa.dt_birth_padre + dt.timedelta(365.26 * MIN_PARENT_AGE))
                           ]
len(matched_padres)
matched_madres = whoa_mama[~whoa_mama.duplicated(['cedula'], keep=False)
                           & (whoa_mama.ced_mad.notnull())
                           #                          & (whoa_mama.dt_birth > whoa_mama.dt_birth_madre + dt.timedelta(365.26 * MIN_PARENT_AGE))
                           ]
len(matched_madres)

n_official = (valid_matched_padres.ced_padre != "").sum()
n_official
errors = valid_matched_padres[(valid_matched_padres.ced_padre != valid_matched_padres.ced_pad)
                              & (valid_matched_padres.ced_padre != "")
                              ]
len(errors)
len(errors)/n_official

### Now invert 4-token names that aren't in legal form

inv_padres = nf[nf.has_padre & ~nf.is_plegal & (
    nf.nlen_padre == 4)][['cedula', 'nombre_padre']]
inv_padres.rename(columns={'cedula': 'ced_kid',
                  'nombre_padre': 'nombre_normform'}, inplace=True)

inv_madres = nf[nf.has_madre & ~nf.is_mlegal & (
    nf.nlen_madre == 4)][['cedula', 'nombre_madre']]
inv_madres.rename(columns={'cedula': 'ced_kid',
                  'nombre_madre': 'nombre_normform'}, inplace=True)
inv_padres['nombre'] = inv_padres.nombre_normform.map(
    lambda x: ' '.join(x.split()[2:] + x.split()[:2]))
inv_madres['nombre'] = inv_madres.nombre_normform.map(
    lambda x: ' '.join(x.split()[2:] + x.split()[:2]))
inv_padres.head()

flipped_pads = nf[['nombre', 'cedula', 'dt_birth']].merge(
    inv_padres[['ced_kid', 'nombre']], on='nombre')
flipped_pads.rename(columns={
                    'cedula': 'ced_pad', 'ced_kid': 'cedula', 'dt_birth': 'dt_birth_padre'}, inplace=True)

flipped_mads = nf[['nombre', 'cedula', 'dt_birth']].merge(
    inv_madres[['ced_kid', 'nombre']], on='nombre')
flipped_mads.rename(columns={
                    'cedula': 'ced_mad', 'ced_kid': 'cedula', 'dt_birth': 'dt_birth_madre'}, inplace=True)

wow_papa = nf.merge(
    flipped_pads[['cedula', 'ced_pad', 'dt_birth_padre']], how='left', on='cedula')
print("# poss padre recs :", len(wow_papa))

wow_mama = nf.merge(
    flipped_mads[['cedula', 'ced_mad', 'dt_birth_madre']], how='left', on='cedula')
print("# poss madre recs :", len(wow_mama))

# 2 mins
alternate_matched_padres = wow_papa[~wow_papa.duplicated(['cedula'], keep=False)
                                    & (wow_papa.ced_pad.notnull())
                                    & (wow_papa.dt_birth > wow_papa.dt_birth_padre + dt.timedelta(365.26 * MIN_PARENT_AGE))
                                    ]
len(alternate_matched_padres)
alternate_matched_madres = wow_mama[~wow_mama.duplicated(['cedula'], keep=False)
                                    & (wow_mama.ced_mad.notnull())
                                    & (wow_mama.dt_birth > wow_mama.dt_birth_madre + dt.timedelta(365.26 * MIN_PARENT_AGE))
                                    ]
len(alternate_matched_madres)
alternate_matched_madres.head()

"""
Combine matched frames, save to disk
"""
p1 = valid_matched_padres[['cedula', 'ced_padre', 'ced_pad']
                          ].rename(columns={'ced_pad': 'padre_matched', 'ced_padre': 'padre_official'})

p2 = alternate_matched_padres[['cedula', 'ced_padre', 'ced_pad']
                              ].rename(columns={'ced_pad': 'padre_matched', 'ced_padre': 'padre_official'})

dp = pd.concat([p1, p2], axis=0)
len(dp)
m1 = valid_matched_madres[['cedula', 'ced_madre', 'ced_mad']
                          ].rename(columns={'ced_mad': 'madre_matched', 'ced_madre': 'madre_official'})

m2 = alternate_matched_madres[['cedula', 'ced_madre', 'ced_mad']
                              ].rename(columns={'ced_mad': 'madre_matched', 'ced_madre': 'madre_official'})

dm = pd.concat([m1, m2], axis=0)
len(dm)
dp.to_csv('../data/interim/matched_padres_' +
          READ_DATE + '.tsv', sep='\t', index=False)
dm.to_csv('../data/interim/matched_madres_' +
          READ_DATE + '.tsv', sep='\t', index=False)

"""
Some plots
"""
chk_pad = matched_padres[(matched_padres.ced_padre == matched_padres.ced_pad)]
pad_age = (chk_pad.dt_birth - chk_pad.dt_birth_padre)
pad_age = pad_age.dt.days / 365.24
chk_mad = matched_madres[(matched_madres.ced_madre == matched_madres.ced_mad)]
mad_age = (chk_mad.dt_birth - chk_mad.dt_birth_madre)
mad_age = mad_age.dt.days / 365.24
fig, ax = plt.subplots(figsize=(9, 6))

# 99th % for age is 53.7 for men, 42.8 for women

ax.set(yscale='log', xlabel="parent's age (at birth of child)", ylabel="# of births")
ax.hist(pad_age, bins=np.arange(-100, 100, 1),
        alpha=0.3, color='b', label='padre age at birth')
ax.hist(mad_age, bins=np.arange(-100, 100, 1),
        alpha=0.3, color='r', label='madre age at birth')
ax.plot([MIN_PARENT_AGE, MIN_PARENT_AGE], [1, 500000],
        'k--', alpha=0.3, label="Age " + str(MIN_PARENT_AGE))
ax.legend()
NANOSEC_TO_YEAR = (365.2425 * 24 * 60 * 60 * 1e9)
matched_padres['padre_age'] = (
    matched_padres.dt_birth - matched_padres.dt_birth_padre).values.astype(float)/NANOSEC_TO_YEAR
fig, ax = plt.subplots(figsize=(9, 9))

ax.set(xlabel="year of birth", ylabel="age of padre")

sub = matched_padres  # .sample(10000000)
ax.plot(sub.dt_birth, sub.padre_age, '.', alpha=0.05)
alternate_matched_madres
maybe_papa = whoa_papa[whoa_papa.ced_pad.notnull()
                       & (whoa_papa.ced_padre != "")
                       & (whoa_papa.dt_birth > whoa_papa.dt_birth_padre + dt.timedelta(365.26 * MIN_PARENT_AGE))
                       ]
len(maybe_papa)
len(maybe_papa[maybe_papa.ced_pad != maybe_papa.ced_padre]) / len(maybe_papa)
maybe_papa[maybe_papa.ced_padre == maybe_papa.ced_pad]
maybe_mama = whoa_mama[whoa_mama.ced_mad.notnull()
                       & (whoa_mama.ced_madre != "")
                       & (whoa_mama.dt_birth > whoa_mama.dt_birth_madre + dt.timedelta(365.26 * MIN_PARENT_AGE))
                       ]
len(maybe_mama)
len(maybe_mama[maybe_mama.ced_mad != maybe_mama.ced_madre]) / len(maybe_mama)
maybe_papa[maybe_papa.ced_pad != maybe_papa.ced_padre].head()
bad_rec = maybe_papa[maybe_papa.ced_pad != maybe_papa.ced_padre].iloc[2]

ceds_chk = {bad_rec.cedula, bad_rec.ced_padre, bad_rec.ced_pad}
maybe_mama[maybe_mama.cedula == bad_rec.cedula]

""" ### Error: padre_prenames is not defined ###
whoa = nf.merge(obv_padres.rename(columns={'cedula': 'ced_kid', 'nombre_padre': 'nombre'}), on='nombre')
padre_bothsur = padre_prenames[(padre_prenames.sur2 != "") & (
    nf.dt_birth >= dt.datetime(1960, 1, 1))]
len(padre_bothsur)
targets = nf[nf.cedula.isin(set(padre_bothsur.cedula))]
targets.head(8)
target = targets.iloc[1]
target
Error: padre_penames is not defined
target_padre = padre_prenames[padre_prenames.cedula == target.cedula].iloc[0]
target_padre
padre_prenames.head()
len(whoa)
whoa = whoa.merge(rf, on='cedula', how='left', suffixes=('_pred', '_obs'))
whoa.sample(30)
nf.head()
sub_pad = nf[(nf.sur_padre == target_padre.sur1) & (nf.sur_madre == target_padre.sur2)
             & (nf.gender == 1)
             & (nf.dt_birth <= dt.datetime(target.dt_birth.year - 13, target.dt_birth.month, target.dt_birth.day))
             ]
sub_pad
subsub = sub_pad[sub_pad.prenames.map(lambda x: "JOSE" in x)
                 ]
subsub
subsub[subsub.nombre_spouse.map(lambda x: target.sur_madre in x)]

whoa_padre = nf[['cedula', 'nombre']].rename({'cedula': 'ced_padre', 'nombre': 'nombre_padre'}, axis=1
                                             ).merge(nf.loc[(nf.nlen_padre == 4), ['cedula', 'nombre_padre']], on='nombre_padre')
print("# naive-matched padre :", len(whoa_padre))

whoa_madre = nf[['cedula', 'nombre']].rename({'cedula': 'ced_madre', 'nombre': 'nombre_madre'}, axis=1
                                             ).merge(nf.loc[(nf.nlen_padre == 4), ['cedula', 'nombre_madre']], on='nombre_madre')
print("# naive-matched madre :", len(whoa_padre)) 
"""
