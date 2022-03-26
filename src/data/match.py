import pandas as pd
import numpy as np
import datetime as dt
from tqdm.auto import tqdm
import utils

def merge_ncleaned_rf(nf, rf):
    for col in ['nombre_spouse', 'ced_spouse', 'ced_madre', 'ced_padre']:
        rf[col] = rf[col].fillna('')
    
    nf.loc[nf.sur_padre.isnull(), 'sur_padre'] = ""
    nf.loc[nf.sur_madre.isnull(), 'sur_madre'] = ""
    nf.loc[nf.prenames.isnull(), 'prenames'] = ""
    nf['nlen_pre'] = nf.prenames.map(lambda x: len(x.split()))
    nf['is_plegal'] = nf.is_plegal.map(
        lambda x: np.nan if x is np.nan else bool(x))
    nf['is_mlegal'] = nf.is_mlegal.map(
        lambda x: np.nan if x is np.nan else bool(x))
    nf.drop(['n_char_nombre', 'n_char_prenames', 'nlen_pre'], axis=1, inplace=True)

    usecols = ['cedula', 'dt_birth', 'dt_death', 'dt_marriage',
               'nombre_spouse', 'ced_spouse', 'ced_padre', 'ced_madre']
    cols_reg = usecols[1:]

    for col in cols_reg:
        if col in nf.columns:
            del nf[col]

    merging = nf.merge(rf, how='left', on='cedula')

    merging.rename(columns = {'nombre_x': 'nombre', 'gender_x':'gender', 'nombre_padre_x': 'nombre_padre', 
                            'nombre_madre_x': 'nombre_madre'}, inplace = True)

    return merging


def exact_name_padre(ncleaned_rf):
    MIN_PARENT_AGE = 12

    # Exact four part names
    obv_padres = ncleaned_rf[ncleaned_rf.has_padre & ncleaned_rf.is_plegal & (ncleaned_rf.nlen_padre == 4)][[
        'cedula', 'nombre_padre']]
    obv_padres.rename(columns={'cedula': 'ced_kid',
                    'nombre_padre': 'nombre'}, inplace=True)
    clean_pads = ncleaned_rf[['nombre', 'cedula', 'dt_birth']].merge(
        obv_padres, on='nombre')
    clean_pads.rename(columns={'cedula': 'ced_pad', 'ced_kid': 'cedula',
                    'dt_birth': 'dt_birth_padre'}, inplace=True)
    whoa_papa = ncleaned_rf.merge(
        clean_pads[['cedula', 'ced_pad', 'dt_birth_padre']], how='left', on='cedula')
    print("# poss padre recs :", len(whoa_papa))

    valid_matched_padres = whoa_papa[~whoa_papa.duplicated(['cedula'], keep=False)
                                     & (whoa_papa.ced_pad.notnull())
                                     & (whoa_papa.dt_birth > whoa_papa.dt_birth_padre + dt.timedelta(365.26 * MIN_PARENT_AGE))
                                     ]

    # Invert 4-token names with no legal form
    inv_padres = ncleaned_rf[ncleaned_rf.has_padre & ~ncleaned_rf.is_plegal & (
        ncleaned_rf.nlen_padre == 4)][['cedula', 'nombre_padre']]
    inv_padres.rename(columns={'cedula': 'ced_kid',
                    'nombre_padre': 'nombre_normform'}, inplace=True)
    inv_padres['nombre'] = inv_padres.nombre_normform.map(
        lambda x: ' '.join(x.split()[2:] + x.split()[:2]))

    flipped_pads = ncleaned_rf[['nombre', 'cedula', 'dt_birth']].merge(
        inv_padres[['ced_kid', 'nombre']], on='nombre')
    flipped_pads.rename(columns={
                        'cedula': 'ced_pad', 'ced_kid': 'cedula', 'dt_birth': 'dt_birth_padre'}, inplace=True)

    wow_papa = ncleaned_rf.merge(
        flipped_pads[['cedula', 'ced_pad', 'dt_birth_padre']], how='left', on='cedula')

    alternate_matched_padres = wow_papa[~wow_papa.duplicated(['cedula'], keep=False)
                                        & (wow_papa.ced_pad.notnull())
                                        & (wow_papa.dt_birth > wow_papa.dt_birth_padre + dt.timedelta(365.26 * MIN_PARENT_AGE))
                                        ]

    p1 = valid_matched_padres[['cedula', 'ced_padre', 'ced_pad']
                              ].rename(columns={'ced_pad': 'padre_matched', 'ced_padre': 'padre_official'})
    p2 = alternate_matched_padres[['cedula', 'ced_padre', 'ced_pad']
                                ].rename(columns={'ced_pad': 'padre_matched', 'ced_padre': 'padre_official'})
    dp = pd.concat([p1, p2], axis=0)

    return dp


def exact_name_madre(ncleaned_rf):
    MIN_PARENT_AGE = 12

    # Exact four part names
    obv_madres = ncleaned_rf[ncleaned_rf.has_madre & ncleaned_rf.is_mlegal & (ncleaned_rf.nlen_madre == 4)][[
        'cedula', 'nombre_madre']]
    obv_madres.rename(columns={'cedula': 'ced_kid',
                               'nombre_madre': 'nombre'}, inplace=True)

    clean_mads = ncleaned_rf[['nombre', 'cedula', 'dt_birth']].merge(
        obv_madres, on='nombre')
    clean_mads.rename(columns={'cedula': 'ced_mad', 'ced_kid': 'cedula',
                               'dt_birth': 'dt_birth_madre'}, inplace=True)

    whoa_mama = ncleaned_rf.merge(
        clean_mads[['cedula', 'ced_mad', 'dt_birth_madre']], how='left', on='cedula')
    print("# poss madre recs :", len(whoa_mama))

    valid_matched_madres = whoa_mama[~whoa_mama.duplicated(['cedula'], keep=False)
                                     & (whoa_mama.ced_mad.notnull())
                                     & (whoa_mama.dt_birth > whoa_mama.dt_birth_madre + dt.timedelta(365.26 * MIN_PARENT_AGE))
                                     ]

    # Invert 4-token names with no legal form
    inv_madres = ncleaned_rf[ncleaned_rf.has_madre & ~ncleaned_rf.is_mlegal & (
        ncleaned_rf.nlen_madre == 4)][['cedula', 'nombre_madre']]
    inv_madres.rename(columns={'cedula': 'ced_kid',
                               'nombre_madre': 'nombre_normform'}, inplace=True)
    inv_madres['nombre'] = inv_madres.nombre_normform.map(
        lambda x: ' '.join(x.split()[2:] + x.split()[:2]))

    flipped_mads = ncleaned_rf[['nombre', 'cedula', 'dt_birth']].merge(
        inv_madres[['ced_kid', 'nombre']], on='nombre')
    flipped_mads.rename(columns={
                        'cedula': 'ced_mad', 'ced_kid': 'cedula', 'dt_birth': 'dt_birth_madre'}, inplace=True)

    wow_mama = ncleaned_rf.merge(
        flipped_mads[['cedula', 'ced_mad', 'dt_birth_madre']], how='left', on='cedula')

    alternate_matched_madres = wow_mama[~wow_mama.duplicated(['cedula'], keep=False)
                                        & (wow_mama.ced_mad.notnull())
                                        & (wow_mama.dt_birth > wow_mama.dt_birth_madre + dt.timedelta(365.26 * MIN_PARENT_AGE))
                                        ]

    m1 = valid_matched_madres[['cedula', 'ced_madre', 'ced_mad']
                              ].rename(columns={'ced_mad': 'madre_matched', 'ced_madre': 'madre_official'})
    m2 = alternate_matched_madres[['cedula', 'ced_madre', 'ced_mad']
                                  ].rename(columns={'ced_mad': 'madre_matched', 'ced_madre': 'madre_official'})
    dm = pd.concat([m1, m2], axis=0)

    return dm


def exact_name(n_cleaned):
    matched_padres = exact_name_padre(n_cleaned)
    matched_madres = exact_name_madre(n_cleaned)
    return matched_padres, matched_madres


def create_names(parsed, rf):
    parsed = parsed.astype(utils.get_dtypes_names())

    # for col in utils.get_cols_cat():
    #     parsed[col].cat.add_categories('', inplace=True)
    #     parsed[col].fillna('', inplace=True)

    parsed['junk'].fillna('', inplace=True)
    names = parsed.merge(rf, on='cedula', how='left')

    return names

def ceds_found(names, matched, ced_parent):
    nm = names.cedula.isin(set(matched.cedula))
    ced = (names[ced_parent] != '')
    ceds_found = set(names[nm | ced ].cedula)
    return ceds_found

def parsed(parent, ceds_found):
    parent = parent.astype(utils.get_dtypes_parsed())

    # for col in utils.get_cols_cat_parsed():
    #     parent[col].cat.add_categories('', inplace=True)
    # parent.fillna('', inplace=True)
    parsed = parent[~parent.cedula.isin(ceds_found)]

    return parsed

def match_padre_namedata(par, sub):

    if par.sur2:
        sub = sub[sub.sur_madre == par.sur2]

    # if we have 2 prenames, use them in sequence
    if par.pre2:
        sub = sub[(sub.pre1 == par.pre1) & (sub.pre2 == par.pre2)]

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

def matched_by_name(parsed, names, gender, file_out):

    guys = names[names.gender == gender]
    print('guys_columns',guys.columns)
    count = parsed.sur1.value_counts()

    with open(file_out, 'wt') as f:
        results = []
        past = set()
        if gender == 'F':
            for ind, chk_name in enumerate(count[count >= 1].index):#tqdm(enumerate(count[count >= 1].index)):

                if ind % 1000 == 0:
                    print("  >>>>>>>>>>>> ITER " + str(ind))
                if pd.isnull(chk_name) or chk_name == '':
                    continue
                
                # copying only takes ~15 mins overhead, and probably makes subsequent searching faster.  Do it.
                sub_citizens = guys[guys.sur_padre == chk_name].copy(deep=True)
                sub_madres = parsed[parsed.sur1 == chk_name]
                if len(sub_citizens) >= 1:
                    # show the progress if there are a lot of names
                    print(chk_name, len(sub_madres))
                    for par in sub_madres.itertuples(): #tqdm(sub_madres.itertuples()):
                        if par.cedula in past:
                            break
                        out = match_madre_namedata(par, sub_citizens)
                        results.append((par.cedula, out))
                        past.add(par.cedula)
                        f.write(par.cedula + '\t' + out + '\n')
        elif gender == 'M':
            for ind, chk_name in tqdm(enumerate(count[count >= 1].index)):
                if ind % 1000 == 0:
                    print("  >>>>>>>>>>>> ITER " + str(ind))

                if pd.isnull(chk_name) or chk_name == '':
                    continue 
                                   
                # copying only takes ~15 mins overhead, and probably makes subsequent searching faster.  Do it.
                sub_citizens = guys[guys.sur_padre == chk_name].copy(deep=True)
                sub_padres = parsed[parsed.sur1 == chk_name]
                if len(sub_citizens) >= 1:
                    # show the progress if there are a lot of names
                    print(chk_name, len(sub_padres))
                    for par in tqdm(sub_padres.itertuples()):
                        if par.cedula in past:
                            break
                        out = match_padre_namedata(par, sub_citizens)
                        results.append((par.cedula, out))
                        past.add(par.cedula)
                        f.write(par.cedula + '\t' + out + '\n')
