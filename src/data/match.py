import pandas as pd
import numpy as np
import datetime as dt

def merge_ncleaned_rf(names_cleaned, rf):
    for col in ['nombre_spouse', 'ced_spouse', 'ced_madre', 'ced_padre']:
        rf[col] = rf[col].fillna('')
    
    names_cleaned.loc[names_cleaned.sur_padre.isnull(), 'sur_padre'] = ""
    names_cleaned.loc[names_cleaned.sur_madre.isnull(), 'sur_madre'] = ""
    names_cleaned.loc[names_cleaned.prenames.isnull(), 'prenames'] = ""
    names_cleaned['nlen_pre'] = names_cleaned.prenames.map(lambda x: len(x.split()))
    names_cleaned['is_plegal'] = names_cleaned.is_plegal.map(
        lambda x: np.nan if x is np.nan else bool(x))
    names_cleaned['is_mlegal'] = names_cleaned.is_mlegal.map(
        lambda x: np.nan if x is np.nan else bool(x))
    names_cleaned.drop(['n_char_nombre', 'n_char_prenames', 'nlen_pre'], axis=1, inplace=True)

    usecols = ['cedula', 'dt_birth', 'dt_death', 'dt_marriage',
               'nombre_spouse', 'ced_spouse', 'ced_padre', 'ced_madre']
    cols_reg = usecols[1:]

    for col in cols_reg:
        if col in names_cleaned.columns:
            del names_cleaned[col]

    return names_cleaned.merge(rf, how = 'left', on = 'cedula')


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


def exact_name_padre(ncleaned_rf):
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
