"""
Extract and clean surnames
"""
# The core insight is that each record contains the citizen's name, as well as a field for their padre's name, and another for their madre's name.  Those names are "supposed" to follow the legal format (sur1 sur2 pre1 pre2) but often don't.

# So while it's tricky to tell just from the citizen's record what tokens are their padre's surname and which are their madre's, we can make progress by comparing the fields.  We know that the citizen's name is always in the correct/legal form, so the first token *must* be part of the padre's surname.  Once the padre's name has been identified and removed, the remainder *must* start with the madre's surname.  Whatever remains is prenames(which also have to be separated out).

# This is complicated by several things.  Common offenders are multi-token names, when both parents have the same surname, and when the madre's first name is the same as the daughter's.  Likewise, common Catholic dedications can add some confusion.

# These are made much easier by having underscores, so that multi-token names become a single token(e.g. a three-token name like `DE LA CRUZ` becomes the single-token `DE_LA_CRUZ`).  The core function here, `parse_fullrow()` was originally written before I'd started collapsing tokens.  There's some assumptions built in that are violated, so a second version has been tweaked to handle the underscores.

# Later data exploration found all kinds of weirdness.  People without an entry for father, or who had only the father's surname.  People without an entry for *mother*, although not sure how.  A handful of people whose names are in common form (pre1 pre2 sur1 sur2), even in the primary field.  Mis-spellings of the parent's surname(S/Z substitutions are especially common), or in some cases mis-spellings of the * citizen's * name(which then _becomes_ their name, and the name their children are stuck with).

# This notebook outputs several files
# chief among them are:
#     - `NAMECOUNTS` - has every observed name, and the counts of when used as surname vs prename
#     - `ALLNAMES` - like the above, but has names both with spaces and with underscores
#     - `names_cleaned` - citizen names, along with extracted parent surnames, and some useful metadata
#     - `NEWFREQFILE` - Sorry, future me, it's a horrible filename; it's actually the parsed-out name of each citizen

# Because of data errors, I couldn only extract surnames for ~90 % of citizens.  The other 2 million will have to get handled later.

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
N_ROWS = None  # 100000
READ_DATE = '20211028'

LOC_RAW = "../data/testdata/"
LOC_INTERIM = "../data/testdata/interim/"

LOC_RAW = "D:/Windows/Desktop/tmp_initial_namegraph/"
LOC_INTERIM = LOC_RAW + "out/"

TODAY = dt.datetime.now().strftime("%Y%m%d")
TODAY

nan_values = ['-1.#IND', '1.#QNAN', '1.#IND', '-1.#QNAN', '#N/A N/A', '#N/A', 'N/A', 'n/a',  # 'NA' is sometimes name
              '<NA>', '#NA', 'NULL', 'null', 'NaN', '-NaN', 'nan', '-nan', '']

dtypes_reg = {'cedula': str,
              'nombre': str,
              'gender': 'category',
              'dt_birth': str,
              'nationality': 'category',
              'nombre_spouse': str,
              'dt_death': str,
              'orig_cedula': str,
              'marital_status': 'category',
              'nombre_padre': str,
              'nombre_madre': str,
              'ced_spouse': str,
              'ced_padre': str,
              'ced_madre': str,
              'dt_marriage': str,
              }


if True:
    # read cleaned-up input file
    print("LOADING REG DATA FOR : " + READ_DATE)
    dtypes_reg = {'cedula': str, 'nombre': str, 'gender': 'category', 'nationality': 'category',
                  'orig_cedula': str, 'marital_status': 'category',
                  'nombre_spouse': str, 'nombre_padre': str, 'nombre_madre': str,
                  'ced_spouse': str, 'ced_padre': str, 'ced_madre': str
                  }

    rf = pd.read_csv(LOC_RAW + "simpsons_test_cases.tsv", sep='\t', dtype=dtypes_reg,
                     parse_dates=['dt_birth', 'dt_death', 'dt_marriage'],
                     keep_default_na=False, na_values=nan_values,
                     nrows=N_ROWS
                     )
    if "index" in rf.columns:
        rf.drop(columns=['index'], inplace=True)


# replace NaN with empty string
text_cols = ['nombre', 'nombre_spouse', 'nombre_padre',
             'nombre_madre', 'ced_spouse', 'ced_padre', 'ced_madre']
rf[text_cols] = rf[text_cols].fillna("")

print("# records loaded :", len(rf))
# 5 mins

"""
Funcs to parse primary citizen row-by-row
"""


def check_nombre_doubling(nombre):
    tokens = nombre.split()
    num_parts = len(tokens)
    if num_parts == len(set(tokens)):
        # no possible doubling, use name as-is
        return ""

    else:
        # check for doublings, looking at both the front and the end of the string
        surname = ""
        for i in range(1, num_parts-1):
            if tokens[:i] == tokens[i: 2*i]:
                # legal form
                surname = ' '.join(tokens[:i])
                break
            elif tokens[-i:] == tokens[-2*i:-i]:
                # sequential form
                surname = ' '.join(tokens[-i:])
                break
    return surname

def get_substrings(nombre, START=0):
    nlen = len(nombre)
    n_max = nlen - START
    for chunk_len in range(n_max, 0, -1):
        n_subs = n_max - chunk_len + 1
        for ind in range(0, n_subs):
            sub = ' '.join(nombre[START+ind: ind+chunk_len])

            # only return a single-token substring if the length is at least 2 (avoids matching on "DE", etc)
            if (chunk_len > 1) or (len(sub) > 2):
                yield sub


re_de_pre = re.compile(r"(^.*)\s(DEL?\s\w+.*$)")
re_one_y_two = re.compile(r"(^|\s)\w{2,} [IYE]{1} \w{2,}(\s|$)")
re_de_la_ao = re.compile(r"(^|\s)DE LA [AO]+(\s|$)")

re_de_pre = re.compile(r"(^.*)\s(DEL?\s\w+.*$)")
re_one_y_two = re.compile(r"(^|\s)\w{2,} [IYE]{1} \w{2,}(\s|$)")
re_de_la_ao = re.compile(r"(^|\s)DE LA [AO]+(\s|$)")

re_de_pre_UNDERSCORE = re.compile(r"(^.*)\s(DEL?_\w+.*$)")

def parse_fullrow(row):
    # updated function, which expects multi-token names to be handled already with underscores
    
    out = {'cedula': '', 'sur_padre': "", 'sur_madre': "", "prenames": "",
           "has_padre": False, "is_plegal": False,
           "has_madre": False, "is_mlegal": False}
    out['cedula'] = row.cedula

    if not row.nombre_padre and not row.nombre_madre:
        return out

    nombre = row.nombre
    nomset = set(nombre.split())
    madre = ""
    padre = ""
    prenames = ""

    # check if madre/padre have overlapping tokens (requires special handling)
    mset = set(row.nombre_madre.split())
    pset = set(row.nombre_padre.split())
    both = pset & mset
    flag_overlap = len(both) > 0

    if flag_overlap:
        surname = check_nombre_doubling(nombre)
        if surname != "":
            madre = surname
            padre = surname
        else:
            for guess in get_substrings(row.nombre_madre.split()):
                if (guess in nombre
                        and not nombre.endswith(guess)):
                    poss_padre = nombre.split(guess)[0].strip()
                    poss_set = set(poss_padre.split())
                    guess_set = set(guess.split())
                    if (poss_padre in row.nombre_padre
                        and not nombre.endswith(poss_padre)
                        and poss_set.issubset(nomset) and poss_set.issubset(pset)
                        and guess_set.issubset(nomset) and guess_set.issubset(mset)
                        ):
                        padre = poss_padre
                        madre = guess
                        break

    # if the overlap method wasn't successful, parse them separately
    if not madre and not padre:
        parts = nombre.split()

        #### FATHERS NAME ####
        # start by trying the first LEN-1 tokens as a single name, then LEN-2 tokens, etc
        # this matches longest chunk found, so it should pick up compound names like DE LA CRUZ
        try:
            if row.nombre_padre:
                poss_padre = check_nombre_doubling(row.nombre_padre)
                poss_pset = set(poss_padre.split())
                if (poss_padre
                    and (poss_padre in row.nombre)
                    and not nombre.endswith(poss_padre)
                        and poss_pset.issubset(nomset)):
                    padre = poss_padre
                    parts = ''.join(nombre.split(
                        padre, maxsplit=1)).strip().split()
                else:
                    # start by trying everything except the last element (always a prename), and work down
                    for ind in range(len(parts)-1, 0, -1):
                        guess = ' '.join(parts[:ind])
                        poss_pset = set(guess.split())
                        if ((guess in row.nombre_padre)
                                    and poss_pset.issubset(nomset) and poss_pset.issubset(pset)
                                ):
                            # update before checking mother's name
                            padre = guess
                            parts = nombre.split(padre, maxsplit=1)[1].split()
                            break
        except:
            out['sur_padre'] = "WTF PADRE PROBLEM"
            return out

        #### MOTHERS NAME ####
        # having removed the padre name from the front of the string, try similar trick with the madre name
        try:
            if row.nombre_madre:
                poss_madre = check_nombre_doubling(row.nombre_madre)
                poss_mset = set(poss_madre.split())
                if (poss_madre
                    and (poss_madre in row.nombre)
                    and not nombre.endswith(poss_madre)
                        and poss_mset.issubset(nomset) and poss_mset.issubset(mset)):
                    madre = poss_madre
                elif not madre:
                    nombre_madre = row.nombre_madre

                    if nombre_madre.startswith(parts[0]):
                        # in legal form, so strike any catholic addons from both citizen and mother
                        # complicated bc surnames like "GOMEZ DE LA TORRE" mean we have to skip the zeroth token
                        m_de_pre_nombre = re_de_pre_UNDERSCORE.match(
                            ' '.join(parts[1:]))
                        if m_de_pre_nombre:
                            # keep 'parts' as a list
                            parts = parts[:1] + \
                                m_de_pre_nombre.group(1).split()

                        mom_parts = nombre_madre.split()
                        m_de_pre_madre = re_de_pre.match(
                            ' '.join(mom_parts[1:]))
                        if m_de_pre_madre:
                            # keep 'nombre_madre' as string
                            nombre_madre = mom_parts[0] + \
                                " " + m_de_pre_madre.group(1)

                    for ind in range(len(parts)-1, 0, -1):
                        guess = ' '.join(parts[:ind])
                        poss_mset = set(guess.split())
                        if ((guess in nombre_madre)
                            and not nombre.endswith(guess)
                                and poss_mset.issubset(nomset) and poss_mset.issubset(mset)):
                            # now check which is the better fit
                            if (guess == nombre_madre):
                                # when madre is in short legal form and daughter has the same prename1
                                # it can look like the mother's surname is "GONZALEZ MARIA", etc.
                                # So ignore these, ??? because we need to handle them later
                                pass
                            else:
                                madre = guess
                                break

        except:
            out['sur_madre'] = "WTF MADRE PROBLEM"
            return out

    # get prenames explicitly, as remainder after removing prenames.
    # this bypasses some funny stuff I have to do above
    deduced_surnames = (padre + " " + madre).strip()
    if deduced_surnames and row.nombre.startswith(deduced_surnames):
        prenames = ''.join([x.strip()
                           for x in row.nombre.split(deduced_surnames)])
    else:
        prenames = "WTF SURNAME PROBLEM"

    if padre:
        out['has_padre'] = True
        out['sur_padre'] = padre
        out['is_plegal'] = row.nombre_padre.startswith(padre)
    if madre:
        out['has_madre'] = True
        out['sur_madre'] = madre
        out['is_mlegal'] = row.nombre_madre.startswith(madre)
    out['prenames'] = prenames

    return out

"""
Extract surnames/prenames
"""
if full_run:
    surnames_extracted = rf.progress_apply(
        lambda row: parse_fullrow(row), axis=1, result_type='expand')
    surnames_extracted.to_csv(
        LOC_INTERIM + "surnames_extracted_" + TODAY + ".tsv", sep='\t', index=False)

else:
    dtypes_names = {'cedula': str, 'sur_padre': str, 'sur_madre': str, 'prenames': str,
                    'has_padre': bool, 'has_madre': bool, 'is_plegal': 'category', 'is_mlegal': 'category'
                    }
    surnames_extracted = pd.read_csv(LOC_INTERIM + "surnames_extracted_" + READ_DATE + ".tsv",
                                     sep='\t', dtype=dtypes_names)

    # replace NaN in text columns
    textcols = ['sur_padre', 'sur_madre', 'prenames']
    surnames_extracted[textcols] = surnames_extracted[textcols].fillna("")

    # clean up bool columns
    if surnames_extracted.is_plegal.dtype != bool:
        surnames_extracted['is_plegal'] = surnames_extracted.is_plegal.map(
            lambda x: x in {'True', True}).astype(bool)
    if surnames_extracted.is_mlegal.dtype != bool:
        surnames_extracted['is_mlegal'] = surnames_extracted.is_mlegal.map(
            lambda x: x in {'True', True}).astype(bool)

print("# parsed :", len(surnames_extracted))

# set column order
surnames_extracted = surnames_extracted[['cedula', 'sur_padre', 'has_padre', 'is_plegal',
                                         'sur_madre', 'has_madre', 'is_mlegal', 'prenames']]

"""
test surname
"""

new_types = {'cedula': str,
             'sur_padre': str,
             'sur_madre': str,
             'prenames': str,
             'has_padre': bool,
             'is_plegal': bool,
             'has_madre': bool,
             'is_mlegal': bool,
             }

test_cases_surnames = pd.read_csv(LOC_RAW + "simpsons_test_cases_surname.tsv",
                                  sep='\t', dtype=new_types, keep_default_na=False, na_values=nan_values)

for i in range(2):
    print(".................................")
print("surname test")

for row in test_cases_surnames.to_dict(orient='records'):
    try:
        res = parse_fullrow(rf[rf.cedula == row['cedula']].iloc[0])
        #print(res)
        #print(row)
    except IndexError:
        print(row['cedula'], ": NOT IN THIS SUBFRAME")
        continue
    if res == row:
        print(row['cedula'], ": OK")
    else:
        print(row['cedula'], ": FAILED")
        print(row)
        print(res)

# 3.5 hours

"""
Clean up extracted
"""
# Clean surnames

## the "nf" (name frame) is just the subset of well-behaved names

# confirm that the indexing is still correct
if not (rf.index == surnames_extracted.index).all():
    rf.reset_index(inplace=True, drop=True)
assert (rf.cedula == surnames_extracted.cedula).all()

# join the parsed names to the originals (but only retain the well-behaved ones)
nf = pd.concat([rf[['cedula','nombre','nombre_padre','nombre_madre','gender']], surnames_extracted.iloc[:,1:]], axis=1)

# delete those without a madre or padre
nf = nf.loc[(nf.sur_padre.notnull()) & (nf.sur_padre != "") & nf.has_padre & 
            (nf.sur_madre.notnull()) & (nf.sur_madre != "") & nf.has_madre & 
            (nf.prenames.notnull() & (nf.prenames != "")),
        ['cedula','nombre','prenames', 'gender', 
         'nombre_padre','sur_padre','has_padre', 'is_plegal',
         'nombre_madre','sur_madre','has_madre', 'is_mlegal']]

len(nf)

# Look for very uncommon names, make sure they're legit, as opposed to parsing errors

sur_counts = nf.sur_padre.value_counts()
sur_counts.sort_index(inplace=True)

sur_counts = pd.concat([sur_counts,
                        pd.Series(data=sur_counts.index.map(
                            lambda x: len(x.split())), index=sur_counts.index)
                        ], axis=1)
sur_counts.columns = ['n_obs', 'nlen']

sur_counts

# 7k cases of names w <10 obs, 5100 with only 1
# 3900 cases w < 10 obs, only 2600 with just 1 (following improvements)

#sur_counts[(sur_counts.nlen > 1) & (sur_counts.n_obs < 10)].sample(60)
# 7k cases of names w <10 obs, 5100 with only 1
# 3900 cases w < 10 obs, only 2600 with just 1 (following improvements)

#sur_counts[(sur_counts.nlen > 1) & (sur_counts.n_obs < 10)].sample(60)

"""
Clean prenames
"""

# # There are plenty of prenames which have some sort of honorific(e.g. "DEL CARMEN").  The goal here is to find any of those, and render them into a single token(e.g. "DEL_CARMEN").

# # By definition, this only applies to names with at least 3 tokens(since the honorific is minimum of 2, and the prename is a minimum of 1).

# # Occasionally, someone will have something like "SANTA DEL CARMEN".  In practice, they're referred to as "CARMEN".

# # Because of cleaning in NB 1.0, this section has almost nothing to catch.

# # in all cases, we look for a word boundary as the first group, then our funky name as the second

re_von = re.compile(u"(\s)(V[AO]N \w{2,})(\s|$)")              # these results are subset of "re_vande"
re_vande = re.compile(u"(\s)(V[AO]N DE[RN]? \w{2,})(\s|$)")
re_sant = re.compile(u"(\s)(SANT?A? \w{2,})(\s|$)")            # SAN and SANTA (SANTO doesn't form compounds)
re_dela = re.compile(u"(\s)(DE L[AO]S? ?\w{2,})(\s|$)")   # these results are subset of "re_laos"
re_laos = re.compile(u"(\s)(L[AEO]S? \w{2,})(\s|$)")
re_del  = re.compile(u"(\s)(DEL \w{2,})(\s|$)")
re_de   = re.compile(r"(\s)(DE \w{2,})(\s|$)")


def regex_funky_prenames(nombre):
    """ This is a little slow (~4mins / million rows), but pretty thorough.  """
    
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
        if "ZOILA CRUZ" in poss_funks:
            print("WTF :", nombre)
        for funk in poss_funks:
            funky_prenames.add(funk)
        return True
    else:
        return False

def fix_funk(nombre, funks):
    """ The 'funks' list should be sorted in descending length, to prevent substrings from being clobbered.
    
    NB: there's a potential bug in here, bc the list is sorted according to character length, but checks
    here are being done according to number of tokens.  But very unlikely to cause an issue, so ignoring for now
    """
    nlen = len(nombre.split())
    if nlen <= 2:
        return nombre
    
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


if True:  # full_run:
    funky_prenames = set()
    nf['is_funky'] = nf.prenames.map(regex_funky_prenames)
    funky_prenames = list(funky_prenames)

    with open(LOC_INTERIM + "funky_prenames_" + TODAY + ".txt", 'w') as f:
        for funk in funky_prenames:
            f.write(funk + "\n")

funky_prenames.sort(reverse=True, key=len)
print("# funkies :", len(funky_prenames))

nf[nf.is_funky]
funky_prenames

nf.loc[nf.is_funky, 'prenames'] = nf[nf.is_funky].prenames.progress_map(
    lambda x: fix_funk(x, funky_prenames))

# now that there are only a few hundred funkies (most are handled in data-cleaning), this is faster by 100x

nf['nlen_padre'] = nf.nombre_padre.map(lambda x: len(x.split()))
nf['nlen_madre'] = nf.nombre_madre.map(lambda x: len(x.split()))

nf['n_char_nombre'] = nf.nombre.map(len)
nf['n_char_prenames'] = nf.prenames.map(len)
#  1 min

fig, ax = plt.subplots()

bins = np.arange(80)
ax.set(yscale='log')

ax.hist(nf.n_char_nombre, bins=bins, color='blue', alpha=0.3)
ax.hist(nf.n_char_prenames, bins=bins, color='red', alpha=0.3)

"""
Use surname/prename frequency to adjust results
"""
def parse_prename(prenames):
    """ The surnames are parsed, but the prenames must be split up.  
    This is possible once the multi-part prenames have been given underscores 
    """
    
    out = {'pre1':"", 'pre2':"", 'pre3':"", 'junk':""}
    
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

def count_all_names(freq):
    tmp = pd.concat([freq.sur_padre, freq.sur_madre], axis=0).value_counts()
    count_sur = pd.DataFrame({'obsname':tmp.index, 'n_sur':tmp.values})
    tmp = pd.concat([freq.pre1, freq.pre2], axis=0).value_counts()
    count_pre = pd.DataFrame({'obsname':tmp.index, 'n_pre':tmp.values})

    count_names = count_sur.merge(count_pre, on='obsname', how='outer')
    count_names.fillna(0, inplace=True)

    # add null record, so that null names get weight factor of 1
    count_names.loc[count_names.obsname == "", ['n_sur','n_pre']] = 0

    count_names['n_sur'] = count_names.n_sur + 0.5
    count_names['n_pre'] = count_names.n_pre + 0.5

    count_names['sratio'] = count_names.n_sur / count_names.n_pre
    count_names['pratio'] = count_names.n_pre / count_names.n_sur
    
    return count_names

def is_name_multimatch(nombre):
    mdel   = re_del.search(nombre)
    msant  = re_sant.search(nombre)
    
    mlaos  = re_laos.search(nombre)
    mdela  = re_dela.search(nombre)
    
    mde  = re_de.search(nombre)
    
    mvon   = re_von.search(nombre)
    mvande = re_vande.search(nombre)
    
    if mdel or msant or mlaos or mdela or mde or mvon or mvande:
        return True
    else:
        return False

if full_run:
    freq = pd.concat([nf[['cedula', 'nombre', 'sur_padre', 'sur_madre']], 
                     nf.progress_apply(lambda row: parse_prename(row.prenames), axis=1, result_type='expand')], axis=1)

    freq['nlen'] = freq[['sur_padre','sur_madre','pre1','pre2','pre3','junk']
                     ].replace("", np.nan).notnull().astype(int).sum(axis=1)
    print("saving...")
    freq.to_csv(LOC_INTERIM + "FREQFILE_" + TODAY + ".tsv", sep='\t', index=False)
else:
    freq = pd.read_csv(LOC_INTERIM +  "FREQFILE_" + READ_DATE + ".tsv", sep='\t', dtype=str)

print("# recs :", len(freq))
# 90 mins

"""
test freqfile
"""

new_types = {'cedula': str,
             'nombre': str,
             'sur_padre': str,
             'sur_madre': str,
             'pre1': str,
             'pre2': str,
             'pre3': str,
             'junk': str,
             'nlen': int,
             }

test_cases_freq = pd.read_csv(LOC_RAW + "simpsons_test_cases_freqfile.tsv",
                                  sep='\t', dtype=new_types, keep_default_na=False, na_values=nan_values)

for i in range(2):
    print(".................................")
print("freqfile test")

for row in test_cases_freq.to_dict(orient='records'):
    try:
        for key in row.keys():
            if pd.isnull(row[key]):
                row[key] = ''
        res = freq[freq.cedula == row['cedula']].iloc[0].to_dict()
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
NAMECOUNTS
"""

count_names = count_all_names(freq)
count_names['nlen'] = count_names.obsname.map(lambda x: len(x.split()))
count_names['is_multimatch'] = count_names.obsname.map(is_name_multimatch)

if full_run:
    count_names.to_csv(LOC_INTERIM + "NAMECOUNTS_" +
                       TODAY + ".tsv", sep='\t', index=False)

"""
Test NAMECOUNTS
"""
new_types = {'obsname': str,
             'n_sur': float,
             'n_pre': float,
             'sratio': float,
             'pratio': float,
             'nlen': float,
             'is_multimatch': bool
             }

test_cases_NCOUNTS = pd.read_csv(LOC_RAW + "simpsons_test_cases_namecounts.tsv",
                              sep='\t', dtype=new_types, keep_default_na=False, na_values=nan_values)

for i in range(2):
    print(".................................")
print("NAMECOUNTS test")
for row in test_cases_NCOUNTS.to_dict(orient='records'):
    try:
        res = count_names[count_names.obsname ==
                          row['obsname']].iloc[0].to_dict()  
    except IndexError:
        print(row['obsname'], ": NOT IN THIS SUBFRAME")
        continue
    if res == row:
        print(row['obsname'], ": OK")
    else:
        print(row['obsname'], ": FAILED")
        print(row)
        print(res)

### Check single-token surnames

solo_sur = nf[nf.sur_padre.map(lambda x: len(x.split()) == 1)]
solo_counts = solo_sur.sur_padre.value_counts()
fig, ax = plt.subplots()
bins = np.arange(0,100000,100)
ax.set(yscale='log')
ax.hist(solo_counts, bins=bins);
solo_counts[:10]
surs_solo_rare = set(solo_counts[solo_counts < 10].index)
len(surs_solo_rare)
# how many records are used by "rare" surnames?
nf.sur_padre.isin(surs_solo_rare).sum()   # 101k (out of 20M), so 0.5%

### Check the two-token surnames, many are actually a sur-pre combo

dual_sur = count_names[(count_names.nlen == 2) & ~count_names.is_multimatch]
dual_sur = dual_sur.apply(lambda x: x.obsname.split(),
                          axis=1, result_type='expand')
column = ['probably_sur', 'probably_pre']
dual_sur = pd.DataFrame(columns=column)

dual_sur = dual_sur.merge(count_names[['obsname', 'sratio']],
                          left_on='probably_sur', right_on='obsname').drop(columns=['obsname'])

dual_sur = dual_sur.merge(count_names[['obsname', 'pratio']],
                          left_on='probably_pre', right_on='obsname').drop(columns=['obsname'])

dual_sur['evidence'] = dual_sur.sratio * dual_sur.pratio

dual_sur
n_samp = np.min((30, len(dual_sur[dual_sur.evidence < 100])))
dual_sur[(dual_sur.evidence < 100)].sample(
    n_samp)  # & (tmp.probably_pre != "MARIA")]
dual_sur[(dual_sur.probably_pre == 'MARIA') &
         (dual_sur.probably_sur == 'BANO')]
fig, ax = plt.subplots(figsize=(9, 6))

ax.set(xscale='log', xlabel='evidence name is actually sur+pre', ylabel='frequency')

logbins = np.logspace(-6, 10, 129)
ax.hist(dual_sur.evidence, bins=logbins, color='blue', alpha=0.3)
ax.hist(dual_sur[dual_sur.probably_pre == 'MARIA'].evidence,
        bins=logbins, color='red', alpha=0.3)
#
sub = dual_sur[dual_sur.probably_pre.isin(
    {'MARIA', 'JORGE', 'CARMEN', 'JOSE'})]

ax.hist(sub.evidence, bins=logbins, color='red', alpha=0.3)
dual_sur[(dual_sur.evidence < 20) & (dual_sur.probably_pre == 'MARIA')]
tst = "MANSSUR	YAMILE"
tst = ' '.join(tst.split('\t'))
nf[nf.sur_madre == tst]
dual_sur[dual_sur.evidence > 100]
# ~ 120k sur_madre need this treatment.  But only ~300 padres (and most of those seem to actually be correct)
needs_repair = dual_sur[dual_sur.evidence > 1000]
needs_repair = set(needs_repair.probably_sur + ' ' + needs_repair.probably_pre)


def repair_dual_surmadre(row):
    out = {'sur_madre': "", 'prenames': ""}
    sur_madre, pre1 = row.sur_madre.split()

    out['prenames'] = pre1 + ' ' + row.prenames
    out['sur_madre'] = sur_madre
    return out


nf.loc[nf.sur_madre.isin(needs_repair), ['sur_madre', 'prenames']
       ] = nf[nf.sur_madre.isin(needs_repair)].progress_apply(lambda row: repair_dual_surmadre(row), axis=1, result_type='expand')

"""
Spousal names
"""
# Look for "DE $husband" within the mother's name, then remove and re-parse.

# Should probably expand this to include checks for spouse amongst women.  Probably plenty of isuses there.


def poss_husb(row):
    # tried both simple search and regex; no difference in speed
    return " DE " + row.sur_padre in row.nombre_madre

nf['maybe_husb'] = nf.progress_apply(lambda row: poss_husb(row), axis=1)

# 60 minutes  (this is a check of the mother's name, so have to run it for everyone; spouse would be only women)
sub = nf[nf.maybe_husb].copy(deep=True)
len(sub)
#sub.sample(20)
print(sub.head(10))
# these are almost all when a woman and her mother both use husband's last name, so algo picks it up as mother's
nf[nf.sur_madre == 'DE']


def remove_husband(row):
    out = row.copy(deep=True)
    try:
        madre = ''.join(row.nombre_madre.split(" DE " + row.sur_padre))
    except AttributeError:
        print("ERROR :", row)
#        return None
    out.nombre_madre = madre
    return out

removed = sub.apply(lambda row: remove_husband(row),
                    axis=1, result_type='expand')
newparse = removed.progress_apply(
    lambda row: parse_fullrow(row), axis=1, result_type='expand')
sub[(newparse.sur_madre != sub.sur_madre)]
newparse[(newparse.sur_madre != sub.sur_madre)]
removed

# cleans mother's name of the "DE HUSBAND" junk
re_despouse = re.compile(u"\sDE (\w{2,})(\s|$)")
re_despouse = re.compile(u"(^|\s)\w{2,}\sDE (\w{2,})(\s|$)")

def clean_spousename(row):
    m = re_despouse.search(row.nombre_madre)

    # group1 might be husband's name, compare to already-extracted husband name
    if m and (m.group(1) in row.nombre_padre):
        # if it matches, replace with space (and strip, to avoid leaving trailing space)

        # NB : improve this by finding the longest husband-name that matches (as in the parse_fullrow alg)
        # ALSO, re-match the child/mother name
        return (re_despouse.sub(" ", row.nombre_madre)).strip()
    else:
        return row.nombre_madre

"""
Fix errors from the first pass
"""
missed = rf[~rf.cedula.isin(set(nf.cedula)) & (rf.nombre_padre != "") & (rf.nombre_madre != "")]
len(missed)

"""
Following cleanup, re-check the prename parsing
"""
if full_run:
    nf.to_csv(LOC_INTERIM + 'names_cleaned_' + TODAY + '.tsv', sep='\t', index=False)
# 90 sec

if full_run:
    newfreq = pd.concat([nf[['cedula', 'nombre', 'sur_padre', 'sur_madre']], 
                     nf.progress_apply(lambda row: parse_prename(row.prenames), axis=1, result_type='expand')], axis=1)

    newfreq['nlen'] = newfreq[['sur_padre','sur_madre','pre1','pre2','pre3','junk']
                     ].replace("", np.nan).notnull().astype(int).sum(axis=1)
    print("saving...")
    newfreq.to_csv(LOC_INTERIM + "NEWFREQFILE_" + TODAY + ".tsv", sep='\t', index=False)
#    newfreq.to_csv("NEWfreqFILE_" + READ_DATE + ".tsv", sep='\t', index=False)
else:
    newfreq = pd.read_csv(LOC_INTERIM + "NEWFREQFILE_" + READ_DATE + ".tsv", sep='\t', dtype=str)

print("# recs :", len(newfreq))

# 90 mins

count_names = count_all_names(newfreq)
count_names['nlen'] = count_names.obsname.map(lambda x: len(x.split()))
count_names['is_multimatch'] = count_names.obsname.map(is_name_multimatch)

if full_run:
    count_names.to_csv(LOC_INTERIM + "NAMECOUNTS_" + TODAY + ".tsv", sep='\t', index=False)
    
len(newfreq)

"""
Test names_cleaned
"""
new_types = {'cedula': str, 'nombre': str, 'prenames': str, 'gender': str,
             'nombre_padre': str, 'sur_padre': str, 'has_padre': bool, 'is_plegal': bool,
             'nombre_madre': str, 'sur_madre': str, 'has_madre': bool, 'is_mlegal': bool,
             'is_funky': bool, 'nlen_padre': int, 'nlen_madre': int, 'n_char_nombre': int,
             'n_char_prename':int, 'maybe_husb': bool
             }

test_cases_NCLEANED = pd.read_csv(LOC_RAW + "simpsons_test_cases_names_cleaned.tsv",
                                 sep='\t', dtype=new_types, keep_default_na=False, na_values=nan_values)

for i in range(2):
    print(".................................")
print("names_cleaned test")

for row in test_cases_NCLEANED.to_dict(orient='records'):
    try:
        res = nf[nf.cedula == row['cedula']].iloc[0].to_dict()
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
Replace spaces with underscores
"""
ncounts = count_all_names(freq)

def merge_underscore_names(ncounts):
    under_prenames = set(ncounts[ncounts.obsname.map(lambda x: "_" in x)].obsname)

    for upre in tqdm(under_prenames):

        u_rec = ncounts[ncounts.obsname == upre].iloc[0]

        norm_pre = ' '.join(upre.split("_"))
        norm_rec = ncounts[ncounts.obsname == norm_pre]
        if len(norm_rec) == 1:
            norm_rec = norm_rec.iloc[0]
            ncounts.loc[ncounts.obsname == norm_pre, 'n_sur'] = u_rec.n_sur + norm_rec.n_sur - 0.5
            ncounts.loc[ncounts.obsname == norm_pre, 'n_pre'] = u_rec.n_pre + norm_rec.n_pre - 0.5
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

if full_run:
    allnames = merge_underscore_names(ncounts)
    allnames.to_csv(LOC_INTERIM+'ALLNAMES_' + TODAY + '.tsv', sep='\t', index=False)
else:
    allnames = pd.read_csv(LOC_INTERIM+"ALLNAMES_" + READ_DATE + ".tsv", sep='\t')

print("# allnames :", len(allnames))
# 15 mins

"""
Test names_cleaned
"""
new_types = {'obsname': str,
             'n_sur': float,
             'n_pre': float,
             'sratio': float,
             'pratio': float,
             'nlen': float,
             'is_multimatch': bool
             }
test_cases_ALLNAMES = pd.read_csv(LOC_RAW + "simpsons_test_cases_allnames.tsv",
                                  sep='\t', dtype=new_types, keep_default_na=False, na_values=nan_values)

for i in range(2):
    print(".................................")
print("allnames test")

for row in test_cases_ALLNAMES.to_dict(orient='records'):
    try:

        res = allnames[allnames.obsname == row['obsname']].iloc[0].to_dict()
    except IndexError:
        print(row['obsname'], ": NOT IN THIS SUBFRAME")
        continue
    if res == row:
        print(row['obsname'], ": OK")
    else:
        print(row['obsname'], ": FAILED")
        print(row)
        print(res)

"""
Old junk, I think
"""
def parse_parent_prenames(row):
    
    out = {'padre_sur2':None, 'padre_pre1':None, 'padre_pre2':None,
           'madre_sur2':None, 'madre_pre1':None, 'madre_pre2':None
          }
    
    ptokens = row.nombre_padre.split()
    if row.nlen_padre == 2:
        if row.sur_padre == ptokens[0]:
            # legal order; prename is final token
            out['padre_pre1'] = ptokens[1]
        elif row.sur_padre == ptokens[1]:
            # sequential order
            out['padre_pre1'] = ptokens[0]
        else:
            print("WTF PADRE n2 ??", row)
            
    elif row.nlen_padre == 4:
        if row.sur_padre == ptokens[0]:
            # legal order
            out['padre_sur2'] = ptokens[1]
            out['padre_pre1'] = ptokens[2]
            out['padre_pre2'] = ptokens[3]
        elif row.sur_padre == ptokens[2]:
            # sequential ourder
            out['padre_sur2'] = ptokens[3]
            out['padre_pre1'] = ptokens[0]
            out['padre_pre2'] = ptokens[1]
        else:
            print("WTF PADRE n4 ?", row)
nf.nlen_padre.value_counts()

# Identify and fix funky names ("VAN DER HOOK", "DE LA CRUZ", etc)

# This takes a while to run, bc ~N**2 

# in all cases, we look for a word boundary as the first group, then our funky name as the second
re_von = re.compile(u"(^|\s)(V[AO]N \w{2,})(\s|$)")              # these results are subset of "re_vande"
re_vande = re.compile(u"(^|\s)(V[AO]N DE[RN]? \w{2,})(\s|$)")
re_sant = re.compile(u"(^|\s)(SANT?A? \w{2,})(\s|$)")
re_dela = re.compile(u"(^|\s)(DE L[AO]S? ?\w{2,})(\s|$)")   # these results are subset of "re_laos"
re_laos = re.compile(u"(^|\s)(L[AEO]S? \w{2,})(\s|$)")
re_del  = re.compile(u"(^|\s)(DEL \w{2,})(\s|$)")


def regex_funky(row):
    """ This is a little slow (~4mins / million rows), but pretty thorough.  """
    
    mdel   = re_del.search(row.nombre)
    msant  = re_sant.search(row.nombre)
    
    mlaos  = re_laos.search(row.nombre)
    mdela  = re_dela.search(row.nombre)
    
    mvon   = re_von.search(row.nombre)
    mvande = re_vande.search(row.nombre)
    
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
    
    for poss_funk in poss_funks:
        
        if row.nombre.endswith(poss_funk):
            # assume it's a prename
            funky_prenames.append(poss_funk)
            
        elif row.nombre_padre and (poss_funk in row.nombre_padre):
            # father's surname
            funky_surnames.append(poss_funk)
            
        elif row.nombre_madre and (poss_funk in row.nombre_madre):
            # mother's surname
            funky_surnames.append(poss_funk)
            
        elif not row.nombre_padre and not row.nombre_madre:
            # no parent data available, so just assume it's correct
            funky_surnames.append(poss_funk)

        else:
            # something's fucky (possible misspelling ?)
            funky_junk.append(poss_funk)

# sub = rf.sample(1000000)

funky_surnames = list()
funky_prenames = list()
funky_junk = list()
# _ = sub.apply(lambda row: regex_funky(row), axis=1)

funkcount = Counter(funky_junk)
surcount = Counter(funky_surnames)
precount = Counter(funky_prenames)
def check_weirds(row, multi, df, nameset):
    
    MIN_RECS = 2
    new_names = set()
    parts = [x.strip() for x in row.nombre.split(multi)[0].split()]
    for ind in np.arange(len(parts)):
        guess = ' '.join(parts[ind:] + [multi])
        if guess in nameset:
            # it's already known to be a last nome, just skip
            break
            
        # don't draw conclusions unless we have data from both parents
        if (row.nombre_padre and row.nombre_madre):
            if (   (guess in row.nombre_padre and not multi in row.nombre_madre)
                or (guess in row.nombre_madre and not multi in row.nombre_padre) ):
                if df.nombre.map(lambda x: x.startswith(guess)).sum() >= MIN_RECS:
                    print("\t", guess)
                    new_names.add(guess)
        
        """
                if (guess in row.nombre_padre) or (guess in row.nombre_madre):
            if df.nombre.map(lambda x: x.startswith(guess)).sum() >= MIN_RECS:
                print("\t", guess)
                new_names.add(guess)
        """
    return new_names

nameset = set(surcount)

# funky/multi-word surnames which have occurred more than once
multis = [x for x,y in surcount.items() if y > 1]
for multi in multis:
    
    # check how often the proposed surname is the first element/surname  
    # If that happens rarely, then there might be another word in front
    sub = rf[rf.nombre.map(lambda x: x.startswith(multi))]
    thresh = np.max([2, np.floor(np.sqrt(surcount[multi]))])
    if len(sub) <= thresh:
        print(multi)
        for ind, row in rf[rf.nombre.map(lambda x: multi in x)].iterrows():
            new_names = check_weirds(row, multi, rf, nameset)
            nameset = nameset | new_names

#  ~ 5 mins/million
junky = list()
my_ynames = list()

def get_ynames(row):
    names = row.nombre.split()
    ind_y = names.index('Y')
    
    pre = names[ind_y - 1]
    post = names[ind_y + 1]
    
    if pre == post:
        # sometimes "SANCHEZ Y SANCHEZ" is used to indicate that both parents had the same name.  
        pass
    else:
        possible = pre + ' Y ' + post

        if possible in row.nombre_padre or possible in row.nombre_madre:
#            funky_surnames.append(possible)
            my_ynames.append(possible)
        else:
            junky.append(possible)

has_y = rf[rf.nombre.map(lambda x: " Y " in x)]
has_y.apply(lambda row: get_ynames(row), axis=1);

junk_count = Counter(junky)
Counter(my_ynames)
funky_surnames = list(set(surcount) | set(my_ynames) | nameset)
funky_surnames.sort(reverse=True, key=len)
len(funky_surnames)
re_laroy = re.compile(r"(^)LA ROSA(\s|$)")
rf[rf.nombre.map(lambda x: True if re_laroy.search(x) else False)]
funky_prenames = list(set(precount))
funky_prenames.sort(reverse=True, key=len)
len(funky_prenames)
funks = funky_surnames + funky_prenames
funks.sort(reverse=True, key=len)
funks

def fix_funk(nombre, funks):
    """ The 'funks' list should be sorted in descending length, to prevent substrings from being clobbered."""
    nlen = len(nombre.split())
    for funk in funks:
        flen = len(funk.split())
        if (nlen - flen > 1):
            if (funk in nombre):
                defunk = '_'.join(funk.split())
                nombre = defunk.join(nombre.split(funk))
                nlen = len(nombre.split())
        else:
            # since the list is sorted, once we have a match that uses all the tokens, just skip ahead
            continue
    return nombre

def fix_funk(nombre, funks):
    """ The 'funks' list should be sorted in descending length, to prevent substrings from being clobbered."""
    nlen = len(nombre.split())
    for funk in funks:
        flen = len(funk.split())
        if (nlen - flen > 1):
            if (funk in nombre):
                defunk = '_'.join(funk.split())
                nombre = defunk.join(nombre.split(funk))
                nlen = len(nombre.split())
        else:
            # since the list is sorted, once we have a match that uses all the tokens, just skip ahead
            continue
    return nombre
from collections import namedtuple
re_under = re.compile(r"(^|\s)(\w+_\w*)(\s|$)")
re_under.findall("DE_LA_CRUZ SOME WAN")
#help(re_under.finditer)
len(rf)

"""
Split names, writ output
"""
def write_nameparts(nombre):

    rec = {'sur1': '',
           'sur2': '',
           'pre1': '',
           'pre2': '',
           'pre3': '',
           'junk': ''
           }

    try:
        tokens = nombre.split()
    except AttributeError:
        # when something besids a string is passed in
        tokens = []

    # get rid of rogue commas
    if ',' in nombre:
        print(nombre)
        nombre = ' '.join(nombre.split(','))

    tlen = len(tokens)

    if tlen < 2:
        return pd.Series(rec)

    elif tlen == 2:
        rec['sur1'] = tokens[0]
        rec['pre1'] = tokens[1]

    elif tlen == 3:
        # assume only 1 surname - it's correct for extranjeros, and ecudoranos will get fixed by the algorithm
        rec['sur1'] = tokens[0]
        rec['pre1'] = tokens[1]
        rec['pre2'] = tokens[2]

    else:
        rec['sur1'] = tokens[0]
        rec['sur2'] = tokens[1]
        rec['pre1'] = tokens[2]

        if tlen > 4:
            rec['pre3'] = tokens[4]
        if tlen > 5:
            rec['junk'] = ' '.join(tokens[5:])
    return pd.Series(rec)


def parse_column(rf, col):
    namecols = ['sur1', 'sur2', 'pre1', 'pre2', 'pre3', 'junk']

    now = dt.datetime.now()
    fname = LOC_INTERIM + col + "_" + now.strftime("%Y%m%dT%H%M") + ".tsv"

    datacol = rf[col]
    cedulas = rf['cedula']
    nlens = datacol.map(lambda x: len(x.split())).astype(str)
    with open(fname, 'w') as f:
        f.write('\t'.join(['cedula', 'nombre', 'nlen'] + namecols) + '\n')
        for ind, nombre in tqdm(datacol.items()):
            vals = write_nameparts(nombre)
            f.write('\t'.join([cedulas[ind], nombre,
                    nlens[ind]] + [v for v in vals]) + '\n')

parse_column(rf, 'nombre')

parse_column(rf, 'nombre_padre')

parse_column(rf, 'nombre_madre')

""" 
for i, target in datacol.items():
    pass
datacol = rf['nombre']
for namefile in namefiles:
    tokens = namefile.split("/")
    path = "/".join(tokens[:-1])
    nameout = tokens[-1][:-4] + '_PARSED.tsv'
    print(nameout)

    df = load_large_dta(namefile)
    df.rename(columns={'name': 'nombre', 'name_len': 'nlen'}, inplace=True)
    df['nlen'] = df.nlen.astype(str)
#    df = pd.concat([df, df['nombre'].apply(split_firstlast)], axis=1)
#    df = pd.concat([df, df['nombre'].apply(split_nameparts)], axis=1)

    with open(nameout, 'w') as f:
        f.write('\t'.join(['cc', 'nombre', 'nlen'] + namecols) + '\n')
        for ind, row in tqdm(df.iterrows()):
            vals = write_nameparts(row)
            f.write(
                '\t'.join([row.cc, row.nombre, row.nlen] + [v for v in vals]) + '\n')

    if i == 0:
        ff = df.copy(deep=True)
    i += 1 
"""





