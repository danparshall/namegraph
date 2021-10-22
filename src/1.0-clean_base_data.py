"""
Clean base data
"""
# This notebook takes the input data file, produces a cleaned version.  One of the biggest challenges is names with spaces, so I attempt to replace the spaces with underscores.  How do I know which names have spaces?  By having cleaned and processed the data through NB 3 already.  After NB 3 I save the results, and then use that file as an input here.  It makes other things much, much easier.

# I also check for labels/notes that have been put into the fields.  E.g. duplicate records, people who need to provide birth certificates, etc.

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

full_run = True
N_ROWS = None  # 1000000
READ_DATE = '20200823'

LOC_RAW = "../data/raw/"

LOC_RAWFILE = LOC_RAW + 'RAW_SAMPLE.tsv'

LOC_INTERIM = "../data/interim/"

TODAY = dt.datetime.now().strftime("%Y%m%d")
TODAY

"""
Load special names data
"""
LOC_SPECIAL = LOC_INTERIM + "special_names.txt"

specials = set()
with open(LOC_SPECIAL) as f:
    for line in f:
        if line:
            line = line.strip()

            # eliminate comments
            if line.startswith('#'):
                continue
            line = line.split('#')[0].strip()

            specials.add(line)
len(specials)

allnames = pd.read_csv(LOC_INTERIM + "ALLNAMES_20200824.tsv", sep='\t')
len(allnames)

compound_names = set()

# in all cases, we look for a word boundary as the first group, then our funky name as the second
# these results are subset of "re_vande"
re_von = re.compile(u"(^|\s)(V[AO]N \w{2,})(\s|$)")
re_vande = re.compile(u"(^|\s)(V[AO]N DE[RN]? \w{2,})(\s|$)")
# SAN and SANTA (SANTO doesn't form compounds)
re_sant = re.compile(u"(^|\s)(SANT?A? \w{2,})(\s|$)")
# these results are subset of "re_laos"
re_dela = re.compile(u"(^|\s)(DE L[AO]S? ?[AO]? ?\w{2,})(\s|$)")
re_laos = re.compile(u"(^|\s)(L[AEO]S? \w{2,})(\s|$)")
re_del = re.compile(u"(^|\s)(DEL \w{2,})(\s|$)")
re_de = re.compile(r"(^|\s)(DE \w{2,})(\s|$)")


def regex_compound_names(nombre):
    """ This is a little slow (~4mins / million rows), but pretty thorough.  """

    mdel = re_del.search(nombre)
    msant = re_sant.search(nombre)

    mlaos = re_laos.search(nombre)
    mdela = re_dela.search(nombre)

    mvon = re_von.search(nombre)
    mvande = re_vande.search(nombre)

    mde = re_de.search(nombre)

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

"""
Load Reg data
"""


def sub_unders(nombre):
    # this function adds underscores to compound names; gets run in "fix_nombre()"
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
re_starplus = re.compile(r'(^[\s*]+)([\w\s]+)')
# triggers on one or more of "·./+$\(){}[]<>"
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
#    if " DE EL " in nombre:
#        nombre = " DEL ".join(nombre.split(" DE EL " ))

    nombre = sub_unders(nombre)
    return nombre


nan_values = ['-1.#IND', '1.#QNAN', '1.#IND', '-1.#QNAN', '#N/A N/A', '#N/A', 'N/A', 'n/a',  # 'NA' is sometimes name
              '<NA>', '#NA', 'NULL', 'null', 'NaN', '-NaN', 'nan', '-nan', '']

date_cols = ['dt_birth', 'dt_death', 'dt_marriage']

dtypes_namedata = {'cedula': str, 'nombre': str, 'nombre_spouse': str, 'orig_cedula': str,
                   'nombre_padre': str, 'nombre_madre': str, 'ced_spouse': str, 'ced_padre': str, 'ced_madre': str,
                   'is_nat': bool, 'is_nat_padre': bool, 'is_nat_madre': bool,
                   }

rf = pd.read_csv(LOC_RAWFILE, sep='\t', encoding='latin',
                 parse_dates=date_cols, dtype=dtypes_namedata,
                 keep_default_na=False, na_values=nan_values,
                 nrows=N_ROWS,
                 )

if full_run:

    # cleanup names
    namecols = ['nombre', 'nombre_spouse', 'nombre_padre', 'nombre_madre']
    for col in namecols:
        print(col)
        rf[col] = rf[col].map(fix_nombre)

    # save cleaned input data
    rf.to_csv(LOC_RAW + "NAMES__c01__" + TODAY + ".tsv", sep='\t', index=False)


# replace NaN with empty string
text_cols = ['nombre', 'nombre_spouse', 'nombre_padre',
             'nombre_madre', 'ced_spouse', 'ced_padre', 'ced_madre']
#rf[text_cols] = rf[text_cols].fillna("")

print("# records loaded :", len(rf))
# 5 mins

"""
Look for duplicates
"""
dups = rf[rf.duplicated(subset=['nombre', 'dt_birth'], keep=False)].sort_values(
    ['dt_birth', 'nombre'])
len(dups)

dups.gender.value_counts()

dups[(dups.nombre_spouse != "") & (dups.gender == '1')][-50:]   # later dups are caught earlier

dups[:50]

old = rf[(rf.orig_cedula != rf.cedula)]
len(old)

old[old.apply(lambda row: (row.cedula[1:] != row.orig_cedula), axis=1)]

rf.sample(10)

"""
Cleanup
"""
if full_run:
    bigamos = rf[rf.nombre.map(lambda x: "BIGAMO" in x)
                 | rf.nombre_padre.map(lambda x: "BIGAMO" in x)
                 | rf.nombre_madre.map(lambda x: "BIGAMO" in x)
                 ]
    bigamos.to_csv(LOC_RAW + "BIGAMOS_" + TODAY + ".tsv", sep='\t')
else:
    bigamos = pd.read_csv(LOC_RAW + "BIGAMOS_" +
                          READ_DATE + ".tsv", sep='\t', dtype=str)

print("# bigamists :", len(bigamos))

re_pago = re.compile(r"PAG[AO]? MULT")

if full_run:
    pagos = rf[rf.nombre.map(lambda x: True if re_pago.search(x) else False)
               | rf.nombre_padre.map(lambda x: True if re_pago.search(x) else False)
               | rf.nombre_madre.map(lambda x: True if re_pago.search(x) else False)
               ]
    pagos.to_csv(LOC_RAW + "PAGOS_" + TODAY + ".tsv", sep='\t')
else:
    pagos = pd.read_csv(LOC_RAW + "PAGOS_" + READ_DATE +
                        ".tsv", sep='\t', dtype=str)

print("# pagos :", len(pagos))

re_nacimento = re.compile(r"(^|\s|\*)(NAC|NCM)[\w\s]*\d{2,4}")

if full_run:
    nacs = rf[rf.nombre_padre.map(lambda x: True if re_nacimento.search(x) else False)
              | rf.nombre_madre.map(lambda x: True if re_nacimento.search(x) else False)
              | rf.nombre_spouse.map(lambda x: True if re_nacimento.search(x) else False)
              | rf.nombre.map(lambda x: True if re_nacimento.search(x) else False)
              ]
    nacs.to_csv(LOC_RAW + "NACIMENTOS_" + TODAY +
                ".tsv", sep='\t', index=False)
else:
    nacs = pd.read_csv(LOC_RAW + "NACIMENTOS_" +
                       READ_DATE + ".tsv", sep='\t', dtype=str)

print("# NACIMENTOS :", len(nacs))

# "CED 2 VECES", "CED DOS VEC", "CEDL2VECES", "DOS CED", "2 CEDULAS"
re_cedula = re.compile(r"(DOB|DUP|DOS|2)\w{0,}\s?CE?DU?L?A?")

if full_run:
    dobles = rf[rf.nombre_padre.map(lambda x: True if re_cedula.search(x) else False)
                | rf.nombre_madre.map(lambda x: True if re_cedula.search(x) else False)
                | rf.nombre.map(lambda x: True if re_cedula.search(x) else False)
                | rf.nombre_spouse.map(lambda x: True if re_cedula.search(x) else False)
                ]
    dobles.to_csv(LOC_RAW + "DOBLES_" + TODAY + ".tsv", sep='\t', index=False)
else:
    dobles = pd.read_csv(LOC_RAW + "DOBLES_" + READ_DATE +
                         ".tsv", sep='\t', dtype=str)
print("# doble-cedula recs ", len(dobles))

# "PEDIR PARTIDA NACIMIENTO" and abbrev (e.g. "PP NAC", "P P NACI")
# WARNING! has some false positives (e.g. "PULLOPAXI NACIMBA ROBERTH GABRIEL")
#re_ppnac = re.compile(r"P\w{0,}\s?P\w{0,}\s[DE]{0,2}\s?NAC\w*")

# improved, shouldn't have false positives
re_ppnac = re.compile(r"P(P|\w{0,}\sP\w{0,})\s[DE]{0,2}\s?(NAC|NCM)\w*")

if full_run:
    ppnaci = rf[rf.nombre_padre.map(lambda x: True if re_ppnac.search(x) else False)
                | rf.nombre_madre.map(lambda x: True if re_ppnac.search(x) else False)
                | rf.nombre.map(lambda x: True if re_ppnac.search(x) else False)
                | rf.nombre_spouse.map(lambda x: True if re_ppnac.search(x) else False)
                ]
    ppnaci.to_csv(LOC_RAW + "PPNAXI_" + TODAY + ".tsv", sep="\t", index=False)
else:
    ppnaci = pd.read_csv(LOC_RAW + "PPNAXI_" + READ_DATE +
                         ".tsv", sep='\t', dtype=str)

print("# PPNACI ", len(ppnaci))

ced_map = {'0': 0, '1': 2, '2': 4, '3': 6, '4': 8,
           '5': 1, '6': 3, '7': 5, '8': 7, '9': 9}


def isvalid_cedula(ced):

    try:
        if len(ced) != 10:
            print("CED NOT 10 :", ced)
            return False
    except TypeError:
        print("NONETYPE :", ced)

    oddsum = sum([int(x) for x in ced[1] + ced[3] + ced[5] + ced[7]])
    evensum = sum([ced_map[x]
                  for x in ced[0] + ced[2] + ced[4] + ced[6] + ced[8]])

    # last digit is a checksum
    is_valid = str(np.mod(10 - np.mod(oddsum + evensum, 10), 10)) == ced[9]

    if not is_valid:
        print("INVALID :", ced)
    return is_valid

re_digits = re.compile(r"[\w\s]+\d+")

if False:  # full_run:
    digits = rf[rf.nombre_padre.map(lambda x: True if re_digits.search(x) else False)
                | rf.nombre_madre.map(lambda x: True if re_digits.search(x) else False)
                | rf.nombre_spouse.map(lambda x: True if re_digits.search(x) else False)
                | rf.nombre.map(lambda x: True if re_digits.search(x) else False)
                ]
    digits.to_csv(LOC_RAW + "DIGITS_" + TODAY + ".tsv", sep='\t', index=False)
elif True:
    digits = pd.read_csv(LOC_RAW + "DIGITS_20200822.tsv", sep='\t', dtype=str)
else:
    digits = pd.read_csv(LOC_RAW + "DIGITS_" + READ_DATE +
                         ".tsv", sep='\t', dtype=str)

print("# recs with digits :", len(digits))
# ~15 minutes

"""
Remove all "bad" records from reg frame, then save
"""
badlist = [set(bigamos.cedula), set(dobles.cedula), set(pagos.cedula), set(nacs.cedula),
           set(ppnaci.cedula), set(digits.cedula)]

tot = 0
for x in badlist:
    tot += len(x)
print(tot)  # 10763

ceds_junk = set()

#[ceds_junk.add(y) for y in x for x in badlist]

for x in badlist:
    for y in x:
        ceds_junk.add(y)
len(ceds_junk)  # 8541

rf = rf[~rf.cedula.isin(ceds_junk)]

print("# decent recs :", len(rf))
rf.reset_index(inplace=True, drop=True)
rf.to_csv(LOC_RAW + "NAMES__c01__" + TODAY + ".tsv", sep='\t', index=False)

# 5 min
