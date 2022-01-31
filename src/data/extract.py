import pandas as pd
import numpy as np
import re


# enable progress bar on long operations
from tqdm.auto import tqdm
from itertools import product
tqdm.pandas()


"""
NOTES:


TODO:
Handle "FALLECIDA EN ...." and other things when part of the "nombre" field.



"""



def check_nombre_doubling(nombre):
    """ Checks name for possible repeats, extracts surname if doubled (which would be for both parents).

    It's entirely possible for someone to be named "JOSE JULIO GARCIA GARCIA", or "DE LA CUEVA DE LA CUEVA JOSE JULIO".
    This function looks for repeated tokens at the beginning or end of a string, and assumes that they correspond to surnames.
    """
    tokens = nombre.split()
    n_tokens = len(tokens)
    if n_tokens == len(set(tokens)):
        # no possible doubling, use name as-is
        return ""
    else:
        # check for doublings, looking at both the front and the end of the string
        surname = ""
        for i in range(1, round((n_tokens-1)/2)):
            # note that we have to check both for single-token and multi-token surnames
            # indexing goes for i, then 2i, to look first for "GARCIA GARCIA", then "DA VINCI DA VINCI", etc
            if tokens[:i] == tokens[i: 2*i]:
                # legal form, start at the beginning of the string
                surname = ' '.join(tokens[:i])
                break
            elif tokens[-i:] == tokens[-2*i:-i]:
                # social form, start at end of string and work backwards
                surname = ' '.join(tokens[-i:])
                break
    return surname



def get_substrings(nombre, START=0):
    """ Generator returning successively shorter contiguous subsets of tokens.

    Given a string consisting of tokens 'X Y Z', this function returns:
        'X Y Z',
        'X Y',
        'Y Z',
        'X',
        'Y',
        'Z'

    NB: single-token substrings of length 2 or less (e.g., "A", "DE", etc) are not returned    
    """
    tokens = nombre.split()
    nlen = len(tokens)
    n_max = nlen - START
    for chunk_len in range(n_max, 0, -1):
        n_subs = n_max - chunk_len + 1
        for ind in range(0, n_subs):
            sub = ' '.join(tokens[START+ind: ind+chunk_len])

            # only return a single-token substring if the length is at least 3 (avoids matching on "DE", etc)
            if (chunk_len > 1) or (len(sub) > 2):
                yield sub

def desinterpret_surname(surname):
    """ Correct the detection of monosylabus ('DE') and non coherent surnames (endswith 'DE', 'DEL', 'LA')
    
    Manage the cases when both the mother and the daughter use honorific. Mother's surname cannot end in ' DE' or ' LA'
    Sometimes when the names are in short form (one prename and one surname), it detects mother or father surname as
    monosylabus words as 'DE'.
    Args:
        surname: detected mother or father's surname given by parse functions.

    Returns:
        surname: mother or father's surname without honorific and monosylabus
    """
    while surname.endswith(" DE") or surname.endswith(" LA") or surname.endswith(" DEL"):
        surname = surname[:-3]
        surname = surname.strip()
        # madre = ''.join(madre.rsplit(" DE", 1))
    
    if surname == 'DE' or surname == 'DEL':
        surname = ''
    return surname

def fix_spelling_errors(nombre):
    keyletters = ['ZS','VB','IY']

    # Convert input string into a list so we can easily substitute letters
    seq = list(nombre)
    for key in keyletters:
        indices = [i for i, c in enumerate(seq) if c in key]
        for t in product(key, repeat = len(indices)):
            for i, c in zip(indices, t):
                seq[i] = c
            yield ''.join(seq)

def parse_padre(row, parts, nomset, pset):
    """ Identifies surname of father by comparing the 'nombre' and 'nombre_padre' fields within a given row.
    
    Args:
        row: row of the reg dataframe, containing both the 'nombre' and 'nombre_padre' columns
        parts: list of tokens from the nombre field
        nomset: set of tokens from the nombre field
        pset: set of tokens from the nombre_padre field

    Returns:
        padre: surname of father
        parts: list of tokens from the nombre field, after removing those for the father's surname
    """
    # start by trying the first LEN-1 tokens as a single name, then LEN-2 tokens, etc
    # this matches longest chunk found, so it should pick up compound names like DE LA CUEVA
    padre = ""
    if row.nombre_padre:
        
        # we try names that might have doubling first, befor moving to the more common situation
        poss_padre = check_nombre_doubling(row.nombre_padre)
        poss_pset = set(poss_padre.split())
        
        if (poss_padre
            and (poss_padre in row.nombre)
            and not row.nombre.endswith(poss_padre)
            and poss_pset.issubset(nomset)
            ):
            padre = poss_padre
            parts = ''.join(row.nombre.split(padre, maxsplit=1)).strip().split()

        else:
            # start by trying everything except the last element (always a prename), and work down
            for ind in range(len(parts)-1, 0, -1):
                guess = ' '.join(parts[:ind])
                poss_pset = set(guess.split())
                if (guess in row.nombre_padre
                    and poss_pset.issubset(nomset)
                    and poss_pset.issubset(pset)
                    ):
                    padre = guess
                    parts = row.nombre.split(padre, maxsplit=1)[1].split()  # update before checking mother's name
                    break

    padre = desinterpret_surname(padre)

    return padre, parts

def wrap_parse_padre(row, parts, nomset, pset):
    padre, parts = parse_padre(row, parts, nomset, pset)

    if padre == '':
        for combinacion in fix_spelling_errors(row.nombre):
            nomset = set(combinacion.split())
            parts = combinacion.split()
            padre, parts = parse_padre(row, parts, nomset, pset)
            if padre != '':
                break
    
    return padre, parts

def parse_madre(row, parts, nomset, mset):
    """ Identifies surname of mother by comparing the 'nombre' and 'nombre_madre' fields within a given row.
    
    Args:
        row: row of the reg dataframe, containing both the 'nombre' and 'nombre_madre' columns
        parts: list of tokens from the nombre field, after removing father's surname
        nomset: set of tokens from the nombre field
        mset: set of tokens from the nombre_madre field

    Returns:
        madre: surname of mother

    NOTE: when mother's name is in short legal form (such as "LOPEZ MARIA"), and daughter 
    shares mother's prename (such as "GARCIA LOPEZ MARIA JUANA"), then this function will
    interpret the mother's combined name as the mother's surname (i.e., "LOPEZ MARIA").
    Not sure there's a good solution to handle this while also robustly handling multi-token
    surnames such as "DE LA CUEVA".
    """
    madre = ""
    if row.nombre_madre:
        poss_madre = check_nombre_doubling(row.nombre_madre)
        poss_mset = set(poss_madre.split())
        if (poss_madre
            and (poss_madre in row.nombre)
            and not row.nombre.endswith(poss_madre)
                and poss_mset.issubset(nomset) and poss_mset.issubset(mset)):
            madre = poss_madre
        else:
            nombre_madre = row.nombre_madre

            if nombre_madre.startswith(parts[0]):
            # try to remove any catholic addons from both citizen and mother
            # this isn't a concern when in social form
            # complicated bc surnames like "GOMEZ DE LA TORRE" mean we have to skip the zeroth token

                # if names have underscores
                m_de_pre_nombre = re_de_pre_UNDERSCORE.match(
                    ' '.join(parts[1:]))
                if m_de_pre_nombre:
                    # keep 'parts' as a list
                    parts = parts[:1] + m_de_pre_nombre.group(1).split()

                # names without underscores
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
                    and not row.nombre.endswith(guess)
                        and poss_mset.issubset(nomset) and poss_mset.issubset(mset)
                        # NOTE: possibly check here to make sure nombre_madre isn't identical to 'guess'; may resolve the mother/daughter overlap issue
                        ):
                    # now check which is the better fit
                    if (guess == nombre_madre):
                        # when madre is in short legal form and daughter has the same prename1
                        # it can look like the mother's surname is "GONZALEZ MARIA", etc.
                        # So ignore these, ??? because we need to handle them later
                        pass
                    else:
                        madre = guess
                        break

    madre = desinterpret_surname(madre)
    return madre

def wrap_parse_madre(row, parts, nomset, pset):
    madre = parse_madre(row, parts, nomset, pset)

    if madre == '':
        for combinacion in fix_spelling_errors(row.nombre):
            nomset = set(combinacion.split())
            parts = combinacion.split()
            madre = parse_madre(row, parts, nomset, pset)
            if madre != '':
                break

    return madre

def parse_overlaps(row, nomset, pset, mset):
    """ Handles special case when tokens in 'nombre_padre' and 'nombre_madre' overlap.

    NOTE: when mother's name is in short legal form (such as "LOPEZ MARIA"), and daughter 
    shares mother's prename (such as "GARCIA LOPEZ MARIA JUANA"), then this function will
    interpret the mother's combined name as the mother's surname (i.e., "LOPEZ MARIA").
    Not sure there's a good solution to handle this while also robustly handling multi-token
    surnames such as "DE LA CUEVA".
    """
    surname = check_nombre_doubling(row.nombre)
    if surname != "":
        madre = surname
        padre = surname
    else:
        padre = ""
        madre = ""
        for guess in get_substrings(row.nombre_madre):
            if (guess in row.nombre
                and not row.nombre.endswith(guess)
               ):
                poss_padre = row.nombre.split(guess)[0].strip()
                poss_set = set(poss_padre.split())
                guess_set = set(guess.split())
                if (poss_padre in row.nombre_padre
                    and not row.nombre.endswith(poss_padre)
                    and poss_set.issubset(nomset) and poss_set.issubset(pset)
                    and guess_set.issubset(nomset) and guess_set.issubset(mset)
                    ):
                    padre = poss_padre
                    madre = guess
                    break

    padre = desinterpret_surname(padre)
    madre = desinterpret_surname(madre)
    return padre, madre



re_de_pre = re.compile(r"(^.*)\s(DEL?\s\w+.*$)")
re_de_pre_UNDERSCORE = re.compile(r"(^.*)\s(DEL?_\w+.*$)")
def parse_fullrow(row):
    """ Identifies surnames of citizen, by comparing to 'nombre_padre' and 'nombre_madre'

    The core insight is that the citizen's name is always written in legal form, so we
    can be sure that the first token(s) in the citizen's name match up to the father.
    Once we've identified the father's surname, we remove it from the string, and we
    now know that the new first token(s) correspond to the mother's surname.

    This function can handle multi-token names which don't include underscores.
    TODO: Use better exception handling.

    Args:
        row: row of the reg dataframe, containing the 'nombre', 'nombre_padre', and 'nombre_madre' columns

    Returns:
        out: a dict with the decomposed surnames and prenames (to be used as row of the names dataframe)
    """
    out = {"cedula": row.cedula, 
            "sur_padre": "", "sur_madre": "", "prenames": "",
            "has_padre": False, "is_plegal": False,
            "has_madre": False, "is_mlegal": False}

    if not row.nombre_padre and not row.nombre_madre:
        return out

    nomset = set(row.nombre.split())
    mset = set(row.nombre_madre.split())
    pset = set(row.nombre_padre.split())

    # check if madre/padre have overlapping tokens (requires special handling)
    both = pset & mset
    flag_overlap = len(both) > 0
    if flag_overlap:
        padre, madre = parse_overlaps(row, nomset, pset, mset)
    else:
        madre = ""
        padre = ""

    # if the overlap method wasn't successful, parse father/mother surname separately
    if not madre and not padre:
        parts = row.nombre.split()

        #### FATHERS NAME ####
        try:
            padre, parts = wrap_parse_padre(row, parts, nomset, pset)
        except:
            out['sur_padre'] = "HAS PADRE PROBLEM"
            return out

        #### MOTHERS NAME ####
        # having removed the padre name from the front of the string, try similar trick with the madre name
        try:
            madre = wrap_parse_madre(row, parts, nomset, mset)
        except:
            out['sur_madre'] = "HAS MADRE PROBLEM"
            return out

    # get prenames explicitly, as remainder after removing prenames.
    # this bypasses some funny stuff I have to do above
    deduced_surnames = (padre + " " + madre).strip()
    if deduced_surnames and row.nombre.startswith(deduced_surnames):
        prenames = ''.join([x.strip()
                           for x in row.nombre.split(deduced_surnames)])
    else:
        prenames = "HAS SURNAME PROBLEM"

    if padre:
        out['has_padre'] = True
        out['sur_padre'] = padre
        out['is_plegal'] = row.nombre_padre.startswith(padre) or row.nombre_padre.endswith(padre) or (padre in row.nombre_padre)
    if madre:
        out['has_madre'] = True
        out['sur_madre'] = madre
        out['is_mlegal'] = row.nombre_madre.startswith(madre) or row.nombre_madre.endswith(madre) or (madre in row.nombre_madre)
    out['prenames'] = prenames
    return out



# There are plenty of prenames which have some sort of honorific(e.g. "DEL CARMEN").  The goal here is to find any of those, and render them into a single token(e.g. "DEL_CARMEN").
# By definition, this only applies to names with at least 3 tokens(since the honorific is minimum of 2, and the prename is a minimum of 1).
# Occasionally, someone will have something like "SANTA DEL CARMEN".  In practice, they're referred to as "CARMEN".
# Because of cleaning in NB 1.0, this section has almost nothing to catch.
# in all cases, we look for a word boundary as the first group, then our funky name as the second
re_von = re.compile(u"(\s|^)(V[AO]N \w{2,})(\s|$)")              # these results are subset of "re_vande"
re_vande = re.compile(u"(\s|^)(V[AO]N DE[RN]? \w{2,})(\s|$)")
re_sant = re.compile(u"(\s|^)(SANT?A? \w{2,})(\s|$)")            # SAN and SANTA (SANTO doesn't form compounds)
re_dela = re.compile(u"(\s|^)(DE L[AO]S? ?\w{2,})(\s|$)")   # these results are subset of "re_laos"
re_laos = re.compile(u"(\s|^)(L[AEO]S? \w{2,})(\s|$)")
re_del  = re.compile(u"(\s|^)(DEL \w{2,})(\s|$)")
re_de   = re.compile(u"(\s|^)(DE \w{2,})(\s|$)")

def clean_names(rf, surnames_extracted, funky_prenames = list()):
    """ Uses extracted surnames as reference, to extract the prenames and clean them up.
    
    Args:
        rf: the registration dataframe
        surnames_extracted : surname dataframe created from "parse_fullrow"
        funky_prenames : list of currently-identified multi-token prenames (modified as the function runs)

    Returns:
        nf : the "names" dataframe, containing the extracted surnames and cleaned prenames
        funky_prenames : updated list of multi-token prenames

    """

    # set column order
    surnames_extracted = surnames_extracted[['cedula', 'sur_padre', 'has_padre', 'is_plegal',
                                             'sur_madre', 'has_madre', 'is_mlegal', 'prenames']]

    ## the "nf" (name frame) is just the subset of well-behaved names
    # confirm that the indexing is still correct
    if not (rf.index == surnames_extracted.index).all():
        rf.reset_index(inplace=True, drop=True)
    assert (rf.cedula == surnames_extracted.cedula).all()

    # join the parsed names to the originals (but only retain the well-behaved ones)
    nf = pd.concat([rf[['cedula','nombre','nombre_padre','nombre_madre','gender']], surnames_extracted.iloc[:,1:]], axis=1)

    nf = nf.loc[(nf.sur_padre.notnull()) & (nf.sur_padre != "") & nf.has_padre & 
                (nf.sur_madre.notnull()) & (nf.sur_madre != "") & nf.has_madre & 
                (nf.prenames.notnull() & (nf.prenames != "")),
            ['cedula','nombre','prenames', 'gender', 
             'nombre_padre','sur_padre','has_padre', 'is_plegal',
             'nombre_madre','sur_madre','has_madre', 'is_mlegal']]

    # "funky_prenames" gets modified as a side-effect
    funky_prenames = set(funky_prenames)   # used as set within this function, but length-sorted list externally
    nf['is_funky'] = nf.prenames.map(lambda x: regex_funky_prenames(x, funky_prenames))
    funky_prenames = list(funky_prenames)  # return to a list
    funky_prenames.sort(reverse=True, key=len)
    print("# funkies :", len(funky_prenames))
    nf.loc[nf.is_funky, 'prenames'] = nf.loc[nf.is_funky, 'prenames'].progress_map(lambda x: fix_funk(x, funky_prenames))

    # now that there are only a few hundred funkies (most are handled in data-cleaning), this is faster by 100x
    nf['nlen_padre'] = nf.nombre_padre.map(lambda x: len(x.split()))
    nf['nlen_madre'] = nf.nombre_madre.map(lambda x: len(x.split()))
    nf['n_char_nombre'] = nf.nombre.map(len)
    nf['n_char_prenames'] = nf.prenames.map(len)
    return nf, funky_prenames



def regex_funky_prenames(prenames, funky_prenames):
    """ Uses a series of regexes to identify prenames which are potentially unusual.
    
    Args:
        prenames : string containing the prenames (ie. nombre after the surnames have been removed)
        funky_prenames : set with currently-identified multi-token names.  More names can be added within this function

    Returns:
        is_funky : boolean indicating if the prename had a multi-token component
    NOTE: adds to "funky_prenames" as a side-effect

    This is a little slow (~4mins / million rows), but pretty thorough.
    """
    mdel   = re_del.search(prenames)
    msant  = re_sant.search(prenames)
    mlaos  = re_laos.search(prenames)
    mdela  = re_dela.search(prenames)
    mvon   = re_von.search(prenames)
    mvande = re_vande.search(prenames)
    mde    = re_de.search(prenames)
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
            funky_prenames.add(funk)
        is_funky = True
    else:
        is_funky = False
    return is_funky



def fix_funk(prenames, funky_prenames):
    """ Fixes multi-token names by replacing the spaces with underscores.
    
    The 'funky_prenames' list should be sorted in descending length, to prevent substrings from being clobbered.
    
    NB: there's a potential bug in here, bc the list is sorted according to character length, but checks
    here are being done according to number of tokens.  But very unlikely to cause an issue, so ignoring for now
    """
    nlen = len(prenames.split())
    if nlen <= 2:
        return prenames
    
    for funk in funky_prenames:
        flen = len(funk.split())
        if (nlen > flen):
            if (funk in prenames):
                defunk = '_'.join(funk.split())
                prenames = defunk.join(prenames.split(funk))
                nlen = len(prenames.split())
        else:
            # since the list is sorted, once we have a match that uses all the tokens, just skip ahead
            continue
    return prenames



def parse_prenames(nf):
    """ Parse the prenames column, to return dataframe with the fully-extracted names of each citizen.
    
    
    """
    def parse_prename_row(prenames):
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
    
    # apply the parsing function to each row; merge with surnames
    parsed = pd.concat([nf[['cedula', 'sur_padre', 'sur_madre']], 
                     nf.progress_apply(lambda row: parse_prename_row(row.prenames), axis=1, result_type='expand')], axis=1)

    # counts number of components in name (e.g. n_surs + n_pres)
    parsed['nlen'] = parsed[['sur_padre','sur_madre','pre1','pre2','pre3','junk']
                           ].replace("", np.nan).notnull().astype(int).sum(axis=1)
    return parsed



def make_allnames(parsed):
    """ Produce relative counts of all names (i.e. number of times appearing as surname vs prename)
    
    """
    def count_all_names(parsed):
        tmp = pd.concat([parsed.sur_padre, parsed.sur_madre], axis=0).value_counts()
        count_sur = pd.DataFrame({'obsname':tmp.index, 'n_sur':tmp.values})
        tmp = pd.concat([parsed.pre1, parsed.pre2], axis=0).value_counts()
        count_pre = pd.DataFrame({'obsname':tmp.index, 'n_pre':tmp.values})
        count_names = count_sur.merge(count_pre, on='obsname', how='outer')
        count_names.fillna(0, inplace=True)
        
        count_names.loc[count_names.obsname == "", ['n_sur','n_pre']] = 0 # make sure null names get weight factor of 1
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
        
    name_counts = count_all_names(parsed)
    name_counts['nlen'] = name_counts.obsname.map(lambda x: len(x.split()))
    name_counts['is_multimatch'] = name_counts.obsname.map(is_name_multimatch)
    return name_counts



def fix_mixed_presur_names(nf, name_counts):
    """ Fix cases where what appears to be multi-token surname is actually a surname + prename.
    
    """
    EVIDENCE_TO_FLIP = 1000  # how strongly we need to believe that we've mis-parsed the name; found by eyeballing, should be updated
    dual_sur = name_counts[(name_counts.nlen == 2) & ~name_counts.is_multimatch]
    dual_sur = dual_sur.apply(lambda x: x.obsname.split(), axis=1, result_type='expand')
    dual_sur.columns = ['probably_sur', 'probably_pre']
    dual_sur = dual_sur.merge(name_counts[['obsname', 'sratio']], left_on='probably_sur', right_on='obsname').drop(columns=['obsname'])
    dual_sur = dual_sur.merge(name_counts[['obsname', 'pratio']], left_on='probably_pre', right_on='obsname').drop(columns=['obsname'])
    dual_sur['evidence'] = dual_sur.sratio * dual_sur.pratio

    needs_repair = dual_sur[dual_sur.evidence > EVIDENCE_TO_FLIP]
    needs_repair = set(needs_repair.probably_sur + ' ' + needs_repair.probably_pre)

    print("Repairing {} mixed pre/sur records".format(len(needs_repair)))
    def repair_dual_surmadre(row):
        out = {'sur_madre': "", 'prenames': ""}
        sur_madre, pre1 = row.sur_madre.split()
        out['prenames'] = pre1 + ' ' + row.prenames
        out['sur_madre'] = sur_madre
        return out

    fix_rows = nf.sur_madre.isin(needs_repair)
    nf.loc[fix_rows, ['sur_madre', 'prenames']] = nf[fix_rows].progress_apply(
                                                    lambda row: repair_dual_surmadre(row), axis=1, result_type='expand')
    return nf



def fix_husband_honorific(nf, rf, funky_prenames):
    """ Repairs cases when mother's field includes husband's surname as an honorific.
    
    TODO: include fix for when the citizen is a woman using both her surname, and her
    husband's surname as an honorific (this shouldn't be recorded, but sometimes is)

    TODO: need to fix cases which show up as "sur_madre" == "DE".  These are almost all 
    when a woman and her mother both use husband's honorific, so algo picks it up as
    mother's surname
    """
    # first, identify likely cases where a mother is listed with the husband's surname as an honorific
    # 60 minutes  (this is a check of the mother's name, so have to run it for everyone; spouse would be only women)
    def poss_husb(row):
        # tried both simple search and regex; no difference in speed
        return " DE " + row.sur_padre in row.nombre_madre
    maybe_husb = nf.progress_apply(lambda row: poss_husb(row), axis=1)

    # second, remove the honorific from mother's name
    def remove_husband(row):
        out = row.copy(deep=True)
        try:
            madre = ''.join(row.nombre_madre.split(" DE " + row.sur_padre))
        except AttributeError:
            print("ERROR :", row)
        out.nombre_madre = madre
        return out
    sub = nf[maybe_husb].copy(deep=True)
    ceds_to_fix = set(sub.cedula)
    rf_removed = sub.apply(lambda row: remove_husband(row), axis=1, result_type='expand')

    # third, re-parse the names and update the original frames
    rf.loc[rf.cedula.isin(ceds_to_fix), 'nombre_madre'] = rf_removed.nombre_madre
    surnames_fixed = rf_removed.apply(lambda row: parse_fullrow(row), axis=1, result_type='expand')
    nf_fixed, funky_prenames = clean_names(rf_removed, surnames_fixed)

    ceds_were_fixed = set(nf_fixed[nf_fixed.nombre.notnull()].cedula)
    cols_fixed = ['nombre', 'prenames', 'nombre_madre', 'sur_madre', 'has_madre', 'is_mlegal', 'nlen_madre', 'n_char_nombre', 'n_char_prenames']
    for col in cols_fixed:
        nf.loc[nf.cedula.isin(ceds_were_fixed), col] = nf_fixed.loc[:, col]
    return nf, rf



def merge_underscore_names(ncounts):
    """ Find any names which are identical (other than underscores) and correctly merge their counts.

    """

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
