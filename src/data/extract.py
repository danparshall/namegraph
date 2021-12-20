import pandas as pd
import re


# enable progress bar on long operations
from tqdm.auto import tqdm
tqdm.pandas()


"""
NOTES:


TODO:
Handle "FALLECIDA EN ...." and other things when part of the "nombre" field.



"""



def check_nombre_doubling(nombre):
    """ Checks name for possible repeats.  Extracts surname if doubled (which would be for both parents).

    It's entirely possible for someone to be named "GARCIA GARCIA JOSE JULIO", or "DE LA CUEVA DE LA CUEVA JOSE JULIO".
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



def parse_padre(row, parts, nomset, pset):
    # start by trying the first LEN-1 tokens as a single name, then LEN-2 tokens, etc
    # this matches longest chunk found, so it should pick up compound names like DE LA CUEVA
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
                    # update before checking mother's name
                    padre = guess
                    parts = row.nombre.split(padre, maxsplit=1)[1].split()
                    break
    else:
        padre = ""
    return padre, parts



def parse_madre(row, parts, nomset, mset):
    if row.nombre_madre:
        poss_madre = check_nombre_doubling(row.nombre_madre)
        poss_mset = set(poss_madre.split())
        if (poss_madre
            and (poss_madre in row.nombre)
            and not row.nombre.endswith(poss_madre)
                and poss_mset.issubset(nomset) and poss_mset.issubset(mset)):
            madre = poss_madre
        elif not madre:
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
                    parts = parts[:1] + \
                        m_de_pre_nombre.group(1).split()

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
        else:
            madre = ""
    return madre



def parse_overlaps(row, nomset, pset, mset):
    surname = check_nombre_doubling(row.nombre)
    if surname != "":
        madre = surname
        padre = surname
    else:
        padre = ""
        madre = ""
        for guess in get_substrings(row.nombre_madre.split()):
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
    return padre, madre



re_de_pre = re.compile(r"(^.*)\s(DEL?\s\w+.*$)")
re_de_pre_UNDERSCORE = re.compile(r"(^.*)\s(DEL?_\w+.*$)")
def parse_fullrow(row):
    # this function expects multi-token names to be handled already with underscores

    out = {"cedula": row.cedula, 
            "sur_padre": "", "sur_madre": "", "prenames": "",
            "has_padre": False, "is_plegal": False,
            "has_madre": False, "is_mlegal": False}

    if not row.nombre_padre and not row.nombre_madre:
        return out

    # check if madre/padre have overlapping tokens (requires special handling)
    nomset = set(row.nombre.split())
    mset = set(row.nombre_madre.split())
    pset = set(row.nombre_padre.split())
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
            padre, parts = parse_padre(row, parts, nomset, pset)
        except:
            out['sur_padre'] = "WTF PADRE PROBLEM"
            return out

        #### MOTHERS NAME ####
        # having removed the padre name from the front of the string, try similar trick with the madre name
        try:
            madre = parse_madre(row, parts, nomset, mset)
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


def clean_names(rf, surnames_extracted, funky_prenames = set()):
    """ Uses extracted surnames as reference, to extract the prenames and clean them up
    
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
    nf['is_funky'] = nf.prenames.map(lambda x: regex_funky_prenames(x, funky_prenames))
    funky_prenames = list(funky_prenames)
    funky_prenames.sort(reverse=True, key=len)
    print("# funkies :", len(funky_prenames))
    nf.loc[nf.is_funky, 'prenames'] = nf[nf.is_funky].prenames.progress_map(lambda x: fix_funk(x, funky_prenames))

    # now that there are only a few hundred funkies (most are handled in data-cleaning), this is faster by 100x
    nf['nlen_padre'] = nf.nombre_padre.map(lambda x: len(x.split()))
    nf['nlen_madre'] = nf.nombre_madre.map(lambda x: len(x.split()))
    nf['n_char_nombre'] = nf.nombre.map(len)
    nf['n_char_prenames'] = nf.prenames.map(len)
    return nf, funky_prenames



def regex_funky_prenames(nombre, funky_prenames):
    """ This is a little slow (~4mins / million rows), but pretty thorough.
        NB: adds to "funky_prenames" as a side-effect
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



def fix_mixed_presur_names(nf, name_counts):
    dual_sur = name_counts[(name_counts.nlen == 2) & ~name_counts.is_multimatch]
    dual_sur = dual_sur.apply(lambda x: x.obsname.split(), axis=1, result_type='expand')
    dual_sur.columns = ['probably_sur', 'probably_pre']
    dual_sur = dual_sur.merge(name_counts[['obsname', 'sratio']], left_on='probably_sur', right_on='obsname').drop(columns=['obsname'])
    dual_sur = dual_sur.merge(name_counts[['obsname', 'pratio']], left_on='probably_pre', right_on='obsname').drop(columns=['obsname'])
    dual_sur['evidence'] = dual_sur.sratio * dual_sur.pratio

    needs_repair = dual_sur[dual_sur.evidence > 1000]
    needs_repair = set(needs_repair.probably_sur + ' ' + needs_repair.probably_pre)

    print("Repairing {} mixed pre/sur records".format(len(needs_repair)))
    def repair_dual_surmadre(row):
        out = {'sur_madre': "", 'prenames': ""}
        sur_madre, pre1 = row.sur_madre.split()

        out['prenames'] = pre1 + ' ' + row.prenames
        out['sur_madre'] = sur_madre
        return out

    fix_rows = nf.sur_madre.isin(needs_repair)
    nf.loc[fix_rows, ['sur_madre', 'prenames']
          ] = nf[fix_rows].progress_apply(lambda row: repair_dual_surmadre(row), axis=1, result_type='expand')
    return nf



def fix_husband_addition(nf, rf, funky_prenames):

    # first, identify likely cases where a mother is listed with the husband's surname as an honorific
    # 60 minutes  (this is a check of the mother's name, so have to run it for everyone; spouse would be only women)
    # NOTE: need to fix cases which show up as "sur_madre" == "DE".
    # these are almost all when a woman and her mother both use husband's honorific, so algo picks it up as mother's surname
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
    ceds_fix = set(sub.cedula)
    removed = sub.apply(lambda row: remove_husband(row), axis=1, result_type='expand')

    # third, re-parse the names and update the original frames
    rf.loc[rf.cedula.isin(ceds_fix), 'nombre_madre'] = removed.nombre_madre
    ceds_were_fixed = set(nf_fixed[nf_fixed.nombre.notnull()].cedula)
    cols_fixed = ['nombre', 'prenames', 'nombre_madre', 'sur_madre', 'has_madre', 'is_mlegal', 'nlen_madre', 'n_char_nombre', 'n_char_prenames']
    for col in cols_fixed:
        nf.loc[nf.cedula.isin(ceds_were_fixed), col] = nf_fixed.loc[:, col]
    return nf, rf



def merge_underscore_names(ncounts):
    """
    Find any names which are identical (other than underscores) and correctly merge their counts.
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