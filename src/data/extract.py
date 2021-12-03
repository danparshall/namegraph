import pandas as pd
import re


# enable progress bar on long operations
from tqdm.auto import tqdm
tqdm.pandas()




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



re_de_pre_UNDERSCORE = re.compile(r"(^.*)\s(DEL?_\w+.*$)")
def parse_fullrow(row):
    # this function expects multi-token names to be handled already with underscores

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
        # this matches longest chunk found, so it should pick up compound names like DE LA CUEVA
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
                        m_de_pre_madre = re_de_pre_UNDERSCORE.match(
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



