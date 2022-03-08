import extract
import re
import numpy as np

def parse_pres(pres, pname, funky_prenames, row=None):

    if len(pres.split()) == 1:
        pname['pre1'] = pres
    elif len(pres.split()) == 2:
        pname['pre1'], pname['pre2'] = pres.split()
    else:
        pres = extract.fix_funk(pres, funky_prenames).split()
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


re_beg_von = re.compile(u"^(V[AO]N \w{2,})(\s|$)")
re_beg_vande = re.compile(u"^(V[AO]N DE[RN]? \w{2,})(\s|$)")
# SANTA and SAN (in lieu of SANTO)
re_beg_sant = re.compile(u"^(SANT?A? \w{2,})(\s|$)")
# these results are subset of "re_beg_laos"
re_beg_dela = re.compile(u"^(DE L[AO]S? ?\w{2,})(\s|$)")
re_beg_laos = re.compile(u"^(L[AEO]S? \w{2,})(\s|$)")
re_beg_del = re.compile(u"^(DEL \w{2,})(\s|$)")
re_beg_de = re.compile(r"^(DE \w{2,})(\s|$)")

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

def wts(allnames):
    wts_pre = dict()
    wts_sur = dict()

    for _, row in allnames.iterrows():
        wts_pre[row.obsname] = row.pratio
        wts_sur[row.obsname] = row.sratio
    
    return wts_pre, wts_sur


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

def best_split_by_evidence(my_pres, row, wts_pre, wts_sur, verbose=False):
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


def extract_prename_parent(row, target_col, wts_pre, wts_sur, funky_prenames):
    """ 
    Extract information about the parents
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
        pname = parse_pres(pres, pname, funky_prenames, row)

    elif not is_pstart:
        # name is in normal form, sur2 follows sur1, everything before sur1 is a prename
        parts = [x.strip() for x in row[target_col].split(sur1, maxsplit=1)]

        if len(parts) > 1:
            pname['sur2'] = parts[1]
            pres = row[target_col].split(sur1)[0]
            pname = parse_pres(pres, pname, funky_prenames, row)
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
            pname = parse_pres(pres, pname, funky_prenames, row)

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

            best_split, flag_split = best_split_by_evidence(my_pres, row, wts_pre, wts_sur,)
            pname['flag'] = flag_split

            if best_split:
                pres = my_pres[best_split:]
                pname = assign_pres(pres, pname)
                surs = my_pres[:best_split]
                if surs:
                    pname['sur2'] = surs[0]

    return pname
