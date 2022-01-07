_nan_values = ['-1.#IND', '1.#QNAN', '1.#IND', '-1.#QNAN', '#N/A N/A', '#N/A', 'N/A', 'n/a',
               '<NA>', '#NA', 'NULL', 'null', 'NaN', '-NaN', 'nan', '-nan', ]

_date_cols = ['dt_birth', 'dt_death', 'dt_marriage']

_dtypes_reg = {'cedula': str, 'nombre': str, 'gender': 'category',
              'marital_status': 'category', 'place_birth': str,
              'nombre_spouse': str, 'nombre_padre': str, 'nombre_madre': str,
              'ced_spouse': str, 'ced_padre': str, 'ced_madre': str,
              'is_nat': bool, 'is_nat_padre': bool, 'is_nat_madre': bool
              }

_dtypes_surname = {  'cedula': str, 'sur_padre': str, 'sur_madre': str, 'prenames': str,
                    'has_padre': bool, 'is_plegal': bool, 'has_madre': bool, 'is_mlegal': bool,
                    }

_dtypes_cleaned = {  'cedula': str, 'nombre': str, 'prenames': str, 'gender': str,
                    'nombre_padre': str, 'sur_padre': str, 'has_padre': bool, 'is_plegal': bool,
                    'nombre_madre': str, 'sur_madre': str, 'has_madre': bool, 'is_mlegal': bool,
                    'is_funky': bool, 'nlen_padre': int, 'nlen_madre': int, 'n_char_nombre': int,
                    'n_char_prename': int, 'maybe_husb': bool
                    }

_dtypes_allnames = { 'obsname': str, 'n_sur': float, 'n_pre': float, 'sratio': float, 'pratio': float
                    }

_dtypes_namecounts = {'obsname': str, 'n_sur': float, 'n_pre': float, 'sratio': float,
                    'pratio': float, 'nlen': float, 'is_multimatch': bool
                    }

_dtypes_newfreqfile = {  'cedula': str, 'nombre': str, 'sur_padre': str, 'sur_madre': str,
                        'pre1': str, 'pre2': str, 'pre3': str, 'junk': str, 'nlen': int,
                        }

_dtypes_padres = {'cedula': str, 'sur1': str, 'sur2': str, 'pre1': str,
             'pre2': str, 'pre3': str, 'junk': str, 'flag': bool
             }

# Google's style guide recommend to call globals through public module-level functions
def get_nan_values():
    return _nan_values

def get_date_cols():
    return _date_cols

def get_dtypes_reg():
    return _dtypes_reg

def get_dtypes_surname():
    return _dtypes_surname

def get_dtypes_cleaned():
    return _dtypes_cleaned

def get_dtypes_allnames():
    return _dtypes_allnames

def get_dtypes_namecounts():
    return _dtypes_namecounts

def get_dtypes_newfreqfile():
    return _dtypes_newfreqfile

def get_dtypes_padres():
    return _dtypes_padres
