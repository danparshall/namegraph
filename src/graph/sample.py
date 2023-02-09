import pandas as pd

NROWS = None

_date_cols = ['dt_birth', 'dt_death', 'dt_wedding']
_dtypes_reg = {'cedula': str, 'nombre': str, 'gender': 'category',
               'marital_status': 'category', 'place_birth': str,
               'nombre_spouse': str, 'nombre_padre': str, 'nombre_madre': str,
               'ced_spouse': str, 'ced_padre': str, 'ced_madre': str,
               'is_nat': bool, 'is_nat_padre': bool, 'is_nat_madre': bool}
_nan_values = ['-1.#IND', '1.#QNAN', '1.#IND', '-1.#QNAN', '#N/A N/A', '#N/A', 'N/A', 'n/a',
               '<NA>', '#NA', 'NULL', 'null', 'NaN', '-NaN', 'nan', '-nan', ]

reg_cleaned = pd.read_csv("/home/juan.russy/shared/FamilyNetwork/RegCleaned.tsv", 
                          sep = '\t', 
                          encoding='latin-1',
                         #  parse_dates=_date_cols,
                         #  dtype=_dtypes_reg,
                          usecols = ['cedula', 'nombre_spouse'],
                          keep_default_na=False, 
                         #  na_values=_nan_values,
                          nrows=NROWS
                         )

reg_cleaned = reg_cleaned[reg_cleaned['nombre_spouse'] != '']
print(reg_cleaned.shape)
print(reg_cleaned.head(40))
reg_cleaned.to_csv(
    "/home/juan.russy/shared/proof_run_FamNet/output/15-match_by_cedula_spouse.tsv", sep='\t', index=False)

# reg_cleaned = reg_cleaned.sample(n = 100000, random_state = 1)
# reg_cleaned.to_csv(
#     "/home/juan.russy/shared/proof_run_FamNet/sample_100k_regCleaned.tsv", sep='\t', index=False)
