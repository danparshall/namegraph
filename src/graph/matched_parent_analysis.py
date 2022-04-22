import pandas as pd

# Path in the cluster
dir_name = ''

# Concat exact name matching for padre/madre
matched_padres = pd.read_csv(dir_name + '09-matched_padres.tsv', sep = '\t')
matched_padres = matched_padres[['cedula', 'padre_matched']]
matched_padres.rename(columns={'padre_matched': 'cedula_matched'},
                  inplace=True)

matched_madres = pd.read_csv(dir_name + '10-matched_madres.tsv', sep = '\t')
matched_madres = matched_madres[['cedula', 'madre_matched']]
matched_madres.rename(columns={'madre_matched': 'cedula_matched'},
                  inplace=True)

matched_exact_name = pd.concat([matched_padres, matched_madres])

# Structure of the concat dataframe
print(matched_exact_name.columns)
print(matched_exact_name.shape)
print(matched_exact_name.head(10))

matched_exact_name.to_csv(dir_name + 'PROOF_GRAPH_matched_exact_name.txt',
                          header=True, index=None, sep=' ', mode='a')
