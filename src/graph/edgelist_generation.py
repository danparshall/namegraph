import pandas as pd

# Path in the cluster
dir_name = ''

#%% Match by cedula padres
print('match by cedula padres')
matched_padres_by_cedula = pd.read_csv(dir_name + '09-match_by_cedula_padres.tsv', 
                                       sep='\t')
matched_padres_by_cedula.rename(columns={'ced_padre': 'cedula_matched'},
                                inplace=True)
# Structure of the dataframe
print(matched_padres_by_cedula.columns)
print(matched_padres_by_cedula.shape)
print(matched_padres_by_cedula.head(10))

#%% Match by Cedula madres
print('match by cedula madres')
matched_madres_by_cedula = pd.read_csv(dir_name + '10-match_by_cedula_madres.tsv', 
                                       sep = '\t')
matched_madres_by_cedula.rename(columns={'ced_madre': 'cedula_matched'},
                                inplace=True)
# Structure of the dataframe
print(matched_madres_by_cedula.columns)
print(matched_madres_by_cedula.shape)
print(matched_madres_by_cedula.head(10))

matched_1 = pd.concat([matched_padres_by_cedula, matched_madres_by_cedula], 
                      ignore_index=True, sort=True)
del matched_padres_by_cedula, matched_madres_by_cedula

#%% Match Exact Name Padres
print('match exact name padres')
matched_padres_exact_name = pd.read_csv(dir_name + '11-matched_padres.tsv', 
                                        sep='\t')
matched_padres_exact_name.rename(columns={'padre_matched': 'cedula_matched'},
                                 inplace=True)

# Structure of the padre dataframe
print(matched_padres_exact_name.columns)
print(matched_padres_exact_name.shape)
print(matched_padres_exact_name.head(10))

matched_1 = pd.concat([matched_1, matched_padres_exact_name], ignore_index=True,
                      sort=True)
del matched_padres_exact_name

#%% Match Exact Name Madres
print('match exact name madres')
matched_madres_exact_name = pd.read_csv(dir_name + '12-matched_madres.tsv', 
                                        sep='\t')
matched_madres_exact_name.rename(columns={'madre_matched': 'cedula_matched'},
                                 inplace=True)

# Structure of the madre dataframe
print(matched_madres_exact_name.columns)
print(matched_madres_exact_name.shape)
print(matched_madres_exact_name.head(10))

matched_1 = pd.concat([matched_1, matched_madres_exact_name], ignore_index=True,
                        sort = True)
del matched_madres_exact_name

matched_1['weight'] = 1

#%% Match by Cedula Spouse
print('match by cedula spouse')
match_spouse_by_cedula = pd.read_csv(dir_name + '15-match_by_cedula_spouse.tsv',
                                     sep = '\t')
match_spouse_by_cedula.rename(columns={'nombre_spouse': 'cedula_matched'}, 
                              inplace=True) # The original dataset has nombre and cedula of the spouse columns backwards
# Structure of the spouse dataframe
print(match_spouse_by_cedula.columns)
print(match_spouse_by_cedula.shape)
print(match_spouse_by_cedula.head(10))

match_spouse_by_cedula['weight'] = 0

#%% Concat all dataframes weight 1
print('MATCH')
matched = pd.concat([matched_1, match_spouse_by_cedula], ignore_index=True,
                    sort=True)

# Structure of the concat dataframe
print(matched.columns)
print(matched.shape)
print(matched.head(10))

matched.to_csv(dir_name + 'Matched_Graph.txt',
                header = False, index = None, sep = '\t')
