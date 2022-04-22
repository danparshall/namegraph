import pandas as pd
df = pd.read_csv(
    '/home/juan.russy/shared/FamilyNetwork/original/RC_v2021.tsv', sep='\t', usecols=['cedula', 'ced_madre', 'ced_padre'])  

df_padre = df.dropna(subset = ['ced_padre'])
df_madre = df.dropna(subset = ['ced_madre'])

## PADRES
# 11'783.702 records with cedula
print(df_padre.shape)
print("-"*50)
#  4'672.179 distinct records
print(df_padre.ced_padre.value_counts())
print("-"*50)

lista_padres = df['ced_padre'].drop_duplicates()
# In lista 2'278.266
df_padre_in = df_padre[df_padre['cedula'].isin(lista_padres)]
print("IN", "-"*50, '\n', df_padre_in.shape)
# Not In Lista 9'505.436
df_padre_not = df_padre[~df_padre['cedula'].isin(lista_padres)]
print("NOT", "-"*50, '\n', df_padre_not.shape)
# Why In Lista and Not In Lista sum 11'783.702 and not 4'672.179
# when they are suppose to be distinct records in cedula


## MADRES
# 12’378.745 records with cedula
print(df_madre.shape)
print("-"*50)
#  5’153.392 distinct records
print(df_madre.ced_madre.value_counts())
print("-"*50)

lista_madres = df['ced_madre'].drop_duplicates()
df_madre_in = df_madre[df_madre['cedula'].isin(lista_madres)]
print("IN", "-"*50, '\n', df_madre_in.shape)
df_madre_not = df_madre[~df_madre['cedula'].isin(lista_madres)]
print("NOT", "-"*50, '\n', df_madre_not.shape)
# Why In Lista and Not In Lista sum 12’378.745 and not 5’153.392
# when they are suppose to be distinct records in cedula
