import matplotlib.pyplot as plt
import networkit as nk
import sys
import pandas as pd
import numpy as np
sys.path.append("../data")
import utils

# Paths
dir_name = '/home/juan.russy/shared/proof_run_FamNet/'
raw_path = '/home/juan.russy/shared/FamilyNetwork/'
filepath_raw = 'RegCleaned.tsv'
base_filename = 'output/Matched_Graph.txt'
filename = dir_name + base_filename
print(filename)

# Functions
def load_registry(filepath_raw, N_ROWS=None):
    """ Load original dataset with family information

    Args:
        filepath_raw    : path with original dataframe file
        NROWS           : Argument for number of rows in read_csv

    Returns:
        rf              : original dataframe with each columns type configured
    """
    rf = pd.read_csv(filepath_raw, sep='\t', 
                     encoding='utf-8',
                     parse_dates = ['dt_birth', 'dt_death'],  
                     dtype=utils.get_dtypes_reg(), 
                     # Interest in dt_birth and dt_death
                     usecols=['cedula', 'dt_birth', 'dt_death'],
                     na_values=['', ],  # identifies '' as NaN
                     keep_default_na=False,
                     nrows=N_ROWS
                     ) 
    return rf


#%% EdgeListReader to create 'graphio' object
print('Read EdgeList')
reader = nk.graphio.EdgeListReader(  
    separator='\t', firstNode=0, continuous=False, directed=False)
G = reader.read(filename)  # 'graph' object
# Networkit's graphs have id from (0, n-1)
# Dictionary containing mapping from Networkit ID to cedula
print('Creacion Dataframe')
map_nkID_ced = reader.getNodeMap()
# print('len map', len(map_nkID_ced.keys()))
map_df = pd.DataFrame.from_dict(map_nkID_ced, orient='index')
map_df = map_df.reset_index(level=0)
map_df.columns = ['id', 'nk_id']


#%% Birth by year - Largest Component 10 M
print('Largest Connected Component')
# compactGraph = False do not reset node ids
cc = nk.components.ConnectedComponents.extractLargestConnectedComponent(G, compactGraph=False)
print('number of nodes:', cc.numberOfNodes())

# Query of the nodes
print('Query Filtro')
list_nodes = [x for x in G.iterNodes()]
map_df.query('nk_id in @list_nodes')
ced_LCC = map_df['id'].tolist()
print(len(ced_LCC))

# Load Registry
print('Load Registry')
rf = load_registry(raw_path + filepath_raw)
rf = rf.query('cedula in @ced_LCC')
print(rf.head())
print(rf.shape)

# Filter people born before 1880
print('Data Sample')
data_sample = rf[rf['dt_birth'] > np.datetime64('1880-01-01 00:00:00')]
data_sample.loc[:, 'year'] = pd.DatetimeIndex(data_sample['dt_birth']).year 
print(data_sample.head())

# # Generate dataset
print('People')
people = data_sample.set_index('year').groupby(level=0).count()
print(people.head())
# Plots
plt.figure(0)
plt.scatter(people.index.values, people.dt_birth)
plt.title("Births by year - Largest Component 10M")
plt.savefig(dir_name + 'plots/births_by_year_largest_component_10M', dpi=200)


#%% Birth by year - Nodes Degree 0 (Do not appear on the graph)
print("Nodes Degree 0")
ced_LCC = map_df['id'].tolist() 
print('len cedulas', len(ced_LCC))

print('Load Registry')
rf = load_registry(raw_path + filepath_raw)
rf = rf.query('cedula not in @ced_LCC')
print(rf.head())
print(rf.shape)

# Filter people born before 1880
print('Data Sample')
data_sample = rf[rf['dt_birth'] > np.datetime64('1880-01-01 00:00:00')]
data_sample.loc[:, 'year'] = pd.DatetimeIndex(data_sample['dt_birth']).year
print(data_sample.head())

# # Generate dataset
print('People')
people = data_sample.set_index('year').groupby(level=0).count()
print(people.head())
# Plots
plt.figure(0)
plt.scatter(people.index.values, people.dt_birth)
plt.title("Births by year - Nodes Degree 0")
plt.savefig(dir_name + 'plots/births_by_year_nodes_degree_0', dpi=200)
