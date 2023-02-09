import matplotlib.pyplot as plt
import networkit as nk
import sys
import pandas as pd
import numpy as np

# Paths
dir_name = '/home/juan.russy/shared/proof_run_FamNet/'
base_filename = 'interim/PROOF_GRAPH_matched_exact_name.txt'  # EdgeList File
filename = dir_name + base_filename
print(filename)

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


#%% Largest Component 10 M
print('Largest Connected Component')
# compactGraph = False do not reset node ids
cc = nk.components.ConnectedComponents.extractLargestConnectedComponent(G, compactGraph=False)
print('number of nodes:', cc.numberOfNodes())

diam = nk.distance.Diameter(cc, algo = 1)
diam.run()
print('diameter', diam.getDiameter())