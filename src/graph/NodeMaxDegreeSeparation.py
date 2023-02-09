import time
import networkit as nk
import pandas as pd
import numpy as np
from tqdm import tqdm
import itertools

NROWS = None

# Paths
dir_name = '/home/juan.russy/shared/proof_run_FamNet/output/'
base_filename = 'Matched_Graph.txt'
gml_filename = 'example.gml'
mapDf_filename = 'Matched_Graph_Map.tsv'
filepath_raw = '/home/juan.russy/shared/FamilyNetwork/RegCleaned.tsv'
merged_filename = 'merged_consanguinity.tsv'
# Read EdgeList
print('Read EdgeList')
start_time = time.time()
reader = nk.graphio.EdgeListReader(  # 'graphio' object
    separator='\t', firstNode=0, continuous=False, directed=False)
G = reader.read(dir_name + base_filename)  # 'graph' object
print("EdgeList %s seconds ---" % round(time.time() - start_time, 2))

# Functions
def load_registry(filepath_raw, NROWS=None):
    """ Load original dataset with cedula and nombre information

    Args:
        filepath_raw : path with original dataframe file
        NROWS : Argument for number of rows in read_csv

    Returns:
        rf : dataframe with cedula and nombre columns
    """
    reg_cleaned = pd.read_csv(filepath_raw,
                              sep='\t',
                              encoding='latin-1',
                              usecols=['cedula', 'nombre'],
                              keep_default_na=False,
                              nrows=NROWS
                              )
    return reg_cleaned


def path_arrange(path):
    """ Structure the path to the form R-<inner_nodes>-S
    """

    if len(path) == 1:
        return path
    else:
        return path[: path.rfind('-')] + '-S'


def EdgeSeparation(G, node, MaxDegree):
    """ Nodes with less than MaxDegree of separation from node 
    
    BFS is used to do the search, the function manages three queue's with the
    information of interest.

    Args:
        G: Weighted or Unweighted networkit's graph
        node: Integer with node ID
        MaxDegree: Maximum degree of separation from node allowed
    Returns:
        info_nodes_df: Dataframe with the name, degree of separation and paths
        of the nodes that follow the MaxDegree rule
    """
    path = 'R'  # Initial path
    degree_separation = 0  # Initial degree of separation

    visited = set()  # Visited nodes set
    # Queue's
    queue_name, queue_degree, queue_path = ([] for _ in range(3))

    # Initial node
    queue_name.append(node)
    queue_degree.append(degree_separation)
    queue_path.append(path)
    visited.add(node)

    # List with all the nodes information (they are not deleted)
    all_names, all_degrees, all_paths = ([] for _ in range(3))
    # BFS
    while queue_name:
        # Obtain information of the last queue node
        node = queue_name.pop(0)
        degree = queue_degree.pop(0)
        temp_path = queue_path.pop(0)

        # Append that information
        all_names.append(node)
        all_degrees.append(degree)
        all_paths.append(temp_path)

        # If degree is less than MAX DEGREE
        # Add neighbors to the queue
        if degree < MaxDegree:

            # Iterate over node neighbors 
            for neighbor in G.iterNeighbors(node):
                if neighbor not in visited:
                    queue_name.append(neighbor)
                    queue_degree.append(degree + 1)
                    path = temp_path + '-' + str(neighbor)
                    queue_path.append(path)
                    visited.add(neighbor)
    
    all_paths = list(map(lambda path: path_arrange(path), all_paths))
    # Generate dataframe from the ALL lists
    info_nodes_df = pd.DataFrame(
        {'node_name': all_names, 'degree_separation': all_degrees, 'path': all_paths})

    return info_nodes_df


def ConsanguinitySeparation(graph, NODE, MaxDegree):
    """ Obtain all nodes from a certain consanguinity degree of separation

    Consanguinity follows specific rules: between the couple and the person 
    there is zero degree of separation and with parents and children there 
    is one degree of separation
    
    BFS is used to do the search and we use the weights of the edges to 
    measure the consangunity degree of separation.

    Args:
        graph: Weighted networkit's graph
        NODE: Integer with node ID
        MaxDegree: Maximum consanguinity degree of separation from node allowed
    Returns:
        info_nodes_df: Dataframe with the name, consanguinity degree of separation 
        and paths of the nodes that follow the MaxDegree rule
    """
    path = 'R'  # Initial path
    degree_separation = 0  # Initial degree of separation

    visited = set()  # Visited nodes set
    # Queue's
    queue_name, queue_degree, queue_path = ([] for _ in range(3))

    # Initial node
    queue_name.append(NODE)
    queue_degree.append(degree_separation)
    queue_path.append(path)
    visited.add(NODE)

    # List with all the nodes information (they are not deleted)
    all_names, all_degrees, all_paths = ([] for _ in range(3))

    # BFS
    while queue_name:
        # Obtain information of the last queue node
        node = queue_name.pop(0)
        degree = queue_degree.pop(0)
        temp_path = queue_path.pop(0)

        # Append that information
        all_names.append(node)
        all_degrees.append(degree)
        all_paths.append(temp_path)

        # If degree is less than MAX DEGREE
        # Add ALL neighbors to the queue
        if degree <= MaxDegree:
            WeightCondition = True
            
            # Iterate over node neighbors
            for neighbor, weight in graph.iterNeighborsWeights(node):
                # If degree is equal to MAX DEGREE
                # Add neighbors with weight 0.0 only
                if degree == MaxDegree:
                    WeightCondition = weight == 0.0

                if neighbor not in visited and WeightCondition:
                    queue_name.append(neighbor)
                    queue_degree.append(degree + weight)
                    path = temp_path + '-' + str(neighbor)
                    queue_path.append(path)
                    visited.add(neighbor)

    all_paths = list(map(lambda path: path_arrange(path), all_paths))
    # Generate dataframe from the ALL lists
    info_nodes_df = pd.DataFrame(
        {'node_name': all_names, 'degree_separation': all_degrees, 'path': all_paths})
    info_nodes_df['reference'] = NODE
    return info_nodes_df


def Map_IDnk_cedula(reader):
    """ Obtain the mapping between networkit ID and Cedula

    Networkit automatically changes node ID's for integers between
    1 and N-1. This function gives you a dataframe with this
    information.

    Args:
        reader: 'graph' object from networkit
    Returns:
        map_df: Mapping dataframe  
    """
    map_nkID_ced = reader.getNodeMap()
    # print('len map', len(map_nkID_ced.keys()))
    map_df = pd.DataFrame.from_dict(map_nkID_ced, orient='index')
    map_df = map_df.reset_index(level=0)
    map_df.columns = ['cedula', 'node_name']
    return map_df


def find_cedula_in_dict(dct, cedula):
    if cedula in dct.keys():
        return dct[cedula]
    else:
        return None


def ConsanguinitySeparationMap(reader, graph, cedula, MaxDegree, filepath_raw = filepath_raw):
    start_time = time.time()
    # Generate map
    map_dct = reader.getNodeMap()
    map_df = Map_IDnk_cedula(reader)
    print("Map %s seconds ---" % round(time.time() - start_time, 2))

    start_time = time.time()
    consanguinity = {}
    for ced in tqdm(cedula):
        # Filter cedula of interest
        node = find_cedula_in_dict(map_dct, ced)
        # print('node', node)
        if node == None:
            continue
        # Found nodes base on MaxDegree consanguinity separation
        consanguinity_temp = ConsanguinitySeparation(graph, node, MaxDegree)
        consanguinity[ced] = consanguinity_temp
    consanguinity = pd.concat(consanguinity, ignore_index=True)
    print("Loop consanguinity %s seconds ---" % round(time.time() - start_time, 2))

    # Load cedula and names columns of original dataset
    start_time = time.time()
    rf = load_registry(filepath_raw, NROWS)
    print("Load registry %s seconds ---" % round(time.time() - start_time, 2))
    
    # Merge with the map to obtain cedula number
    start_time = time.time()
    output = consanguinity.merge(map_df, how='left')
    map_df.columns = ['cedula_reference', 'reference']
    output = output.merge(map_df, how='left', on = 'reference')
    output.drop(['node_name', 'reference'], axis = 1)
    print("Merge Map %s seconds ---" % round(time.time() - start_time, 2))

    # Merge with rf to obtain name
    start_time = time.time()
    output = output.merge(rf, how='left')
    print("Merge rf %s seconds ---" % round(time.time() - start_time, 2))
    output.to_csv(dir_name + merged_filename, sep='\t', index = False)


print('Consanguinity')
filepath_firms = '/home/juan.russy/shared/FamilyNetwork/sample_firm_size4.tsv'
df_firms = pd.read_csv(filepath_firms, sep='\t', encoding='utf-8', 
                       usecols=['employerIDh'], keep_default_na=False, nrows=NROWS)
ConsanguinitySeparationMap(reader, G, df_firms['employerIDh'].tolist(), 7)  # '000001a435117d61'

# start_time = time.time()
# consanguinity = ConsanguinitySeparation(G, 9653085, 8)
# print("--- %s seconds ---" % round(time.time() - start_time, 2))
# print(EdgeSeparation(G, 2, 9))