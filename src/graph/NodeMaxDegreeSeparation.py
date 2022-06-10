import networkit as nk
import pandas as pd
import numpy as np

# Paths
dir_name = ''
base_filename = ''
filename = dir_name + base_filename
print(filename)

# Read EdgeList
print('Read EdgeList')
reader = nk.graphio.EdgeListReader(  # 'graphio' object
    separator='\t', firstNode=0, continuous=False, directed=False)
G = reader.read(filename)  # 'graph' object

print('Node separation')
def NodeMaxDegreeSeparation(NODE, MAX_DEGREE):
    """ Nodes with less than MAX_DEGREE of separation from NODE 
    
    BFS is used to do the search, the innovation is that the function manages
    three queue's with the information of interest.

    Args:
        NODE: Integer with node ID.
        MAX_DEGREE: Maximum degree of separation from NODE allowed
    Returns:
        info_nodes_df: Dataframe with the name, degree of separation and paths
        of the nodes that follow the MAX_DEGREE rule
    """
    visited = set()  # Nodes that have been visited
    path = 'R-{}-S'.format(NODE)  # Initial path
    degree_separation = 0  # Initial degree of separation

    # Queue's
    queue_name = []
    queue_degree = []
    queue_path = []
    
    # Initial NODE
    queue_name.append(NODE)
    queue_degree.append(degree_separation)
    queue_path.append(path)
    visited.add(NODE)

    # List with all the nodes information (they are not deleted)
    all_names = []
    all_degrees = []
    all_paths = []

    # BFS
    while queue_name:
        # Obtain information of the last queue node
        node = queue_name.pop(0)
        degree = queue_degree.pop(0)
        temp_path = queue_path.pop(0)

        print('-'*50)
        print('node', node)
        print('degree', degree)
        print('Path', temp_path)

        # Append that information
        all_names.append(node)
        all_degrees.append(degree)
        all_paths.append(temp_path)

        # If degree is less than MAX DEGREE
        # Add neighbors to the queue
        if degree < MAX_DEGREE:

            # Iterate over node neighbors 
            # (In the future weights also are going to be included)
            for neighbor in G.iterNeighbors(node):
                if neighbor not in visited:
                    queue_name.append(neighbor)
                    queue_degree.append(degree + 1)
                    path = temp_path[:-2] + '-' + str(neighbor) + temp_path[-2:]
                    queue_path.append(path)
    
    # Generate dataframe from the ALL lists
    info_nodes_df = pd.DataFrame(
        {'node_name': all_names, 'degree_separation': all_degrees, 'path': all_paths})

    return info_nodes_df

print(NodeMaxDegreeSeparation(2, 3))
