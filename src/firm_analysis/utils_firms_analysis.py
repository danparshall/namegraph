import pandas as pd
import networkit as nk
from time import time


def read_edgelist(graph_path):
    """ Read edgelist from path
    
    Args:
        graph_path: path to edgelist
    Returns:
        reader: 'graphio' object
        G: 'graph' object
    """
    start_time = time()
    # 'graphio' object
    reader = nk.graphio.EdgeListReader(separator='\t', firstNode=0, continuous=False, directed=False)
    # 'graph' object
    G = reader.read(graph_path)  
    print("EdgeList %s seconds ---" % round(time() - start_time, 2))
    return reader, G


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
    """ Find id node in networkit from cedula
    
    Args:
        dct: dictionary with mapping between cedula and id node
        cedula: cedula of interest
    Returns:
        id node in networkit
    """
    if cedula in dct.keys():
        return dct[cedula]
    else:
        return None


def distance(reader, G, ced1, ced2):
    """ Distance between two nodes in a graph

    Args:
        reader: 'graphio' object
        G: 'graph' object
        ced1: cedula of node 1
        ced2: cedula of node 2
    Returns:
        dist: distance between node 1 and node 2
    """
    node1 = find_cedula_in_dict(reader.getNodeMap(), ced1)
    node2 = find_cedula_in_dict(reader.getNodeMap(), ced2)

    apsp = nk.distance.APSP(G).run()
    dist = apsp.getDistance(node1, node2)
    return dist


def harmonic_centrality(reader, G, cedula):
    """ Harmonic centrality of a node in networkit

    Args:
        reader: 'graphio' object
        G: 'graph' object
        cedula: cedula of node of interest
    Returns:
        hc: harmonic centrality of node of interest
    """
    node = find_cedula_in_dict(reader.getNodeMap(), cedula)
    hc = nk.centrality.HarmonicCloseness(G).run()
    hc = hc.score(node)
    return hc
