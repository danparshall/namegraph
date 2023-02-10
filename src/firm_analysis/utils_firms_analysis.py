import pandas as pd
import networkit as nk
import numpy as np
from tqdm import tqdm
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


def find_cedula_in_dict(reader, cedula):
    """ Find id node in networkit from cedula
    
    Args:
        reader: 'graphio' object
        cedula: cedula of interest
    Returns:
        id node in networkit
    """
    dct = reader.getNodeMap()
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
    node1 = find_cedula_in_dict(reader, ced1)
    node2 = find_cedula_in_dict(reader, ced2)

    apsp = nk.distance.APSP(G).run()
    dist = apsp.getDistance(node1, node2)
    return dist


def avg_dist_employees_owner(reader, G, employees_nihs, id_owner_hs):
    """ Find the average distance between employees and owner

    Args:
        reader: 'graphio' object
        G: 'graph' object
        employees_nihs: dataframe with employees and owners
        id_owner_hs : cedula of the owner
    Returns:
        mean distance between employees and owner
    """
    # Get id_employee_hs values for current id_owner_hs
    employees = employees_nihs[employees_nihs.id_owner_hs ==
                               id_owner_hs].id_employee_hs

    distances = []
    for id_employee_hs in employees:
        dist = distance(reader, G, id_employee_hs, id_owner_hs)
        distances.append(dist)
    return np.mean(distances)


def harmonic_centrality(reader, G, cedula):
    """ Harmonic centrality of a node in networkit

    Args:
        reader : 'graphio' object
        G : 'graph' object
        cedula : cedula of node of interest
    Returns:
        harmonic centrality of node of interest
    """
    node = find_cedula_in_dict(reader, cedula)
    hc = nk.centrality.HarmonicCloseness(G).run()
    hc = hc.score(node)
    return hc


def avg_closeness_employees(reader, graph, employees_nihs, id_owner_hs):
    """ Find the average closeness centrality amongst all employees of a firm

    Args:
        reader : 'graphio' object
        graph : 'graph' object
        employees_nihs : dataframe with employees and owners
        id_owner_hs : cedula of the owner
    Returns:
        mean closeness between employees and owner
    """
    # Get id_employee_hs values for current id_owner_hs
    employees = employees_nihs[employees_nihs.id_owner_hs ==
                               id_owner_hs].id_employee_hs

    closeness = []
    for id_employee_hs in employees:
        closeness.append(
            harmonic_centrality(reader, graph, id_employee_hs))
    return np.mean(closeness)


def fraction_employees(reader, graph, employees_nihs, id_owner_hs, MaxDegree):
    """ Fraction with less than a degree of separation from the owner

    Args:
        reader : 'graphio' object
        graph : 'graph' object
        employees_nihs : dataframe with employees and owners
        id_owner_hs : cedula of the owner
        MaxDegree : maximum degree of separation
    Returns:
        fraction of employees with a degree of separation of 6 from the owner
    """

    # Get id_employee_hs values for current id_owner_hs
    employees = employees_nihs[employees_nihs.id_owner_hs ==
                               id_owner_hs].id_employee_hs

    # Count number of employees with less than degree consanguinity
    count = 0
    for id_employee_hs in employees:
        distance = distance(reader, graph, id_employee_hs, id_owner_hs)
        if distance <= MaxDegree:
            count += 1

    # Obtain fraction of all employees
    return count / len(employees)


def fraction_citizens(reader, graph, cedula, MaxDegree):
    """ Fraction of citizens with less than a degree of separation from the owner
    
    Args:
        reader : 'graphio' object
        graph : 'graph' object
        cedula : cedula of the owner
        MaxDegree : maximum degree of separation
    Returns:
        fraction of citizens with a degree of separation of 6 from the owner
    """
    # Generate map between networkit ID and cedula
    node = find_cedula_in_dict(reader, cedula)

    # Queue's
    queue_name, queue_degree = ([] for _ in range(2))

    # Initial node
    degree_separation = 0  # Initial degree of separation
    queue_name.append(node)
    queue_degree.append(degree_separation)

    # Visited nodes set
    visited = set()
    visited.add(node)

    # List with all the nodes information (they are not deleted)
    all_names, all_degrees = ([] for _ in range(2))

    # BFS
    while queue_name:
        # Obtain information of the last queue node
        node_queue = queue_name.pop(0)
        degree = queue_degree.pop(0)

        # Append that information
        all_names.append(node_queue)
        all_degrees.append(degree)

        # If degree is less than MAX DEGREE, add all neighbors to the queue
        if degree <= MaxDegree:
            WeightCondition = True

            # Iterate over node_queue neighbors
            for neighbor, weight in graph.iterNeighborsWeights(node_queue):
                # If degree is equal to MAX DEGREE
                # Add neighbors with weight 0.0 only
                if degree == MaxDegree:
                    WeightCondition = weight == 0.0

                if neighbor not in visited and WeightCondition:
                    queue_name.append(neighbor)
                    queue_degree.append(degree + weight)
                    visited.add(neighbor)

    # Obtain fraction of all citizens
    return len(all_names) / graph.numberOfNodes()


def nodes_within_distance(reader, G, cedula, MaxDistance = 100000):
    """ Find nodes within a certain distance of a node of interest

    Args:
        reader: 'graphio' object
        G: 'graph' object
        cedula: cedula of node of interest
        MaxDistance: maximum weighted distance
    Returns:
        nodes_within_distance: list of nodes within a certain distance of node of interest
    """
    node = find_cedula_in_dict(reader, cedula)
    bfs = nk.distance.BFS(G, node)
    bfs.run()

    nodes_within_distance = []
    for i in range(G.numberOfNodes()):
        if bfs.distance(i) <= MaxDistance:
            nodes_within_distance.append(i)
    print('Node distance', nodes_within_distance)
    return nodes_within_distance
