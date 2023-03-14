import pandas as pd
import networkit as nk
import numpy as np
from tqdm import tqdm
import time
from operator import itemgetter

_output_path = "/home/juan.russy/shared/FamilyNetwork/employees_included_owner.tsv"


def read_edgelist(graph_path):
    """ Read an edgelist from a file.
    
    The edgelist file is expected to have one edge per line, with
    the two nodes of the edge separated by a tab.
    
    Args:
        graph_path: path to edgelist
    
    Returns:
        reader: 'graphio' object
        G: 'graph' object
    """
    start_time = time.time()
    reader = nk.graphio.EdgeListReader(
        separator='\t', firstNode=0, continuous=False, directed=False)
    G = reader.read(graph_path)
    print("EdgeList %s seconds ---" % round(time.time() - start_time, 2))
    return reader, G


def read_consanguinity(sep_map_path):
    """ Read the consanguinity map file for owners and employees

    Args:
        sep_map_path: Path to the consanguinity map csv
    Returns:
        sme: Dictionary with the consanguinity map
    """
    cols = ['cedula_reference', 'cedula', 'degree_separation']
    # Subgraph six degrees of separation for all employees
    smo = pd.read_csv(sep_map_path, sep='\t', encoding='utf-8', usecols=cols)
    smo = {person: {'family': list(group['cedula']),
                    'degree': list(group['degree_separation'])}
            for person, group in smo.groupby('cedula_reference')}

    return smo


def owners_as_only_employee(employees_nihs, output_path=_output_path):
    """ Include owners as employees and Remove firms which unique employee is the owner

    Args:
        employees_nihs: Dataframe with the employees of each firm
        output_path: Path to save the output dataframe
    Returns:
        employees_nihs: Dataframe with all correctionis
    """
    # Remove firms which unique employee is the owner
    employees_nihs.query('id_owner_hs != id_employee_hs', inplace=True)
    # Include owners as employees
    cols = ['id_firm_hs', 'id_owner_hs']
    temp = employees_nihs.drop_duplicates(subset=cols)
    temp = temp.assign(id_employee_hs=temp['id_owner_hs'])
    employees_nihs.append(temp, ignore_index=True)
    employees_nihs.to_csv(output_path, sep='\t', index=False)

    return employees_nihs


def EdgeSeparation(G, NODE, MaxDegree):
    """ Nodes with less than MaxDegree of separation from node 
    
    BFS is used to do the search, the function manages three queue's with the
    information of interest.

    Args:
        G: Weighted or Unweighted networkit's graph
        NODE: Integer with node ID
        MaxDegree: Maximum degree of separation from node allowed
    Returns:
        info_nodes_df: Dataframe with the name, degree of separation and paths
        of the nodes that follow the MaxDegree rule
    """
    degree_separation = 0  # Initial degree of separation

    visited = set()  # Visited nodes set
    # Queue's
    queue_name, queue_degree = ([] for _ in range(2))

    # Initial node
    queue_name.append(NODE)
    queue_degree.append(degree_separation)
    visited.add(NODE)

    # List with all the nodes information (they are not deleted)
    all_names, all_degrees = ([] for _ in range(2))
    # BFS
    while queue_name:
        # Obtain information of the last queue node
        node = queue_name.pop(0)
        degree = queue_degree.pop(0)

        # Append that information
        all_names.append(node)
        all_degrees.append(degree)

        # If degree is less than MAX DEGREE
        # Add neighbors to the queue
        if degree < MaxDegree:
            # Iterate over node neighbors
            for neighbor in G.iterNeighbors(node):
                if neighbor not in visited:
                    queue_name.append(neighbor)
                    queue_degree.append(degree + 1)
                    visited.add(neighbor)

    # Generate dataframe from the ALL lists
    dct_info = {'node_name': all_names, 'degree_separation': all_degrees}
    info_nodes_df = pd.DataFrame(dct_info)
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
    map_df = pd.DataFrame.from_dict(map_nkID_ced, orient='index')
    map_df = map_df.reset_index(level=0)
    map_df.columns = ['cedula', 'node_name']

    return map_df


def find_cedula_in_dict(dct, cedula):
    """ Find id node in networkit from cedula
    
    Args:
        dct: dictionary with mapping between cedula and id node in networkit
        cedula: cedula of interest
    Returns:
        id node in networkit
    """
    if cedula in dct.keys():
        return dct[cedula]
    else:
        return None


def found_spouse(cedula, dict_spouses, inv_spouses):
    """ Find partner of a node in the spouse relation dataframe
    
    Args:
        cedula: cedula of interest
        dict_spouses: spouse relation dictionary
    Returns:
        registry: cedula of the partner, None if not found
    """

    if cedula in dict_spouses:
        registry = dict_spouses[cedula]
    elif cedula in inv_spouses:
        registry = inv_spouses[cedula]
    else:
        registry = None
    return registry


def ConsanguinitySeparationMap(reader, graph, cedula, dict_spouses, MaxDegree):
    """ Find consanguinity separation of nodes in cedula

    Args:
        reader: 'graphio' object
        graph: 'graph' object
        cedula: list of cedula
        spouse: spouse relation dataframe
        MaxDegree: Maximum degree of separation from node allowed
    Returns:
        output: dataframe with the consanguinity separation of nodes in cedula
    """
    # Generate Map Node ID - Cedula
    map_dct = reader.getNodeMap()
    map_df = Map_IDnk_cedula(reader)

    # Obtain the inverse of the dict_spouses dictionary
    inv_spouses = {value: key for key, value in dict_spouses.items()}
    # Find nodes to a certain consanguinity separation
    consanguinity = {}

    for ced in tqdm(cedula):
        # Obtain node id of cedula, continue if do not exist in graph
        node = find_cedula_in_dict(map_dct, ced)
        if node == None:
            continue

        # Found nodes with less than MaxDegree consanguinity separation
        consang = EdgeSeparation(graph, node, MaxDegree)

        # Find spouse and search consanguinity separation
        ced_spouse = found_spouse(ced, dict_spouses, inv_spouses)
        node_spouse = find_cedula_in_dict(map_dct, ced_spouse)
        if node_spouse != None:
            consang_spouse = EdgeSeparation(graph, node_spouse, MaxDegree)
            consang_spouse['reference'] = node
            consang = pd.concat([consang, consang_spouse], ignore_index=True)
        # Obtain nodes with less degree of separation and drop duplicates
        consang = consang.sort_values(by = 'degree_separation')
        consang = consang.drop_duplicates(subset = ['node_name', 'reference'])
        # Append to dictionary
        consanguinity[ced] = consang
    consanguinity = pd.concat(consanguinity, ignore_index=True)

    # Merge with the map to obtain cedula number
    output = consanguinity.merge(map_df, how='left', on = 'node_name')
    map_df.columns = ['cedula_reference', 'reference']
    output = output.merge(map_df, how='left', on = 'reference')
    print('Output', output.head(), sep='\n')

    return output


def obtain_subgraph(G, nodes):
    """ Obtain subgraph of a node with a certain degree of separation

    Args:
        G: 'graph' object
        nodes: list of nodes
    Returns:
        subgraph: subgraph of the nodes
    """
    return nk.graphtools.subgraphFromNodes(G, nodes)


# def centrality_measures(id_owner, ConsangunitySeparation, employees_nihs, num_citizens):
def centrality_measures(id_owner, smo, employees_nihs, num_citizens):
    """ Obtain centrality measures for a reference node

    Harmonic centrality, average distance between employers and owners,
    fraction of employees and fraction of citizens within six degrees of
    consanguinity.

    For harmonic centrality measure, distance greater than six are set to
    infinity. In case of spouse or the owner with consanguinity degree 0, 
    distance is set to 1.

    Average distance between employers and owners is calculated only for
    employees within the consanguinity network.

    Count of employees that belong to the consanguinity network is done 
    within the loop.

    Args:
        id_owner : id of the reference node
        smo : dictionary with owners as keys and employees and degrees of 
        separation as values.
        employees_nihs : dictionary with owners as keys and employees as values.
        num_citizens : number of citizens in the network

    Returns:
        harmonic_centrality : harmonic centrality between owner and employees
        avg_dist_employees_owner : average distance between employees and owner
        fraction_employees : fraction of employees within consanguinity network
        fraction_family : fraction of family members that are employees
        fraction_citizens : fraction of citizens within consanguinity network
    """
    # Get employees cedulas for reference node
    employees = employees_nihs.loc[ employees_nihs.id_owner_hs == id_owner,
        'id_employee_hs'].values

    consanguinity = []
    indices = []
    dist_from_owner = []

    # Cedula owner exists and obtain indices of employees 
    # that are family
    if id_owner in smo.keys():
        consanguinity = smo[id_owner]['family']
        indices = [
            consanguinity.index(term)
            for term in employees if term in consanguinity]

    # Obtain consangunity separation for employees
    if indices:
        dist_from_owner = smo[id_owner]['degree']
        dist_from_owner = itemgetter(*indices)(dist_from_owner)
    dist_from_owner = np.array(dist_from_owner)

    # Centrality measures
    harmonic_centrality = 0
    avg_dist_employees_owner = np.nan
    fraction_num = 0

    if not dist_from_owner.size == 0:
        inv_dist_from_owner = np.where( 
            dist_from_owner == 0, 1, 1 / dist_from_owner)
        harmonic_centrality = inv_dist_from_owner.sum()
        avg_dist_employees_owner = np.mean(dist_from_owner)
        fraction_num = dist_from_owner.size

    # Fraction of employees that are family members
    fraction_employees = fraction_num / employees.size
    # Fraction of family members that are employees
    fraction_family = fraction_num / (len(consanguinity) - 1)
    # Fraction of citizens, substract one because of owner
    fraction_citizens = (len(consanguinity) - 1) / num_citizens

    return  (harmonic_centrality, avg_dist_employees_owner,
            fraction_employees, fraction_family, fraction_citizens)


def avg_closeness_employees(id_owner, smo, sme, employees_nihs):
    """ Find the average closeness centrality amongst all employees of a firm

    Args:
        id_owner: id of the reference node
        ConsangunitySeparation: dataframe with consanguinity separation
        employees_nihs: dataframe with relation between employees and owners
    Returns:
        mean closeness between employees and owner
    """
    employees = employees_nihs[
        employees_nihs.id_owner_hs == id_owner].id_employee_hs
    firm_people = employees.append(pd.Series([id_owner]))

    harmonic_centrality = []

    for employee in employees:
        # Cedula owner exists and obtain indices of employees
        # that are family
        if id_owner in smo.keys():
            consanguinity = smo[id_owner]['family']
            indices = [
                consanguinity.index(term)
                for term in employees if term in consanguinity]

        # Obtain consangunity separation for employees
        if indices:
            dist_from_owner = smo[id_owner]['degree']
            dist_from_owner = itemgetter(*indices)(dist_from_owner)
        dist_from_owner = np.array(dist_from_owner)