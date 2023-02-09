import pandas as pd
import numpy as np
import utils_firms_analysis as ufa

# Paths
employees_nihs_path = "/home/juan.russy/shared/FamilyNetwork/employees_non_incorporated​_hs.tsv"
graph_path = '/home/juan.russy/shared/proof_run_FamNet/output/Matched_Graph.txt'

# Employees
employees_nihs = pd.read_csv(employees_nihs_path, sep='\t', encoding='latin-1')
# print(employees_nihs.head(3))
# Count number of employees per firm
owner_only = employees_nihs.groupby(['id_owner_hs']).id_employee_hs.count()
firms_nihs = employees_nihs.groupby(['id_firm_hs', 'id_owner_hs']).id_employee_hs.count()
firms_nihs = pd.DataFrame(firms_nihs).reset_index(drop=False)
# Rename id_employee_hs column to 'num_employees'
firms_nihs.columns = ['id_firm_hs', 'id_owner_hs', 'num_employees']
# print('Firms', firms_nihs.head(3), sep = '\n')

# Read EdgeList
reader, G = ufa.read_edgelist(graph_path)

def avg_dist_employees_owner(id_owner_hs):
    """ Find the average distance between employees and owner

    Args:
        id_owner_hs : cedula of the owner
    Returns:
        mean distance between employees and owner
    """
    # Get id_employee_hs values for current id_owner_hs
    employees = employees_nihs[employees_nihs.id_owner_hs ==
                               id_owner_hs].id_employee_hs

    distances = []
    for id_employee_hs in employees:
        distance = ufa.distance(reader, G, id_employee_hs, id_owner_hs)
        distances.append(distance)
    return np.mean(distances)


# Average distance between employees and owner
firms_nihs = firms_nihs.id_owner_hs.apply(avg_dist_employees_owner)

# Harmonic closeness of the owner
firms_nichs = firms_nihs.id_owner_hs.apply(ufa.harmonic_centrality)


def avg_closeness_employees(id_owner_hs):
    """ Find the average closeness centrality amongst all employees of a firm

    Args:
        id_owner_hs : cedula of the owner
    Returns:
        mean closeness between employees and owner
    """
    # Get id_employee_hs values for current id_owner_hs
    employees = employees_nihs[employees_nihs.id_owner_hs ==
                               id_owner_hs].id_employee_hs

    closeness = []
    for id_employee_hs in employees:
        closeness.append(ufa.harmonic_centrality(id_employee_hs))
    return np.mean(closeness)


firms_nichs = firms_nihs.id_owner_hs.apply(avg_closeness_employees)
print(firms_nihs.head(3))

