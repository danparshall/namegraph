import pandas as pd
import numpy as np
import utils_firms_analysis as ufa

# Paths
employees_nihs_path = "/home/juan.russy/shared/FamilyNetwork/employees_non_incorporated​_hs.tsv"
graph_path = '/home/juan.russy/shared/proof_run_FamNet/output/Matched_Graph.txt'

# Employees
employees_nihs = pd.read_csv(employees_nihs_path, sep='\t', encoding='latin-1')
# Count number of employees per firm
owner_only = employees_nihs.groupby(['id_owner_hs']).id_employee_hs.count()
firms_nihs = employees_nihs.groupby(['id_firm_hs', 'id_owner_hs']).id_employee_hs.count()
firms_nihs = pd.DataFrame(firms_nihs).reset_index(drop=False)
# Rename id_employee_hs column to 'num_employees'
firms_nihs.columns = ['id_firm_hs', 'id_owner_hs', 'num_employees']

# Read EdgeList
reader, G = ufa.read_edgelist(graph_path)

# Average distance between employees and owner
firms_nihs['avg_dist_employees'] = firms_nihs.id_owner_hs.apply(
    lambda id: ufa.avg_dist_employees_owner(reader, G, employees_nihs, id))

# Harmonic closeness of the owner
firms_nihs['h_closeness_owner'] = firms_nihs.id_owner_hs.apply(
    lambda id: ufa.harmonic_centrality(reader, G, id))

# Average closeness of employees
firms_nihs['avg_closeness_employees'] = firms_nihs.id_owner_hs.apply(
    lambda id: ufa.avg_closeness_employees(reader, G, employees_nihs, id))

MaxDegreeSeparation = 6
# Fraction of employees with less than a degree of separation from the owner
firms_nihs['fraction_employees'] = firms_nihs.id_owner_hs.apply(
    lambda id: ufa.fraction_employees(reader, G, employees_nihs, id))
# Fraction of citizens with less than a degree of separation from the owner
firms_nihs['fraction_citizens'] = firms_nihs.id_owner_hs.apply(
    lambda id: ufa.fraction_citizens(reader, G, id, MaxDegreeSeparation))

# ced_owner = firms_nihs.iloc[0, 1]
# print('Ced owner: ', ced_owner)
# MaxDistance = 2
# ufa.nodes_within_distance(reader, G, ced_owner, MaxDistance)