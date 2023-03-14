import time
import pandas as pd
import utils_firms_analysis as ufa

# %% Parameters
MAX_DEGREE = 6

# Generate consangunity map for each owner and employee, or load dataset.
separation_map_owners = False
separation_map_employees = False 
# Calculate centrality measures
centrality = True
# Avg closeness of employees
avg_closeness_employees = False
# Generate output
generate_output = True

# Paths
main_path = '/home/juan.russy/shared/FamilyNetwork/{}'
output_path = '/home/juan.russy/shared/proof_run_FamNet/output/{}'

graph_path = output_path.format("Matched_Graph.txt")
employees_nihs_path = main_path.format("employees_included_owner.tsv")
partner_path = output_path.format("15-match_by_cedula_spouse.tsv")
sep_map_owners = main_path.format(F"consangunity_map_owners_{MAX_DEGREE}.tsv")
sep_map_employees = main_path.format(
    F"consangunity_map_employees_{MAX_DEGREE}.tsv")

#%% Data
# Firms, owners and employees dataset
employees_nihs = pd.read_csv(employees_nihs_path, sep='\t', encoding='utf-8')
# Dictionary of partners
spouse = pd.read_csv(partner_path, sep='\t', encoding='utf-8')
dict_spouses = dict(zip(spouse['ced1'], spouse['ced2']))
# Consanguinity graph
reader, G = ufa.read_edgelist(graph_path)
# Final dataset
id_cols = ['id_firm_hs', 'id_owner_hs']
firms_nihs = employees_nihs.groupby(id_cols).id_employee_hs.count()
firms_nihs = pd.DataFrame(firms_nihs).reset_index(drop=False)
firms_nihs.columns = ['id_firm_hs', 'id_owner_hs', 'num_employees']

# %% Subgraph six degrees of separation for all owners
if separation_map_owners:
    smo = ufa.ConsanguinitySeparationMap(
        reader, G, firms_nihs.id_owner_hs, dict_spouses, MAX_DEGREE)
    smo.to_csv(sep_map_owners, sep='\t', encoding='utf-8', index=False)

# %% Subgraph six degrees of separation for all employees
if separation_map_employees:
    sme = ufa.ConsanguinitySeparationMap(
        reader, G, employees_nihs.id_employee_hs, dict_spouses, MAX_DEGREE)
    sme.to_csv(sep_map_employees, sep='\t', encoding='utf-8', index=False)    

#%% Centrality measures
centrality_cols = ['harmonic_centrality', 'avg_dist_employees_owner',
    'fraction_employees', 'fraction_family', 'fraction_citizens']

if centrality:
    # Subgraph six degrees of separation for all owners
    smo = ufa.read_consanguinity(sep_map_owners)

    # Initialize centrality_measures arguments
    citizens = G.numberOfNodes()
    # Calculate centrality measures
    centrality_measures = firms_nihs.id_owner_hs.apply(
        lambda id: ufa.centrality_measures(id, smo, employees_nihs, citizens))
    # Format and concatenate
    centrality_measures = pd.DataFrame(
        centrality_measures.tolist(), columns=centrality_cols)
    firms_nihs = pd.concat([firms_nihs, centrality_measures], axis=1)

if avg_closeness_employees:
    # Subgraph six degrees of separation for all employees
    sme = ufa.read_consanguinity(sep_map_employees)

    # Calculate average closeness of employees
    firms_nihs['avg_closeness_employees'] = firms_nihs.id_owner_hs.apply(
        lambda id: ufa.avg_closeness_employees(id, smo, sme, employees_nihs))

if generate_output:
    firms_nihs.to_csv(main_path.format("firms_nihs_centrality.tsv"),
        sep='\t', encoding='utf-8', index=False)
