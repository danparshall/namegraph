import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

main = "/home/juan.russy/shared/proof_run_FamNet/interim/{}"
data_padres = pd.read_csv(main.format('12-PADRES_matched_by_name.tsv'), sep='\t', header=None)

def count_matches(match_string):
    print(match_string)
    if match_string.startswith('Found'):
        return match_string.split()[1]
    elif not match_string.startswith('PAPAS:'):
        return match_string.count(';') + 1

data_padres['num_matches'] = data_padres[1].map(count_matches)

data_padres['num_matches'] = pd.to_numeric(
    data_padres['num_matches'], errors="coerce")

# Including 0
plt.figure(0)
plt.hist(data_padres['num_matches'], edgecolor='black',
         bins=[-0.5, 0.5, 1.5, 5, 10, 20, 40, 60, 80, 100, 1000])
plt.title('Number of matching padres, Including 0')
plt.savefig(main.format("plots/num_matchs_padres_with_0"), dpi = 200)

# Not including 0
plt.figure(1)
plt.hist(data_padres['num_matches'], edgecolor='black' ,
          bins=[0.5, 1.5, 5, 10, 20, 40, 60, 80, 100, 1000])
plt.title('Number of matching padres, Not including 0')
plt.savefig(main.format("plots/num_matchs_padres_without_0"), dpi=200)
