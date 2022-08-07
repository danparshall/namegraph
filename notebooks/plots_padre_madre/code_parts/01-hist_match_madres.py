import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

main = "/home/juan.russy/shared/proof_run_FamNet/interim/{}"
data_madres = pd.read_csv(main.format('11-MADRES_matched_by_name.tsv'), sep='\t', header=None)
data_madres.dropna(how="any", inplace=True)

def count_matches(match_string):
    print(match_string)
    if match_string.startswith('Found'):
        return match_string.split()[1]
    elif not match_string.startswith('PAPAS:'):
        return match_string.count(';') + 1

data_madres['num_matches'] = data_madres[1].map(count_matches)

data_madres['num_matches'] = pd.to_numeric(
    data_madres['num_matches'], errors="coerce")

# Including 0
plt.figure(0)
plt.hist(data_madres['num_matches'], edgecolor='black',
         bins=[-0.5, 0.5, 1.5, 5, 10, 20, 40, 60, 80, 100])
plt.title('Number of matching madres, Including 0')
plt.savefig(main.format("plots/num_matchs_madres_with_0_until_100"), dpi = 200)

# Not including 0
plt.figure(1)
plt.hist(data_madres['num_matches'], edgecolor='black' ,
          bins=[0.5, 1.5, 5, 10, 20, 40, 60, 80, 100])
plt.title('Number of matching madres, Not including 0')
plt.savefig(main.format("plots/num_matchs_madres_without_0_until_100"), dpi=200)
