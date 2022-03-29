import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Paths
main = "/home/juan.russy/shared/FamilyNetwork/original/{}"
out = "/home/juan.russy/shared/proof_run_FamNet/interim/plots/{}"

# Load dataset
normal = True  # Percentage or Aggregate
NROWS = None  # Constant for test
# https://towardsdatascience.com/%EF%B8%8F-load-the-same-csv-file-10x-times-faster-and-with-10x-less-memory-%EF%B8%8F-e93b485086c7

req_cols = ['dt_birth', 'dt_death']  # Load only require columns
data_sample = pd.read_csv(main.format('RC_v2021.tsv'), sep='\t', encoding='utf-8',
    usecols = req_cols,
    parse_dates= ['dt_birth', 'dt_death'], 
    # identifies '' as NaN
    na_values=['',], keep_default_na=False,
    nrows=NROWS,)

# Filter people born before 1880
data_sample = data_sample[data_sample['dt_birth'] > np.datetime64('1880-01-01 00:00:00')]
data_sample['year'] = pd.DatetimeIndex(data_sample['dt_birth']).year

# Generate dataset
if normal:
    death_people = data_sample.set_index('year').groupby(level=0).count()
    print('size_birth', death_people.dt_birth.size)
    print('size_death', death_people.dt_death.size)
    intento = death_people['dt_birth'].values / data_sample['dt_death'].values
    # Plots
    plt.figure(1)
    plt.scatter(data_sample.index.values, intento)
    plt.title("Percentage of population death by year")
    plt.savefig(out.format('percentage_population_death_by_year'))
else:
    death_people = data_sample.set_index('year').groupby(level=0).count()
    # Plots
    plt.figure(0)
    plt.scatter(death_people.index.values, death_people.dt_death)
    plt.title("Population death by year")
    plt.savefig(out.format('population_death_by_year'))
