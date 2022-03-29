import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Paths
main = "/home/juan.russy/shared/FamilyNetwork/original/{}"
out = "/home/juan.russy/shared/proof_run_FamNet/interim/plots/{}"

# Load dataset
normal = True
NROWS = None  # Constant for test
# https://towardsdatascience.com/%EF%B8%8F-load-the-same-csv-file-10x-times-faster-and-with-10x-less-memory-%EF%B8%8F-e93b485086c7

req_cols = ['dt_birth']  # Load only require columns
data_sample = pd.read_csv(main.format('RC_v2021.tsv'), sep='\t', encoding='utf-8',
    usecols = req_cols,
    parse_dates= ['dt_birth'], 
    # identifies '' as NaN
    na_values=['',], keep_default_na=False,
    nrows=NROWS,)

# Filter people born before 1880
data_sample = data_sample[data_sample['dt_birth'] > np.datetime64('1880-01-01 00:00:00')]
data_sample['year'] = pd.DatetimeIndex(data_sample['dt_birth']).year

# Generate dataset
if normal:
    data_sample = data_sample['year'].value_counts(normalize=True)
    # Plots
    plt.figure(0)
    plt.scatter(data_sample.index, data_sample.values)
    plt.title("Percentage of the population by birth year")
    plt.savefig(out.format('percentage_population_by_year'))

else:
    data_sample = data_sample['year'].value_counts()

    # Plots
    plt.figure(0)
    plt.scatter(data_sample.index, data_sample.values)
    plt.title("Population by year")
    plt.savefig(out.format('population_by_year'))
