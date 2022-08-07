import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Paths
main = "/home/juan.russy/shared/FamilyNetwork/original/{}"
out = "/home/juan.russy/shared/proof_run_FamNet/interim/plots/{}"

# Load dataset
NROWS = None  # Constant for test
# https://towardsdatascience.com/%EF%B8%8F-load-the-same-csv-file-10x-times-faster-and-with-10x-less-memory-%EF%B8%8F-e93b485086c7

req_cols = ['dt_birth', 'ced_padre', 'ced_madre']  # Load only require columns
data_sample = pd.read_csv(main.format('RC_v2021.tsv'), sep='\t', encoding='utf-8',
    usecols = req_cols,
    parse_dates= ['dt_birth'], 
    dtype={'ced_padre': str, 'ced_madre': str},
    # identifies '' as NaN
    na_values=['',], keep_default_na=False,
    nrows=NROWS,)

# Filter people born before 1880
data_sample = data_sample[data_sample['dt_birth'] > np.datetime64('1880-01-01 00:00:00')]
data_sample['year'] = pd.DatetimeIndex(data_sample['dt_birth']).year

# Generate dataset
num_nan_data_sample = data_sample.set_index('year').isna().groupby(level=0).sum()
num_nan_data_sample.to_csv(out.format("by_year_missing_values_aggregate.csv"))

# Plots
plt.figure(0)
plt.scatter(num_nan_data_sample.index.values, num_nan_data_sample['ced_padre'].values)
plt.title("Missing ced_padre - Aggregate by year")
plt.savefig(out.format('missing_ced_padre_aggregate_by_year'))

plt.figure(1)
plt.scatter(num_nan_data_sample.index.values,
            num_nan_data_sample['ced_madre'].values)
plt.title("Missing ced_madre - Aggregate by year")
plt.savefig(out.format('missing_ced_madre_aggregate_by_year'))
