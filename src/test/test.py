import pandas as pd
import sys
sys.path.append("../data")
import utils

def test_data(org_data, test_data_path, dtypes, date_col=None):
    test = pd.read_csv(test_data_path, sep='\t', parse_dates=date_col,
                       dtype=dtypes, keep_default_na=False, na_values=utils.get_nan_values())
    print("Test", test_data_path)

    for row in test.to_dict(orient='records'):
        try:
            # for key in row.keys():
            #     if pd.isnull(row[key]):
            #         row[key] = ''
            res = org_data[org_data.cedula == row['cedula']].iloc[0].to_dict()
        except IndexError:
            print(row['cedula'], ": NOT IN THIS SUBFRAME")
            continue
        if res == row:
            print(row['cedula'], ": OK")
        else:
            print(row['cedula'], ": FAILED")
            print(row)
            print(res)


def test_names(org_data, test_data_path, dtypes, date_col=None):
    test = pd.read_csv(test_data_path, sep='\t', parse_dates=date_col,
                       dtype=dtypes, keep_default_na=False, na_values=utils.get_nan_values())
    print("Test", test_data_path)

    for row in test.to_dict(orient='records'):
        try:
            res = org_data[org_data.obsname ==
                            row['obsname']].iloc[0].to_dict()
        except IndexError:
            print(row['obsname'], ": NOT IN THIS SUBFRAME")
            continue
        if res == row:
            print(row['obsname'], ": OK")
        else:
            print(row['obsname'], ": FAILED")
            print(row)
            print(res)
