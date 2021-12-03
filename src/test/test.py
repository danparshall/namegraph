import pandas as pd

nan_values = ['-1.#IND', '1.#QNAN', '1.#IND', '-1.#QNAN', '#N/A N/A', '#N/A', 'N/A', 'n/a',
              '<NA>', '#NA', 'NULL', 'null', 'NaN', '-NaN', 'nan', '-nan', ]

def test_data(org_data, test_data_path, dtypes, date_col=None):
    test = pd.read_csv(test_data_path, sep='\t', parse_dates=date_col,
                       dtype=dtypes, keep_default_na=False, na_values=nan_values)
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
