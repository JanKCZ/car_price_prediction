import pandas as pd
df = pd.read_csv('/Users/jankolnik/Downloads/car_list_all_v1_updated_sauto.csv')

pd.set_option('display.max_colwidth', None)
print(df['add_id-href'].head())

