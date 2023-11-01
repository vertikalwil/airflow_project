import gspread
import pandas as pd
import time

df  = pd.read_csv('~/data_science/airflow_docker/data_csv/clean_tangerang_pbi.csv')

#for credential, replace with your credential
sa = gspread.service_account(filename='credential.json')
sh = sa.open("tangerang_pbi")
sheet = sh.worksheet("tangerang_pbi")

list_values = [list(item) for item in list(df.to_numpy())]

range = f'A2:N{len(df) + 1}'
sheet.update(range_name = range, values = list_values)