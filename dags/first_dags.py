from datetime import datetime
import os
import fungsi 
import ast
import pandas as pd

from airflow.operators.bash import BashOperator
from airflow.providers.ssh.operators.ssh import SSHOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow import DAG
from airflow.operators.python import PythonOperator


default_args = {
    'owner':'me'
}

def data_transform():
    df = fungsi.data_transform_func()
    df.to_csv('/opt/airflow/data_csv/tangerang_transformed.csv', index = False)

def data_pbi():
    hook = PostgresHook(postgres_conn_id="postgres_docker")
    df = hook.get_pandas_df(sql="select * from scrap;")
    df = fungsi.data_pbi_func(df)
    df.to_csv('/opt/airflow/data_csv/clean_tangerang_pbi.csv', index=False)

def data_evidently():
    hook = PostgresHook(postgres_conn_id="postgres_docker")
    df = hook.get_pandas_df(sql="select * from scrap;")
    df = fungsi.data_evidently_func(df)
    df.to_csv('/opt/airflow/data_csv/clean_tangerang_evidently.csv', index=False)


with DAG(
    dag_id='ETL',
    default_args=default_args,
    start_date=datetime.utcnow(),
    description = 'desc'

) as dag:
 
    task5a = SSHOperator(
         task_id = 'generate_evidently_report',
         ssh_conn_id = 'host_machine',
         command = 'cd /home/vertikal/data_science/power_bi && . .venv/bin/activate && python3 evidently.ai/generate_report.py && deactivate ',
         cmd_timeout = None)

    task4a = PythonOperator(
        task_id='pull_and_clean_data_evidently',
        python_callable=data_evidently)   
 
    
    task6 = SSHOperator(
         task_id = 'refresh_visualization_pbi',
         ssh_conn_id = 'host_machine',
         command = 'cd /home/vertikal/data_science/power_bi && . .venv/bin/activate && python3 pbi_refresh.py && deactivate ',
         cmd_timeout = None)
    
    
    task5 = SSHOperator(
         task_id = 'upload_to_drive_pbi',
         ssh_conn_id = 'host_machine',
         command = 'cd /home/vertikal/data_science/power_bi && . .venv/bin/activate && python3 drive_upload.py && deactivate ',
         cmd_timeout = None)
    
    task4 = PythonOperator(
        task_id='pull_and_clean_data_pbi',
        python_callable=data_pbi)
    
 
    task3 = PostgresOperator(
        task_id='upload_data_postgres',
        postgres_conn_id='postgres_docker',
        autocommit=True,
        sql = """

        COPY scrap FROM '/var/lib/postgresql/data/data_upload/tangerang_transformed.csv' DELIMITER ',' CSV HEADER

        """)
    
     
    task2 = PythonOperator(
        task_id='transform_data',
        python_callable=data_transform)
    
    
    task1 = SSHOperator(
         task_id = 'scrap_data',
         ssh_conn_id = 'host_machine',
         command = 'cd /home/vertikal/data_science/web_scrap/rumah/rumah && bash script2.sh ',
         cmd_timeout = None
         )
   
task1 >> task2 >> task3
task3 >> [task4, task4a]
task4 >> task5 >> task6
task4a >> task5a
    