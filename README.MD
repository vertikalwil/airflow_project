## ABOUT
This project involves creating data pipelines using Apache Airflow. For more details, please visit: vertikalwil.github.io

## SETUP (FOR WINDOWS)
1. Install WSL2 

* [Download & install WSL2](https://github.com)

2. Install Docker

* [Download & install Docker](https://docs.docker.com/desktop/install/windows-install/)

* Go to Settings --> General Tab --> select WSL2

3. Run Airflow in Docker

* [Airflow with Docker](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html)

* Create dags, config, logs, and plugins directory

* To initialize run 'airflow init'

* To open airflow run ''' docker-compose up ''' and then open ''' http://localhost:8080/ ''' on your web browser

4. Install python packages into Docker's image

* Make sure Dockerfile & requirements.txt in directory

* Run ''' docker build . '''

* Restart airflow by running ''' docker-compose down ''' and ''' docker-compose up'''

## SCRIPTS

* Task scripts --> first_dags.py

* Python scripts (data transformation, data cleaning, etc) --> fungsi.py

* Power BI dashboard refresh --> pbi_refresh.py

* Google sheet data update --> drive_upload.py
