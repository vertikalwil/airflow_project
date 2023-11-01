FROM apache/airflow:2.7.2
COPY requirements.txt /requirements.txt
RUN python -m pip install --no-cache-dir --user -r /requirements.txt