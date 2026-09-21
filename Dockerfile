FROM apache/airflow:2.8.1-python3.11

USER airflow
COPY requirements.txt /tmp/requirements.txt
RUN pip install --default-timeout=2000 --retries 10 --no-cache-dir -r /tmp/requirements.txt
