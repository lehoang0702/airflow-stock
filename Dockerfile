FROM apache/airflow:slim-2.9.3-python3.11

USER airflow
COPY requirements.txt /tmp/requirements.txt
RUN pip install --default-timeout=2000 --retries 10 --no-cache-dir \
    "apache-airflow==${AIRFLOW_VERSION}" -r /tmp/requirements.txt \
    && pip check \
    && python -c "import airflow, mlflow, fastapi; print('Airflow/MLflow imports OK')"

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
USER airflow
