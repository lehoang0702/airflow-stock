FROM apache/airflow:2.8.1-python3.11

USER airflow
RUN pip install --no-cache-dir \
    yfinance \
    pyarrow \
    fastparquet \
    xgboost \
    scikit-learn \
    joblib \
    requests \
    transformers \
    matplotlib \
    boto3 \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
