"""
STORAGE & S3 INTEGRATION — QUANTUM DASHBOARD
=============================================
Quản lý kết nối MinIO S3, tìm kiếm latest keys, và đọc Parquet / CSV datasets.
"""

import os
import io
import re
import logging
import boto3
from botocore.config import Config
import pandas as pd

from dashboard.config import BUCKET_NAME


def get_s3_client():
    """Tạo kết nối tới MinIO S3 API."""
    endpoint = os.environ.get("S3_ENDPOINT", os.environ.get("MINIO_ENDPOINT_URL", "http://minio:9000"))
    try:
        client = boto3.client(
            's3',
            endpoint_url=endpoint,
            aws_access_key_id=os.environ.get("MINIO_ROOT_USER", "minioadmin"),
            aws_secret_access_key=os.environ.get("MINIO_ROOT_PASSWORD", "minioadmin"),
            config=Config(connect_timeout=2, read_timeout=10, retries={'total_max_attempts': 2}),
        )
        return client
    except Exception as e:
        logging.warning(f"⚠️ S3 connection error on {endpoint}: {e}")
        try:
            return boto3.client(
                's3',
                endpoint_url="http://localhost:9000",
                aws_access_key_id="minioadmin",
                aws_secret_access_key="minioadmin",
                config=Config(connect_timeout=2, read_timeout=10, retries={'total_max_attempts': 2}),
            )
        except Exception:
            return None


def get_latest_s3_key(s3, prefix: str):
    """Tìm object key mới nhất theo prefix trong bucket."""
    try:
        contents = [item for page in s3.get_paginator('list_objects_v2').paginate(
            Bucket=BUCKET_NAME, Prefix=prefix) for item in page.get('Contents', [])]
        if not contents:
            return None
        latest = max(contents, key=lambda x: x['LastModified'])
        return latest['Key']
    except Exception as e:
        logging.warning(f"Error listing S3 objects for {prefix}: {e}")
        return None


def read_s3_dataframe(s3, key: str) -> pd.DataFrame:
    """Tự động đọc Parquet hoặc CSV từ MinIO S3."""
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
    raw = io.BytesIO(obj['Body'].read())
    if key.endswith('.parquet'):
        return pd.read_parquet(raw)
    return pd.read_csv(raw)


def extract_percentage(text: str, default=None):
    """Trích xuất phần trăm từ chuỗi 'MUA (55.36%)'."""
    if not text or not isinstance(text, str):
        return default
    m = re.search(r'([0-9.]+)\s*%', text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return default
