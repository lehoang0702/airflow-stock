"""The reported price-column failure must not break SAFE MODE exports."""
import ast
from datetime import datetime, timedelta, timezone
import io
import logging
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dags'))
from pipeline_data import normalize_price_history, prepare_target, clean_features, temporal_masks, rolling_temporal_masks, add_cross_sectional_features
from chart_utils import load_price_data_from_minio
from mlflow_utils import log_model_compat, log_xgboost_model


class MemoryS3:
    def __init__(self, datasets):
        self.objects = {key: frame.to_csv(index=False) for key, frame in datasets.items()}

    def check_for_bucket(self, bucket):
        return True

    def list_keys(self, bucket_name, prefix):
        return [key for key in self.objects if key.startswith(prefix)]

    def read_key(self, key, bucket_name):
        return self.objects[key]

    def load_string(self, string_data, key, bucket_name, replace):
        self.objects[key] = string_data


@pytest.mark.parametrize('columns', [
    ['date', 'symbol', 'close'], ['Date', 'Ticker', 'Close'],
    ['datetime', 'ma_co_phieu', 'adj_close'], ['Date', 'Ticker', 'Adj Close'],
])
def test_price_history_accepts_real_dataset_column_variants(columns):
    raw = pd.DataFrame([['2026-09-28', 'AAPL', 123.45]], columns=columns)
    actual = normalize_price_history(raw)
    assert list(actual.columns) == ['Date', 'Ticker', 'Close']
    assert actual.iloc[0]['Close'] == 123.45
    assert actual.iloc[0]['Date'] == pd.Timestamp('2026-09-28')


def test_missing_or_invalid_price_is_absent_instead_of_fabricated():
    assert normalize_price_history(pd.DataFrame({'date': ['2026-09-28'], 'symbol': ['AAPL']})).empty
    raw = pd.DataFrame({'date': ['2026-09-28'] * 5, 'symbol': ['AAPL'] * 5,
                        'close': [None, -1, 0, np.inf, 123.45]})
    assert normalize_price_history(raw)['Close'].tolist() == [123.45]


def test_price_loader_skips_invalid_newest_file_and_normalizes_xgboost_fallback():
    s3 = MemoryS3({
        'xgboost/xgboost_stock_20tickers_10y_20260928.csv':
            pd.DataFrame({'date': ['2026-09-28'], 'symbol': ['AAPL'], 'close': [123.45]}),
        'xgboost/xgboost_stock_20tickers_10y_20260929.csv':
            pd.DataFrame({'date': ['2026-09-29'], 'symbol': ['AAPL'], 'feature': [9]}),
    })
    result = load_price_data_from_minio(s3, 'test')
    assert result['Close'].tolist() == [123.45]
    assert result['Ticker'].tolist() == ['AAPL']


def test_mlflow_model_name_supports_both_old_and_new_apis():
    def old(model, artifact_path=None, input_example=None):
        return artifact_path
    def new(model, artifact_path=None, name=None, input_example=None):
        assert artifact_path is None
        return name
    assert log_model_compat(old, object()) == 'model'
    assert log_model_compat(new, object()) == 'model'


def test_xgboost_mlflow_schema_accepts_missing_float_inputs_without_integer_warning():
    import warnings
    mlflow = pytest.importorskip('mlflow')
    import mlflow.xgboost
    import xgboost as xgb
    model = xgb.XGBClassifier(n_estimators=2, max_depth=1, n_jobs=1)
    features = pd.DataFrame({'flag': [0, 1, 0, 1], 'value': [1., 2., 3., 4.]})
    model.fit(features, [0, 1, 0, 1])
    with warnings.catch_warnings(record=True) as caught, patch.object(mlflow.xgboost, 'log_model', autospec=True) as log:
        log_xgboost_model(mlflow, model, features)
    kwargs = log.call_args.kwargs
    assert kwargs['input_example'].dtypes.eq('float64').all()
    assert all(column.type.name == 'double' for column in kwargs['signature'].inputs.inputs)
    assert not any('integer column' in str(warning.message) for warning in caught)


@pytest.mark.parametrize('has_training_prices', [True, False])
def test_real_xgboost_training_exports_safe_mode_with_lowercase_or_missing_prices(has_training_prices):
    pytest.importorskip('mlflow')
    import mlflow
    import chart_utils
    import hyperparameter_tuner
    import model_registry
    import mlflow_utils
    dates = pd.bdate_range('2023-01-02', periods=320)
    close = 100 + np.sin(np.arange(320))
    frame = pd.DataFrame({'date': dates, 'symbol': 'AAPL', 'close': close,
                          'momentum': np.cos(np.arange(320)),
                          'target': np.r_[close[1:] > close[:-1], False].astype(int)})
    if not has_training_prices:
        frame = frame.drop(columns='close')
    s3 = MemoryS3({'xgboost/xgboost_stock_20tickers_10y_20260928.csv': frame})
    path = Path(__file__).resolve().parents[1] / 'dags/dag_xgboost_train_and_predict.py'
    node = next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == 'train_and_predict_xgboost_task')
    messages = []
    namespace = {'datetime': datetime, 'timedelta': timedelta, 'timezone': timezone,
                 'pd': pd, 'np': np, 'io': io, 'logging': logging,
                 'MINIO_CONN_ID': 'test', 'BUCKET_NAME': 'test', 'SECTOR_MAP': {'AAPL': 'Technology'},
                 'TELEGRAM_TOKEN': '', 'TELEGRAM_CHAT_ID': '',
                 'S3Hook': lambda **kwargs: s3, 'prepare_target': prepare_target,
                 'clean_features': clean_features, 'temporal_masks': temporal_masks, 'rolling_temporal_masks': rolling_temporal_masks, 'add_cross_sectional_features': add_cross_sectional_features,
                 'normalize_price_history': normalize_price_history,
                 'send_telegram_safe': lambda url, data=None, **kwargs: messages.append(data)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    params = {'n_estimators': 3, 'max_depth': 1, 'n_jobs': 1}
    gate = {'status': 'SAFE_MODE', 'selected_version': None, 'reason': 'Quality gate failed'}
    # Real XGBoost fitting and prediction; isolate only storage, tracking and messaging.
    with patch.object(hyperparameter_tuner, 'tune_xgboost_hyperparameters', return_value=params), \
         patch.object(hyperparameter_tuner, 'save_tuned_hyperparameters_to_minio'), \
         patch.object(model_registry, 'save_model_to_minio'), \
         patch.object(model_registry, 'select_champion_model', return_value=gate), \
         patch.object(mlflow, 'set_tracking_uri'), patch.object(mlflow, 'set_experiment'), \
         patch.object(mlflow, 'start_run'), patch.object(mlflow, 'log_params'), patch.object(mlflow, 'log_metrics'), \
         patch.object(mlflow_utils, 'log_xgboost_model'), \
         patch.object(chart_utils, 'load_price_data_from_minio', return_value=
                      pd.DataFrame({'Date': [dates[-1]], 'Ticker': ['AAPL'], 'close': [close[-1]]})), \
         patch.object(chart_utils, 'generate_individual_charts', return_value={}), \
         patch.object(chart_utils, 'save_charts_zip_minio_and_telegram'):
        namespace['train_and_predict_xgboost_task']()
    key = next(key for key in s3.objects if key.startswith('xgboost/du_bao_tang_truong_'))
    output = pd.read_csv(io.StringIO(s3.objects[key]))
    assert output['prob_num'].isna().all()
    assert output['model_version'].isna().all()
    assert output['inference_source'].eq('unavailable').all()
    assert output['khuyen_nghi'].eq('ĐỨNG NGOÀI').all()
    text = messages[0]['text']
    assert 'SAFE MODE' in text
    assert 'TOP 5' not in text
    if not has_training_prices:
        assert output['gia_5d_du_kien'].isna().all()
        assert output['loi_nhuan_5d_du_kien'].isna().all()
