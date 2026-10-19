import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dags'))
"""On-demand forecasts must come only from successful model inference."""

import ast
from datetime import datetime, timezone
import io
import logging
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from concurrent.futures import ThreadPoolExecutor, wait
from unittest.mock import patch

import numpy as np
import pandas as pd
from sentiment_utils import sentiment_scores
from signal_formulas import finbert_signal, finbert_signal_text, weighted_ensemble, ENSEMBLE_SOURCE, ENSEMBLE_FORMULA_VERSION, ENSEMBLE_WEIGHTS


def load_dashboard_functions():
    path = Path(__file__).resolve().parents[1] / 'dashboard/inference.py'
    tree = ast.parse(path.read_text())
    wanted = {'_combine_available_model_predictions', 'analyze_ticker_on_demand', '_get_history', '_parse_scalar_prob'}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    import time
    import threading
    namespace = {
        'logging': logging, 'sentiment_scores': sentiment_scores,
        'finbert_signal': finbert_signal, 'finbert_signal_text': finbert_signal_text, 'weighted_ensemble': weighted_ensemble,
        'ENSEMBLE_SOURCE': ENSEMBLE_SOURCE, 'ENSEMBLE_FORMULA_VERSION': ENSEMBLE_FORMULA_VERSION,
        'datetime': datetime, 'timezone': timezone, 'math': __import__('math'),
        'time': time, 'threading': threading, 'pd': pd,
        'get_s3_client': lambda: None, 'BUCKET_NAME': 'test',
        'predict_signal': lambda *args: (None, None),
        'ThreadPoolExecutor': ThreadPoolExecutor, 'wait': wait,
        '_MODEL_EXECUTOR': ThreadPoolExecutor(max_workers=6),
        '_GLOBAL_FETCH_LOCK': threading.Lock(),
        '_STRIPED_LOCKS': [threading.Lock() for _ in range(16)],
        '_NUM_STRIPES': 16,
        '_FETCH_CACHE': {},
        'FETCH_CACHE_TTL': 3600,
        'MAX_FETCH_CACHE': 128,
        '_MODEL_CACHE': {'finbert': {'sentiments': {}}},
        'TICKERS_META': {},
        'EXTENDED_TICKERS_META': {'META': {'name': 'Meta', 'sector': 'Technology'}},
        'SECTOR_TRANSLATION': {'Technology': 'Công nghệ'},
        '_ON_DEMAND_PREDICTIONS': {},
        '_CHART_CACHE': {},
        '_CACHE': {'market': {'data': None}},
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


class ModelAvailabilityTests(unittest.TestCase):
    def test_only_available_models_contribute_to_ensemble(self):
        combine = load_dashboard_functions()['_combine_available_model_predictions']
        value, available = combine({'xgboost': None, 'lstm': 60.0, 'finbert': 40.0})
        self.assertEqual(value, 50.8)
        self.assertEqual(available, {'lstm': 60.0, 'finbert': 40.0})
        self.assertEqual(combine({'xgboost': None, 'lstm': None, 'finbert': None}), (None, {}))

    def run_analysis(self, xgb, lstm, bert):
        namespace = load_dashboard_functions()
        history = pd.DataFrame({'Close': [100.0, 101.0]})
        fake_yfinance = SimpleNamespace(Ticker=lambda ticker: SimpleNamespace(history=lambda *args, **kwargs: history))
        namespace['_predict_xgboost_real'] = lambda ticker, price: xgb
        namespace['_predict_lstm_real'] = lambda ticker: lstm
        namespace['_predict_finbert_real'] = lambda ticker: {'sentiment_positive': .4, 'sentiment_negative': .2, 'sentiment_neutral': .4, 'status': 'available', 'news_count': 3} if bert is not None else {}
        with patch.dict(sys.modules, {'yfinance': fake_yfinance}):
            result = namespace['analyze_ticker_on_demand']('META')
        return result, namespace

    def test_on_demand_uses_another_model_without_inventing_missing_scores(self):
        result, _ = self.run_analysis(None, 60.0, 40.0)
        self.assertEqual(result['status'], 'success')
        pred = result['prediction']
        self.assertIsNone(pred['prob_xgb'])
        self.assertEqual(pred['pp1_xgboost'], 'Không khả dụng')
        self.assertEqual(pred['prob_lstm'], 60.0)
        self.assertEqual(pred['prob_bert'], 60.0)
        self.assertEqual(pred['finbert_sentiment']['sentiment_positive'], .4)
        self.assertEqual(pred['prob_ensemble'], 60.0)
        self.assertIsNone(pred['plan_target'])
        self.assertIsNone(pred['plan_stop_loss'])
        self.assertIsNone(pred['plan_rr'])
        self.assertEqual(pred['real_models_used'], ['LSTM', 'FinBERT'])

    def test_sentiment_remains_available_when_price_models_fail(self):
        result, _ = self.run_analysis(None, None, 40.0)
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['prediction']['real_models_used'], ['FinBERT'])
        self.assertEqual(result['prediction']['trang_thai'], 'Chỉ có FinBERT khả dụng')
        self.assertEqual(result['prediction']['prob_ensemble'], 60.0)

    def test_on_demand_reports_unavailable_when_all_models_fail(self):
        result, namespace = self.run_analysis(None, None, None)
        self.assertEqual(result['status'], 'error')
        self.assertIn('mô hình AI khả dụng', result['message'])
        self.assertEqual(namespace['_ON_DEMAND_PREDICTIONS'], {})


class EnsembleDagAvailabilityTests(unittest.TestCase):
    def test_dag_uses_surviving_models_without_fifty_percent_placeholder(self):
        path = Path(__file__).resolve().parents[1] / 'dags/dag_ensemble_master.py'
        tree = ast.parse(path.read_text())
        node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == 'compare_and_ensemble_task')
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d')

        class MemoryS3:
            def __init__(self):
                self.objects = {
                    f'lstm/du_bao_tang_truong_lstm_{stamp}.csv':
                        'ma_co_phieu,nhom_nganh,khuyen_nghi,xac_suat_tang_gia,prob_num,issued_at,reference_date,inference_source,model_version\nAAPL,Technology,MUA,60.00%,0.6,2026-09-30T22:00:00Z,2026-09-30,trained_model_v2,v1\n',
                    f'finbert/du_bao_tang_truong_finbert_{stamp}.csv':
                        'ma_co_phieu,nhom_nganh,khuyen_nghi,xac_suat_tang_gia,prob_num,issued_at,reference_date,inference_source,model_version,news_policy,news_count,sentiment_model_version,sentiment_positive,sentiment_negative,sentiment_neutral\nAAPL,Technology,BÁN,40.00%,0.4,2026-09-30T22:00:00Z,2026-09-30,trained_model_v2,v1,observed_news_v1,2,ProsusAI/finbert,0.2,0.6,0.2\n',
                }

            def get_conn(self):
                return None

            def list_keys(self, bucket_name, prefix):
                return [key for key in self.objects if key.startswith(prefix)]

            def read_key(self, key, bucket_name):
                return self.objects[key]

            def load_string(self, string_data, key, bucket_name, replace):
                self.objects[key] = string_data

        s3 = MemoryS3()
        namespace = {
            'S3Hook': lambda aws_conn_id: s3,
            'MINIO_CONN_ID': 'test', 'BUCKET_NAME': 'test',
            'datetime': datetime, 'timezone': timezone, 'math': __import__('math'),
        'get_s3_client': lambda: None, 'BUCKET_NAME': 'test',
        'predict_signal': lambda *args: (None, None), 'timedelta': __import__('datetime').timedelta, 'pd': pd, 'np': np, 'io': io, 'logging': logging,
            'SECTOR_MAP': {'AAPL': 'Technology'},
            'TELEGRAM_TOKEN': '', 'TELEGRAM_CHAT_ID': '',
            'requests': SimpleNamespace(post=lambda *args, **kwargs: None),
        }
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
        fake_charts = SimpleNamespace(
            generate_individual_ensemble_charts=lambda **kwargs: {},
            save_charts_zip_minio_and_telegram=lambda **kwargs: None,
            load_price_data_from_minio=lambda *args: pd.DataFrame(),
            calculate_trading_plan=lambda *args: {},
        )
        with patch.dict(sys.modules, {'chart_utils': fake_charts}):
            namespace['compare_and_ensemble_task']()

        key = f'comparison/bang_so_sanh_4_phuong_phap_{stamp}.csv'
        comparison = pd.read_csv(io.StringIO(s3.objects[key]))
        row = comparison.iloc[0]
        self.assertEqual(row['pp1_xgboost'], 'Không khả dụng')
        self.assertEqual(row['pp2_lstm'], 'MUA (60.00%)')
        self.assertEqual(row['pp3_finbert'], 'TIÊU CỰC (30.00%)')
        self.assertEqual(row['sentiment_negative'], .6)
        self.assertEqual(row['pp4_tong_hop'], 'ĐỨNG NGOÀI (46.15%)')
        self.assertEqual(row['ensemble_model_version'], 'weighted_35_35_30_v1')
        # Legacy sentiment is excluded, while surviving price models remain.
        news_key = f'finbert/du_bao_tang_truong_finbert_{stamp}.csv'
        legacy = pd.read_csv(io.StringIO(s3.objects[news_key])).drop(columns='news_policy')
        s3.objects[news_key] = legacy.to_csv(index=False)
        with patch.dict(sys.modules, {'chart_utils': fake_charts}):
            namespace['compare_and_ensemble_task']()
        row = pd.read_csv(io.StringIO(s3.objects[key])).iloc[0]
        self.assertEqual(row['pp2_lstm'], 'MUA (60.00%)')
        self.assertEqual(row['pp3_finbert'], 'Không khả dụng')
        self.assertEqual(row['pp4_tong_hop'], 'MUA (60.00%)')
