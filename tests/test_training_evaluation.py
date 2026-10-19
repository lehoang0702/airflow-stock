import sys
import os
import ast
import types
from unittest.mock import patch
import pytest
import pandas as pd
import numpy as np

# Add dags/ to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dags')))

from model_registry import XGBoostModelBuilder, LSTMModelBuilder

dag_path = os.path.join(os.path.dirname(__file__), '..', 'dags', 'dag_model_evaluation.py')
with open(dag_path, 'r') as f:
    tree = ast.parse(f.read())

fn_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'evaluation_temporal_masks')
fn_code = compile(ast.Module(body=[fn_node], type_ignores=[]), '<ast>', 'exec')
ns = {}
exec(fn_code, ns)
dag_model_evaluation = types.SimpleNamespace(evaluation_temporal_masks=ns['evaluation_temporal_masks'])


def test_evaluation_temporal_masks_metadata():
    df = pd.DataFrame({
        'Date_Std': pd.date_range('2023-01-01', periods=10),
        'Ticker_Std': ['AAPL'] * 10,
        'Target_Std': [1] * 10
    })

    # Rolling metadata triggers rolling_temporal_masks
    hyperparameters = {
        'split_policy': 'rolling_v1',
        'train_start': '2023-01-01',
        'validation_start': '2023-01-05',
        'test_start': '2023-01-08',
        'test_end': '2023-01-10',
        'prediction_horizon': 1
    }

    with patch('pipeline_data.rolling_temporal_masks') as mock_rolling:
        mock_rolling.return_value = (
            pd.Series([True]*5 + [False]*5),
            pd.Series([False]*5 + [True]*2 + [False]*3),
            pd.Series([False]*7 + [True]*3),
            hyperparameters
        )
        train_mask, test_mask = dag_model_evaluation.evaluation_temporal_masks(df, hyperparameters)
        assert mock_rolling.called
        assert train_mask.sum() == 5
        assert test_mask.sum() == 3

    # Legacy: no metadata (or only prediction_horizon) delegates to temporal_masks
    with patch('pipeline_data.temporal_masks') as mock_temporal:
        mock_temporal.return_value = (pd.Series([True]*5 + [False]*5), pd.Series([False]*5 + [True]*5))
        train_mask, test_mask = dag_model_evaluation.evaluation_temporal_masks(df, {'prediction_horizon': 1})
        assert mock_temporal.called

    # Fails closed on incomplete/unknown configuration
    incomplete_meta = {
        'split_policy': 'rolling_v1',
        'train_start': '2023-01-01'
        # missing other fields
    }
    with pytest.raises(ValueError, match="Incomplete/unknown"):
        dag_model_evaluation.evaluation_temporal_masks(df, incomplete_meta)


def test_evaluation_temporal_masks_saved_boundary_replay():
    def make_df(n):
        return pd.DataFrame({
            'Date_Std': pd.date_range('2023-01-01', periods=n),
            'Ticker_Std': ['AAPL'] * n,
            'Target_Std': [1.0] * n
        })
    df_orig = make_df(10)
    from pipeline_data import rolling_temporal_masks
    _, _, test_mask_orig, meta = rolling_temporal_masks(df_orig, train_years=1, validation_sessions=2, test_sessions=3, horizon=1)
    
    df_extended = make_df(20)
    train_mask, test_mask = dag_model_evaluation.evaluation_temporal_masks(df_extended, meta)
    
    assert df_extended.loc[test_mask, 'Date_Std'].max() <= pd.Timestamp(meta['test_end'])
    assert (df_extended.loc[test_mask, 'Date_Std'] == df_orig.loc[test_mask_orig, 'Date_Std']).all()


def test_registry_cross_sectional_features():
    df = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=5),
        'symbol': ['AAPL'] * 5,
        'close': [100.0, 101.0, 102.0, 101.5, 103.0],
        'volume': [1000] * 5,
        'log_return': [0.01] * 5,
        'rsi_14': [50.0] * 5,
        'volume_sma_ratio': [1.0] * 5,
        'ret_close_sma5': [0.02] * 5,
        'macd_hist': [0.5] * 5
    })
    xgb_builder = XGBoostModelBuilder()
    df_out, cols = xgb_builder.preprocess(df.copy())
    assert any('cs_' in c for c in cols)

    lstm_builder = LSTMModelBuilder()
    df_out2, cols2 = lstm_builder.preprocess(df.copy())
    assert any('cs_' in c for c in cols2)


def test_ast_evaluation_dag():
    dag_path = os.path.join(os.path.dirname(__file__), '..', 'dags', 'dag_model_evaluation.py')
    with open(dag_path, 'r') as f:
        tree = ast.parse(f.read())

    # Ensure evaluation_temporal_masks exists
    func_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    assert 'evaluation_temporal_masks' in func_names

    # Check for batch size in LSTM evaluation and time-series training subset
    source = open(dag_path).read()
    assert "DataLoader(" in source
    assert "df_train = df_processed[train_mask" in source
    assert "builder.model_card.metrics = {**builder.model_card.metrics" in source
