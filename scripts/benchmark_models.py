"""Compare saved incumbents with validation-selected challengers without publishing.

Run in the project's Python environment with access to MinIO:
    PYTHONPATH=dags:. python scripts/benchmark_models.py --output output/model_review.json
"""
import argparse
import hashlib
import io
import json
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dags'))
from model_registry import XGBoostModelBuilder, LSTMModelBuilder
from pipeline_data import rolling_temporal_masks, create_sequences
from model_training import (recency_weights, select_decision_threshold,
                            select_lstm_configuration, predict_lstm_loader)


def metrics(y, probabilities, threshold=0.5):
    return {'accuracy': float(accuracy_score(y, probabilities >= threshold)),
            'accuracy_at_0_5': float(accuracy_score(y, probabilities >= 0.5)),
            'balanced_accuracy': float(balanced_accuracy_score(y, probabilities >= threshold)),
            'auc_roc': float(roc_auc_score(y, probabilities)) if len(np.unique(y)) == 2 else None,
            'logloss': float(log_loss(y, probabilities, labels=[0, 1])),
            'decision_threshold': float(threshold), 'samples': len(y)}


def load_inputs(s3, bucket, kind):
    def keys(prefix):
        return [item['Key'] for page in s3.get_paginator('list_objects_v2').paginate(
            Bucket=bucket, Prefix=prefix) for item in page.get('Contents', [])]
    found_keys = (keys(f'{kind}/{kind}_stock_100tickers_5y_')
                  or keys(f'{kind}/{kind}_stock_20tickers_5y_')
                  or keys(f'{kind}/{kind}_stock_20tickers_10y_'))
    data_key = sorted(key for key in found_keys if key.endswith('.csv'))[-1]
    card_key = sorted(key for key in keys(f'models/{kind}/')
                      if key.endswith('/model_card.json'))[-1]
    def read(key):
        return s3.get_object(Bucket=bucket, Key=key)['Body'].read()
    card = json.loads(read(card_key))
    data_key = card['hyperparameters'].get('dataset_key', data_key)
    dataset_bytes = read(data_key)
    digest = hashlib.sha256(dataset_bytes).hexdigest()
    recorded_digest = card['hyperparameters'].get('dataset_sha256')
    if recorded_digest and digest != recorded_digest:
        raise ValueError('Recorded training data changed; benchmark cannot replay this artifact')
    frame = pd.read_csv(io.BytesIO(dataset_bytes))
    prefix = card_key.rsplit('/', 1)[0]
    columns = json.loads(read(f'{prefix}/feature_columns.json'))
    return frame, card, columns, read, prefix, data_key, digest


def benchmark_xgboost(inputs, trials):
    import xgboost as xgb
    from hyperparameter_tuner import tune_xgboost_hyperparameters
    raw, card, old_columns, read, prefix, data_key, digest = inputs
    builder = XGBoostModelBuilder()
    old_df, _ = builder.preprocess(raw.copy(), feature_policy=card['hyperparameters'].get('feature_policy', 'legacy'))
    _, _, old_test, _ = rolling_temporal_masks(old_df, split_metadata=card['hyperparameters'])
    old = xgb.XGBClassifier()
    old.load_model(bytearray(read(f'{prefix}/model.json')))
    baseline_p = old.predict_proba(old_df.loc[old_test, old_columns])[:, 1]

    df, columns = builder.preprocess(raw.copy(), feature_policy='stationary_v2')
    train, validation, test, split = rolling_temporal_masks(df, split_metadata=card['hyperparameters'])
    params = tune_xgboost_hyperparameters(df.loc[train, columns], df.loc[train, 'Target_Std'],
                                         n_trials=trials, dates=df.loc[train, 'Date_Std'])
    half_life = params.pop('recency_half_life', 0)
    params.update(eval_metric='auc', early_stopping_rounds=20)
    model = xgb.XGBClassifier(**params)
    model.fit(df.loc[train, columns], df.loc[train, 'Target_Std'],
              sample_weight=recency_weights(df.loc[train, 'Date_Std'], half_life),
              eval_set=[(df.loc[validation, columns], df.loc[validation, 'Target_Std'])], verbose=False)
    validation_p = model.predict_proba(df.loc[validation, columns])[:, 1]
    threshold = select_decision_threshold(df.loc[validation, 'Target_Std'], validation_p,
                                           dates=df.loc[validation, 'Date_Std'])
    # All choices are frozen before the first challenger test prediction.
    y = df.loc[test, 'Target_Std'].to_numpy()
    assert np.array_equal(y, old_df.loc[old_test, 'Target_Std'].to_numpy())
    probabilities = model.predict_proba(df.loc[test, columns])[:, 1]
    return {'dataset': data_key, 'dataset_sha256': digest,
            'recorded_baseline_metrics': card['metrics'],
            'baseline_dataset_verified': bool(card['hyperparameters'].get('dataset_sha256')),
            'incumbent': card['model_version'], 'split': split,
            'baseline': metrics(y, baseline_p, card['hyperparameters'].get('decision_threshold', 0.5)),
            'challenger': metrics(y, probabilities, threshold),
            'validation': metrics(df.loc[validation, 'Target_Std'].to_numpy(), validation_p, threshold),
            'selected_parameters': {**params, 'recency_half_life': half_life,
                                    'best_iteration': model.best_iteration},
            'features': columns}


def benchmark_lstm(inputs, epochs):
    import pickle
    import torch
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import DataLoader, TensorDataset
    torch.set_num_threads(2)
    raw, card, old_columns, read, prefix, data_key, digest = inputs
    hp = {**LSTMModelBuilder.DEFAULT_HYPERPARAMS, **card['hyperparameters']}
    builder = LSTMModelBuilder(hp)
    old_df, _ = builder.preprocess(raw.copy(), feature_policy=hp.get('feature_policy', 'legacy'))
    _, _, old_test, _ = rolling_temporal_masks(old_df, split_metadata=hp)
    old_scaler = pickle.loads(read(f'{prefix}/scaler.pkl'))
    old_df[old_columns] = old_scaler.transform(old_df[old_columns].values)
    old_X, old_y = create_sequences(old_df, old_columns, hp['lookback'], mask=old_test)
    old = builder.deserialize_model(read(f'{prefix}/model.pt'), len(old_columns))
    baseline_p = predict_lstm_loader(old, DataLoader(TensorDataset(torch.tensor(old_X)), batch_size=256))
    del old, old_X, old_df

    df, columns = builder.preprocess(raw.copy(), feature_policy='stationary_v2')
    train, validation, test, split = rolling_temporal_masks(df, split_metadata=hp)
    scaler = StandardScaler().fit(df.loc[train, columns].values)
    df[columns] = scaler.transform(df[columns].values)
    X_train, y_train = create_sequences(df, columns, hp['lookback'], mask=train)
    X_val, y_val, val_indices = create_sequences(df, columns, hp['lookback'], mask=validation, return_indices=True)
    model, config, validation_p, history = select_lstm_configuration(
        builder._build_model_class(), X_train, y_train, X_val, y_val, epochs=epochs)
    threshold = select_decision_threshold(y_val, validation_p, dates=df.loc[val_indices, 'Date_Std'])
    # Test windows are created only after model and threshold selection.
    X_test, y_test = create_sequences(df, columns, hp['lookback'], mask=test)
    assert np.array_equal(y_test, old_y)
    probabilities = predict_lstm_loader(model, DataLoader(
        TensorDataset(torch.tensor(X_test)), batch_size=256))
    return {'dataset': data_key, 'dataset_sha256': digest,
            'recorded_baseline_metrics': card['metrics'],
            'baseline_dataset_verified': bool(card['hyperparameters'].get('dataset_sha256')),
            'incumbent': card['model_version'], 'split': split,
            'baseline': metrics(y_test, baseline_p, hp.get('decision_threshold', 0.5)),
            'challenger': metrics(y_test, probabilities, threshold),
            'validation': metrics(y_val, validation_p, threshold),
            'selected_parameters': config, 'validation_search': history, 'features': columns}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=['xgboost', 'lstm'], default=['xgboost', 'lstm'])
    parser.add_argument('--trials', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    from dashboard.storage import get_s3_client
    s3 = get_s3_client()
    report = {'created_at': pd.Timestamp.now(tz='UTC').isoformat(),
              'selection': 'training CV / validation only; same saved holdout; no publication',
              'models': {}}
    for kind in args.models:
        inputs = load_inputs(s3, 'stock-data', kind)
        result = (benchmark_xgboost(inputs, args.trials) if kind == 'xgboost'
                  else benchmark_lstm(inputs, args.epochs))
        result['delta'] = {metric: result['challenger'][metric] - result['baseline'][metric]
                           for metric in ('accuracy', 'auc_roc', 'balanced_accuracy', 'logloss')
                           if result['baseline'][metric] is not None and result['challenger'][metric] is not None}
        report['models'][kind] = result
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(json.dumps({kind: {key: result[key] for key in ('baseline', 'challenger', 'delta')}}), flush=True)


if __name__ == '__main__':
    main()
