"""Read-only paired research benchmark; outputs reports, never serving artifacts.

Uses the LSTM crawler's adjusted OHLCV as the shared source for BOTH models.
Historical backtests are exploratory, including periods used in prior training.
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
from pipeline_data import create_sequences, rolling_temporal_masks
from research_data import prepare_shared_panel, audit_original_labels, walk_forward_splits, PRICE_POLICY
from model_training import recency_weights, select_decision_threshold, select_lstm_configuration, predict_lstm_loader


def metrics(y, probability, threshold=0.5):
    y, probability = np.asarray(y), np.asarray(probability)
    return {'accuracy': float(accuracy_score(y, probability >= threshold)),
            'accuracy_at_0_5': float(accuracy_score(y, probability >= .5)),
            'balanced_accuracy': float(balanced_accuracy_score(y, probability >= threshold)),
            'auc_roc': float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else None,
            'logloss': float(log_loss(y, probability, labels=[0, 1])),
            'positive_ratio': float(y.mean()),
            'predicted_positive_ratio': float((probability >= threshold).mean()),
            'decision_threshold': float(threshold), 'samples': len(y)}


def candidate_specs(kind, years):
    reference = {'name': 'reference', 'group': 'full', 'years': 3, 'lookback': 30}
    candidates = [reference]
    if kind == 'xgboost':
        candidates.append({'name': 'core_3y', 'group': 'core', 'years': 3, 'lookback': 1})
    else:
        candidates.extend({'name': f'core_{lookback}d', 'group': 'core', 'years': 3,
                           'lookback': lookback} for lookback in (10, 20, 60))
    candidates.extend({'name': f'enriched_{year}y', 'group': 'enriched', 'years': year,
                       'lookback': 20} for year in years)
    return candidates


def fit_candidate(kind, panel, groups, metadata, spec, trials, epochs, seeds, diagnostics=False):
    """Returns a fitted predictor without accessing any test observations."""
    meta = dict(metadata)
    meta['train_start'] = (pd.Timestamp(meta['validation_start']) - pd.DateOffset(years=spec['years'])).isoformat()
    train, validation, _, _ = rolling_temporal_masks(panel, split_metadata=meta)
    fraction = spec.get('train_fraction', 1.)
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError('Training fraction must be in (0, 1]')
    train_dates = np.sort(panel.loc[train, 'Date_Std'].unique())
    if fraction < 1:
        retain = max(1, int(np.ceil(len(train_dates) * fraction)))
        train &= panel.Date_Std.ge(pd.Timestamp(train_dates[-retain]))
    columns = groups[kind][spec['group']]
    if panel.loc[train, 'Target_Std'].nunique() != 2:
        raise ValueError('Training requires both classes')
    if kind == 'xgboost':
        import xgboost as xgb
        from hyperparameter_tuner import tune_xgboost_hyperparameters
        params = tune_xgboost_hyperparameters(panel.loc[train, columns], panel.loc[train, 'Target_Std'],
                                             n_trials=trials, dates=panel.loc[train, 'Date_Std'])
        half_life = params.pop('recency_half_life', 0)
        params.update(eval_metric='auc', early_stopping_rounds=20)
        model = xgb.XGBClassifier(**params)
        model.fit(panel.loc[train, columns], panel.loc[train, 'Target_Std'],
                  sample_weight=recency_weights(panel.loc[train, 'Date_Std'], half_life),
                  eval_set=[(panel.loc[validation, columns], panel.loc[validation, 'Target_Std'])], verbose=False)
        val_indices = panel.index[validation].to_numpy()
        val_probability = model.predict_proba(panel.loc[validation, columns])[:, 1]
        def predict(mask):
            indices = panel.index[mask].to_numpy()
            return indices, model.predict_proba(panel.loc[mask, columns])[:, 1]
        config = {**params, 'recency_half_life': half_life, 'best_iteration': model.best_iteration}
    else:
        import torch
        from sklearn.preprocessing import StandardScaler
        from torch.utils.data import DataLoader, TensorDataset
        from model_registry import LSTMModelBuilder
        from model_training import LSTM_CANDIDATES
        scaler = StandardScaler().fit(panel.loc[train, columns].values)
        data = panel[['Ticker_Std', 'Date_Std', 'Target_Std']].copy()
        data[columns] = scaler.transform(panel[columns].values).astype(np.float32)
        X_train, y_train = create_sequences(data, columns, spec['lookback'], mask=train)
        X_val, y_val, val_indices = create_sequences(data, columns, spec['lookback'], mask=validation, return_indices=True)
        models, probabilities, configs = [], [], []
        for seed in seeds:
            model, config, probability, search = select_lstm_configuration(
                LSTMModelBuilder()._build_model_class(), X_train, y_train, X_val, y_val,
                candidates=None if spec['name'] == 'reference' else [LSTM_CANDIDATES[1]],
                epochs=epochs, patience=5, seed=seed, collect_history=diagnostics)
            models.append(model)
            probabilities.append(probability)
            configs.append({'seed': seed, 'selected': config, 'search': search})
        val_probability = np.mean(probabilities, axis=0)
        def predict(mask):
            X, _, indices = create_sequences(data, columns, spec['lookback'], mask=mask, return_indices=True)
            loader = DataLoader(TensorDataset(torch.as_tensor(X)), batch_size=256)
            return indices, np.mean([predict_lstm_loader(model, loader) for model in models], axis=0)
        config = {'seeds': configs}
    y_val = panel.loc[val_indices, 'Target_Std'].to_numpy()
    threshold = select_decision_threshold(y_val, val_probability, dates=panel.loc[val_indices, 'Date_Std'])
    record = {'spec': spec, 'features': columns, 'configuration': config,
              'train_samples': int(train.sum()), 'validation': metrics(y_val, val_probability, threshold)}
    if diagnostics:
        indices, train_p = predict(train)
        train_y = panel.loc[indices, 'Target_Std'].to_numpy()
        record['training'] = metrics(train_y, train_p, threshold)
        record['training_period'] = {'start': panel.loc[train, 'Date_Std'].min().isoformat(),
                                     'end': panel.loc[train, 'Date_Std'].max().isoformat(),
                                     'sessions': int(panel.loc[train, 'Date_Std'].nunique())}
        record['probability_diagnostics'] = {
            'training': probability_diagnostics(train_y, train_p, threshold),
            'validation': probability_diagnostics(y_val, val_probability, threshold)}
    return record, predict, val_indices, val_probability


def probability_diagnostics(y, probability, threshold=.5):
    from sklearn.metrics import confusion_matrix
    y, probability = np.asarray(y), np.asarray(probability)
    if len(y) != len(probability) or not len(y) or not np.isfinite(probability).all():
        raise ValueError('Probability diagnostics need aligned finite observations')
    if not np.isin(y, [0, 1]).all() or not ((probability >= 0) & (probability <= 1)).all():
        raise ValueError('Invalid binary labels or probabilities')
    bins = []
    for index in range(10):
        lower = index / 10
        upper = (index + 1) / 10
        mask = (probability >= lower) & ((probability < upper) if index < 9 else (probability <= upper))
        bins.append({'lower': float(lower), 'upper': upper, 'samples': int(mask.sum()),
                     'mean_probability': float(probability[mask].mean()) if mask.any() else None,
                     'observed_positive_ratio': float(y[mask].mean()) if mask.any() else None})
    return {'confusion_matrix': confusion_matrix(y, probability >= threshold, labels=[0, 1]).tolist(),
            'probability_quantiles': np.quantile(probability, [0, .05, .25, .5, .75, .95, 1]).tolist(),
            'probability_std': float(probability.std()), 'calibration_bins': bins}


def select_candidate(records, max_logloss_increase=.005):
    """Selection uses validation metrics only, with the reference always eligible."""
    baseline_loss = records[0]['validation']['logloss']
    eligible = [i for i, record in enumerate(records)
                if record['validation']['auc_roc'] is not None and
                record['validation']['logloss'] <= baseline_loss + max_logloss_increase]
    if not eligible:
        return 0
    return max(eligible, key=lambda i: (records[i]['validation']['auc_roc'],
                                        -records[i]['validation']['logloss'], -i))


def paired_block_intervals(predictions, seed=42, repeats=400, block_sessions=5):
    """Paired moving-block bootstrap, preserving same-date ticker dependence."""
    rng = np.random.default_rng(seed)
    pieces = [g.sort_values(['Date', 'Ticker']) for _, g in predictions.groupby('fold')]
    if any(g.Date.nunique() < 2 * block_sessions for g in pieces):
        return {'status': 'insufficient_dates'}
    deltas = {'accuracy': [], 'auc_roc': []}
    for _ in range(repeats):
        samples = []
        for group in pieces:
            dates = np.sort(group.Date.unique())
            starts = rng.integers(0, len(dates) - block_sessions + 1,
                                  size=int(np.ceil(len(dates) / block_sessions)))
            positions = np.concatenate([np.arange(start, start + block_sessions) for start in starts])[:len(dates)]
            rows = group.groupby('Date').indices
            samples.append(group.iloc[np.concatenate([rows[pd.Timestamp(dates[i])] for i in positions])])
        sample = pd.concat(samples)
        deltas['accuracy'].append(float((sample.selected_label == sample.y).mean() -
                                        (sample.reference_label == sample.y).mean()))
        if sample.y.nunique() == 2:
            deltas['auc_roc'].append(float(roc_auc_score(sample.y, sample.selected_probability) -
                                          roc_auc_score(sample.y, sample.reference_probability)))
    return {'method': 'paired moving-block bootstrap within folds; all tickers retained per date',
            'block_sessions': block_sessions, 'repeats': repeats,
            'delta_95pct': {key: np.quantile(values, [.025, .975]).tolist() if values else None
                            for key, values in deltas.items()}}


def run_fold(kind, panel, groups, split, fold, args):
    train, validation, test, meta = split
    records, reference, selected = [], None, None
    # fit_candidate fits and selects exclusively on train/validation masks.
    expected_val = set(panel.index[validation])
    for spec in candidate_specs(kind, args.train_years):
        logging.info('%s fold %s candidate %s', kind, fold, spec['name'])
        result, predictor, val_indices, val_p = fit_candidate(
            kind, panel, groups, meta, spec, args.trials, args.epochs, args.seeds)
        if set(val_indices) != expected_val:
            raise ValueError('Candidates must share every validation observation')
        records.append(result)
        if len(records) == 1:
            reference = (result, predictor)
        if select_candidate(records) == len(records) - 1:
            selected = (result, predictor)
    if selected is None:
        selected = reference
    # Freeze the selected spec before generating any held-out predictions.
    output = panel.loc[test, ['Date_Std', 'Ticker_Std', 'Sector', 'Target_Std']].rename(
        columns={'Date_Std': 'Date', 'Ticker_Std': 'Ticker', 'Target_Std': 'y'}).copy()
    fold_report = {'fold': fold, 'split': meta, 'selection': selected[0]['spec'], 'candidates': records}
    for name, (result, predictor) in [('reference', reference), ('selected', selected)]:
        indices, probability = predictor(test)
        if set(indices) != set(output.index):
            raise ValueError('Every model must predict the identical held-out observations')
        probability = pd.Series(probability, index=indices).loc[output.index].to_numpy()
        threshold = result['validation']['decision_threshold']
        output[f'{name}_probability'] = probability
        output[f'{name}_label'] = (probability >= threshold).astype(int)
        fold_report[name] = metrics(output.y, probability, threshold)
    p_train = float(panel.loc[train, 'Target_Std'].mean())
    output['majority_probability'] = p_train
    output['majority_label'] = int(p_train > .5)
    # Constant probabilities retain training prevalence and give AUC=0.5.
    fold_report['majority'] = metrics(output.y, np.full(len(output), p_train),
                                    .5 if p_train != .5 else np.nextafter(.5, 1.))
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    logistic = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=500, random_state=42))
    logistic.fit(panel.loc[train, groups[kind]['core']], panel.loc[train, 'Target_Std'])
    val_p = logistic.predict_proba(panel.loc[validation, groups[kind]['core']])[:, 1]
    threshold = select_decision_threshold(panel.loc[validation, 'Target_Std'], val_p,
                                         dates=panel.loc[validation, 'Date_Std'])
    p = logistic.predict_proba(panel.loc[test, groups[kind]['core']])[:, 1]
    output['logistic_probability'], output['logistic_label'] = p, (p >= threshold).astype(int)
    fold_report['logistic'] = metrics(output.y, p, threshold)
    output['fold'] = fold
    return fold_report, output


def summarize(predictions, folds):
    report = {}
    for name in ('reference', 'selected', 'majority', 'logistic'):
        result = metrics(predictions.y, predictions[f'{name}_probability'])
        result['accuracy'] = float((predictions[f'{name}_label'] == predictions.y).mean())
        result['balanced_accuracy'] = float(balanced_accuracy_score(predictions.y, predictions[f'{name}_label']))
        result['predicted_positive_ratio'] = float(predictions[f'{name}_label'].mean())
        result['decision_threshold'] = 'validation threshold per fold'
        aucs = [f[name]['auc_roc'] for f in folds if f[name]['auc_roc'] is not None]
        result['fold_auc_mean'] = float(np.mean(aucs)) if aucs else None
        report[name] = result
    report['paired_uncertainty'] = paired_block_intervals(predictions)
    for grouping in ('Ticker', 'Sector'):
        report[f'by_{grouping.lower()}'] = {}
        for value, group in predictions.groupby(grouping):
            result = metrics(group.y, group.selected_probability)
            result['accuracy'] = float((group.selected_label == group.y).mean())
            result['balanced_accuracy'] = float(balanced_accuracy_score(group.y, group.selected_label))
            result['predicted_positive_ratio'] = float(group.selected_label.mean())
            result['decision_threshold'] = 'validation threshold per fold'
            report[f'by_{grouping.lower()}'][str(value)] = result
    return report


def read_snapshot(s3, kind, key=None):
    if key is None:
        keys = [o['Key'] for page in s3.get_paginator('list_objects_v2').paginate(
            Bucket='stock-data', Prefix=f'{kind}/{kind}_stock_20tickers_10y_')
            for o in page.get('Contents', []) if o['Key'].endswith('.csv')]
        if not keys:
            raise ValueError(f'No {kind} snapshots')
        key = sorted(keys)[-1]
    raw = s3.get_object(Bucket='stock-data', Key=key)['Body'].read()
    return pd.read_csv(io.BytesIO(raw)), {'key': key, 'sha256': hashlib.sha256(raw).hexdigest()}


def runtime_provenance():
    from importlib.metadata import version
    root = Path(__file__).resolve().parents[1]
    files = ['scripts/walk_forward_benchmark.py', 'dags/research_data.py',
             'dags/model_training.py', 'dags/hyperparameter_tuner.py',
             'dags/pipeline_data.py', 'dags/feature_engineering.py', 'dags/model_registry.py']
    return {'python': sys.version,
            'packages': {name: version(name) for name in ('numpy', 'pandas', 'scikit-learn', 'torch', 'xgboost', 'optuna')},
            'source_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset-key')
    parser.add_argument('--audit-key')
    parser.add_argument('--end', required=True, help='Last allowed outcome date, fixed before running experiments')
    parser.add_argument('--folds', type=int, default=3)
    parser.add_argument('--test-sessions', type=int, default=63)
    parser.add_argument('--validation-sessions', type=int, default=63)
    parser.add_argument('--train-years', nargs='+', type=int, default=[1, 3, 5])
    parser.add_argument('--trials', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 7])
    parser.add_argument('--models', nargs='+', choices=['xgboost', 'lstm'], default=['xgboost', 'lstm'])
    args = parser.parse_args()
    if min(args.trials, args.epochs, *args.train_years) < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Positive budgets/windows and unique seeds are required')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    import torch
    torch.set_num_threads(2)
    from dashboard.storage import get_s3_client
    s3 = get_s3_client()
    raw, source = read_snapshot(s3, 'lstm', args.dataset_key)
    old, audit_source = read_snapshot(s3, 'xgboost', args.audit_key)
    panel, groups = prepare_shared_panel(raw)
    splits = list(walk_forward_splits(panel, args.folds, args.test_sessions,
                                     args.validation_sessions, args.end, max(3, *args.train_years)))
    # Reference uses three years; alternative windows change ONLY their train start.
    splits = [rolling_temporal_masks(panel, split_metadata={**split[3],
              'train_start': (pd.Timestamp(split[3]['validation_start']) - pd.DateOffset(years=3)).isoformat()})
              for split in splits]
    report = {'created_at': pd.Timestamp.now(tz='UTC').isoformat(), 'status': 'running',
              'interpretation': 'Exploratory historical backtest, not a pristine prospective test. No deployment.',
              'price_policy': PRICE_POLICY, 'source': source, 'audit_source': audit_source,
              'runtime': runtime_provenance(),
              'label_audit': audit_original_labels(panel, old),
              'protocol': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'models': {}}
    args.output.mkdir(parents=True, exist_ok=True)
    def save():
        path = args.output / 'results.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
        temporary.replace(path)
    save()
    for kind in args.models:
        folds, predictions = [], []
        report['models'][kind] = {'folds': folds}
        for number, split in enumerate(splits, start=1):
            fold, prediction = run_fold(kind, panel, groups, split, number, args)
            folds.append(fold)
            predictions.append(prediction)
            pd.concat(predictions).to_csv(args.output / f'{kind}_predictions.csv', index=False)
            save()
            logging.info('%s fold %s completed: %s', kind, number,
                         json.dumps({k: fold[k] for k in ('selection', 'reference', 'selected', 'majority', 'logistic')}))
        report['models'][kind]['summary'] = summarize(pd.concat(predictions), folds)
        save()
    report['status'] = 'complete'
    save()


if __name__ == '__main__':
    main()
