"""Bounded, local-only T+1 trial of simpler classifiers on frozen prices.

All recipes and thresholds are selected before test predictions are generated.
These are retrospective windows, including dates examined in earlier research.
No serving/MLflow writes, network calls, alerts, or auto-promotion are performed.
"""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'dags'), str(ROOT)]
from model_training import select_decision_threshold
from research_data import CORE, REGIME, prepare_shared_panel, walk_forward_splits
from research_improvements import independent_stopping_masks, select_generalizing_candidate
from scripts.improve_models import candidate_specs, fit_candidate, pooled_metrics
from scripts.walk_forward_benchmark import (fit_candidate as fit_reference, metrics,
    paired_block_intervals, probability_diagnostics, runtime_provenance)


def linear_model(c=.01):
    # Scaling is fitted on the fit rows only; clipping limits extreme observations.
    return make_pipeline(StandardScaler(),
        FunctionTransformer(np.clip, kw_args={'a_min': -5., 'a_max': 5.}),
        LogisticRegression(C=c, max_iter=1000, random_state=42))


def stable_features(panel, columns, fit, limit=8):
    """Retain coefficients with the same sign in three disjoint train periods.

    Each block fits its own scaler and classifier. Boundary labels whose outcome
    belongs to the next block are purged. No validation/test values are inspected.
    An empty stable set is explicitly allowed and falls back to the train prior.
    """
    if not panel.index.equals(fit.index) or panel.index.has_duplicates:
        raise ValueError('Aligned unique indices required')
    dates = np.sort(panel.loc[fit, 'Date_Std'].unique())
    if len(dates) < 60 or limit < 1:
        raise ValueError('At least 60 fit sessions and a positive limit required')
    outcomes = panel.sort_values(['Ticker_Std', 'Date_Std']).groupby('Ticker_Std').Date_Std.shift(-1).reindex(panel.index)
    coefficients, blocks = [], []
    for dates_block in np.array_split(dates, 3):
        mask = fit & panel.Date_Std.isin(dates_block) & outcomes.le(pd.Timestamp(dates_block[-1]))
        if panel.loc[mask, 'Target_Std'].nunique() != 2:
            return [], {'reason': 'single_class_train_block', 'blocks': blocks}
        model = linear_model(.1).fit(panel.loc[mask, columns], panel.loc[mask, 'Target_Std'])
        coefficients.append(model[-1].coef_[0])
        blocks.append({'start': str(pd.Timestamp(dates_block[0])),
                       'end': str(pd.Timestamp(dates_block[-1])), 'rows': int(mask.sum())})
    values = np.array(coefficients)
    agrees = (values > 1e-5).all(axis=0) | (values < -1e-5).all(axis=0)
    scores = np.min(np.abs(values), axis=0)
    ordered = sorted(np.flatnonzero(agrees), key=lambda i: (-scores[i], columns[i]))[:limit]
    selected = [columns[i] for i in ordered]
    return selected, {'blocks': blocks, 'selected': selected,
        'coefficients': {name: values[:, i].tolist() for i, name in enumerate(columns)},
        'rule': 'same nonzero sign in all three train blocks; rank by minimum absolute coefficient',
        'max_features': limit}


def fit_linear(panel, columns, fit, validation, name, stability=None):
    if columns:
        model = linear_model().fit(panel.loc[fit, columns], panel.loc[fit, 'Target_Std'])
    else:
        model = DummyClassifier(strategy='prior').fit(np.zeros((int(fit.sum()), 1)), panel.loc[fit, 'Target_Std'])
    def predict(mask):
        index = panel.index[mask].to_numpy()
        X = panel.loc[mask, columns] if columns else np.zeros((len(index), 1))
        return index, model.predict_proba(X)[:, 1]
    record = describe(panel, validation, predict, name)
    record.update(features=columns, stability=stability, fit_rows=int(fit.sum()),
                  parameters={'C': .01, 'clip_standardized': [-5., 5.]})
    return record, predict


def describe(panel, validation, predict, name):
    index, p = predict(validation)
    threshold = select_decision_threshold(panel.loc[index, 'Target_Std'], p,
                                          dates=panel.loc[index, 'Date_Std'])
    return {'spec': {'name': name}, 'validation': metrics(panel.loc[index, 'Target_Std'], p, threshold)}


def average_predictors(first, second):
    def predict(mask):
        index, p = first(mask)
        other, q = second(mask)
        if not np.array_equal(index, other):
            raise ValueError('Blend observations must align')
        return index, .5 * p + .5 * q
    return predict


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def summaries(frame, names):
    return {name: pooled_metrics(frame, name) for name in names}


def run(snapshot, output, end='2026-10-02', folds=5, control_trials=30):
    if (output / 'results.json').exists():
        raise ValueError('Use a new output directory to preserve the previous run')
    output.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(snapshot)
    # Exclude intraday/unclosed observations before recomputing features/labels.
    raw = raw[pd.to_datetime(raw.Date).le(pd.Timestamp(end))].copy()
    panel, groups = prepare_shared_panel(raw)
    splits = list(walk_forward_splits(panel, folds=folds, test_sessions=63,
                                     validation_sessions=63, end=end, train_years=3))
    names = ['reference', 'control', 'linear_core', 'linear_stable', 'additive', 'blend', 'selected', 'majority']
    report = {'status': 'training', 'created_at': pd.Timestamp.now(tz='UTC').isoformat(),
        'source': {'path': str(snapshot), 'sha256': hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                   'last_allowed_outcome': end},
        'runtime': runtime_provenance(),
        'protocol': {'folds': folds, 'test_sessions': 63, 'validation_sessions': 63,
            'stopping_sessions': 63, 'train_years': 3, 'seeds': [42, 7],
            'control_trials': control_trials, 'threshold_policy': 'existing validation-only conservative accuracy rule',
            'candidate_names': names, 'min_validation_auc_gain': .005, 'max_validation_logloss_increase': .005,
            'reference': 'current XGBoost tuning recipe, full reconstructed features, outer validation early stopping',
            'control': 'fixed full XGBoost, independent stopping tail, two seeds',
            'selection': 'validation AUC gain >= .005 and logloss degradation <= .005 versus control',
            'interpretation': 'retrospective exploratory; same frozen OHLCV/labels for all models; no live promotion'},
        'folds': []}
    report['runtime']['source_sha256']['scripts/compact_model_trial.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['runtime']['source_sha256']['scripts/improve_models.py'] = hashlib.sha256((ROOT / 'scripts/improve_models.py').read_bytes()).hexdigest()
    report['runtime']['source_sha256']['dags/research_improvements.py'] = hashlib.sha256((ROOT / 'dags/research_improvements.py').read_bytes()).hexdigest()
    atomic_json(output / 'results.json', report)
    fitted = []
    for number, (train, validation, test, metadata) in enumerate(splits, 1):
        logging.info('Fold %s/%s: fitting current XGBoost reference', number, folds)
        reference, ref_predict, _, _ = fit_reference('xgboost', panel, groups, metadata,
            {'name': 'reference', 'group': 'full', 'years': 3, 'lookback': 1}, control_trials, 1, [42], diagnostics=True)
        fit, stop, stop_metadata = independent_stopping_masks(panel, train, 63)
        records, predictors = [], []
        control_spec = candidate_specs('xgboost')[0]
        control, control_predict = fit_candidate('xgboost', panel, groups['xgboost']['full'],
            fit, stop, validation, control_spec, [42, 7], 1, 1)
        control['spec']['name'] = 'control'
        records.append(control)
        predictors.append(control_predict)
        logging.info('Fold %s: fitting compact candidates', number)
        chosen, stability = stable_features(panel, CORE + REGIME, fit)
        for name, columns, detail in [('linear_core', CORE, None), ('linear_stable', chosen, stability)]:
            record, predict = fit_linear(panel, columns, fit, validation, name, detail)
            records.append(record)
            predictors.append(predict)
        spec = {'name': 'additive', 'group': 'enriched', 'parameters': {
            'n_estimators': 400, 'max_depth': 1, 'learning_rate': .03,
            'subsample': .8, 'colsample_bytree': 1., 'min_child_weight': 30,
            'reg_alpha': .1, 'reg_lambda': 10., 'n_jobs': 2, 'tree_method': 'hist'}}
        record, predict = fit_candidate('xgboost', panel, CORE + REGIME, fit, stop,
                                       validation, spec, [42, 7], 1, 1)
        records.append(record)
        predictors.append(predict)
        blend = average_predictors(predictors[2], predictors[3])
        record = describe(panel, validation, blend, 'blend')
        record['parameters'] = {'linear_stable_weight': .5, 'additive_weight': .5}
        records.append(record)
        predictors.append(blend)
        for record, predict in zip(records, predictors):
            index, p = predict(fit)
            record['training'] = metrics(panel.loc[index, 'Target_Std'], p, record['validation']['decision_threshold'])
        selected = select_generalizing_candidate(records)
        fold = {'fold': number, 'split': metadata, 'stopping_split': stop_metadata,
            'reference': reference, 'candidates': records, 'selected_index': selected,
            'selection': records[selected]['spec']['name'],
            'train_prior': float(panel.loc[train, 'Target_Std'].mean())}
        report['folds'].append(fold)
        fitted.append((test, ref_predict, predictors))
        # Persist choices without using or reporting any test metrics.
        atomic_json(output / 'results.json', report)
        logging.info('Fold %s selection frozen: %s; stable features: %s', number, fold['selection'], chosen)
    report['status'] = 'selections_frozen'
    atomic_json(output / 'selection_manifest.json', report)
    report['selection_manifest_sha256'] = hashlib.sha256((output / 'selection_manifest.json').read_bytes()).hexdigest()
    frames = []
    for fold, (test, reference, predictors) in zip(report['folds'], fitted):
        frame = panel.loc[test, ['Date_Std', 'Ticker_Std', 'Target_Std']].rename(
            columns={'Date_Std': 'Date', 'Ticker_Std': 'Ticker', 'Target_Std': 'y'}).copy()
        frame['fold'] = fold['fold']
        candidates = [(fold['reference'], reference)] + list(zip(fold['candidates'], predictors))
        candidates.append((fold['candidates'][fold['selected_index']], predictors[fold['selected_index']]))
        fold['test'] = {}
        for name, (record, predict) in zip(names[:-1], candidates):
            index, p = predict(test)
            if not np.array_equal(index, frame.index.to_numpy()):
                raise ValueError('Every candidate must predict every identical test row')
            threshold = record['validation']['decision_threshold']
            frame[f'{name}_probability'], frame[f'{name}_label'] = p, (p >= threshold).astype(int)
            fold['test'][name] = metrics(frame.y, p, threshold)
            fold['test'][name]['diagnostics'] = probability_diagnostics(frame.y, p, threshold)
        prior = fold['train_prior']
        frame['majority_probability'], frame['majority_label'] = prior, int(prior >= .5)
        fold['test']['majority'] = metrics(frame.y, np.full(len(frame), prior))
        frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)
    if predictions.duplicated(['Date', 'Ticker']).any():
        raise ValueError('Test windows overlap')
    predictions.to_csv(output / 'predictions.csv', index=False)
    report['summary'] = summaries(predictions, names)
    report['recent_two_folds'] = summaries(predictions[predictions.fold > folds - 2], names)
    report['uncertainty'] = {}
    for comparator in ('reference', 'control', 'majority'):
        comparison = predictions.copy()
        comparison['reference_probability'] = comparison[f'{comparator}_probability']
        comparison['reference_label'] = comparison[f'{comparator}_label']
        report['uncertainty'][f'selected_vs_{comparator}'] = paired_block_intervals(comparison)
    report['status'] = 'complete'
    atomic_json(output / 'results.json', report)
    logging.info('Completed. Summary: %s', report['summary'])
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--end', default='2026-10-02')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--control-trials', type=int, default=30)
    args = parser.parse_args()
    if args.folds < 2 or args.control_trials < 1:
        parser.error('At least two folds and one tuning trial required')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    run(args.snapshot, args.output, args.end, args.folds, args.control_trials)
