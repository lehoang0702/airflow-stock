"""Bounded XGBoost/LSTM research with untouched outer validation and test.

Trains on frozen common prices. Local reports only; no serving writes or alerts.
Historical windows are already examined and cannot establish prospective gains.
"""
import argparse
import gc
import hashlib
import json
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'dags'), str(ROOT)]
from pipeline_data import create_sequences
from model_training import fit_lstm_with_validation, predict_lstm_loader, select_decision_threshold
from research_data import prepare_shared_panel, walk_forward_splits, CORE, REGIME
from research_improvements import (add_context_features, independent_stopping_masks,
                                   select_generalizing_candidate)
from scripts.walk_forward_benchmark import metrics, paired_block_intervals, read_snapshot, runtime_provenance
from scripts.diagnose_models import inventory_news


def candidate_specs(kind):
    if kind == 'xgboost':
        from hyperparameter_tuner import DEFAULT_XGB_PARAMS
        control = {key: value for key, value in DEFAULT_XGB_PARAMS.items()
                   if key not in ('recency_half_life', 'random_state')}
        regularized = {**control, 'n_estimators': 500, 'max_depth': 2,
                       'learning_rate': .02, 'min_child_weight': 20, 'gamma': 1.,
                       'reg_alpha': 1., 'reg_lambda': 20.,
                       'subsample': .8, 'colsample_bytree': .8}
    else:
        control = {'hidden_dim': 48, 'num_layers': 2, 'dropout': .2,
                   'lr': .001, 'weight_decay': .001}
        regularized = {'hidden_dim': 16, 'num_layers': 1, 'dropout': 0.,
                       'lr': .0005, 'weight_decay': .01}
    return [{'name': 'full_control', 'group': 'full', 'parameters': control},
            {'name': 'full_regularized', 'group': 'full', 'parameters': regularized},
            {'name': 'context_regularized', 'group': 'context', 'parameters': regularized}]


def fit_candidate(kind, panel, columns, fit, stopping, validation, spec, seeds, epochs, lookback):
    """All fitted state depends only on fit/stopping, then select on validation."""
    params, models, checkpoints = spec['parameters'], [], []
    val_indices = panel.index[validation].to_numpy()
    if kind == 'xgboost':
        import xgboost as xgb
        for seed in seeds:
            model = xgb.XGBClassifier(**{**params, 'random_state': seed,
                                        'eval_metric': 'auc', 'early_stopping_rounds': 20})
            model.fit(panel.loc[fit, columns], panel.loc[fit, 'Target_Std'],
                      eval_set=[(panel.loc[stopping, columns], panel.loc[stopping, 'Target_Std'])], verbose=False)
            models.append(model)
            checkpoints.append({'seed': seed, 'best_iteration': int(model.best_iteration)})
        def predict(mask):
            indices = panel.index[mask].to_numpy()
            p = np.mean([m.predict_proba(panel.loc[indices, columns])[:, 1] for m in models], axis=0)
            return indices, p
    else:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.preprocessing import StandardScaler
        from model_registry import LSTMModelBuilder
        scaler = StandardScaler().fit(panel.loc[fit, columns])
        scaled = panel[['Ticker_Std', 'Date_Std', 'Target_Std']].copy()
        scaled[columns] = scaler.transform(panel[columns]).astype(np.float32)
        X_fit, y_fit = create_sequences(scaled, columns, lookback, mask=fit)
        X_stop, y_stop = create_sequences(scaled, columns, lookback, mask=stopping)
        stop_loader = DataLoader(TensorDataset(torch.as_tensor(X_stop), torch.as_tensor(y_stop)),
                                 batch_size=256, shuffle=False)
        Model = LSTMModelBuilder()._build_model_class()
        dataset = TensorDataset(torch.as_tensor(X_fit), torch.as_tensor(y_fit))
        for seed in seeds:
            np.random.seed(seed)
            torch.manual_seed(seed)
            model = Model(input_dim=len(columns), **{k: params[k] for k in ('hidden_dim', 'num_layers', 'dropout')})
            loader = DataLoader(dataset, batch_size=128, shuffle=True,
                                generator=torch.Generator().manual_seed(seed))
            best_epoch, epochs_run, stop_auc, model = fit_lstm_with_validation(
                model, loader, stop_loader, epochs=epochs, patience=5,
                lr=params['lr'], weight_decay=params['weight_decay'])
            models.append(model)
            checkpoints.append({'seed': seed, 'best_epoch': best_epoch,
                                'epochs_run': epochs_run, 'stopping_auc': stop_auc})
        def predict(mask):
            X, _, indices = create_sequences(scaled, columns, lookback, mask=mask, return_indices=True)
            loader = DataLoader(TensorDataset(torch.as_tensor(X)), batch_size=256, shuffle=False)
            return indices, np.mean([predict_lstm_loader(m, loader) for m in models], axis=0)
        del X_fit, y_fit, X_stop, y_stop, dataset, loader, stop_loader
    indices, probability = predict(validation)
    if set(indices) != set(val_indices):
        raise ValueError('Candidates must retain every outer validation row')
    threshold = select_decision_threshold(panel.loc[indices, 'Target_Std'], probability,
                                         dates=panel.loc[indices, 'Date_Std'])
    record = {'spec': spec, 'features': columns, 'checkpoints': checkpoints,
              'fit_rows': int(fit.sum()), 'stopping_rows': int(stopping.sum()),
              'validation': metrics(panel.loc[indices, 'Target_Std'], probability, threshold)}
    return record, predict


def pooled_metrics(frame, name):
    result = metrics(frame.y, frame[f'{name}_probability'])
    labels = frame[f'{name}_label']
    result.update(accuracy=float(accuracy_score(frame.y, labels)),
                  balanced_accuracy=float(balanced_accuracy_score(frame.y, labels)),
                  predicted_positive_ratio=float(labels.mean()),
                  decision_threshold='frozen validation threshold per fold')
    result['fold_auc_mean'] = float(np.mean([
        metrics(g.y, g[f'{name}_probability'])['auc_roc'] for _, g in frame.groupby('fold')]))
    return result


def verify_reference(frame, old, meta, source, previous):
    """Never compare a saved baseline with a changed source, window or cohort."""
    if source != previous['source']:
        raise ValueError('Frozen source does not match the saved reference')
    expected = frame[['Date', 'Ticker', 'y']].sort_values(['Date', 'Ticker']).reset_index(drop=True)
    actual = old[['Date', 'Ticker', 'y']].sort_values(['Date', 'Ticker']).reset_index(drop=True)
    pd.testing.assert_frame_equal(expected, actual, check_dtype=False)
    if any(meta[key] != previous['split'][key] for key in ('train_start', 'validation_start', 'test_start', 'test_end', 'prediction_horizon')):
        raise ValueError('Saved reference boundaries must match exactly')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path, default=ROOT / 'output/walk_forward')
    parser.add_argument('--dataset-key', default='lstm/lstm_stock_20tickers_10y_20261005.csv')
    parser.add_argument('--end', required=True)
    parser.add_argument('--folds', type=int, default=3)
    parser.add_argument('--test-sessions', type=int, default=63)
    parser.add_argument('--validation-sessions', type=int, default=63)
    parser.add_argument('--stopping-sessions', type=int, default=63)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--lookback', type=int, default=30)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 7])
    parser.add_argument('--models', nargs='+', choices=['xgboost', 'lstm'], default=['xgboost', 'lstm'])
    args = parser.parse_args()
    if min(args.epochs, args.lookback, args.folds, args.test_sessions, args.validation_sessions) < 1 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error('Positive budgets and unique seeds are required')
    if (args.output / 'results.json').exists():
        parser.error('Use a new output directory to preserve the prior experiment')
    import torch
    torch.set_num_threads(2)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    from dashboard.storage import get_s3_client
    s3 = get_s3_client()
    raw, source = read_snapshot(s3, 'lstm', args.dataset_key)
    panel, groups = prepare_shared_panel(raw)
    panel, extra = add_context_features(panel)
    for model in groups.values():
        model['context'] = CORE + REGIME + extra
    splits = list(walk_forward_splits(panel, args.folds, args.test_sessions,
                                     args.validation_sessions, args.end, 3))
    previous = json.loads((args.reference / 'results.json').read_text())
    report = {'status': 'running', 'created_at': pd.Timestamp.now(tz='UTC').isoformat(),
              'source': source, 'runtime': runtime_provenance(),
              'protocol': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'interpretation': 'Exploratory previously examined historical windows; no serving publication.',
              'selection_policy': {'min_validation_auc_gain': .005, 'max_validation_logloss_increase': .005,
                                   'stopping': 'purged tail of outer train; validation never used for stopping'},
              'news_readiness': inventory_news(s3, pd.Timestamp.now(tz='UTC')), 'models': {}}
    for name in ('dags/research_improvements.py', 'scripts/improve_models.py'):
        report['runtime']['source_sha256'][name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    def save():
        temporary = args.output / 'results.tmp'
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
        temporary.replace(args.output / 'results.json')
    save()
    for kind in args.models:
        old = pd.read_csv(args.reference / f'{kind}_predictions.csv', parse_dates=['Date'])
        frames = []
        folds = report['models'][kind] = {'folds': []}
        for number, (train, validation, test, meta) in enumerate(splits, 1):
            fit, stopping, stop_meta = independent_stopping_masks(panel, train, args.stopping_sessions)
            records, predictors = [], []
            fold = {'fold': number, 'split': meta, 'stopping_split': stop_meta, 'candidates': records}
            folds['folds'].append(fold)
            for spec in candidate_specs(kind):
                logging.info('%s fold %s candidate %s', kind, number, spec['name'])
                record, predict = fit_candidate(kind, panel, groups[kind][spec['group']],
                                                fit, stopping, validation, spec, args.seeds,
                                                args.epochs, args.lookback)
                records.append(record)
                predictors.append(predict)
                logging.info('Validation %s: %s', spec['name'], record['validation'])
                save()
                gc.collect()
            selected = select_generalizing_candidate(records)
            fold['selected_index'], fold['selection'] = selected, records[selected]['spec']['name']
            # Persist all choices before the first test prediction.
            save()
            frame = panel.loc[test, ['Date_Std', 'Ticker_Std', 'Sector', 'Target_Std']].rename(
                columns={'Date_Std': 'Date', 'Ticker_Std': 'Ticker', 'Target_Std': 'y'}).copy()
            for name, index in [('control', 0), ('selected', selected)]:
                indices, p = predictors[index](test)
                if len(indices) != len(frame) or set(indices) != set(frame.index):
                    raise ValueError('Every test observation must be retained')
                p = pd.Series(p, index=indices).reindex(frame.index).to_numpy()
                threshold = records[index]['validation']['decision_threshold']
                frame[f'{name}_probability'], frame[f'{name}_label'] = p, (p >= threshold).astype(int)
                fold[name] = metrics(frame.y, p, threshold)
            reference = old[old.fold.eq(number)]
            verify_reference(frame, reference, meta, source,
                             {'source': previous['source'], 'split': previous['models'][kind]['folds'][number-1]['split']})
            reference = reference.set_index(['Date', 'Ticker']).reindex(pd.MultiIndex.from_frame(frame[['Date', 'Ticker']]))
            for name in ('reference', 'majority'):
                frame[f'{name}_probability'] = reference[f'{name}_probability'].to_numpy()
                frame[f'{name}_label'] = reference[f'{name}_label'].to_numpy()
            frame['fold'] = number
            frames.append(frame)
            pd.concat(frames).to_csv(args.output / f'{kind}_predictions.csv', index=False)
            save()
            logging.info('Completed %s fold %s: %s', kind, number, fold['selection'])
            del predictors, predict
            gc.collect()
        predictions = pd.concat(frames, ignore_index=True)
        summary = {name: pooled_metrics(predictions, name) for name in ('reference', 'control', 'selected', 'majority')}
        summary['uncertainty_vs_reference'] = paired_block_intervals(predictions)
        for comparator in ('control', 'majority'):
            comparison = predictions.copy()
            comparison['reference_probability'] = comparison[f'{comparator}_probability']
            comparison['reference_label'] = comparison[f'{comparator}_label']
            summary[f'uncertainty_vs_{comparator}'] = paired_block_intervals(comparison)
        summary['by_ticker'] = {ticker: pooled_metrics(g, 'selected') for ticker, g in predictions.groupby('Ticker')}
        summary['by_sector'] = {sector: pooled_metrics(g, 'selected') for sector, g in predictions.groupby('Sector')}
        folds['summary'] = summary
        save()
        logging.info('%s summary: %s', kind, {k: summary[k] for k in ('reference', 'control', 'selected', 'majority')})
    report['status'] = 'complete'
    save()


if __name__ == '__main__':
    main()
