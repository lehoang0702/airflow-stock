"""Train/validation/test diagnostics with chronological learning curves.

Uses frozen inputs and the paired benchmark procedure. Writes local reports only.
"""
import argparse
import io
import json
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'dags'), str(ROOT)]
from research_data import prepare_shared_panel, walk_forward_splits
from scripts.walk_forward_benchmark import (fit_candidate, metrics, probability_diagnostics,
                                             read_snapshot, runtime_provenance)
from news_data import news_inventory, NEWS_POLICY, select_eligible_news


def inventory_news(s3, cutoff):
    entries, all_news, scored = [], [], []
    prefixes = ('finbert/finbert_news_20tickers_', 'news_history/', 'news_sentiment_history/')
    for prefix in prefixes:
        keys = [item['Key'] for page in s3.get_paginator('list_objects_v2').paginate(
            Bucket='stock-data', Prefix=prefix) for item in page.get('Contents', [])
            if item['Key'].endswith('.csv')]
        for key in sorted(keys):
            raw = s3.get_object(Bucket='stock-data', Key=key)['Body'].read()
            frame = pd.read_csv(io.BytesIO(raw))
            report = news_inventory(frame, cutoff)
            report['key'] = key
            import hashlib
            report['sha256'] = hashlib.sha256(raw).hexdigest()
            entries.append(report)
            if prefix == 'news_sentiment_history/':
                scored.append(frame)
            else:
                all_news.append(frame)
    combined = pd.concat(all_news, ignore_index=True) if all_news else pd.DataFrame()
    sentiment = pd.concat(scored, ignore_index=True) if scored else pd.DataFrame()
    eligible = select_eligible_news(combined, cutoff, lookback_hours=24 * 365 * 30)
    eligible_scored = select_eligible_news(sentiment, cutoff, lookback_hours=24 * 365 * 30, require_sentiment=True)
    dates = pd.to_datetime(eligible_scored.first_seen_at, utc=True).dt.normalize() if not eligible_scored.empty else pd.Series(dtype=object)
    return {'news_policy': NEWS_POLICY, 'objects': entries, 'eligible_unique_versions': len(eligible),
            'eligible_scored_versions': len(eligible_scored), 'eligible_scored_days': int(dates.nunique()),
            'historical_news_experiment_status': 'ready_for_protocol_review' if dates.nunique() >= 189 else 'insufficient_observed_history',
            'requirement': 'At least 189 distinct observed sentiment days to match the existing three 63-session test windows; additional prior training/validation coverage is required.',
            'can_backdate_collection_from_publication': False}


def assessment(records):
    """Descriptive flag, not proof of the cause or a deployment decision."""
    full = [row for row in records if row['spec']['train_fraction'] == 1]
    gaps = [row['training']['auc_roc'] - row['validation']['auc_roc'] for row in full
            if row['training']['auc_roc'] is not None and row['validation']['auc_roc'] is not None]
    return {'full_data_mean_train_validation_auc_gap': float(np.mean(gaps)) if gaps else None,
            'interpretation': 'Descriptive train/validation gap; feature insufficiency and overfitting require follow-up evidence.'}


def plot_diagnostics(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for axis, (kind, model_report) in zip(axes, report['models'].items()):
        rows = model_report['records']
        fractions = sorted({r['spec']['train_fraction'] for r in rows})
        for split, color in [('training', '#2563eb'), ('validation', '#e97316'), ('test', '#16a34a')]:
            values = [[r[split]['auc_roc'] for r in rows if r['spec']['train_fraction'] == fraction
                       and r[split]['auc_roc'] is not None] for fraction in fractions]
            means = [np.mean(v) if v else np.nan for v in values]
            axis.plot(np.array(fractions) * 100, means, marker='o', color=color, label=split)
        axis.axhline(.5, color='#64748b', linestyle='--', linewidth=1)
        axis.set(title=kind.upper(), xlabel='Latest fraction of 3-year training window (%)', ylabel='Mean ROC-AUC across folds', ylim=(.4, 1))
        axis.grid(alpha=.2)
        axis.legend()
    fig.savefig(output / 'learning_curves.png', dpi=180)
    plt.close(fig)
    lstm = report['models'].get('lstm')
    if not lstm:
        return
    full = [r for r in lstm['records'] if r['spec']['train_fraction'] == 1]
    fig, axes = plt.subplots(1, len(full), figsize=(4 * len(full), 4), squeeze=False, constrained_layout=True)
    for axis, record in zip(axes[0], full):
        for seed in record['configuration']['seeds']:
            history = seed['selected']['learning_history']
            epochs = [h['epoch'] for h in history]
            axis.plot(epochs, [h['train_auc'] for h in history], label=f"train seed {seed['seed']}")
            axis.plot(epochs, [h['validation_auc'] for h in history], linestyle='--', label=f"val seed {seed['seed']}")
        axis.set(title=f"LSTM fold {record['fold']}", xlabel='Epoch', ylabel='ROC-AUC', ylim=(.4, 1))
        axis.legend(fontsize=8)
        axis.grid(alpha=.2)
    fig.savefig(output / 'lstm_epoch_curves.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset-key', required=True)
    parser.add_argument('--end', required=True)
    parser.add_argument('--folds', type=int, default=3)
    parser.add_argument('--fractions', type=float, nargs='+', default=[.25, .5, 1.])
    parser.add_argument('--trials', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 7])
    parser.add_argument('--models', nargs='+', choices=['xgboost', 'lstm'], default=['xgboost', 'lstm'])
    args = parser.parse_args()
    if 1. not in args.fractions or any(not 0 < f <= 1 for f in args.fractions) or len(set(args.fractions)) != len(args.fractions):
        parser.error('Unique fractions in (0, 1], including 1, are required')
    if min(args.trials, args.epochs, args.folds) < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Positive budgets and unique seeds are required')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    import torch
    torch.set_num_threads(2)
    from dashboard.storage import get_s3_client
    s3 = get_s3_client()
    raw, source = read_snapshot(s3, 'lstm', args.dataset_key)
    panel, groups = prepare_shared_panel(raw)
    splits = list(walk_forward_splits(panel, folds=args.folds, end=args.end))
    report = {'status': 'running', 'created_at': pd.Timestamp.now(tz='UTC').isoformat(),
              'source': source, 'runtime': runtime_provenance(),
              'protocol': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'interpretation': 'Exploratory diagnostics on previously examined historical windows; no serving publication.',
              'news_inventory': inventory_news(s3, pd.Timestamp.now(tz='UTC')), 'models': {}}
    import hashlib
    report['runtime']['source_sha256'].update({name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ('scripts/diagnose_models.py', 'dags/news_data.py')})
    args.output.mkdir(parents=True, exist_ok=True)
    def save():
        temporary = args.output / 'diagnostics.tmp'
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
        temporary.replace(args.output / 'diagnostics.json')
    save()
    for kind in args.models:
        records = []
        report['models'][kind] = {'records': records}
        for fold, (train, validation, test, meta) in enumerate(splits, 1):
            predictions = []
            for fraction in sorted(args.fractions):
                spec = {'name': 'reference', 'group': 'full', 'years': 3, 'lookback': 30, 'train_fraction': fraction}
                logging.info('%s fold %s training fraction %s', kind, fold, fraction)
                record, predict, val_indices, val_probability = fit_candidate(
                    kind, panel, groups, meta, spec, args.trials, args.epochs, args.seeds, diagnostics=True)
                record.update({'fold': fold, 'split': meta})
                # All budgets, feature choices and thresholds were frozen above.
                indices, p = predict(test)
                y = panel.loc[indices, 'Target_Std'].to_numpy()
                threshold = record['validation']['decision_threshold']
                record['test'] = metrics(y, p, threshold)
                record['probability_diagnostics']['test'] = probability_diagnostics(y, p, threshold)
                detail = panel.loc[indices, ['Date_Std', 'Ticker_Std', 'Sector', 'Target_Std']].copy()
                detail['probability'], detail['prediction'], detail['fold'], detail['train_fraction'] = p, (p >= threshold).astype(int), fold, fraction
                predictions.append(detail)
                record['test_by_ticker'] = {ticker: metrics(g.Target_Std, g.probability, threshold) for ticker, g in detail.groupby('Ticker_Std')}
                record['test_by_sector'] = {sector: metrics(g.Target_Std, g.probability, threshold) for sector, g in detail.groupby('Sector')}
                records.append(record)
                save()
                logging.info('Completed %s fold %s fraction %s: train AUC %.4f, validation AUC %.4f, test AUC %.4f',
                             kind, fold, fraction, record['training']['auc_roc'], record['validation']['auc_roc'], record['test']['auc_roc'])
            pd.concat(predictions).to_csv(args.output / f'{kind}_fold_{fold}_predictions.csv', index=False)
        report['models'][kind]['assessment'] = assessment(records)
    report['status'] = 'complete'
    save()
    plot_diagnostics(report, args.output)


if __name__ == '__main__':
    main()
