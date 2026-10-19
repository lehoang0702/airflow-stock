"""Validation-only model selection; test observations never enter these helpers."""
import copy
import logging

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score


def select_decision_threshold(y_validation, probabilities, min_gain=0.005, dates=None):
    """Conservative accuracy tuning without trading away balanced accuracy."""
    y = np.asarray(y_validation).reshape(-1)
    p = np.asarray(probabilities).reshape(-1)
    if not len(y) or len(y) != len(p) or not np.isfinite(p).all():
        raise ValueError('Threshold selection requires finite, matched validation observations')
    if not np.isin(y, [0, 1]).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError('Invalid validation labels or probabilities')
    default = p >= 0.5
    if len(np.unique(y)) < 2:
        return 0.5
    baseline = accuracy_score(y, default)
    balanced = balanced_accuracy_score(y, default)
    blocks = []
    if dates is not None:
        import pandas as pd
        dates = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
        if len(dates) != len(y) or dates.isna().any():
            raise ValueError('Validation dates must align with labels')
        unique_dates = np.sort(dates.unique())
        if len(unique_dates) < 12:
            return 0.5
        blocks = [dates.isin(block).to_numpy() for block in np.array_split(unique_dates, 3)]
    candidates = []
    for threshold in np.linspace(0.4, 0.6, 41):
        prediction = p >= threshold
        if len(np.unique(prediction)) < 2:
            continue
        score = accuracy_score(y, prediction)
        stable = all(accuracy_score(y[block], prediction[block]) >=
                     accuracy_score(y[block], default[block]) for block in blocks)
        if stable and score >= baseline + min_gain and balanced_accuracy_score(y, prediction) >= balanced:
            candidates.append((score, -abs(threshold - 0.5), float(threshold)))
    return max(candidates)[2] if candidates else 0.5


def recency_weights(dates, half_life_sessions=0):
    """Equal weight per date across tickers, exponential decay by trading session."""
    import pandas as pd
    dates = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
    if dates.empty or dates.isna().any() or half_life_sessions < 0:
        raise ValueError('Recency weighting requires valid dates and a nonnegative half-life')
    if half_life_sessions == 0:
        return np.ones(len(dates), dtype=np.float32)
    sessions = np.sort(dates.unique())
    age = len(sessions) - 1 - np.searchsorted(sessions, dates.to_numpy())
    weights = np.exp2(-age / float(half_life_sessions))
    return (weights / weights.mean()).astype(np.float32)


def fit_lstm_with_validation(model, train_loader, val_loader, epochs=50, lr=0.001,
                             patience=5, max_grad_norm=1.0, weight_decay=1e-3,
                             min_delta=1e-4, history=None):
    import torch
    if len(train_loader.dataset) == 0 or len(val_loader.dataset) == 0:
        raise ValueError('Train or val dataset is empty')
    if epochs <= 0 or patience <= 0:
        raise ValueError('epochs and patience must be > 0')
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-5)
    best_metric, best_state, best_epoch, stale = -float('inf'), None, 0, 0
    if history is not None:
        from torch.utils.data import DataLoader
        # DataLoader draws a worker seed even without shuffling. Keep diagnostic
        # iteration independent of the RNG used by dropout during training.
        diagnostic_loader = DataLoader(train_loader.dataset, batch_size=256, shuffle=False,
                                       generator=torch.Generator().manual_seed(0))
    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            if not torch.isfinite(loss):
                raise ValueError('Loss is not finite')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        probabilities, labels = predict_lstm_loader(model, val_loader, include_labels=True)
        if not np.isfinite(probabilities).all():
            raise ValueError('Val metric is not finite')
        metric = (roc_auc_score(labels, probabilities) if len(np.unique(labels)) == 2
                  else -log_loss(labels, probabilities, labels=[0, 1]))
        if not np.isfinite(metric):
            raise ValueError('Val metric is not finite')
        if history is not None:
            train_p, train_y = predict_lstm_loader(model, diagnostic_loader, include_labels=True)
            history.append({'epoch': epoch + 1,
                            'train_auc': float(roc_auc_score(train_y, train_p)) if len(np.unique(train_y)) == 2 else None,
                            'train_logloss': float(log_loss(train_y, train_p, labels=[0, 1])),
                            'validation_auc': float(metric) if len(np.unique(labels)) == 2 else None,
                            'validation_logloss': float(log_loss(labels, probabilities, labels=[0, 1]))})
        scheduler.step(metric)
        if metric > best_metric + min_delta:
            best_metric, best_epoch = float(metric), epoch
            best_state, stale = copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return best_epoch, epoch + 1, best_metric, model


def predict_lstm_loader(model, loader, include_labels=False):
    import torch
    predictions, labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            predictions.append(torch.sigmoid(model(batch[0])).cpu().numpy())
            if include_labels:
                labels.append(batch[1].cpu().numpy())
    probabilities = np.concatenate(predictions)
    return (probabilities, np.concatenate(labels)) if include_labels else probabilities


LSTM_CANDIDATES = (
    {'hidden_dim': 48, 'num_layers': 2, 'dropout': 0.2, 'lr': 0.001, 'weight_decay': 1e-3},
    {'hidden_dim': 32, 'num_layers': 1, 'dropout': 0.0, 'lr': 0.0005, 'weight_decay': 5e-3},
    {'hidden_dim': 64, 'num_layers': 1, 'dropout': 0.0, 'lr': 0.001, 'weight_decay': 3e-3},
)


def select_lstm_configuration(model_class, X_train, y_train, X_validation, y_validation,
                               candidates=None, epochs=50, patience=5, max_logloss_increase=0.005,
                               seed=42, collect_history=False):
    """Bounded deterministic search ranked by validation AUC then log loss."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    train_data = TensorDataset(torch.as_tensor(X_train), torch.as_tensor(y_train))
    validation_loader = DataLoader(
        TensorDataset(torch.as_tensor(X_validation), torch.as_tensor(y_validation)),
        batch_size=256, shuffle=False)
    best, history = None, []
    for config in (LSTM_CANDIDATES if candidates is None else candidates):
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = model_class(input_dim=X_train.shape[-1], **{
            key: config[key] for key in ('hidden_dim', 'num_layers', 'dropout')})
        train_loader = DataLoader(train_data, batch_size=128, shuffle=True,
                                  generator=torch.Generator().manual_seed(seed))
        learning_history = [] if collect_history else None
        diagnostic_options = {'history': learning_history} if collect_history else {}
        best_epoch, epochs_run, metric, model = fit_lstm_with_validation(
            model, train_loader, validation_loader, epochs=epochs, lr=config['lr'],
            patience=patience, weight_decay=config['weight_decay'], **diagnostic_options)
        probabilities = predict_lstm_loader(model, validation_loader)
        loss = float(log_loss(y_validation, probabilities, labels=[0, 1]))
        eligible = not history or loss <= history[0]['val_logloss'] + max_logloss_increase
        record = {**config, 'best_epoch': best_epoch, 'epochs_run': epochs_run,
                  'val_metric': metric, 'val_logloss': loss, 'eligible': eligible}
        if collect_history:
            record['learning_history'] = learning_history
        history.append(record)
        logging.info('LSTM candidate: %s', record)
        if eligible and (best is None or (metric, -loss) > best[0]):
            best = ((metric, -loss), model, record, probabilities)
    if best is None:
        raise ValueError('At least one LSTM candidate is required')
    _, model, config, probabilities = best
    return model, config, probabilities, history
