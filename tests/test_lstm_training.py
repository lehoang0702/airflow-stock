import sys
import os
import ast
import logging
import pytest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dags')))

def extract_module():
    torch = pytest.importorskip('torch')
    import numpy as np

    path = Path(__file__).resolve().parents[1] / 'dags/dag_lstm_train_and_predict.py'
    nodes = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ClassDef) and node.name == 'CalibratedStackedLSTM':
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == 'fit_lstm_with_validation':
            nodes.append(node)

    namespace = {
        'torch': torch,
        'nn': torch.nn,
        'np': np,
        'copy': __import__('copy'),
        'log_loss': __import__('sklearn.metrics', fromlist=['log_loss']).log_loss,
        'roc_auc_score': __import__('sklearn.metrics', fromlist=['roc_auc_score']).roc_auc_score
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['CalibratedStackedLSTM'], namespace['fit_lstm_with_validation']

def test_determinism():
    torch = pytest.importorskip('torch')
    import numpy as np
    from torch.utils.data import TensorDataset, DataLoader

    CalibratedStackedLSTM, fit_lstm_with_validation = extract_module()
    input_dim = 5
    lookback = 3
    hidden_dim = 8
    num_layers = 1

    torch.manual_seed(42)
    np.random.seed(42)
    x_tr = torch.randn(50, lookback, input_dim)
    y_tr = torch.randint(0, 2, (50,)).float()
    train_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=10, shuffle=False)

    x_val = torch.randn(20, lookback, input_dim)
    y_val = torch.randint(0, 2, (20,)).float()
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=10, shuffle=False)

    model1 = CalibratedStackedLSTM(input_dim, hidden_dim, num_layers)
    model2 = CalibratedStackedLSTM(input_dim, hidden_dim, num_layers)
    model2.load_state_dict(model1.state_dict())

    torch.manual_seed(42)
    ep1, runs1, v1, m1 = fit_lstm_with_validation(
        model1, train_loader, val_loader, epochs=2, lr=0.01, patience=2
    )
    out1 = m1(torch.randn(1, lookback, input_dim))

    torch.manual_seed(42)
    ep2, runs2, v2, m2 = fit_lstm_with_validation(
        model2, train_loader, val_loader, epochs=2, lr=0.01, patience=2
    )
    out2 = m2(torch.randn(1, lookback, input_dim))

    assert ep1 == ep2
    assert np.isclose(v1, v2)
    assert torch.allclose(out1, out2)

def test_patience_and_checkpoint_restoration():
    torch = pytest.importorskip('torch')
    from torch.utils.data import TensorDataset, DataLoader
    CalibratedStackedLSTM, fit_lstm_with_validation = extract_module()
    input_dim = 5
    lookback = 3
    hidden_dim = 8
    num_layers = 1

    model = CalibratedStackedLSTM(input_dim, hidden_dim, num_layers)

    # Overfit to train
    x_tr = torch.randn(50, lookback, input_dim)
    y_tr = torch.zeros(50)
    train_ldr = DataLoader(TensorDataset(x_tr, y_tr), batch_size=10, shuffle=False)

    # Val distribution different => worse val perf
    x_v = torch.randn(20, lookback, input_dim)
    y_v = torch.ones(20)
    val_ldr = DataLoader(TensorDataset(x_v, y_v), batch_size=10, shuffle=False)

    best_ep, runs, v, m = fit_lstm_with_validation(
        model, train_ldr, val_ldr, epochs=20, lr=0.05, patience=3
    )

    assert runs < 20
    assert runs == best_ep + 1 + 3

def test_fallback_logloss_one_class():
    torch = pytest.importorskip('torch')
    from torch.utils.data import TensorDataset, DataLoader
    CalibratedStackedLSTM, fit_lstm_with_validation = extract_module()
    input_dim = 5
    lookback = 3
    hidden_dim = 8
    num_layers = 1

    model = CalibratedStackedLSTM(input_dim, hidden_dim, num_layers)

    x_tr = torch.randn(50, lookback, input_dim)
    y_tr = torch.randint(0, 2, (50,)).float()
    train_ldr = DataLoader(TensorDataset(x_tr, y_tr), batch_size=10, shuffle=False)

    x_v = torch.randn(20, lookback, input_dim)
    y_v = torch.zeros(20)  # ONLY ZERO CLASS
    val_ldr = DataLoader(TensorDataset(x_v, y_v), batch_size=10, shuffle=False)

    best_ep, runs, v_metric, m = fit_lstm_with_validation(
        model, train_ldr, val_ldr, epochs=2, lr=0.01, patience=2
    )

    assert isinstance(v_metric, float)
    assert v_metric <= 0  # negative logloss

def test_lstm_training_exceptions():
    torch = pytest.importorskip('torch')
    import numpy as np
    from torch.utils.data import TensorDataset, DataLoader
    CalibratedStackedLSTM, fit_lstm_with_validation = extract_module()
    input_dim = 5
    lookback = 3
    hidden_dim = 8
    num_layers = 1

    model = CalibratedStackedLSTM(input_dim, hidden_dim, num_layers)

    x_tr = torch.randn(50, lookback, input_dim)
    y_tr = torch.randint(0, 2, (50,)).float()
    train_ldr = DataLoader(TensorDataset(x_tr, y_tr), batch_size=10, shuffle=False)

    x_v = torch.randn(20, lookback, input_dim)
    y_v = torch.zeros(20)
    val_ldr = DataLoader(TensorDataset(x_v, y_v), batch_size=10, shuffle=False)

    empty_ldr = DataLoader(TensorDataset(torch.empty(0, lookback, input_dim), torch.empty(0)), batch_size=10)

    with pytest.raises(ValueError, match="empty"):
        fit_lstm_with_validation(model, empty_ldr, val_ldr)

    with pytest.raises(ValueError, match="empty"):
        fit_lstm_with_validation(model, train_ldr, empty_ldr)

    with pytest.raises(ValueError, match="> 0"):
        fit_lstm_with_validation(model, train_ldr, val_ldr, epochs=0)

    with pytest.raises(ValueError, match="> 0"):
        fit_lstm_with_validation(model, train_ldr, val_ldr, patience=-1)

    x_tr_inf = torch.full((50, lookback, input_dim), float('inf'))
    train_ldr_inf = DataLoader(TensorDataset(x_tr_inf, y_tr), batch_size=10, shuffle=False)
    with pytest.raises(ValueError, match="Loss is not finite"):
        fit_lstm_with_validation(model, train_ldr_inf, val_ldr, epochs=1)

    x_v_inf = torch.full((20, lookback, input_dim), float('inf'))
    val_ldr_inf = DataLoader(TensorDataset(x_v_inf, y_v), batch_size=10, shuffle=False)
    with pytest.raises(ValueError, match="Val metric is not finite"):
        fit_lstm_with_validation(model, train_ldr, val_ldr_inf, epochs=1)

def test_scaler_boundary_and_no_test_data_in_fit():
    path = Path(__file__).resolve().parents[1] / 'dags/dag_lstm_train_and_predict.py'
    text = path.read_text()

    assert 'rolling_temporal_masks(' in text
    assert 'scaler.fit(df.loc[train_mask, feature_cols].values)' in text
    selection_source = (path.parent / 'model_training.py').read_text()
    assert 'batch_size=128' in selection_source
    assert 'batch_size=256' in text
    assert 'np.random.seed(42)' in text
    assert 'torch.manual_seed(42)' in text
    assert 'max_grad_norm=1.0' in text
    assert 'CalibratedStackedLSTM, X_train, y_train, X_val, y_val' in text

    # Ensure train mask and test mask separate
    assert 'create_sequences(scaled_df, feature_cols, lookback, mask=train_mask)' in text
    assert 'create_sequences(scaled_df, feature_cols, lookback, mask=test_mask)' in text
