"""Log genuine models across the project's supported MLflow versions."""
import inspect


def log_model_compat(log_model_fn, model, name='model', **kwargs):
    field = 'name' if 'name' in inspect.signature(log_model_fn).parameters else 'artifact_path'
    return log_model_fn(model, **{field: name}, **kwargs)


def log_xgboost_model(mlflow, model, features):
    """Declare nullable-friendly float inputs and the model's real output dtype."""
    from mlflow.models import ModelSignature
    from mlflow.types import Schema, ColSpec, TensorSpec
    # Workaround: xgboost 2.1.x + scikit-learn 1.9.x compatibility.
    # sklearn 1.9 removed `_estimator_type` from ClassifierMixin but xgboost
    # 2.1 still reads it in save_model → TypeError.  Set it explicitly.
    if not hasattr(model, '_estimator_type'):
        model._estimator_type = 'classifier'
    example = features.iloc[:1].astype('float64')
    output = model.predict(example)
    signature = ModelSignature(
        inputs=Schema([ColSpec('double', name=str(column)) for column in example.columns]),
        outputs=Schema([TensorSpec(output.dtype, (-1,))]))
    return log_model_compat(mlflow.xgboost.log_model, model, input_example=example, signature=signature)
