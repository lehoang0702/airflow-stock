from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import io
import os
import sys

# Thêm dags vào sys.path để import model_registry
sys.path.append(os.path.join(os.path.dirname(__file__), 'dags'))
from model_registry import XGBoostModelBuilder, ModelCard

try:
    from airflow.providers.amazon.aws.hooks.s3 import S3Hook
    s3_hook = S3Hook(aws_conn_id='minio_conn')
except Exception:
    s3_hook = None

app = FastAPI(title="Stock Prediction API", version="1.0")

class PredictRequest(BaseModel):
    model_type: str
    version: str = "latest"
    features: dict

@app.post("/predict")
def predict(req: PredictRequest):
    if req.model_type != 'xgboost':
        raise HTTPException(status_code=400, detail="Only xgboost supported currently")
    
    if s3_hook is None:
        raise HTTPException(status_code=500, detail="S3Hook not initialized")
        
    try:
        from model_registry import load_model_from_minio
        
        # Load registry if latest
        version = req.version
        if version == "latest":
            registry_csv = s3_hook.read_key("models/registry.csv", bucket_name="stock-data")
            df_reg = pd.read_csv(io.StringIO(registry_csv))
            df_xgb = df_reg[df_reg['model_type'] == 'xgboost']
            if df_xgb.empty:
                raise HTTPException(status_code=404, detail="No xgboost models found in registry")
            version = df_xgb.iloc[-1]['version']
            
        artifacts = load_model_from_minio(s3_hook, "stock-data", "xgboost", version)
        if not artifacts.get('model_bytes') or not artifacts.get('feature_columns'):
            raise HTTPException(status_code=404, detail="Model artifacts not found")
            
        # Build DataFrame
        df_input = pd.DataFrame([req.features])
        expected_cols = artifacts['feature_columns']
        
        # Điền missing columns
        for col in expected_cols:
            if col not in df_input.columns:
                df_input[col] = 0.0
                
        df_input = df_input[expected_cols]
        
        # Load XGBoost model
        import xgboost as xgb
        model = xgb.XGBClassifier()
        model.load_model(bytearray(artifacts['model_bytes']))
        
        # Predict
        prob = float(model.predict_proba(df_input)[0][1])
        return {"model_type": "xgboost", "version": version, "probability": prob}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
