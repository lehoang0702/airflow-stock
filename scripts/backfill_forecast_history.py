"""
Script backfill dữ liệu forecast_history cho FinBERT, XGBoost và LSTM trên MinIO
để phục vụ huấn luyện direction head và đánh giá mô hình.
"""
import io
import re
import sys
import logging
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

# Đảm bảo đường dẫn dags
sys.path.append('/opt/airflow/dags')
from chart_utils import load_price_data_from_minio
from forecast_history import save_forecast_history, load_forecast_history, observed_forecasts
from learned_signals import train_signal_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

BUCKET_NAME = 'stock-data'
MINIO_CONN_ID = 'minio_conn'


def run_backfill():
    s3_hook = S3Hook(aws_conn_id=MINIO_CONN_ID)
    logging.info("🚀 Bắt đầu nạp dữ liệu giá cổ phiếu từ MinIO...")
    prices = load_price_data_from_minio(s3_hook, BUCKET_NAME)
    prices['Date'] = pd.to_datetime(prices['Date'], utc=True).dt.tz_localize(None).dt.normalize()
    logging.info(f"📊 Đã nạp dữ liệu giá: {len(prices)} dòng, {prices['Ticker'].nunique()} mã.")

    # =========================================================================
    # 1. BACKFILL FINBERT FORECAST HISTORY
    # =========================================================================
    logging.info("🔍 Kiểm tra danh sách file tin tức FinBERT...")
    all_keys = s3_hook.list_keys(bucket_name=BUCKET_NAME, prefix='finbert/') or []
    
    # Nhóm theo date_nodash, ưu tiên parquet hơn csv
    news_by_date = {}
    for k in all_keys:
        m = re.search(r'finbert_news_20tickers_(\d{8})\.(parquet|csv)', k)
        if m:
            date_nodash = m.group(1)
            ext = m.group(2)
            if date_nodash not in news_by_date or ext == 'parquet':
                news_by_date[date_nodash] = k

    existing_finbert = load_forecast_history(s3_hook, BUCKET_NAME, 'finbert')
    existing_dates = set()
    if not existing_finbert.empty and 'ngay_du_bao' in existing_finbert:
        existing_dates = set(existing_finbert['ngay_du_bao'].dropna().str.replace('-', ''))
    logging.info(f"Đã có {len(existing_dates)} ngày trong forecast_history/finbert: {sorted(list(existing_dates))}")

    dates_to_process = sorted([d for d in news_by_date.keys() if d not in existing_dates])
    logging.info(f"Cần backfill FinBERT cho {len(dates_to_process)} ngày: {dates_to_process}")

    if dates_to_process:
        logging.info("🧠 Khởi tạo mô hình ProsusAI/finbert...")
        tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
        model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
        model.eval()
        labels = {str(label).lower(): int(index) for index, label in model.config.id2label.items()}

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        for date_nodash in dates_to_process:
            key = news_by_date[date_nodash]
            date_iso = f"{date_nodash[:4]}-{date_nodash[4:6]}-{date_nodash[6:]}"
            logging.info(f"⏳ Đang xử lý FinBERT cho ngày {date_iso} từ {key}...")

            obj = s3_hook.get_key(key, bucket_name=BUCKET_NAME)
            raw_bytes = obj.get()['Body'].read()
            if key.endswith('.parquet'):
                df_news = pd.read_parquet(io.BytesIO(raw_bytes))
            else:
                df_news = pd.read_csv(io.BytesIO(raw_bytes), encoding='utf-8', encoding_errors='replace')

            # Xác định phiên tham chiếu giá gần nhất trước hoặc bằng ngày dự báo
            sub_prices = prices[prices['Date'] <= pd.Timestamp(date_iso)]
            references = (sub_prices.sort_values('Date').groupby('Ticker')['Date'].last().to_dict()
                          if not sub_prices.empty else {})

            results = []
            for (ticker, sector), group in df_news.groupby(['ma_co_phieu', 'nhom_nganh']):
                texts = []
                for _, row in group.iterrows():
                    title = str(row.get('tieu_de', '')).strip()
                    if not title or title == 'nan':
                        continue
                    summary = str(row.get('tom_tat', '')).strip()
                    text = title + ('. ' + summary if summary and summary != 'nan' else '')
                    texts.append(text)

                if not texts or ticker not in references:
                    continue

                outputs = []
                # Batch processing
                batch_size = 32
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
                    with torch.no_grad():
                        logits = model(**inputs).logits
                        probs = torch.softmax(logits, dim=-1).cpu().numpy()
                        outputs.extend(probs)

                mean_probs = np.mean(outputs, axis=0)
                sentiment = {f'sentiment_{label}': float(mean_probs[labels[label]])
                             for label in ('positive', 'negative', 'neutral')}

                results.append({
                    'ngay_du_bao': date_iso,
                    'ma_co_phieu': ticker,
                    'nhom_nganh': sector,
                    'khuyen_nghi': 'CHƯA CÓ MÔ HÌNH DỰ BÁO GIÁ',
                    'prob_num': None,
                    'xac_suat_tang_gia': None,
                    'issued_at': f"{date_iso}T20:00:00+00:00",
                    'reference_date': pd.Timestamp(references[ticker]).strftime('%Y-%m-%d'),
                    'prediction_horizon': 5,
                    'model_version': None,
                    'sentiment_model_version': 'ProsusAI/finbert',
                    'news_count': len(outputs),
                    'inference_source': 'sentiment_model_v2',
                    **sentiment
                })

            if results:
                frame = pd.DataFrame(results)
                saved_key = save_forecast_history(s3_hook, BUCKET_NAME, 'finbert', frame)
                logging.info(f"✅ Đã lưu {len(frame)} bản ghi FinBERT ngày {date_iso} vào {saved_key}")

    # =========================================================================
    # 2. BACKFILL XGBOOST FORECAST HISTORY
    # =========================================================================
    logging.info("🔍 Kiểm tra danh sách file dự báo XGBoost...")
    xgb_pred_keys = sorted([k for k in all_keys if 'xgboost/' in k and 'du_bao_tang_truong_' in k and k.endswith('.csv')])
    # Cũng tìm trong bucket nếu all_keys chỉ chứa finbert
    xgb_keys = s3_hook.list_keys(bucket_name=BUCKET_NAME, prefix='xgboost/') or []
    xgb_pred_keys = sorted([k for k in xgb_keys if 'du_bao_tang_truong_' in k and k.endswith('.csv')])
    
    existing_xgb = load_forecast_history(s3_hook, BUCKET_NAME, 'xgboost')
    existing_xgb_dates = set()
    if not existing_xgb.empty and 'ngay_du_bao' in existing_xgb:
        existing_xgb_dates = set(existing_xgb['ngay_du_bao'].dropna().str.replace('-', ''))
    logging.info(f"Đã có {len(existing_xgb_dates)} ngày trong forecast_history/xgboost.")

    for k in xgb_pred_keys:
        m = re.search(r'du_bao_tang_truong_(\d{8})\.csv', k)
        if not m:
            continue
        date_nodash = m.group(1)
        if date_nodash in existing_xgb_dates:
            continue
        date_iso = f"{date_nodash[:4]}-{date_nodash[4:6]}-{date_nodash[6:]}"
        logging.info(f"⏳ Backfill XGBoost từ {k} cho ngày {date_iso}...")
        obj = s3_hook.get_key(k, bucket_name=BUCKET_NAME)
        df_xgb = pd.read_csv(io.BytesIO(obj.get()['Body'].read()))
        if df_xgb.empty:
            continue
        
        # Tạo prob_num nếu chưa có
        if 'prob_num' not in df_xgb.columns and 'xac_suat_tang_gia' in df_xgb.columns:
            df_xgb['prob_num'] = df_xgb['xac_suat_tang_gia'].astype(str).str.replace('%', '').apply(pd.to_numeric, errors='coerce') / 100.0

        sub_prices = prices[prices['Date'] <= pd.Timestamp(date_iso)]
        ref_dates = sub_prices.sort_values('Date').groupby('Ticker')['Date'].last().to_dict() if not sub_prices.empty else {}
        
        df_xgb['reference_date'] = df_xgb['ma_co_phieu'].map(ref_dates).apply(lambda d: pd.Timestamp(d).strftime('%Y-%m-%d') if pd.notna(d) else date_iso)
        df_xgb['issued_at'] = f"{date_iso}T20:00:00+00:00"
        df_xgb['prediction_horizon'] = 1
        df_xgb['model_version'] = f"v{date_nodash}"
        df_xgb['inference_source'] = 'trained_model_v2'
        
        saved_key = save_forecast_history(s3_hook, BUCKET_NAME, 'xgboost', df_xgb)
        logging.info(f"✅ Đã lưu {len(df_xgb)} bản ghi XGBoost ngày {date_iso} vào {saved_key}")

    # =========================================================================
    # 3. BACKFILL LSTM FORECAST HISTORY
    # =========================================================================
    logging.info("🔍 Kiểm tra danh sách file dự báo LSTM...")
    lstm_keys = s3_hook.list_keys(bucket_name=BUCKET_NAME, prefix='lstm/') or []
    lstm_pred_keys = sorted([k for k in lstm_keys if 'du_bao_tang_truong_lstm_' in k and k.endswith('.csv')])
    
    existing_lstm = load_forecast_history(s3_hook, BUCKET_NAME, 'lstm')
    existing_lstm_dates = set()
    if not existing_lstm.empty and 'ngay_du_bao' in existing_lstm:
        existing_lstm_dates = set(existing_lstm['ngay_du_bao'].dropna().str.replace('-', ''))
    logging.info(f"Đã có {len(existing_lstm_dates)} ngày trong forecast_history/lstm.")

    for k in lstm_pred_keys:
        m = re.search(r'du_bao_tang_truong_lstm_(\d{8})\.csv', k)
        if not m:
            continue
        date_nodash = m.group(1)
        if date_nodash in existing_lstm_dates:
            continue
        date_iso = f"{date_nodash[:4]}-{date_nodash[4:6]}-{date_nodash[6:]}"
        logging.info(f"⏳ Backfill LSTM từ {k} cho ngày {date_iso}...")
        obj = s3_hook.get_key(k, bucket_name=BUCKET_NAME)
        df_lstm = pd.read_csv(io.BytesIO(obj.get()['Body'].read()))
        if df_lstm.empty:
            continue
        
        if 'prob_num' not in df_lstm.columns and 'xac_suat_tang_gia' in df_lstm.columns:
            df_lstm['prob_num'] = df_lstm['xac_suat_tang_gia'].astype(str).str.replace('%', '').apply(pd.to_numeric, errors='coerce') / 100.0

        sub_prices = prices[prices['Date'] <= pd.Timestamp(date_iso)]
        ref_dates = sub_prices.sort_values('Date').groupby('Ticker')['Date'].last().to_dict() if not sub_prices.empty else {}
        
        df_lstm['reference_date'] = df_lstm['ma_co_phieu'].map(ref_dates).apply(lambda d: pd.Timestamp(d).strftime('%Y-%m-%d') if pd.notna(d) else date_iso)
        df_lstm['issued_at'] = f"{date_iso}T20:00:00+00:00"
        df_lstm['prediction_horizon'] = 1
        df_lstm['model_version'] = f"v{date_nodash}"
        df_lstm['inference_source'] = 'trained_model_v2'
        
        saved_key = save_forecast_history(s3_hook, BUCKET_NAME, 'lstm', df_lstm)
        logging.info(f"✅ Đã lưu {len(df_lstm)} bản ghi LSTM ngày {date_iso} vào {saved_key}")

    # =========================================================================
    # 4. HUẤN LUYỆN DIRECTION HEAD CHO FINBERT
    # =========================================================================
    logging.info("🎯 Huấn luyện FinBERT Direction Head từ forecast_history...")
    finbert_history = load_forecast_history(s3_hook, BUCKET_NAME, 'finbert')
    logging.info(f"Tổng số bản ghi FinBERT forecast history: {len(finbert_history)}")
    
    observed, pending, invalid = observed_forecasts(finbert_history, prices, horizon=5)
    logging.info(f"Kết quả đối chiếu giá T+5: Observed={len(observed)}, Pending={pending}, Invalid={invalid}")
    
    if not observed.empty:
        sessions = sorted(pd.to_datetime(observed['reference_date']).dt.normalize().unique())
        logging.info(f"Số phiên quan sát được: {len(sessions)}")
        training_result = train_signal_model(s3_hook.get_conn(), BUCKET_NAME, 'finbert_direction', observed)
        logging.info(f"🏁 Kết quả huấn luyện FinBERT Direction Head: {training_result}")
    else:
        logging.warning("⚠️ Không có bản ghi mature nào để huấn luyện Direction Head.")

    logging.info("🎉 Toàn bộ quá trình backfill và huấn luyện đã hoàn tất!")


if __name__ == '__main__':
    run_backfill()
