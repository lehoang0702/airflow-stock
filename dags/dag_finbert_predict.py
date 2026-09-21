"""
DAG: CHẠY MÔ HÌNH FINBERT & DỰ BÁO SENTIMENT (TÍCH HỢP RETRY TELEGRAM & TIMEOUT AN TOÀN)
Lưu file vào MinIO: stock-xgboost-data/finbert/du_bao_tang_truong_finbert_YYYYMMDD.csv
"""
from datetime import datetime, timedelta
import io
import time
import requests
import logging
import pandas as pd
import numpy as np

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

MINIO_CONN_ID = 'minio_conn'
BUCKET_NAME = 'stock-xgboost-data'

try:
    from alert_utils import telegram_failure_callback, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, send_telegram_safe
except ImportError:
    telegram_failure_callback = None
    TELEGRAM_TOKEN = ""
    TELEGRAM_CHAT_ID = ""
    def send_telegram_safe(url, data=None, files=None, max_retries=3): return False

try:
    from config_shared import SECTOR_MAP
except ImportError:
    from dags.config_shared import SECTOR_MAP

default_args = {
    'owner': 'quant_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
    'on_failure_callback': telegram_failure_callback,
}

def predict_finbert_task(**context):
    import typing_extensions
    if not hasattr(typing_extensions, 'TypeIs'):
        class _TypeIsMeta(type):
            def __getitem__(self, item): return bool
        class _TypeIs(metaclass=_TypeIsMeta): pass
        typing_extensions.TypeIs = _TypeIs

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    s3_hook = S3Hook(aws_conn_id=MINIO_CONN_ID)
    date_nodash = datetime.now().strftime("%Y%m%d")
    today_str = datetime.now().strftime("%Y-%m-%d")

    news_key = f"finbert/finbert_news_20tickers_{date_nodash}.csv"
    if not s3_hook.check_for_key(news_key, bucket_name=BUCKET_NAME):
        keys = s3_hook.list_keys(bucket_name=BUCKET_NAME, prefix="finbert/finbert_news_20tickers_")
        if not keys:
            raise FileNotFoundError("❌ Không tìm thấy file tin tức trên MinIO finbert/")
        news_key = sorted(keys)[-1]

    raw_csv = s3_hook.read_key(news_key, bucket_name=BUCKET_NAME)
    df_news = pd.read_csv(io.StringIO(raw_csv))

    logging.info("🧠 Đang nạp mô hình FinBERT...")
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    results = []
    sentiment_stats = {}
    for (ticker, sector), group in df_news.groupby(['ma_co_phieu', 'nhom_nganh']):
        scores = []
        pos_count, neg_count, neu_count = 0, 0, 0
        for idx, row in group.iterrows():
            title = str(row.get('tieu_de', '') or '')
            summary = str(row.get('tom_tat', '') or '')
            text_to_eval = f"{title}. {summary}".strip() if summary and summary != 'nan' else title
            inputs = tokenizer(text_to_eval, return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.nn.functional.softmax(logits, dim=-1)[0].numpy()
            
            score = float(probs[0] - probs[1])
            scores.append(score)

            if probs[0] > probs[1] and probs[0] > probs[2]:
                pos_count += 1
            elif probs[1] > probs[0] and probs[1] > probs[2]:
                neg_count += 1
            else:
                neu_count += 1

        if scores:
            avg_score = float(np.mean(scores))
            prob_up = float(np.clip((avg_score + 1.0) / 2.0, 0.05, 0.95))
        else:
            avg_score = 0.0
            prob_up = 0.50

        # Chuẩn hóa ngưỡng quyết định (Confidence Threshold Filtering)
        # Lọc nhiễu vùng phân vân 0.45 - 0.55 -> ĐỨNG NGOÀI, chỉ MUA khi >= 0.55, BÁN khi <= 0.45
        if prob_up >= 0.55:
            rec = "MUA"
            trend = "Tăng mạnh" if prob_up >= 0.60 else "Tăng tích lũy"
            conf = "Rất cao" if prob_up >= 0.60 else "Cao"
        elif prob_up <= 0.45:
            rec = "BÁN"
            trend = "Giảm mạnh" if prob_up <= 0.40 else "Giảm phân phối"
            conf = "Rất cao" if prob_up <= 0.40 else "Cao"
        else:
            rec = "ĐỨNG NGOÀI"
            trend = "Đi ngang / Lưỡng lự (Sideway)"
            conf = "Trung bình"

        results.append({
            'ngay_du_bao': today_str,
            'ma_co_phieu': ticker,
            'nhom_nganh': sector,
            'khuyen_nghi': rec,
            'xac_suat_tang_gia': f"{prob_up*100:.2f}%",
            'xu_huong_du_kien': trend,
            'do_tin_cay': conf,
            'do_chinh_xac_mo_hinh': 'Xem bao cao danh gia',
            'chi_so_auc': 'Xem bao cao danh gia',
            'prob_num': prob_up
        })

        sentiment_stats[ticker] = {
            'positive': pos_count,
            'negative': neg_count,
            'neutral': neu_count,
            'total_news': len(scores),
            'avg_score': round(avg_score, 4)
        }

    df_res = pd.DataFrame(results)

    # Lưu model artifact (pipeline config + sentiment stats) lên MinIO
    try:
        from model_registry import ModelCard, save_model_to_minio
        total_news = sum(s['total_news'] for s in sentiment_stats.values())
        model_card = ModelCard(
            model_name='FinBERT Sentiment Analysis (ProsusAI/finbert)',
            model_version=f'v{date_nodash}',
            trained_date=today_str,
            model_type='finbert',
            dataset_size=total_news,
            num_tickers=len(sentiment_stats),
            hyperparameters={
                'model_name': 'ProsusAI/finbert',
                'max_length': 128,
                'buy_threshold': 0.55,
                'sell_threshold': 0.45
            },
            metrics={},
            notes=f'Pre-trained. {total_news} tin tuc phan tich'
        )
        import json as _json
        save_model_to_minio(
            s3_hook, BUCKET_NAME, 'finbert', model_card,
            extra_artifacts={
                'pipeline_config.json': _json.dumps(model_card.hyperparameters, indent=2),
                'sentiment_stats.json': _json.dumps(sentiment_stats, indent=2, ensure_ascii=False)
            }
        )
    except Exception as e:
        logging.warning(f'⚠️ Không thể lưu model registry FinBERT: {e}')

    # 1. Lưu kết quả MinIO (Nhiệm vụ cốt lõi)
    csv_buffer = io.StringIO()
    df_res.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
    predict_key = f"finbert/du_bao_tang_truong_finbert_{date_nodash}.csv"

    s3_hook.load_string(
        string_data=csv_buffer.getvalue(),
        key=predict_key,
        bucket_name=BUCKET_NAME,
        replace=True
    )
    logging.info(f"✅ Đã lưu kết quả FinBERT lên MinIO an toàn: {predict_key}")

    # 2. Gửi Telegram với cơ chế chống rớt mạng
    buy_list = df_res[df_res['khuyen_nghi'] == 'MUA']
    sell_list = df_res[df_res['khuyen_nghi'] == 'BÁN']

    msg_lines = [
        "📰 <b>BẢN TIN DỰ BÁO SENTIMENT (FINBERT NLP)</b>",
        f"📅 <b>Dữ liệu phiên:</b> {today_str}",
        "🎯 <b>Mô hình:</b> ProsusAI/finbert (Đánh giá chi tiết tại DAG Model Evaluation)",
        "━━━━━━━━━━━━━━━━━━━━",
        ""
    ]

    if not buy_list.empty:
        msg_lines.append("🟢 <b>TOP CỔ PHIẾU TIN TỨC TÍCH CỰC (MUA):</b>")
        for _, r in buy_list.iterrows():
            msg_lines.append(f"✅ <b>{r['ma_co_phieu']}</b> ({r['nhom_nganh']}): {r['khuyen_nghi']}")
            msg_lines.append(f"   ■ Tỷ lệ tăng: <b>{r['xac_suat_tang_gia']}</b> | Tin cậy: {r['do_tin_cay']}")
        msg_lines.append("")

    if not sell_list.empty:
        msg_lines.append("🔴 <b>CẢNH BÁO TIN TỨC TIÊU CỰC (BÁN):</b>")
        for _, r in sell_list.iterrows():
            msg_lines.append(f"🔻 <b>{r['ma_co_phieu']}</b>: {r['xu_huong_du_kien']} ({r['xac_suat_tang_gia']})")
    else:
        msg_lines.append("🔴 <b>CẢNH BÁO BÁN:</b> Không có cảnh báo tin xấu.")

    # Gửi tin nhắn Text
    url_msg = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    send_telegram_safe(url_msg, data={'chat_id': TELEGRAM_CHAT_ID, 'text': "\n".join(msg_lines), 'parse_mode': 'HTML'})

    # Gửi file CSV
    url_doc = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument"
    csv_bytes = csv_buffer.getvalue().encode('utf-8-sig')
    files = {'document': (f"du_bao_tang_truong_finbert_{date_nodash}.csv", csv_bytes, 'text/csv')}
    send_telegram_safe(url_doc, data={'chat_id': TELEGRAM_CHAT_ID, 'caption': '📊 Bảng phân tích chi tiết FinBERT Sentiment 20 mã'}, files=files)

    # === CHART DỰ BÁO GIÁ CỔ PHIẾU (20 file riêng lẻ + ZIP) ===
    try:
        from chart_utils import generate_individual_charts, save_charts_zip_minio_and_telegram, load_price_data_from_minio

        # FinBERT không có dữ liệu giá → đọc từ dataset LSTM trên MinIO (có cột Close thực)
        df_prices = load_price_data_from_minio(s3_hook, BUCKET_NAME)

        if not df_prices.empty:
            # Thêm prob_num vào results để chart dùng
            for r in results:
                prob_str = r['xac_suat_tang_gia'].replace('%', '')
                r['prob_num'] = float(prob_str) / 100.0

            # Tạo 20 chart riêng lẻ
            chart_files = generate_individual_charts(
                df_price_history=df_prices,
                ticker_predictions=results,
                model_name='FinBERT',
                today_str=today_str
            )

            # Lưu MinIO + nén ZIP + gửi Telegram
            save_charts_zip_minio_and_telegram(
                chart_files=chart_files,
                s3_hook=s3_hook,
                bucket_name=BUCKET_NAME,
                minio_folder=f"finbert/charts_{date_nodash}/",
                zip_filename=f"charts_finbert_{date_nodash}.zip",
                telegram_token=TELEGRAM_TOKEN,
                telegram_chat_id=TELEGRAM_CHAT_ID,
                caption=f'📰 Biểu đồ dự báo giá 20 mã cổ phiếu - FinBERT ({today_str})\n📦 20 tệp PNG trong tệp ZIP',
                send_telegram_fn=send_telegram_safe
            )
        else:
            logging.warning("⚠️ Không có dữ liệu giá để vẽ chart FinBERT")
    except Exception as e:
        logging.warning(f"⚠️ Không thể tạo chart FinBERT: {e}")
        import traceback
        logging.warning(traceback.format_exc())


with DAG(
    dag_id='dag_finbert_predict',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=['finbert', 'prediction', 'minio', 'telegram', 'nlp']
) as dag:

    dag.doc_md = r"""
# 📰 DAG: Suy Luận Cảm Xúc Tài Chính (FinBERT NLP Sentiment)
---
### 1. Tổng Quan Mô Hình NLP
DAG sử dụng mô hình ngôn ngữ lớn chuyên ngành tài chính **FinBERT** (`ProsusAI/finbert`) dựa trên kiến trúc **BERT (Bidirectional Encoder Representations from Transformers)**:
- Đã được pre-train trên kho ngữ liệu tài chính khổng lồ (Financial PhraseBank, SEC 10-K filings, Reuters).
- Phân loại sắc thái cảm xúc cho từng tiêu đề tin tức thành 3 nhãn: `Positive`, `Negative`, `Neutral`.

### 2. Công Thức Tính Điểm & Chuyển Đổi Xác Suất
1. **Sentiment Raw Score:** $\text{score}_i = P(\text{Positive}) - P(\text{Negative}) \in [-1.0, 1.0]$.
2. **Điểm trung bình theo mã:** $\overline{S} = \frac{1}{N}\sum_{i=1}^N \text{score}_i$.
3. **Xác suất tăng giá ước lượng:** $P_{\text{up}} = \text{clip}\left(\frac{\overline{S} + 1.0}{2.0}, 0.05, 0.95\right)$.
4. **Ngưỡng hành động:**
   - $P_{\text{up}} \ge 54\%$ ➔ Khuyến nghị **MUA** (Tăng mạnh nếu $\ge 60\%$).
   - $P_{\text{up}} \le 46\%$ ➔ Khuyến nghị **BÁN** (Giảm mạnh nếu $\le 40\%$).
   - Khoảng $46\% - 54\%$ ➔ **ĐỨNG NGOÀI (Sideway)**.

### 3. Đầu Ra & Trực Quan Hóa
- **File dự báo:** `stock-xgboost-data/finbert/du_bao_tang_truong_finbert_YYYYMMDD.csv`
- **Báo cáo phân tích:** Gửi bảng phân loại sắc thái tin tức 20 mã và CSV qua Telegram.
- **Model Registry:** Lưu `pipeline_config.json` và `sentiment_stats.json`.
"""

    predict_finbert = PythonOperator(
        task_id='predict_finbert_sentiment',
        python_callable=predict_finbert_task,
        provide_context=True,
    )
