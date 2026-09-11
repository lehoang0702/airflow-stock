"""
DAG: ĐỐI CHIẾU 3 PHƯƠNG PHÁP ĐỘC LẬP & TẠO PHƯƠNG PHÁP 4 TỔNG HỢP
- Lưu file PP4: stock-xgboost-data/ensemble/du_bao_tang_truong_ensemble_YYYYMMDD.csv
- Lưu file So sánh: stock-xgboost-data/comparison/bang_so_sanh_4_phuong_phap_YYYYMMDD.csv
- Gửi báo cáo so sánh 4 phương pháp về Telegram
"""
from datetime import datetime, timedelta
import io
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
    from alert_utils import telegram_failure_callback, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
except ImportError:
    telegram_failure_callback = None
    TELEGRAM_TOKEN = ""
    TELEGRAM_CHAT_ID = ""

default_args = {
    'owner': 'quant_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'on_failure_callback': telegram_failure_callback,
}

try:
    from config_shared import SECTOR_MAP
except ImportError:
    SECTOR_MAP = {
        'AAPL': 'Technology', 'MSFT': 'Technology', 'NVDA': 'Technology', 'GOOGL': 'Technology',
        'AMZN': 'Consumer Discretionary', 'NKE': 'Consumer Discretionary', 'MCD': 'Consumer Discretionary',
        'WMT': 'Consumer Staples', 'PG': 'Consumer Staples', 'KO': 'Consumer Staples',
        'JPM': 'Financials', 'V': 'Financials',
        'UNH': 'Healthcare', 'JNJ': 'Healthcare',
        'CAT': 'Industrials', 'BA': 'Industrials',
        'XOM': 'Energy', 'CVX': 'Energy',
        'NEE': 'Utilities', 'LIN': 'Materials'
    }

def compare_and_ensemble_task(**context):
    s3_hook = S3Hook(aws_conn_id=MINIO_CONN_ID)
    date_nodash = datetime.now().strftime("%Y%m%d")
    today_str = datetime.now().strftime("%Y-%m-%d")

    # 1. Đọc dữ liệu giá lịch sử và kết quả từ 3 phương pháp độc lập
    from chart_utils import (
        generate_individual_ensemble_charts,
        save_charts_zip_minio_and_telegram,
        load_price_data_from_minio,
        calculate_trading_plan
    )

    df_prices = load_price_data_from_minio(s3_hook, BUCKET_NAME)
    latest_prices = {}
    if not df_prices.empty:
        latest_rows = df_prices.sort_values(['Ticker', 'Date']).groupby('Ticker').last().reset_index()
        latest_prices = dict(zip(latest_rows['Ticker'], latest_rows['Close']))

    def load_latest(prefix):
        keys = s3_hook.list_keys(bucket_name=BUCKET_NAME, prefix=prefix)
        if not keys:
            raise FileNotFoundError(f"❌ Không tìm thấy file: {prefix}")

        valid_key = None
        for k in reversed(sorted(keys)):
            try:
                data = s3_hook.read_key(k, bucket_name=BUCKET_NAME)
                df_test = pd.read_csv(io.StringIO(data))
                if len(df_test) >= 5:
                    valid_key = k
                    break
            except Exception:
                continue

        if not valid_key:
            raise ValueError(f"❌ Không tìm thấy file dữ liệu hợp lệ cho {prefix}!")

        # Kiểm tra độ tươi mới của dữ liệu
        file_date_str = valid_key.split('_')[-1].replace('.csv', '')
        if file_date_str != date_nodash:
            logging.warning(f"⚠️ Chú ý: File {valid_key} là phiên {file_date_str} (Hôm nay: {date_nodash})")

        data = s3_hook.read_key(valid_key, bucket_name=BUCKET_NAME)
        logging.info(f"📥 Đã đọc file: {valid_key}")
        return pd.read_csv(io.StringIO(data))

    df_xgb = load_latest("xgboost/du_bao_tang_truong_")
    df_lstm = load_latest("lstm/du_bao_tang_truong_lstm_")
    df_bert = load_latest("finbert/du_bao_tang_truong_finbert_")

    def parse_prob(val):
        if pd.isna(val): return 0.5
        if isinstance(val, (float, int)): return float(val)
        return float(str(val).replace('%', '').strip()) / 100.0

    df_xgb['prob_xgb'] = df_xgb['xac_suat_tang_gia'].apply(parse_prob)
    df_lstm['prob_lstm'] = df_lstm['xac_suat_tang_gia'].apply(parse_prob)
    df_bert['prob_bert'] = df_bert['xac_suat_tang_gia'].apply(parse_prob)

    # 2. Ghép 3 bảng độc lập (dùng outer join để giữ đủ 20 mã)
    merged = pd.merge(
        df_xgb[['ma_co_phieu', 'nhom_nganh', 'khuyen_nghi', 'xac_suat_tang_gia', 'prob_xgb']].rename(
            columns={'khuyen_nghi': 'xgb_khuyen_nghi', 'xac_suat_tang_gia': 'xgb_xac_suat'}
        ),
        df_lstm[['ma_co_phieu', 'khuyen_nghi', 'xac_suat_tang_gia', 'prob_lstm']].rename(
            columns={'khuyen_nghi': 'lstm_khuyen_nghi', 'xac_suat_tang_gia': 'lstm_xac_suat'}
        ),
        on='ma_co_phieu', how='outer'
    )

    merged = pd.merge(
        merged,
        df_bert[['ma_co_phieu', 'khuyen_nghi', 'xac_suat_tang_gia', 'prob_bert']].rename(
            columns={'khuyen_nghi': 'bert_khuyen_nghi', 'xac_suat_tang_gia': 'bert_xac_suat'}
        ),
        on='ma_co_phieu', how='outer'
    )

    # Chuẩn hóa nhóm ngành nếu thiếu
    merged['nhom_nganh'] = merged['nhom_nganh'].fillna(merged['ma_co_phieu'].map(SECTOR_MAP)).fillna('General')

    # 3. TÍNH TOÁN PHƯƠNG PHÁP 4 (TỔNG HỢP ENSEMBLE TỰ ĐỘNG CHUẨN HÓA TRỌNG SỐ)
    def calc_ensemble_prob(row):
        weights = []
        probs = []
        if pd.notna(row['prob_xgb']):
            probs.append(row['prob_xgb'])
            weights.append(0.35)
        if pd.notna(row['prob_lstm']):
            probs.append(row['prob_lstm'])
            weights.append(0.35)
        if pd.notna(row['prob_bert']):
            probs.append(row['prob_bert'])
            weights.append(0.30)
        if not probs:
            return 0.5
        total_w = sum(weights)
        return sum(p * (w / total_w) for p, w in zip(probs, weights))

    merged['prob_ensemble'] = merged.apply(calc_ensemble_prob, axis=1)
    merged['xgb_khuyen_nghi'] = merged['xgb_khuyen_nghi'].fillna('ĐỨNG NGOÀI')
    merged['xgb_xac_suat'] = merged['xgb_xac_suat'].fillna('50.00%')
    merged['lstm_khuyen_nghi'] = merged['lstm_khuyen_nghi'].fillna('ĐỨNG NGOÀI')
    merged['lstm_xac_suat'] = merged['lstm_xac_suat'].fillna('50.00%')
    merged['bert_khuyen_nghi'] = merged['bert_khuyen_nghi'].fillna('ĐỨNG NGOÀI')
    merged['bert_xac_suat'] = merged['bert_xac_suat'].fillna('50.00%')

    def get_ensemble_rec(prob):
        # Chuẩn hóa ngưỡng quyết định (Confidence Threshold Filtering)
        # Lọc nhiễu vùng phân vân 0.45 - 0.55 -> ĐỨNG NGOÀI, chỉ MUA khi >= 0.55, BÁN khi <= 0.45
        if prob >= 0.55:
            return "MUA", "Tăng mạnh" if prob >= 0.60 else "Tăng tích lũy", "Rất cao" if prob >= 0.60 else "Cao"
        elif prob <= 0.45:
            return "BÁN", "Giảm mạnh" if prob <= 0.40 else "Giảm phân phối", "Rất cao" if prob <= 0.40 else "Cao"
        else:
            return "ĐỨNG NGOÀI", "Đi ngang / Lưỡng lự (Sideway)", "Trung bình"

    ensemble_list = []
    compare_list = []

    for _, row in merged.iterrows():
        action, trend, conf = get_ensemble_rec(row['prob_ensemble'])

        ensemble_list.append({
            'ngay_du_bao': today_str,
            'ma_co_phieu': row['ma_co_phieu'],
            'nhom_nganh': row['nhom_nganh'],
            'khuyen_nghi': action,
            'xac_suat_tang_gia': f"{row['prob_ensemble']*100:.2f}%",
            'xu_huong_du_kien': trend,
            'do_tin_cay': conf,
            'do_chinh_xac_mo_hinh': 'Đa mô hình',
            'chi_so_auc': 'Tối ưu'
        })

        recs = [row['xgb_khuyen_nghi'], row['lstm_khuyen_nghi'], row['bert_khuyen_nghi']]
        n_mua = recs.count('MUA')
        n_ban = recs.count('BÁN') + recs.count('BAN')

        if n_mua >= 2:
            status = "🟢 MUA ĐỒNG THUẬN" if n_mua == 3 else "🟢 MUA (ĐA SỐ)"
        elif n_ban >= 2:
            status = "🔴 BÁN ĐỒNG THUẬN" if n_ban == 3 else "🔴 BÁN (ĐA SỐ)"
        else:
            status = "🟡 PHÂN HÓA"

        compare_list.append({
            'ma_co_phieu': row['ma_co_phieu'],
            'nhom_nganh': row['nhom_nganh'],
            'pp1_xgboost': f"{row['xgb_khuyen_nghi']} ({row['xgb_xac_suat']})",
            'pp2_lstm': f"{row['lstm_khuyen_nghi']} ({row['lstm_xac_suat']})",
            'pp3_finbert': f"{row['bert_khuyen_nghi']} ({row['bert_xac_suat']})",
            'pp4_tong_hop': f"{action} ({row['prob_ensemble']*100:.2f}%)",
            'trang_thai_doi_chieu': status,
            'prob_ensemble': row['prob_ensemble']
        })

    df_ens = pd.DataFrame(ensemble_list)
    df_compare = pd.DataFrame(compare_list)

    # 4. Lưu File PP4 lên MinIO
    csv_buf_ens = io.StringIO()
    df_ens.to_csv(csv_buf_ens, index=False, encoding='utf-8-sig')
    ens_key = f"ensemble/du_bao_tang_truong_ensemble_{date_nodash}.csv"
    s3_hook.load_string(string_data=csv_buf_ens.getvalue(), key=ens_key, bucket_name=BUCKET_NAME, replace=True)
    logging.info(f"✅ Đã lưu PP4 Ensemble lên MinIO: {ens_key}")

    # 5. Lưu File So Sánh lên MinIO
    csv_buf_cmp = io.StringIO()
    df_compare.drop(columns=['prob_ensemble']).to_csv(csv_buf_cmp, index=False, encoding='utf-8-sig')
    cmp_key = f"comparison/bang_so_sanh_4_phuong_phap_{date_nodash}.csv"
    s3_hook.load_string(string_data=csv_buf_cmp.getvalue(), key=cmp_key, bucket_name=BUCKET_NAME, replace=True)
    logging.info(f"✅ Đã lưu Bảng So Sánh 4 Phương Pháp lên MinIO: {cmp_key}")

    # 6. Tạo nội dung báo cáo Telegram KÈM KẾ HOẠCH GIAO DỊCH (ENTRY, TARGET, STOP-LOSS)
    all_buys = df_compare[df_compare['trang_thai_doi_chieu'].str.contains('MUA')].sort_values('prob_ensemble', ascending=False)
    conflicts = df_compare[df_compare['trang_thai_doi_chieu'].str.contains('PHÂN HÓA')]
    all_sells = df_compare[df_compare['trang_thai_doi_chieu'].str.contains('BÁN')].sort_values('prob_ensemble', ascending=True)

    msg_lines = [
        "👑 <b>BẢNG ĐỐI CHIẾU 4 PHƯƠNG PHÁP & KẾ HOẠCH GIAO DỊCH</b>",
        "<i>[PP1: XGBoost] | [PP2: LSTM] | [PP3: FinBERT] | [PP4: Tổng Hợp]</i>",
        f"📅 <b>Dữ liệu phiên:</b> {today_str}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        ""
    ]

    if not all_buys.empty:
        msg_lines.append("🟢 <b>CÁC MÃ ĐỒNG THUẬN TĂNG (MUA) & KẾ HOẠCH T+5:</b>")
        for _, r in all_buys.iterrows():
            ticker = r['ma_co_phieu']
            prob_e = float(r['prob_ensemble'])
            cur_p = latest_prices.get(ticker, 0.0)

            msg_lines.append(f"🔹 <b>{ticker}</b> ({r['nhom_nganh']}) - <code>{r['trang_thai_doi_chieu']}</code>")
            if cur_p > 0:
                plan = calculate_trading_plan(cur_p, prob_e)
                msg_lines.append(f"   ├ 💰 Giá vào (Entry): <b>${plan['entry']:.2f}</b>")
                msg_lines.append(f"   ├ 🎯 Mục tiêu T+5: <b>${plan['target']:.2f} (+{plan['pct_target']}%)</b>")
                msg_lines.append(f"   ├ 🛑 Cắt lỗ (SL 2%): <b>${plan['stop_loss']:.2f} (-{plan['sl_pct']}%)</b> | R:R: <b>1:{plan['rr_ratio']}</b>")
            msg_lines.append(f"   ├ <i>XGB: {r['pp1_xgboost']} | LSTM: {r['pp2_lstm']} | BERT: {r['pp3_finbert']}</i>")
            msg_lines.append(f"   └ 👉 <b>PP4 TỔNG HỢP:</b> <b>{r['pp4_tong_hop']}</b>")
        msg_lines.append("")

    if not conflicts.empty:
        msg_lines.append("🟡 <b>CÁC MÃ CÓ SỰ PHÂN HÓA (ĐỐI CHIẾU PP4):</b>")
        for _, r in conflicts.iterrows():
            msg_lines.append(f"🔸 <b>{r['ma_co_phieu']}</b> ({r['nhom_nganh']}):")
            msg_lines.append(f"   ├ XGB: {r['pp1_xgboost']} | LSTM: {r['pp2_lstm']} | BERT: {r['pp3_finbert']}")
            msg_lines.append(f"   └ 👉 <b>PP4 Tổng Hợp chốt:</b> <b>{r['pp4_tong_hop']}</b>")
        msg_lines.append("")

    if not all_sells.empty:
        msg_lines.append("🔴 <b>CÁC MÃ CẢNH BÁO GIẢM / BÁN:</b>")
        for _, r in all_sells.iterrows():
            ticker = r['ma_co_phieu']
            cur_p = latest_prices.get(ticker, 0.0)
            p_text = f" (${cur_p:.2f})" if cur_p > 0 else ""
            msg_lines.append(f"🔻 <b>{ticker}</b>{p_text}: {r['trang_thai_doi_chieu']} ➔ PP4: {r['pp4_tong_hop']}")
    else:
        msg_lines.append("🔴 <b>CẢNH BÁO BÁN:</b> Không có mã nào bị đồng thuận bán.")
    # === CẬP NHẬT DANH MỤC PAPER TRADING T+5 ===
    try:
        from portfolio_tracker import update_virtual_portfolio, format_portfolio_telegram_section
        port_summary = update_virtual_portfolio(
            ensemble_df=df_compare,
            current_prices=latest_prices,
            s3_hook=s3_hook,
            bucket_name=BUCKET_NAME,
            today_str=today_str
        )
        port_text = format_portfolio_telegram_section(port_summary)
        if port_text:
            msg_lines.append("")
            msg_lines.append(port_text)
    except Exception as port_err:
        logging.warning(f"⚠️ Không thể cập nhật Paper Trading: {port_err}")

    # Bắn tin nhắn HTML
    url_msg = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url_msg, data={'chat_id': TELEGRAM_CHAT_ID, 'text': "\n".join(msg_lines), 'parse_mode': 'HTML'})

    # Gửi đính kèm File CSV Bảng So Sánh 4 Phương Pháp
    url_doc = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument"
    csv_bytes = csv_buf_cmp.getvalue().encode('utf-8-sig')
    files = {'document': (f"bang_so_sanh_4_phuong_phap_{date_nodash}.csv", csv_bytes, 'text/csv')}
    requests.post(url_doc, data={'chat_id': TELEGRAM_CHAT_ID, 'caption': '📊 Bảng đối chiếu chi tiết 4 Phương Pháp (3 Độc Lập + 1 Tổng Hợp)'}, files=files)

    # === CHART TỔNG HỢP DỰ BÁO GIÁ CỔ PHIẾU (ZIP) ===
    try:
        if not df_prices.empty:
            xgb_probs = dict(zip(merged['ma_co_phieu'], merged['prob_xgb']))
            lstm_probs = dict(zip(merged['ma_co_phieu'], merged['prob_lstm']))
            bert_probs = dict(zip(merged['ma_co_phieu'], merged['prob_bert']))

            for item in ensemble_list:
                prob_str = item['xac_suat_tang_gia'].replace('%', '')
                item['prob_num'] = float(prob_str) / 100.0

            chart_files = generate_individual_ensemble_charts(
                df_price_history=df_prices,
                xgb_probs=xgb_probs,
                lstm_probs=lstm_probs,
                bert_probs=bert_probs,
                ensemble_predictions=ensemble_list,
                today_str=today_str
            )

            def _send_tg_safe(url, data=None, files=None):
                try:
                    requests.post(url, data=data, files=files, timeout=(10, 30))
                except Exception:
                    pass

            # Lưu MinIO + gửi ZIP qua Telegram
            save_charts_zip_minio_and_telegram(
                chart_files=chart_files,
                s3_hook=s3_hook,
                bucket_name=BUCKET_NAME,
                minio_folder=f"ensemble/charts_{date_nodash}/",
                zip_filename=f"charts_ensemble_{date_nodash}.zip",
                telegram_token=TELEGRAM_TOKEN,
                telegram_chat_id=TELEGRAM_CHAT_ID,
                caption=f'👑 Biểu đồ tổng hợp dự báo 4 mô hình - Ensemble Master ({today_str})\n📦 20 tệp PNG trong tệp ZIP',
                send_telegram_fn=_send_tg_safe
            )
        else:
            logging.warning("⚠️ Không có dữ liệu giá để vẽ chart Ensemble")
    except Exception as e:
        logging.warning(f"⚠️ Không thể tạo chart Ensemble: {e}")


with DAG(
    dag_id='dag_ensemble_master',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=['so_sanh', '4_phuong_phap', 'ensemble', 'xgboost', 'lstm', 'finbert', 'telegram']
) as dag:

    dag.doc_md = r"""
# 👑 DAG: Hợp Nhất Đa Phương Thức (Master Multi-Modal Ensemble)
---
### 1. Tổng Quan Triết Lý Đầu Tư & Hợp Nhất
DAG này là trái tim của hệ thống **Multi-Modal Quantitative Pipeline**, giải quyết bài toán dung hợp các nguồn thông tin phân tán:
- **Phương pháp 1 (Tabular):** `XGBoost` phân tích ma trận chỉ báo kỹ thuật và kinh tế vĩ mô.
- **Phương pháp 2 (Time-Series):** `LSTM` khai thác quy luật chuỗi thời gian nến và động lượng.
- **Phương pháp 3 (NLP):** `FinBERT` nắm bắt tin tức và tâm lý thị trường tức thời.
- **Phương pháp 4 (Master Ensemble):** Hợp nhất có trọng số (Weighted Soft Voting) và biểu quyết đa số (Consensus Voting).

### 2. Trọng Số & Cơ Chế Biểu Quyết
- **Xác suất hợp nhất:**
  $$P_{\text{ensemble}} = 0.40 \times P_{\text{XGB}} + 0.35 \times P_{\text{LSTM}} + 0.25 \times P_{\text{FinBERT}}$$
- **Phân loại tín hiệu:**
  - $P \ge 53\%$ ➔ **MUA** (Độ tin cậy Cao nếu $\ge 56\%$).
  - $P \le 47\%$ ➔ **BÁN** (Độ tin cậy Cao nếu $\le 44\%$).
  - Khoảng $47\% - 53\%$ ➔ **ĐỨNG NGOÀI (Sideway)**.
- **Tính toán Kế hoạch Giao dịch (Trading Plan):** Tự động tính toán Điểm vào (Entry), Điểm chốt lời (Take Profit: +3.5% / +5.0%) và Điểm cắt lỗ (Stop Loss: -2.0% / -2.5%) theo tỷ lệ R:R tối ưu 1:2.

### 3. Đầu Ra MinIO & Kênh Báo Cáo
- **Bảng so sánh 4 phương pháp:** `comparison/bang_so_sanh_4_phuong_phap_YYYYMMDD.csv`
- **File dự báo tổng hợp:** `ensemble/du_bao_tang_truong_ensemble_YYYYMMDD.csv`
- **Bộ biểu đồ 4-in-1:** Lưu ZIP `charts_ensemble_YYYYMMDD.zip` lên MinIO và gửi tệp ZIP + CSV qua Telegram.
"""

    run_compare_ensemble = PythonOperator(
        task_id='compare_and_ensemble_all_models',
        python_callable=compare_and_ensemble_task,
        provide_context=True,
    )
