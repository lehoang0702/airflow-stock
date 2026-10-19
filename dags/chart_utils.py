"""
Chart Utilities - Module dùng chung cho các DAG ML
Thiết kế Dual-Panel (Khung kép):
- Panel 1 (Trái): TOÀN CẢNH LỊCH SỬ (Macro View) từ đầu đến hiện tại.
- Panel 2 (Phải): CẬN CẢNH DỰ BÁO 1 TUẦN (Forecast Zoom) mở rộng không gian dự báo,
  hiển thị rõ các cột mốc: Hiện tại, +2 ngày, +4 ngày, +1 tuần kèm giá và % biến động.
"""
import io
import logging
import zipfile
import numpy as np
import pandas as pd

SECTOR_MAP_VN = {
    'Technology': 'Công nghệ',
    'Consumer Discretionary': 'Tiêu dùng',
    'Consumer Staples': 'Hàng thiết yếu',
    'Financials': 'Tài chính',
    'Healthcare': 'Y tế',
    'Industrials': 'Công nghiệp',
    'Energy': 'Năng lượng',
    'Utilities': 'Tiện ích',
    'Materials': 'Vật liệu cơ bản'
}


def _render_single_ticker_dual_chart(
    fig, ax1, ax2, ticker, ticker_df, pred, model_name, today_str, prediction_days=5
):
    """Draw observed prices and actual model outputs, without invented paths."""
    import matplotlib.dates as mdates
    if ticker_df.empty:
        for ax in (ax1, ax2):
            ax.text(.5, .5, 'Chưa có dữ liệu giá', transform=ax.transAxes, ha='center', color='white')
        return
    history = ticker_df.sort_values('Date').dropna(subset=['Date', 'Close'])
    for ax, data, title in ((ax1, history, 'Lịch sử giá thực tế'),
                            (ax2, history.tail(60), 'Giá đóng cửa các phiên gần nhất')):
        ax.set_facecolor('#161b22')
        ax.plot(pd.to_datetime(data['Date']), data['Close'], color='#58a6ff', label='Giá quan sát')
        ax.set_title(f'{ticker} — {title}', color='white', fontsize=11)
        ax.tick_params(colors='#8b949e', labelsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
        ax.grid(alpha=.15)
        ax.legend(facecolor='#161b22', labelcolor='white', fontsize=8)
    probability = pred.get('prob_num')
    has_probability = probability is not None and np.isfinite(float(probability))
    label = f'{model_name}: {float(probability)*100:.2f}%' if has_probability else f'{model_name}: Chưa có kết quả'
    target = pred.get('gia_5d_du_kien')
    if target is not None and pd.notna(target) and pred.get('return_model_version'):
        label += f' | Giá T+5 từ mô hình: ${float(target):.2f}'
    else:
        label += ' | Chưa có mô hình dự báo đường giá'
    fig.suptitle(label, color='white', fontsize=10)


def _render_ensemble_dual_chart(
    fig, ax1, ax2, ticker, ticker_df, pred, xgb_probs, lstm_probs, bert_probs, today_str, prediction_days=5
):
    _render_single_ticker_dual_chart(fig, ax1, ax2, ticker, ticker_df, pred,
                                     'Master Ensemble — Điểm tổng hợp 35/35/30' if pred.get('ensemble_method') == 'weighted_35_35_30_v1' else 'Ensemble đã huấn luyện', today_str, prediction_days)


def generate_individual_charts(
    df_price_history,
    ticker_predictions,
    model_name,
    today_str,
    prediction_days=5
):
    """
    Tạo 20 file ảnh PNG riêng biệt (khung kép: Toàn cảnh + Kính lúp dự báo 1 tuần).

    Args:
        df_price_history: DataFrame có cột ['Date', 'Ticker', 'Close']
        ticker_predictions: list of dict [{ma_co_phieu, prob_num, khuyen_nghi, nhom_nganh}, ...]
        model_name: 'XGBoost' / 'LSTM' / 'FinBERT'
        today_str: '2024-01-01'
        prediction_days: Số ngày trading dự đoán (mặc định 5)

    Returns:
        dict: {ticker_name: png_bytes}
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    preds_by_ticker = {p['ma_co_phieu']: p for p in ticker_predictions}
    tickers = sorted(preds_by_ticker.keys())

    chart_files = {}

    for ticker in tickers:
        pred = preds_by_ticker[ticker]

        # Khung hình 18x7 inches (widescreen), phân chia tỷ lệ 1 : 1.15
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={'width_ratios': [1, 1.15]})
        fig.patch.set_facecolor('#0d1117')

        ticker_df = df_price_history[df_price_history['Ticker'] == ticker].sort_values('Date').copy()

        _render_single_ticker_dual_chart(
            fig, ax1, ax2, ticker, ticker_df, pred, model_name, today_str, prediction_days
        )

        plt.tight_layout(rect=[0, 0, 1, 0.94])

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)

        chart_files[ticker] = buf.getvalue()

    logging.info(f"📊 Đã tạo {len(chart_files)} chart Dual-Panel riêng lẻ cho {model_name}")
    return chart_files


def generate_individual_ensemble_charts(
    df_price_history,
    xgb_probs,
    lstm_probs,
    bert_probs,
    ensemble_predictions,
    today_str,
    prediction_days=5
):
    """
    Tạo 20 file ảnh PNG riêng biệt cho Ensemble (khung kép: Toàn cảnh + 4 đường dự đoán).

    Returns:
        dict: {ticker_name: png_bytes}
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    preds_by_ticker = {p['ma_co_phieu']: p for p in ensemble_predictions}
    tickers = sorted(preds_by_ticker.keys())

    chart_files = {}

    for ticker in tickers:
        pred = preds_by_ticker[ticker]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={'width_ratios': [1, 1.15]})
        fig.patch.set_facecolor('#0d1117')

        ticker_df = df_price_history[df_price_history['Ticker'] == ticker].sort_values('Date').copy()

        _render_ensemble_dual_chart(
            fig, ax1, ax2, ticker, ticker_df, pred, xgb_probs, lstm_probs, bert_probs, today_str, prediction_days
        )

        plt.tight_layout(rect=[0, 0, 1, 0.94])

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)

        chart_files[ticker] = buf.getvalue()

    logging.info(f"📊 Đã tạo {len(chart_files)} chart Ensemble Dual-Panel riêng lẻ")
    return chart_files


def calculate_trading_plan(last_price, prob_up, prediction_days=5, predicted_target_price=None):
    """Only accept a trained return model's endpoint; risk forecasts stay absent."""
    target = (float(predicted_target_price) if predicted_target_price is not None
              and np.isfinite(predicted_target_price) and predicted_target_price > 0 else None)
    return {'entry': float(last_price), 'target': target,
            'pct_target': (target / last_price - 1) * 100 if target is not None else None,
            'stop_loss': None, 'sl_pct': None, 'rr_ratio': None, 'rec_type': None}


def send_telegram_media_group(telegram_token, telegram_chat_id, media_files, caption=""):
    """
    Gửi album ảnh lướt (Media Group) lên Telegram (tối đa 10 ảnh / album).
    media_files: list of (filename_str, png_bytes)
    """
    import json
    import requests

    if not media_files:
        return

    url = f"https://api.telegram.org/bot{telegram_token}/sendMediaGroup"
    media = []
    files = {}
    for i, (name, b_data) in enumerate(media_files[:10]):
        file_key = f"photo_{i}"
        files[file_key] = (f"{name}.png", b_data, "image/png")
        item = {
            "type": "photo",
            "media": f"attach://{file_key}"
        }
        if i == 0 and caption:
            item["caption"] = caption
            item["parse_mode"] = "HTML"
        media.append(item)

    for attempt in range(1, 3):
        try:
            resp = requests.post(url, data={"chat_id": telegram_chat_id, "media": json.dumps(media)}, files=files, timeout=(20, 90))
            if resp.status_code == 200:
                logging.info(f"✅ Đã gửi album Carousel {len(media)} ảnh lên Telegram thành công!")
                return
            else:
                logging.warning(f"⚠️ Gửi album Telegram trả về status {resp.status_code}: {resp.text}")
        except Exception as e:
            logging.warning(f"⚠️ Thử gửi album lần {attempt} thất bại: {e}")
            import time
            time.sleep(3)


def save_charts_zip_minio_and_telegram(
    chart_files, s3_hook, bucket_name,
    minio_folder, zip_filename,
    telegram_token, telegram_chat_id,
    caption, send_telegram_fn,
    top_tickers=None, album_caption=None
):
    """
    Lưu từng chart PNG lên MinIO + nén ZIP + gửi ZIP qua Telegram.
    """
    # 1. Lưu từng file PNG lên MinIO
    for ticker, png_bytes in chart_files.items():
        key = f"{minio_folder}{ticker}.png"
        s3_hook.load_bytes(
            bytes_data=png_bytes,
            key=key,
            bucket_name=bucket_name,
            replace=True
        )
    logging.info(f"✅ Đã lưu {len(chart_files)} chart PNG lên MinIO: {minio_folder}")

    # 2. Nén ZIP toàn bộ 20 mã
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for ticker, png_bytes in sorted(chart_files.items()):
            zf.writestr(f"{ticker}.png", png_bytes)
    zip_buffer.seek(0)
    zip_bytes = zip_buffer.getvalue()

    # 3. Lưu ZIP lên MinIO
    zip_key = f"{minio_folder}{zip_filename}"
    s3_hook.load_bytes(
        bytes_data=zip_bytes,
        key=zip_key,
        bucket_name=bucket_name,
        replace=True
    )
    logging.info(f"✅ Đã lưu ZIP lên MinIO: {zip_key} ({len(zip_bytes)} bytes)")

    # 4. Gửi ZIP qua Telegram
    url = f"https://api.telegram.org/bot{telegram_token}/sendDocument"
    files = {'document': (zip_filename, zip_bytes, 'application/zip')}
    send_telegram_fn(url, data={'chat_id': telegram_chat_id, 'caption': caption}, files=files)
    logging.info(f"✅ Đã gửi ZIP {len(chart_files)} chart qua Telegram")


def load_price_data_from_minio(s3_hook, bucket_name, prefix="lstm/lstm_stock_20tickers_10y_"):
    """
    Đọc dữ liệu giá lịch sử từ MinIO dataset LSTM (có cột Close thực).
    Fallback: thử XGBoost dataset nếu LSTM không có.
    Hỗ trợ Smart Backfill: Nếu file mới nhất thiếu mã, tự động đọc thêm từ file lịch sử để đủ 20 mã.
    """
    from pipeline_data import normalize_price_history
    EXPECTED_TICKERS = [
        'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'JPM', 'V', 'JNJ', 'UNH', 'XOM',
        'CVX', 'PG', 'KO', 'WMT', 'MCD', 'NKE', 'CAT', 'BA', 'NEE', 'LIN'
    ]
    keys = s3_hook.list_keys(bucket_name=bucket_name, prefix=prefix)

    if not keys and "lstm" in prefix:
        logging.warning(f"⚠️ Không tìm thấy LSTM dataset, thử XGBoost dataset...")
        keys = s3_hook.list_keys(bucket_name=bucket_name, prefix="xgboost/xgboost_stock_20tickers_10y_")

    if not keys:
        logging.warning(f"⚠️ Không tìm thấy dataset giá nào trên MinIO")
        return pd.DataFrame(columns=['Date', 'Ticker', 'Close'])

    sorted_keys = sorted(key for key in keys if key.endswith('.csv'))

    # 1. Tìm file mới nhất có đủ 20 mã
    df = None
    selected_key = None
    for k in reversed(sorted_keys):
        try:
            raw_csv = s3_hook.read_key(k, bucket_name=bucket_name)
            temp_df = pd.read_csv(io.StringIO(raw_csv))
            temp_df = normalize_price_history(temp_df)
            if temp_df['Ticker'].nunique() >= len(EXPECTED_TICKERS):
                df = temp_df
                selected_key = k
                break
            if not temp_df.empty and df is None:
                df = temp_df
                selected_key = k
        except Exception:
            continue

    # 2. Nếu không có file nào đủ 20 mã, lấy file mới nhất và chuẩn bị backfill
    if df is None:
        logging.warning('Không có dataset giá hợp lệ với ngày, mã cổ phiếu và giá đóng cửa.')
        return pd.DataFrame(columns=['Date', 'Ticker', 'Close'])
    logging.info(f"📊 Đã chọn dataset giá hợp lệ: {selected_key} ({len(df):,} dòng)")
    result = df.copy()

    # 3. SMART BACKFILL: Kiểm tra xem có mã nào trong EXPECTED_TICKERS bị thiếu không
    found_tickers = set(result['Ticker'].unique())
    missing_tickers = set(EXPECTED_TICKERS) - found_tickers
    if missing_tickers and len(sorted_keys) > 1:
        logging.warning(f"⚠️ Dataset {selected_key} thiếu {len(missing_tickers)} mã: {missing_tickers}. Đang tự động bù từ các file lịch sử...")
        for k in reversed(sorted_keys):
            if k == selected_key:
                continue
            if not missing_tickers:
                break
            try:
                raw_prev = s3_hook.read_key(k, bucket_name=bucket_name)
                df_prev = normalize_price_history(pd.read_csv(io.StringIO(raw_prev)))
                if not df_prev.empty:
                    for mt in list(missing_tickers):
                        m_df = df_prev[df_prev['Ticker'] == mt]
                        if not m_df.empty:
                            result = pd.concat([result, m_df], ignore_index=True)
                            missing_tickers.remove(mt)
                            logging.info(f"🛡️ Đã bù thành công mã {mt} từ {k} ({len(m_df)} phiên)")
            except Exception as e:
                logging.warning(f"Lỗi khi đọc file bù {k}: {e}")

    logging.info(f"📊 Đã nạp dữ liệu giá hoàn tất: {len(result):,} dòng, {result['Ticker'].nunique()}/{len(EXPECTED_TICKERS)} mã")
    return result
