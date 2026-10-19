"""
CHARTS & VISUALIZATION — QUANTUM DASHBOARD
===========================================
Tạo và hiển thị biểu đồ kỹ thuật khung kép (Dual-Panel) và lấy ảnh biểu đồ từ S3 / on-the-fly.
"""

import io
import time
import logging
from datetime import datetime
import pandas as pd

from dashboard.config import _ON_DEMAND_PREDICTIONS
from sentiment_utils import sentiment_scores

try:
    from chart_utils import _render_ensemble_dual_chart, _render_single_ticker_dual_chart
except ImportError:
    from dags.chart_utils import _render_ensemble_dual_chart, _render_single_ticker_dual_chart

_CHART_CACHE = {}


def generate_live_chart_image(ticker: str, model: str = "ensemble"):
    """
    Hiển thị lịch sử giá quan sát được và xác suất từ mô hình đã huấn luyện.
    Không dựng đường giá tương lai khi chưa có mô hình dự báo tương ứng.
    """
    from dashboard.services import get_ticker_chart

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import yfinance as yf

        ticker = ticker.strip().upper()
        model = model.lower().strip()
        if model not in ('ensemble', 'xgboost', 'lstm', 'finbert'):
            model = 'ensemble'

        pred_data = _ON_DEMAND_PREDICTIONS.get(ticker)
        if not pred_data and model == 'finbert':
            from dashboard.services import get_market_data
            pred_data = next((row for row in get_market_data().get('predictions', []) if row['ticker'] == ticker), None)
        if not pred_data:
            from dashboard.jobs import analysis_job
            analysis_job(ticker=ticker)
            return None

        if model == 'finbert':
            scores = sentiment_scores(pred_data.get('finbert_sentiment', {}))
            if not scores:
                return None
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor('#0d1117')
            ax.set_facecolor('#161b22')
            values = [value * 100 for value in scores.values()]
            bars = ax.barh(['Tích cực', 'Tiêu cực', 'Trung tính'], values,
                           color=['#10b981', '#f87171', '#94a3b8'])
            ax.invert_yaxis()
            ax.set_xlim(0, 110)
            ax.set_xticks([0, 25, 50, 75, 100], ['0%', '25%', '50%', '75%', '100%'])
            ax.tick_params(colors='#c9d1d9')
            ax.bar_label(bars, labels=[f'{value:.1f}%' for value in values], padding=5, color='white')
            ax.set_title(f'{ticker} — FinBERT: Cảm xúc tin tức', color='white', pad=20)
            sources = ', '.join(pred_data.get('finbert_sources', []))
            fig.text(.5, .02, f"{pred_data.get('finbert_news_count', 0)} tin đã phân tích" + (f' · {sources}' if sources else ''),
                     ha='center', color='#c9d1d9', fontsize=9)
            fig.tight_layout(rect=[0, .05, 1, 1])
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=130, facecolor=fig.get_facecolor())
            plt.close(fig)
            return buf.getvalue()

        if model == 'ensemble' and pred_data.get('prob_ensemble') is None:
            return None
        prob_xgb = pred_data.get('prob_xgb')
        prob_lstm = pred_data.get('prob_lstm')
        prob_e = float(pred_data['prob_ensemble']) if pred_data.get('prob_ensemble') is not None else None
        if model != 'ensemble' and pred_data.get({'xgboost': 'prob_xgb', 'lstm': 'prob_lstm'}[model]) is None:
            return None
        sector = pred_data.get('sector', 'Công nghệ')

        df_hist = None
        try:
            tk = yf.Ticker(ticker)
            df_hist = tk.history(period="5y")
        except Exception as err:
            logging.warning(f"Error fetching 5y history for {ticker}: {err}")

        if df_hist is None or df_hist.empty:
            try:
                df_hist = yf.download(ticker, period="5y", progress=False)
            except Exception as err:
                logging.warning(f"Error downloading 5y for {ticker}: {err}")

        if df_hist is None or df_hist.empty:
            t_chart = get_ticker_chart(ticker)
            candles = t_chart.get('candles', []) if t_chart else []
            if candles:
                df_hist = pd.DataFrame(candles)
                df_hist.rename(columns={'date': 'Date', 'close': 'Close'}, inplace=True)

        if df_hist is None or df_hist.empty:
            logging.warning(f"Cannot build dual chart for {ticker}: no price history found.")
            return None

        df_clean = df_hist.reset_index()
        date_col = 'Date' if 'Date' in df_clean.columns else ('Datetime' if 'Datetime' in df_clean.columns else df_clean.columns[0])
        df_clean['Date'] = pd.to_datetime(df_clean[date_col])
        if hasattr(df_clean['Date'].dt, 'tz_localize'):
            try:
                df_clean['Date'] = df_clean['Date'].dt.tz_localize(None)
            except Exception:
                try:
                    df_clean['Date'] = df_clean['Date'].dt.tz_convert(None)
                except Exception:
                    pass

        close_col = 'Close'
        if isinstance(df_clean[close_col], pd.DataFrame):
            df_clean['Close'] = df_clean[close_col].iloc[:, 0]

        ticker_df = df_clean[['Date', 'Close']].sort_values('Date').dropna().copy()
        ticker_df['Ticker'] = ticker

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={'width_ratios': [1, 1.15]})
        fig.patch.set_facecolor('#0d1117')

        today_str = datetime.now().strftime("%Y-%m-%d")

        if model == 'ensemble':
            ens_prob = prob_e / 100.0
            rec = 'MUA' if prob_e >= 55.0 else ('BAN' if prob_e <= 45.0 else 'HOLD')
            pred = {
                'ma_co_phieu': ticker,
                'prob_num': ens_prob,
                'ensemble_method': pred_data.get('ensemble_method'),
                'khuyen_nghi': rec,
                'nhom_nganh': sector
            }
            xgb_probs = {ticker: prob_xgb / 100.0} if prob_xgb is not None else {}
            lstm_probs = {ticker: prob_lstm / 100.0} if prob_lstm is not None else {}

            _render_ensemble_dual_chart(
                fig, ax1, ax2, ticker, ticker_df, pred,
                xgb_probs, lstm_probs, {},
                today_str, prediction_days=5
            )
        else:
            model_name_map = {
                'xgboost': ('XGBoost', prob_xgb),
                'lstm': ('LSTM', prob_lstm)
            }
            m_name, m_prob = model_name_map.get(model, ('XGBoost', prob_xgb))
            p_num = m_prob / 100.0
            rec = 'MUA' if m_prob >= 55.0 else ('BAN' if m_prob <= 45.0 else 'HOLD')
            pred = {
                'ma_co_phieu': ticker,
                'prob_num': p_num,
                'khuyen_nghi': rec,
                'nhom_nganh': sector
            }
            _render_single_ticker_dual_chart(
                fig, ax1, ax2, ticker, ticker_df, pred,
                m_name, today_str, prediction_days=5
            )

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        return buf.getvalue()
    except Exception as e:
        logging.exception(f"Lỗi tạo dual-panel live chart cho mã {ticker} ({model}): {e}")
        return None


def get_chart_image(ticker: str, model: str = "ensemble"):
    """
    Tạo biểu đồ từ kết quả mô hình hiện tại và lưu bộ nhớ đệm trong 300 giây.
    """
    cache_key = f"{ticker}_{model}"
    field = {'xgboost': 'prob_xgb', 'lstm': 'prob_lstm'}.get(model.lower())
    pred = _ON_DEMAND_PREDICTIONS.get(ticker)
    if pred and field and pred.get(field) is None:
        return None
    now_ts = time.time()
    cached = _CHART_CACHE.get(cache_key)
    if cached and (now_ts - cached['time']) < 300:
        return cached['data']

    # Older stored PNGs contain heuristic price paths. Render from validated
    # current inference instead of showing those legacy images.
    img_data = generate_live_chart_image(ticker, model)

    if img_data:
        _CHART_CACHE[cache_key] = {'time': now_ts, 'data': img_data}

    return img_data
