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
from datetime import timedelta

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
    """
    Vẽ biểu đồ khung kép cho 1 mô hình (XGBoost / LSTM / FinBERT).
    Toàn bộ chi tiết hiển thị bằng tiếng Việt chuẩn xác, chỉ giữ tên mô hình tiếng Anh.
    """
    import matplotlib.dates as mdates

    ax1.set_facecolor('#161b22')
    ax2.set_facecolor('#161b22')

    if ticker_df.empty:
        ax1.text(0.5, 0.5, f'{ticker}\nKhông có dữ liệu', ha='center', va='center',
                 transform=ax1.transAxes, color='gray', fontsize=14)
        ax2.text(0.5, 0.5, f'{ticker}\nKhông có dữ liệu', ha='center', va='center',
                 transform=ax2.transAxes, color='gray', fontsize=14)
        return

    dates = pd.to_datetime(ticker_df['Date'].values)
    prices = ticker_df['Close'].values.astype(float)

    last_price = float(prices[-1])
    last_date = dates[-1]
    prob_up = float(pred.get('prob_num', 0.5))
    daily_pct = (prob_up - 0.5) * 0.04

    # Tính các ngày giao dịch tương lai
    future_dates = pd.bdate_range(
        start=pd.Timestamp(last_date) + timedelta(days=1),
        periods=prediction_days
    )
    future_prices = [last_price]
    for d in range(prediction_days):
        future_prices.append(future_prices[-1] * (1 + daily_pct))

    pred_dates = [pd.Timestamp(last_date)] + list(future_dates)
    pred_color = '#3fb950' if prob_up >= 0.5 else '#f85149'

    # Phân loại khuyến nghị tiếng Việt
    rec = pred.get('khuyen_nghi', 'N/A')
    sector_raw = pred.get('nhom_nganh', '')
    sector = SECTOR_MAP_VN.get(sector_raw, sector_raw)

    if rec == 'MUA':
        rec_label = '>> KHUYẾN NGHỊ: MUA <<'
        title_color = '#3fb950'
    elif rec in ('BAN', 'BÁN'):
        rec_label = '>> CẢNH BÁO: BÁN <<'
        title_color = '#f85149'
    else:
        rec_label = '-- KHUYẾN NGHỊ: ĐỨNG NGOÀI --'
        title_color = '#d29922'

    # Tiêu đề tổng quát (Super Title tiếng Việt)
    fig.suptitle(
        f"{ticker} ({sector})   |   Mô hình {model_name}\n"
        f"{rec_label}   |   Xác suất tăng: {prob_up*100:.1f}%   |   Phiên giao dịch: {today_str}",
        fontsize=15, fontweight='bold', color=title_color, y=0.98
    )

    # =========================================================================
    # PANEL 1: TOÀN CẢNH LỊCH SỬ DÀI HẠN (2021 - NAY)
    # =========================================================================
    ax1.set_title("[1] TOÀN CẢNH LỊCH SỬ GIÁ (2021 - NAY)", fontsize=11, fontweight='bold', color='#c9d1d9', pad=10)
    ax1.plot(dates, prices, color='#58a6ff', linewidth=1.2, alpha=0.9, label='Giá đóng cửa lịch sử')
    ax1.axvline(x=pd.Timestamp(last_date), color=pred_color, linestyle='--', linewidth=1.5, alpha=0.8)

    ax1.annotate(
        f"Hiện tại: ${last_price:.2f}",
        xy=(pd.Timestamp(last_date), last_price),
        xytext=(-120, 15), textcoords='offset points',
        fontsize=9, color='white', fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='white', lw=1.0),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#30363d', edgecolor=pred_color, alpha=0.9)
    )

    ax1.set_ylabel("Giá cổ phiếu (USD)", fontsize=9, color='#8b949e')
    ax1.set_xlabel("Năm giao dịch", fontsize=9, color='#8b949e')
    ax1.tick_params(axis='x', rotation=30, labelsize=9, colors='#8b949e')
    ax1.tick_params(axis='y', labelsize=9, colors='#8b949e')
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.grid(True, alpha=0.15, color='#30363d')
    ax1.legend(fontsize=8.5, loc='upper left', facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
    for spine in ax1.spines.values():
        spine.set_color('#30363d')

    # =========================================================================
    # PANEL 2: CẬN CẢNH DỰ BÁO 1 TUẦN (FORECAST ZOOM - T+1 ĐẾN T+5)
    # =========================================================================
    ax2.set_title("[2] CẬN CẢNH DỰ BÁO 1 TUẦN (T+1 ➔ T+5)", fontsize=11, fontweight='bold', color=pred_color, pad=10)

    recent_n = min(len(dates), 8)
    recent_dates = dates[-recent_n:]
    recent_prices = prices[-recent_n:]

    # 1. Đường giá 8 phiên gần nhất
    ax2.plot(recent_dates, recent_prices, color='#58a6ff', linewidth=2.2,
             marker='o', markersize=5, label=f'Giá thực tế {recent_n} phiên gần nhất', alpha=0.9, zorder=2)

    # 2. Vùng highlight dự báo (axvspan)
    ax2.axvspan(pd.Timestamp(last_date), pred_dates[-1], facecolor=pred_color, alpha=0.10, zorder=0)

    # 3. Vạch phân cách Hiện tại
    ax2.axvline(x=pd.Timestamp(last_date), color='#8b949e', linestyle='--', linewidth=1.8, alpha=0.8, zorder=1)

    # 4. Đường dự báo với hiệu ứng glow + marker kim cương
    ax2.plot(pred_dates, future_prices, color=pred_color, linewidth=8, alpha=0.15, zorder=2)
    ax2.plot(pred_dates, future_prices, color=pred_color, linewidth=3.5,
             linestyle='-', marker='D', markersize=8, markeredgecolor='white',
             markeredgewidth=1.2, label=f'Dự báo {model_name} (T+1 ➔ T+5)', zorder=3)

    # 5. Dải biến động kỳ vọng chuẩn hóa theo biến động thực tế (Volatility Cone - ATR)
    if len(prices) >= 20:
        recent_log_ret = np.diff(np.log(prices[-21:]))
        daily_vol = float(np.std(recent_log_ret))
    elif len(prices) >= 5:
        recent_log_ret = np.diff(np.log(prices))
        daily_vol = float(np.std(recent_log_ret))
    else:
        daily_vol = 0.008
    daily_vol = max(0.0035, min(daily_vol, 0.015))

    upper = []
    lower = []
    for i in range(len(future_prices)):
        spread = daily_vol * np.sqrt(i) * 0.95
        upper.append(future_prices[i] * (1 + spread))
        lower.append(future_prices[i] * (1 - spread))

    ax2.fill_between(pred_dates, lower, upper, alpha=0.18, color=pred_color, zorder=1,
                     edgecolor=pred_color, linewidth=0.6, label='Dải dao động kỳ vọng (ATR)')

    # 6. MỐC 0: Hiện tại (T+0)
    cur_date_str = pd.Timestamp(last_date).strftime('%d/%m')
    ax2.annotate(
        f"Hiện tại ({cur_date_str})\n${last_price:.2f}",
        xy=(pd.Timestamp(last_date), last_price),
        xytext=(-65, 25), textcoords='offset points',
        ha='center', fontsize=8.5, fontweight='bold', color='white',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#21262d', edgecolor='#8b949e', alpha=0.95, linewidth=1.0),
        arrowprops=dict(arrowstyle='->', color='#8b949e', lw=1.0),
        zorder=5
    )

    # 7. MỐC 1: +2 Ngày (T+2)
    if len(future_dates) >= 2:
        p_2d = future_prices[2]
        pct_2d = (p_2d - last_price) / last_price * 100
        d2_str = future_dates[1].strftime('%d/%m')
        ax2.axvline(x=future_dates[1], color=pred_color, linestyle=':', linewidth=1.2, alpha=0.6, zorder=1)
        ax2.annotate(
            f"+2 NGÀY ({d2_str})\n${p_2d:.2f} ({pct_2d:+.2f}%)",
            xy=(future_dates[1], p_2d),
            xytext=(0, 32), textcoords='offset points',
            ha='center', fontsize=8.5, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=pred_color, edgecolor='white', alpha=0.9, linewidth=1.0),
            arrowprops=dict(arrowstyle='->', color=pred_color, lw=1.0),
            zorder=5
        )

    # 8. MỐC 2: +4 Ngày (T+4)
    if len(future_dates) >= 4:
        p_4d = future_prices[4]
        pct_4d = (p_4d - last_price) / last_price * 100
        d4_str = future_dates[3].strftime('%d/%m')
        ax2.axvline(x=future_dates[3], color=pred_color, linestyle=':', linewidth=1.2, alpha=0.6, zorder=1)
        ax2.annotate(
            f"+4 NGÀY ({d4_str})\n${p_4d:.2f} ({pct_4d:+.2f}%)",
            xy=(future_dates[3], p_4d),
            xytext=(0, -42), textcoords='offset points',
            ha='center', fontsize=8.5, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=pred_color, edgecolor='white', alpha=0.9, linewidth=1.0),
            arrowprops=dict(arrowstyle='->', color=pred_color, lw=1.0),
            zorder=5
        )

    # 9. MỐC 3: +1 Tuần (T+5)
    if len(future_dates) >= 5:
        p_1w = future_prices[5]
        pct_1w = (p_1w - last_price) / last_price * 100
        d5_str = future_dates[4].strftime('%d/%m')
        ax2.axvline(x=future_dates[4], color=pred_color, linestyle='--', linewidth=1.5, alpha=0.8, zorder=1)
        ax2.annotate(
            f"+1 TUẦN ({d5_str})\n${p_1w:.2f} ({pct_1w:+.2f}%)",
            xy=(future_dates[4], p_1w),
            xytext=(15, 0), textcoords='offset points',
            ha='left', va='center',
            fontsize=9, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=pred_color, edgecolor='white', alpha=0.95, linewidth=1.2),
            arrowprops=dict(arrowstyle='->', color='white', lw=1.2),
            zorder=5
        )

    # Mở rộng trục X sang phải để nhãn "+1 TUẦN" không bị che
    ax2.set_xlim(recent_dates[0] - timedelta(days=1), future_dates[-1] + timedelta(days=4))
    ax2.set_ylabel("Giá kỳ vọng (USD)", fontsize=9, color='#8b949e')
    ax2.set_xlabel("Ngày giao dịch (Ngày/Tháng)", fontsize=9, color='#8b949e')

    ax2.tick_params(axis='x', rotation=30, labelsize=8.5, colors='#8b949e')
    ax2.tick_params(axis='y', labelsize=9, colors='#8b949e')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
    ax2.grid(True, alpha=0.15, color='#30363d')
    ax2.legend(fontsize=8.5, loc='upper left', facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
    for spine in ax2.spines.values():
        spine.set_color('#30363d')


def _render_ensemble_dual_chart(
    fig, ax1, ax2, ticker, ticker_df, pred, xgb_probs, lstm_probs, bert_probs, today_str, prediction_days=5
):
    """
    Vẽ biểu đồ khung kép cho Ensemble Master (Đối chiếu 4 đường dự đoán).
    Toàn bộ nhãn, chú thích, thông số bằng tiếng Việt, chỉ tên mô hình tiếng Anh.
    """
    import matplotlib.dates as mdates

    ax1.set_facecolor('#161b22')
    ax2.set_facecolor('#161b22')

    if ticker_df.empty:
        ax1.text(0.5, 0.5, f'{ticker}\nKhông có dữ liệu', ha='center', va='center',
                 transform=ax1.transAxes, color='gray', fontsize=14)
        ax2.text(0.5, 0.5, f'{ticker}\nKhông có dữ liệu', ha='center', va='center',
                 transform=ax2.transAxes, color='gray', fontsize=14)
        return

    dates = pd.to_datetime(ticker_df['Date'].values)
    prices = ticker_df['Close'].values.astype(float)

    last_price = float(prices[-1])
    last_date = dates[-1]

    future_dates = pd.bdate_range(
        start=pd.Timestamp(last_date) + timedelta(days=1),
        periods=prediction_days
    )
    pred_dates = [pd.Timestamp(last_date)] + list(future_dates)

    ens_prob = float(pred.get('prob_num', 0.5))
    ens_color = '#3fb950' if ens_prob >= 0.5 else '#f85149'

    rec = pred.get('khuyen_nghi', 'N/A')
    sector_raw = pred.get('nhom_nganh', '')
    sector = SECTOR_MAP_VN.get(sector_raw, sector_raw)

    if rec == 'MUA':
        title_color = '#3fb950'
        rec_label = '>> KHUYẾN NGHỊ: MUA <<'
    elif rec in ('BAN', 'BÁN'):
        title_color = '#f85149'
        rec_label = '>> CẢNH BÁO: BÁN <<'
    else:
        title_color = '#d29922'
        rec_label = '-- KHUYẾN NGHỊ: ĐỨNG NGOÀI --'

    fig.suptitle(
        f"{ticker} ({sector})   |   TỔNG HỢP 4 MÔ HÌNH (Ensemble Master)\n"
        f"KẾT LUẬN: {rec_label}   |   Xác suất tăng: {ens_prob*100:.1f}%   |   Phiên giao dịch: {today_str}",
        fontsize=15, fontweight='bold', color=title_color, y=0.98
    )

    # =========================================================================
    # PANEL 1: TOÀN CẢNH LỊCH SỬ GIÁ (2021 - NAY)
    # =========================================================================
    ax1.set_title("[1] TOÀN CẢNH LỊCH SỬ GIÁ (2021 - NAY)", fontsize=11, fontweight='bold', color='#c9d1d9', pad=10)
    ax1.plot(dates, prices, color='#58a6ff', linewidth=1.2, alpha=0.9, label='Giá đóng cửa lịch sử')
    ax1.axvline(x=pd.Timestamp(last_date), color=ens_color, linestyle='--', linewidth=1.5, alpha=0.8)

    ax1.annotate(
        f"Hiện tại: ${last_price:.2f}",
        xy=(pd.Timestamp(last_date), last_price),
        xytext=(-120, 15), textcoords='offset points',
        fontsize=9, color='white', fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='white', lw=1.0),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#30363d', edgecolor=ens_color, alpha=0.9)
    )

    ax1.set_ylabel("Giá cổ phiếu (USD)", fontsize=9, color='#8b949e')
    ax1.set_xlabel("Năm giao dịch", fontsize=9, color='#8b949e')
    ax1.tick_params(axis='x', rotation=30, labelsize=9, colors='#8b949e')
    ax1.tick_params(axis='y', labelsize=9, colors='#8b949e')
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.grid(True, alpha=0.15, color='#30363d')
    ax1.legend(fontsize=8.5, loc='upper left', facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
    for spine in ax1.spines.values():
        spine.set_color('#30363d')

    # =========================================================================
    # PANEL 2: ĐỐI CHIẾU 4 ĐƯỜNG DỰ BÁO 1 TUẦN (CẬN CẢNH)
    # =========================================================================
    ax2.set_title("[2] ĐỐI CHIẾU 4 ĐƯỜNG DỰ BÁO 1 TUẦN (T+1 ➔ T+5)", fontsize=11, fontweight='bold', color=ens_color, pad=10)

    recent_n = min(len(dates), 8)
    recent_dates = dates[-recent_n:]
    recent_prices = prices[-recent_n:]

    ax2.plot(recent_dates, recent_prices, color='#8b949e', linewidth=1.8, marker='o', markersize=4,
             alpha=0.8, label=f'Giá thực tế {recent_n} phiên gần nhất', zorder=2)

    ax2.axvspan(pd.Timestamp(last_date), pred_dates[-1], facecolor=ens_color, alpha=0.08, zorder=0)
    ax2.axvline(x=pd.Timestamp(last_date), color='#8b949e', linestyle='--', linewidth=1.8, alpha=0.8, zorder=1)

    model_colors = {
        'XGBoost': '#3fb950',
        'LSTM': '#58a6ff',
        'FinBERT': '#d2a8ff',
        'Ensemble Master': '#f0883e',
    }

    model_probs = {
        'XGBoost': xgb_probs.get(ticker, 0.5),
        'LSTM': lstm_probs.get(ticker, 0.5),
        'FinBERT': bert_probs.get(ticker, 0.5),
        'Ensemble Master': ens_prob,
    }

    ens_future_prices = []

    for m_name, prob in model_probs.items():
        daily_pct = (prob - 0.5) * 0.04
        f_prices = [last_price]
        for d in range(prediction_days):
            f_prices.append(f_prices[-1] * (1 + daily_pct))

        if m_name == 'Ensemble Master':
            ens_future_prices = f_prices
            ax2.plot(pred_dates, f_prices, color=model_colors[m_name], linewidth=8, alpha=0.15, zorder=2)
            ax2.plot(pred_dates, f_prices, color=model_colors[m_name], linewidth=4.0,
                     linestyle='-', marker='D', markersize=8, markeredgecolor='white',
                     markeredgewidth=1.2, alpha=1.0, label=f'{m_name} ({prob*100:.0f}%)', zorder=4)

            # Dải biên độ dao động kỳ vọng chuẩn hóa theo biến động thực tế (Volatility Cone)
            if len(prices) >= 20:
                recent_log_ret = np.diff(np.log(prices[-21:]))
                daily_vol = float(np.std(recent_log_ret))
            elif len(prices) >= 5:
                recent_log_ret = np.diff(np.log(prices))
                daily_vol = float(np.std(recent_log_ret))
            else:
                daily_vol = 0.008
            daily_vol = max(0.0035, min(daily_vol, 0.015))

            upper = []
            lower = []
            for i in range(len(f_prices)):
                spread = daily_vol * np.sqrt(i) * 0.95
                upper.append(f_prices[i] * (1 + spread))
                lower.append(f_prices[i] * (1 - spread))

            ax2.fill_between(pred_dates, lower, upper, alpha=0.18, color=model_colors[m_name], zorder=1,
                             edgecolor=model_colors[m_name], linewidth=0.6, label='Dải dao động kỳ vọng (ATR)')
        else:
            ax2.plot(pred_dates, f_prices, color=model_colors[m_name], linewidth=2.0,
                     linestyle='--', alpha=0.85, label=f'{m_name} ({prob*100:.0f}%)', zorder=3)

    # Milestone nhãn Hiện tại (T+0)
    cur_date_str = pd.Timestamp(last_date).strftime('%d/%m')
    ax2.annotate(
        f"Hiện tại ({cur_date_str})\n${last_price:.2f}",
        xy=(pd.Timestamp(last_date), last_price),
        xytext=(-65, 25), textcoords='offset points',
        ha='center', fontsize=8.5, fontweight='bold', color='white',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#21262d', edgecolor='#8b949e', alpha=0.95, linewidth=1.0),
        arrowprops=dict(arrowstyle='->', color='#8b949e', lw=1.0),
        zorder=5
    )

    # Milestone +2 ngày (T+2)
    if len(future_dates) >= 2 and ens_future_prices:
        p_2d = ens_future_prices[2]
        pct_2d = (p_2d - last_price) / last_price * 100
        d2_str = future_dates[1].strftime('%d/%m')
        ax2.axvline(x=future_dates[1], color=model_colors['Ensemble Master'], linestyle=':', linewidth=1.2, alpha=0.6, zorder=1)
        ax2.annotate(
            f"+2 NGÀY ({d2_str})\n${p_2d:.2f} ({pct_2d:+.2f}%)",
            xy=(future_dates[1], p_2d),
            xytext=(0, 32), textcoords='offset points',
            ha='center', fontsize=8.5, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=model_colors['Ensemble Master'], edgecolor='white', alpha=0.9, linewidth=1.0),
            arrowprops=dict(arrowstyle='->', color=model_colors['Ensemble Master'], lw=1.0),
            zorder=5
        )

    # Milestone +4 ngày (T+4)
    if len(future_dates) >= 4 and ens_future_prices:
        p_4d = ens_future_prices[4]
        pct_4d = (p_4d - last_price) / last_price * 100
        d4_str = future_dates[3].strftime('%d/%m')
        ax2.axvline(x=future_dates[3], color=model_colors['Ensemble Master'], linestyle=':', linewidth=1.2, alpha=0.6, zorder=1)
        ax2.annotate(
            f"+4 NGÀY ({d4_str})\n${p_4d:.2f} ({pct_4d:+.2f}%)",
            xy=(future_dates[3], p_4d),
            xytext=(0, -42), textcoords='offset points',
            ha='center', fontsize=8.5, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=model_colors['Ensemble Master'], edgecolor='white', alpha=0.9, linewidth=1.0),
            arrowprops=dict(arrowstyle='->', color=model_colors['Ensemble Master'], lw=1.0),
            zorder=5
        )

    # Milestone +1 tuần (T+5)
    if len(future_dates) >= 5 and ens_future_prices:
        p_1w = ens_future_prices[5]
        pct_1w = (p_1w - last_price) / last_price * 100
        d5_str = future_dates[4].strftime('%d/%m')
        ax2.axvline(x=future_dates[4], color=model_colors['Ensemble Master'], linestyle='--', linewidth=1.5, alpha=0.8, zorder=1)
        ax2.annotate(
            f"MỤC TIÊU +1 TUẦN ({d5_str})\n${p_1w:.2f} ({pct_1w:+.2f}%)",
            xy=(future_dates[4], p_1w),
            xytext=(15, 0), textcoords='offset points',
            ha='left', va='center',
            fontsize=9, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=model_colors['Ensemble Master'], edgecolor='white', alpha=0.95, linewidth=1.2),
            arrowprops=dict(arrowstyle='->', color='white', lw=1.2),
            zorder=5
        )

    ax2.set_xlim(recent_dates[0] - timedelta(days=1), future_dates[-1] + timedelta(days=5))
    ax2.set_ylabel("Giá kỳ vọng (USD)", fontsize=9, color='#8b949e')
    ax2.set_xlabel("Ngày giao dịch (Ngày/Tháng)", fontsize=9, color='#8b949e')

    ax2.tick_params(axis='x', rotation=30, labelsize=8.5, colors='#8b949e')
    ax2.tick_params(axis='y', labelsize=9, colors='#8b949e')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
    ax2.grid(True, alpha=0.15, color='#30363d')
    ax2.legend(fontsize=8.5, loc='upper left', facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
    for spine in ax2.spines.values():
        spine.set_color('#30363d')


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


def calculate_trading_plan(last_price, prob_up, prediction_days=5):
    """
    Tính toán kế hoạch giao dịch dựa trên giá hiện tại và xác suất tăng:
    - Entry: last_price
    - Take-Profit (T+5): mục tiêu 1 tuần
    - Stop-Loss: ngưỡng bảo vệ an toàn (2.0%)
    - Tỷ lệ Risk / Reward (R:R)
    """
    daily_pct = (prob_up - 0.5) * 0.04
    target_price = last_price
    for _ in range(prediction_days):
        target_price *= (1 + daily_pct)

    pct_target = (target_price - last_price) / last_price * 100

    if prob_up >= 0.5:
        sl_pct = 0.02
        stop_loss = last_price * (1 - sl_pct)
        risk = max(last_price - stop_loss, 0.01)
        reward = max(target_price - last_price, 0.01)
        rr = reward / risk
        rec_type = "MUA"
    else:
        sl_pct = 0.02
        stop_loss = last_price * (1 + sl_pct)
        risk = max(stop_loss - last_price, 0.01)
        reward = max(last_price - target_price, 0.01)
        rr = reward / risk
        rec_type = "BAN"

    return {
        'entry': round(last_price, 2),
        'target': round(target_price, 2),
        'pct_target': round(pct_target, 2),
        'stop_loss': round(stop_loss, 2),
        'sl_pct': round(sl_pct * 100, 1),
        'rr_ratio': round(rr, 1),
        'rec_type': rec_type
    }


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

    sorted_keys = sorted(keys)

    # 1. Tìm file mới nhất có đủ 20 mã
    df = None
    selected_key = None
    for k in reversed(sorted_keys):
        try:
            raw_csv = s3_hook.read_key(k, bucket_name=bucket_name)
            temp_df = pd.read_csv(io.StringIO(raw_csv))
            cols_lower = {c.lower(): c for c in temp_df.columns}
            ticker_col = cols_lower.get('ticker') or cols_lower.get('symbol') or cols_lower.get('ma_co_phieu')
            close_col = cols_lower.get('close') or cols_lower.get('adj close')
            if close_col and ticker_col and temp_df[ticker_col].nunique() >= len(EXPECTED_TICKERS):
                df = temp_df
                selected_key = k
                break
        except Exception:
            continue

    # 2. Nếu không có file nào đủ 20 mã, lấy file mới nhất và chuẩn bị backfill
    if df is None:
        latest_key = sorted_keys[-1]
        logging.info(f"📊 Đang đọc dữ liệu giá từ: {latest_key}")
        raw_csv = s3_hook.read_key(latest_key, bucket_name=bucket_name)
        df = pd.read_csv(io.StringIO(raw_csv))
        selected_key = latest_key
    else:
        logging.info(f"📊 Đã chọn dataset giá hợp lệ: {selected_key} ({len(df):,} dòng)")

    cols_lower = {c.lower(): c for c in df.columns}
    date_col = cols_lower.get('date') or cols_lower.get('datetime') or df.columns[0]
    ticker_col = cols_lower.get('ticker') or cols_lower.get('symbol') or cols_lower.get('ma_co_phieu')
    close_col = cols_lower.get('close') or cols_lower.get('adj close')

    if not close_col:
        logging.warning(f"⚠️ Dataset KHÔNG có cột Close! Columns: {list(df.columns)}")
        return pd.DataFrame(columns=['Date', 'Ticker', 'Close'])

    result = pd.DataFrame()
    result['Date'] = pd.to_datetime(df[date_col], errors='coerce').dt.tz_localize(None)
    result['Ticker'] = df[ticker_col] if ticker_col else 'UNKNOWN'
    result['Close'] = pd.to_numeric(df[close_col], errors='coerce')
    result = result.dropna(subset=['Date', 'Close'])
    result = result[result['Close'] > 0]

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
                df_prev = pd.read_csv(io.StringIO(raw_prev))
                p_cols = {c.lower(): c for c in df_prev.columns}
                p_tcol = p_cols.get('ticker') or p_cols.get('symbol') or p_cols.get('ma_co_phieu')
                p_ccol = p_cols.get('close') or p_cols.get('adj close')
                p_dcol = p_cols.get('date') or p_cols.get('datetime') or df_prev.columns[0]
                if p_tcol and p_ccol:
                    for mt in list(missing_tickers):
                        sub_m = df_prev[df_prev[p_tcol] == mt].copy()
                        if not sub_m.empty:
                            m_df = pd.DataFrame()
                            m_df['Date'] = pd.to_datetime(sub_m[p_dcol], errors='coerce').dt.tz_localize(None)
                            m_df['Ticker'] = mt
                            m_df['Close'] = pd.to_numeric(sub_m[p_ccol], errors='coerce')
                            m_df = m_df.dropna(subset=['Date', 'Close'])
                            m_df = m_df[m_df['Close'] > 0]
                            if not m_df.empty:
                                result = pd.concat([result, m_df], ignore_index=True)
                                missing_tickers.remove(mt)
                                logging.info(f"🛡️ Đã bù thành công mã {mt} từ {k} ({len(m_df)} phiên)")
            except Exception as e:
                logging.warning(f"Lỗi khi đọc file bù {k}: {e}")

    logging.info(f"📊 Đã nạp dữ liệu giá hoàn tất: {len(result):,} dòng, {result['Ticker'].nunique()}/{len(EXPECTED_TICKERS)} mã")
    return result

