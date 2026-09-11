"""
VIRTUAL PORTFOLIO & BACKTESTING ENGINE (Paper Trading T+5)
============================================================
Động cơ mô phỏng danh mục đầu tư định lượng T+5 tự động:
  - Khởi tạo số vốn ban đầu: $100,000 USD
  - Quản trị vị thế: Phân bổ vốn 10% - 15% cho mỗi mã MUA được Master Ensemble đề xuất
  - Chiến lược chốt:
      * Take Profit: +3.5% (Tín hiệu MUA) hoặc +5.0% (MUA MẠNH)
      * Stop Loss: -2.0% (Tín hiệu MUA) hoặc -2.5% (MUA MẠNH)
      * Time Exit (T+5): Đóng lệnh sau 5 phiên nếu không chạm TP/SL
  - Thống kê định lượng: Win Rate %, Realized PnL, Portfolio Value, Sharpe, Max Drawdown
  - Lưu trữ: MinIO `portfolio/portfolio_state.json` và `portfolio/trade_ledger.csv`
"""

import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np

INITIAL_CAPITAL = 100000.0  # $100,000 USD vốn giả lập
MAX_POSITION_WEIGHT = 0.15   # Tối đa 15% vốn cho 1 mã
STATE_KEY = "portfolio/portfolio_state.json"
LEDGER_KEY = "portfolio/trade_ledger.csv"


def load_or_init_portfolio(s3_hook, bucket_name: str) -> dict:
    """Tải trạng thái danh mục từ MinIO, nếu chưa có thì khởi tạo mới."""
    try:
        if s3_hook.check_for_key(STATE_KEY, bucket_name=bucket_name):
            content = s3_hook.read_key(STATE_KEY, bucket_name=bucket_name)
            state = json.loads(content)
            logging.info(f"✅ Đã tải trạng thái Portfolio từ MinIO. Vốn hiện tại: ${state.get('portfolio_value', INITIAL_CAPITAL):,.2f}")
            return state
    except Exception as e:
        logging.warning(f"⚠️ Không thể đọc state portfolio, khởi tạo mới: {e}")

    return {
        "initial_capital": INITIAL_CAPITAL,
        "cash": INITIAL_CAPITAL,
        "portfolio_value": INITIAL_CAPITAL,
        "total_realized_pnl": 0.0,
        "open_positions": [],
        "closed_trades": [],
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


def update_virtual_portfolio(
    ensemble_df: pd.DataFrame,
    current_prices: dict,
    s3_hook,
    bucket_name: str,
    today_str: str
) -> dict:
    """
    Thực hiện cập nhật danh mục Paper Trading:
      1. Khớp lệnh chốt lời / cắt lỗ / hết hạn T+5 cho các vị thế đang mở
      2. Mở vị thế mới cho các mã MUA đồng thuận có xác suất cao
      3. Tính toán các chỉ số Win Rate, Total PnL, Equity
      4. Lưu file JSON & CSV lên MinIO
    """
    state = load_or_init_portfolio(s3_hook, bucket_name)
    cash = float(state.get("cash", INITIAL_CAPITAL))
    open_positions = state.get("open_positions", [])
    closed_trades = state.get("closed_trades", [])

    newly_closed = []
    active_positions = []

    # 1. DUYỆT VÀ ĐỐI SOÁT CÁC VỊ THẾ ĐANG MỞ
    for pos in open_positions:
        ticker = pos['ticker']
        entry_p = float(pos['entry_price'])
        shares = int(pos['shares'])
        target_p = float(pos['target_price'])
        stop_p = float(pos['stop_loss_price'])
        holding_days = int(pos.get('holding_days', 0)) + 1

        cur_p = float(current_prices.get(ticker, entry_p))
        pos['current_price'] = cur_p
        pos['holding_days'] = holding_days

        exit_triggered = False
        exit_reason = ""
        exit_price = cur_p

        # Kiểm tra điều kiện Take Profit
        if cur_p >= target_p:
            exit_triggered = True
            exit_reason = "TAKE_PROFIT"
            exit_price = target_p

        # Kiểm tra điều kiện Stop Loss
        elif cur_p <= stop_p:
            exit_triggered = True
            exit_reason = "STOP_LOSS"
            exit_price = stop_p

        # Kiểm tra điều kiện T+5 Expiry
        elif holding_days >= 5:
            exit_triggered = True
            exit_reason = "TIME_EXIT_T5"
            exit_price = cur_p

        if exit_triggered:
            revenue = shares * exit_price
            cost = shares * entry_p
            pnl_usd = revenue - cost
            pnl_pct = ((exit_price - entry_p) / entry_p) * 100.0

            cash += revenue
            trade_record = {
                "ticker": ticker,
                "open_date": pos['open_date'],
                "close_date": today_str,
                "holding_days": holding_days,
                "shares": shares,
                "entry_price": round(entry_p, 2),
                "exit_price": round(exit_price, 2),
                "pnl_usd": round(pnl_usd, 2),
                "pnl_pct": round(pnl_pct, 2),
                "exit_reason": exit_reason,
            }
            closed_trades.append(trade_record)
            newly_closed.append(trade_record)
            logging.info(f"🔔 Đóng lệnh {ticker}: {exit_reason} | PnL: ${pnl_usd:+,.2f} ({pnl_pct:+.2f}%)")
        else:
            unrealized_usd = (cur_p - entry_p) * shares
            unrealized_pct = ((cur_p - entry_p) / entry_p) * 100.0
            pos['unrealized_pnl_usd'] = round(unrealized_usd, 2)
            pos['unrealized_pnl_pct'] = round(unrealized_pct, 2)
            active_positions.append(pos)

    # 2. TÌM VÀ MỞ VỊ THẾ MỚI TỪ TÍN HIỆU MUA MASTER ENSEMBLE
    existing_tickers = {p['ticker'] for p in active_positions}
    newly_opened = []

    if ensemble_df is not None and not ensemble_df.empty:
        # Lọc danh sách MUA có xác suất >= 53%
        prob_col = 'prob_ensemble' if 'prob_ensemble' in ensemble_df.columns else 'xac_suat_tang_gia'
        buy_candidates = []

        for _, row in ensemble_df.iterrows():
            ticker = row.get('ma_co_phieu') or row.get('ticker')
            if not ticker or ticker in existing_tickers:
                continue

            prob_val = row.get(prob_col, 0.0)
            if isinstance(prob_val, str):
                prob_val = float(prob_val.replace('%', '')) / 100.0
            else:
                prob_val = float(prob_val)

            action = str(row.get('pp4_tong_hop', '') or row.get('action_signal', ''))
            if prob_val >= 0.53 or 'MUA' in action.upper():
                buy_candidates.append((ticker, prob_val, action))

        # Sắp xếp theo xác suất giảm dần
        buy_candidates.sort(key=lambda x: x[1], reverse=True)

        current_portfolio_val = cash + sum(p['shares'] * current_prices.get(p['ticker'], p['entry_price']) for p in active_positions)
        target_pos_size = current_portfolio_val * MAX_POSITION_WEIGHT

        for ticker, prob_val, action in buy_candidates:
            cur_p = float(current_prices.get(ticker, 0.0))
            if cur_p <= 0 or cash < 2000.0:
                continue  # Không đủ tiền mặt hoặc giá không hợp lệ

            alloc = min(target_pos_size, cash)
            shares = int(alloc // cur_p)
            if shares <= 0:
                continue

            cost = shares * cur_p
            cash -= cost

            # TP/SL theo độ mạnh của tín hiệu
            is_strong = (prob_val >= 0.58) or ('MẠNH' in action.upper())
            tp_pct = 0.05 if is_strong else 0.035
            sl_pct = 0.025 if is_strong else 0.020

            new_pos = {
                "ticker": ticker,
                "open_date": today_str,
                "holding_days": 0,
                "shares": shares,
                "entry_price": round(cur_p, 2),
                "target_price": round(cur_p * (1.0 + tp_pct), 2),
                "stop_loss_price": round(cur_p * (1.0 - sl_pct), 2),
                "current_price": round(cur_p, 2),
                "unrealized_pnl_usd": 0.0,
                "unrealized_pnl_pct": 0.0,
                "prob_ensemble": round(prob_val * 100, 1),
                "action": action
            }
            active_positions.append(new_pos)
            existing_tickers.add(ticker)
            newly_opened.append(new_pos)
            logging.info(f"🛒 Mở vị thế mới {ticker}: {shares} cp @ ${cur_p:.2f} (Tổng: ${cost:,.2f})")

    # 3. TÍNH TOÁN CÁC CHỈ SỐ DANH MỤC
    stock_market_value = sum(p['shares'] * current_prices.get(p['ticker'], p['entry_price']) for p in active_positions)
    total_portfolio_value = cash + stock_market_value
    total_realized_pnl = sum(t['pnl_usd'] for t in closed_trades)
    total_trades_count = len(closed_trades)
    win_trades_count = sum(1 for t in closed_trades if t['pnl_usd'] > 0)
    win_rate = (win_trades_count / total_trades_count * 100.0) if total_trades_count > 0 else 0.0

    wins_total = sum(t['pnl_usd'] for t in closed_trades if t['pnl_usd'] > 0)
    losses_total = abs(sum(t['pnl_usd'] for t in closed_trades if t['pnl_usd'] < 0))
    profit_factor = (wins_total / losses_total) if losses_total > 0 else (wins_total if wins_total > 0 else 1.0)

    total_return_pct = ((total_portfolio_value - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100.0

    updated_state = {
        "initial_capital": INITIAL_CAPITAL,
        "cash": round(cash, 2),
        "stock_market_value": round(stock_market_value, 2),
        "portfolio_value": round(total_portfolio_value, 2),
        "total_return_pct": round(total_return_pct, 2),
        "total_realized_pnl": round(total_realized_pnl, 2),
        "total_trades": total_trades_count,
        "win_trades": win_trades_count,
        "win_rate_pct": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2),
        "open_positions": active_positions,
        "closed_trades": closed_trades[-50:],  # Giữ 50 lệnh gần nhất
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # 4. LƯU LÊN MINIO
    try:
        s3_hook.load_string(
            string_data=json.dumps(updated_state, indent=2, ensure_ascii=False),
            key=STATE_KEY,
            bucket_name=bucket_name,
            replace=True
        )

        # Lưu ledger CSV
        if closed_trades:
            df_ledger = pd.DataFrame(closed_trades)
            s3_hook.load_string(
                string_data=df_ledger.to_csv(index=False),
                key=LEDGER_KEY,
                bucket_name=bucket_name,
                replace=True
            )
        logging.info(f"✅ Đã lưu Virtual Portfolio lên MinIO: Giá trị danh mục ${total_portfolio_value:,.2f}")
    except Exception as e:
        logging.warning(f"⚠️ Lỗi lưu trạng thái Portfolio lên MinIO: {e}")

    return {
        "state": updated_state,
        "newly_opened": newly_opened,
        "newly_closed": newly_closed
    }


def format_portfolio_telegram_section(portfolio_summary: dict) -> str:
    """Tạo đoạn tóm tắt báo cáo Paper Trading để đính kèm tin nhắn Telegram."""
    if not portfolio_summary or "state" not in portfolio_summary:
        return ""

    s = portfolio_summary["state"]
    val = s.get("portfolio_value", INITIAL_CAPITAL)
    ret_pct = s.get("total_return_pct", 0.0)
    win_rate = s.get("win_rate_pct", 0.0)
    n_open = len(s.get("open_positions", []))
    n_closed = s.get("total_trades", 0)

    p_icon = "📈" if ret_pct >= 0 else "📉"
    p_color = "+" if ret_pct >= 0 else ""

    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"💼 <b>DANH MỤC ĐẦU TƯ GIẢ LẬP (PAPER TRADING T+5):</b>",
        f"  ├ 💰 Giá trị quỹ: <b>${val:,.2f}</b> ({p_icon} {p_color}{ret_pct:.2f}%)",
        f"  ├ 🎯 Tỷ lệ thắng (Win Rate): <b>{win_rate:.1f}%</b> ({n_closed} lệnh đã đóng)",
        f"  └ 📦 Vị thế đang mở: <b>{n_open} mã</b> (Tối đa 15%/mã)",
    ]

    new_closed = portfolio_summary.get("newly_closed", [])
    if new_closed:
        lines.append("  ⚡ <b>Khớp lệnh hôm nay:</b>")
        for c in new_closed[:3]:
            icon = "🎯" if c['pnl_usd'] > 0 else "🛑"
            lines.append(f"     {icon} {c['ticker']}: {c['exit_reason']} ({c['pnl_pct']:+.2f}%)")

    return "\n".join(lines)
