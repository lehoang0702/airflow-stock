"""
MODULE: ALERT UTILS — HỆ THỐNG CẢNH BÁO LỖI & THÔNG BÁO TỰ ĐỘNG
================================================================
Cung cấp callback tự động bắt lỗi cho Airflow (on_failure_callback)
và gửi cảnh báo chi tiết về Bot Telegram khi có bất kỳ task nào bị FAIL.
"""

import logging
import traceback
from datetime import datetime
import os
import time
import requests

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "8803904442:AAH4Y-GS0J3ffhAxg5CRdv1avz9Lr7b5Svg")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "7660617934")


def send_telegram_safe(url: str, data: dict = None, files: dict = None, max_retries: int = 3) -> bool:
    """
    Gửi request POST an toàn tới Telegram API (sendMessage / sendDocument / sendPhoto)
    kèm cơ chế retry Exponential Backoff (1s, 2s, 4s).
    """
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, data=data, files=files, timeout=(10, 30))
            if resp.status_code == 200:
                return True
            logging.warning(f"⚠️ Telegram API attempt {attempt+1}/{max_retries} trả về mã lỗi: {resp.status_code}")
            time.sleep(2 ** attempt)
        except Exception as e:
            logging.warning(f"⚠️ Telegram connection attempt {attempt+1}/{max_retries} thất bại: {e}")
            time.sleep(2 ** attempt)
    logging.error("❌ Đã thử gửi Telegram qua send_telegram_safe tối đa số lần nhưng thất bại.")
    return False


def send_telegram_alert(text: str, parse_mode: str = "HTML") -> bool:
    """Gửi thông báo dạng text đến Telegram Bot một cách an toàn."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': text,
        'parse_mode': parse_mode,
        'disable_web_page_preview': True
    }
    try:
        resp = requests.post(url, json=payload, timeout=(5, 15))
        if resp.status_code == 200:
            return True
        logging.warning(f"⚠️ Telegram alert trả về mã lỗi: {resp.status_code} - {resp.text}")
    except Exception as e:
        logging.error(f"❌ Lỗi kết nối Telegram alert: {e}")
    return False


def telegram_failure_callback(context: dict):
    """
    Airflow Callback: Được gọi tự động khi bất kỳ Task nào trong DAG bị FAIL.
    Trích xuất thông tin ngữ cảnh lỗi và gửi cảnh báo ngay về Telegram.
    """
    ti = context.get('task_instance')
    dag_id = ti.dag_id if ti else (context.get('dag').dag_id if context.get('dag') else 'Unknown')
    task_id = ti.task_id if ti else 'Unknown'
    run_id = context.get('run_id', 'Unknown')
    execution_date = context.get('execution_date')
    date_str = execution_date.strftime("%Y-%m-%d %H:%M:%S") if execution_date else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Trích xuất lỗi ngoại lệ
    exception = context.get('exception')
    error_msg = str(exception) if exception else "Không có exception cụ thể (Task terminated)"

    # Bỏ qua nếu task bị dừng chủ động bởi Airflow / người dùng (SIGTERM / Reset DagRun / Clear / Cancel)
    # Tránh báo động giả khi người dùng bấm Clear hoặc khi tiến trình bị hủy chủ động
    if "SIGTERM" in error_msg or "externally set to None" in error_msg:
        logging.info(f"ℹ️ Task {dag_id}.{task_id} bị hủy/dừng chủ động (SIGTERM / external reset). Bỏ qua gửi cảnh báo lỗi.")
        return

    if len(error_msg) > 400:
        error_msg = error_msg[:400] + "..."

    # Trích xuất URL log nếu có
    log_url = ti.log_url if ti else None

    # Định dạng tin nhắn HTML
    msg_lines = [
        "🚨 <b>CẢNH BÁO LỖI PIPELINE (AIRFLOW)!</b>",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"📌 <b>DAG:</b> <code>{dag_id}</code>",
        f"⚙️ <b>Task:</b> <code>{task_id}</code>",
        f"📅 <b>Thời gian:</b> <code>{date_str}</code>",
        f"🆔 <b>Run ID:</b> <code>{run_id}</code>",
        "",
        "⚠️ <b>Chi tiết lỗi:</b>",
        f"<pre>{error_msg}</pre>",
    ]

    if log_url:
        msg_lines.extend([
            "",
            f"🔗 <a href=\"{log_url}\">Xem chi tiết Log trên Airflow Web UI</a>"
        ])

    alert_text = "\n".join(msg_lines)
    send_telegram_alert(alert_text)
    logging.info(f"🚨 Đã gửi cảnh báo lỗi Task {dag_id}.{task_id} về Telegram.")


def send_pipeline_event(title: str, message: str, icon: str = "ℹ️"):
    """Tiện ích gửi thông báo sự kiện tổng quan của pipeline."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text = (
        f"{icon} <b>{title}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📅 <b>Thời gian:</b> {now_str}\n"
        f"💬 <b>Nội dung:</b> {message}"
    )
    return send_telegram_alert(text)
