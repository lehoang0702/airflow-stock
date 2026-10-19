"""
HTTP REQUEST HANDLER — QUANTUM DASHBOARD
========================================
SimpleHTTPRequestHandler định tuyến các API endpoints và phục vụ file tĩnh web frontend.
"""

import json
import math
from http.server import SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

from dashboard.config import STATIC_DIR, TICKERS_META
from dashboard.charts import get_chart_image
from dashboard.services import (
    get_market_data,
    get_ticker_chart,
    get_quality_data,
    get_evaluation_data,
)
from dashboard.jobs import analysis_job
import re


def _json_safe(value):
    """Missing/non-finite metrics must stay absent in standards-compliant JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/api/chart-image":
            symbol = query.get("symbol", ["AAPL"])[0].upper()
            model = query.get("model", ["ensemble"])[0].lower()
            img_bytes = get_chart_image(symbol, model)
            if img_bytes:
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(img_bytes)
            else:
                self.send_response(404)
                self.end_headers()
            return

        if path.startswith("/api/"):
            status_code = 200
            if path == "/api/tickers":
                resp = TICKERS_META
            elif path == "/api/market":
                resp = get_market_data()
            elif path == "/api/ticker":
                symbol = query.get("symbol", ["AAPL"])[0].upper()
                resp = get_ticker_chart(symbol)
            elif path == "/api/quality":
                resp = get_quality_data()
            elif path == "/api/evaluation":
                resp = get_evaluation_data()
            elif path == "/api/analyze":
                symbol = query.get("symbol", [""])[0].strip().upper()
                job_id = query.get('job_id', [None])[0]
                if job_id:
                    resp, status_code = analysis_job(job_id=job_id)
                elif not re.fullmatch(r'[A-Z0-9][A-Z0-9.^=-]{0,14}', symbol):
                    status_code = 400
                    resp = {"status": "error", "message": "Vui lòng nhập mã cổ phiếu hợp lệ."}
                else:
                    resp, status_code = analysis_job(ticker=symbol)
            else:
                status_code = 404
                resp = {"status": "error", "message": "Không tìm thấy API."}

            payload = json.dumps(_json_safe(resp), ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(payload)
            return

        return super().do_GET()
