#!/bin/bash
# Script tạo Cloudflare Tunnel cho dự án Airflow Stock

CLOUDFLARED_BIN="$HOME/.local/bin/cloudflared"
LOG_DIR="/tmp/cloudflare_tunnels"
mkdir -p "$LOG_DIR"

if [ ! -f "$CLOUDFLARED_BIN" ]; then
    echo "Không tìm thấy $CLOUDFLARED_BIN. Đang tải..."
    mkdir -p "$HOME/.local/bin"
    curl -sSL -o "$CLOUDFLARED_BIN" https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
    chmod +x "$CLOUDFLARED_BIN"
fi

PORT=${1:-8050}
SERVICE_NAME=${2:-"Dashboard (Port $PORT)"}
PID_FILE="$LOG_DIR/tunnel_${PORT}.pid"
LOG_FILE="$LOG_DIR/tunnel_${PORT}.log"

# Kiểm tra nếu tunnel đang chạy
if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
    echo "⚠️ Tunnel cho $SERVICE_NAME đang chạy (PID: $(cat "$PID_FILE"))."
    echo "URL hiện tại:"
    grep -o 'https://[-a-zA-Z0-9@:%._\+~#=]*\.trycloudflare\.com' "$LOG_FILE" | tail -n 1
    exit 0
fi

echo "🚀 Đang khởi tạo Cloudflare Tunnel cho $SERVICE_NAME..."
nohup "$CLOUDFLARED_BIN" tunnel --protocol http2 --url "http://localhost:$PORT" > "$LOG_FILE" 2>&1 &
TUNNEL_PID=$!
echo $TUNNEL_PID > "$PID_FILE"

# Chờ URL xuất hiện trong log (tối đa 15 giây)
COUNT=0
URL=""
while [ $COUNT -lt 15 ]; do
    sleep 1
    URL=$(grep -o 'https://[-a-zA-Z0-9@:%._\+~#=]*\.trycloudflare\.com' "$LOG_FILE" | tail -n 1)
    if [ -n "$URL" ]; then
        break
    fi
    COUNT=$((COUNT + 1))
done

if [ -n "$URL" ]; then
    echo "=========================================================="
    echo "✅ KẾT NỐI THÀNH CÔNG CHO: $SERVICE_NAME"
    echo "🔗 URL CÔNG KHAI: $URL"
    echo "👉 Bạn có thể gửi đường link này cho thầy giáo truy cập!"
    echo "=========================================================="
else
    echo "❌ Không lấy được URL tunnel. Xem log tại $LOG_FILE:"
    tail -n 10 "$LOG_FILE"
fi
