#!/bin/bash
# Script dừng các Cloudflare Tunnel đang chạy

LOG_DIR="/tmp/cloudflare_tunnels"

if [ -n "$1" ]; then
    PORT="$1"
    PID_FILE="$LOG_DIR/tunnel_${PORT}.pid"
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        kill "$PID" 2>/dev/null
        rm -f "$PID_FILE"
        echo "🛑 Đã dừng tunnel cho cổng $PORT (PID: $PID)"
    else
        echo "Không tìm thấy tiến trình tunnel cho cổng $PORT."
    fi
else
    # Dừng tất cả tunnel
    echo "🛑 Đang dừng tất cả Cloudflare Tunnel..."
    pkill -f "cloudflared tunnel"
    rm -f "$LOG_DIR"/*.pid 2>/dev/null
    echo "✅ Toàn bộ tunnel đã được dừng!"
fi
