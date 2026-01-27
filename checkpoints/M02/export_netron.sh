#!/bin/bash
MODEL=$1
PORT=8081
OUT="/${MODEL%.pt}.pdf"

# 启动 Netron
netron "$MODEL" --host 127.0.0.1 --port $PORT &
sleep 3

# 使用 Chrome 导出 PDF
google-chrome \
  --headless \
  --disable-gpu \
  --no-sandbox \
  --print-to-pdf="$OUT" \
  http://127.0.0.1:$PORT

# 杀掉 Netron
pkill -f "netron.*$MODEL"

