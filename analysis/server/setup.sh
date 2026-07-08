#!/bin/bash
# =========================================================
# IROCA Vision API — LXC/Ubuntuセットアップスクリプト
# 使い方: sudo bash setup.sh
# =========================================================
set -e
apt-get update -qq
apt-get install -y -qq python3-pip python3-venv libgl1 libglib2.0-0

mkdir -p /opt/iroca-vision/jobs
cd /opt/iroca-vision
python3 -m venv venv
./venv/bin/pip install --quiet -r requirements.txt

cat > /etc/systemd/system/iroca-vision.service << 'UNIT'
[Unit]
Description=IROCA Vision API
After=network.target

[Service]
WorkingDirectory=/opt/iroca-vision
Environment=API_KEY=iroca-vision-2026
Environment=JOB_DIR=/opt/iroca-vision/jobs
ExecStart=/opt/iroca-vision/venv/bin/uvicorn server:app --host 0.0.0.0 --port 8340
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable --now iroca-vision
echo "✅ 起動完了: http://$(hostname -I | awk '{print $1}'):8340/api/health"
echo "次: Cloudflare Tunnel でこのポートを公開してください"
