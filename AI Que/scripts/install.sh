#!/bin/bash
# install.sh — Raspberry Pi setup script for AI Queue Prediction System
# Run as: bash scripts/install.sh

set -e

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
echo "📦  Installing AI Queue Prediction System at: $PROJ_DIR"

# 1. System deps
sudo apt-get update -q
sudo apt-get install -y python3-pip python3-venv chromium-browser unclutter

# 2. Python venv
python3 -m venv "$PROJ_DIR/.venv"
source "$PROJ_DIR/.venv/bin/activate"
pip install --upgrade pip -q
pip install -r "$PROJ_DIR/requirements.txt" -q

echo "✅  Python dependencies installed"

# 3. Generate seed data + train initial model
python "$PROJ_DIR/scripts/seed_data.py"
cd "$PROJ_DIR" && python -c "
from backend import model, database
import pandas as pd
database.init_db()
stats = model.train()
print('Model trained:', stats)
"
echo "✅  Seed data generated and initial model trained"

# 4. systemd service for the FastAPI backend
SERVICE_FILE="/etc/systemd/system/ai-queue.service"
sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=AI Queue Prediction Backend
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJ_DIR
ExecStart=$PROJ_DIR/.venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable ai-queue
sudo systemctl start ai-queue
echo "✅  systemd service 'ai-queue' enabled and started"

# 5. Autostart Chromium in kiosk mode on display
AUTOSTART_DIR="$HOME/.config/autostart"
mkdir -p "$AUTOSTART_DIR"
cat > "$AUTOSTART_DIR/queue-display.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Queue Display
Exec=bash -c 'sleep 10 && chromium-browser --kiosk --noerrdialogs --disable-infobars --disable-session-crashed-bubble http://localhost:8000/'
EOF

# Hide cursor on kiosk screen
cat > "$AUTOSTART_DIR/unclutter.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Unclutter
Exec=unclutter -idle 1
EOF

echo "✅  Chromium kiosk autostart configured"
echo ""
echo "🎉  Setup complete! Reboot the Pi to launch the kiosk display."
echo "    Staff panel: http://localhost:8000/staff"
echo "    Public display: http://localhost:8000/"
echo "    API docs: http://localhost:8000/docs"
