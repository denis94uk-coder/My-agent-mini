#!/bin/bash
###############################################################################
#  My-Agent-Mini — One-Command Setup
#  Tested on: Ubuntu 22.04 / 24.04, Debian 12
#  Runs on: Google Cloud e2-micro (reference deployment), any comparable
#           1 GB VM, or self-hosted hardware. Nothing here is cloud-specific
#           — it needs Ubuntu/Debian, systemd, and outbound HTTPS.
#  Usage: curl -sSL https://raw.githubusercontent.com/denis94uk-coder/my-agent-mini/main/setup.sh | bash
###############################################################################

set -e

echo "╔══════════════════════════════════════════════════════════╗"
echo "║       🤖 My-Agent-Mini — Autonomous Agent Setup          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

APP_DIR="$HOME/my-agent-mini"
REPO_URL="https://github.com/denis94uk-coder/my-agent-mini.git"

# ── Step 1: System packages ──
echo "📦 Installing system packages..."
sudo apt update -qq
sudo apt install -y -qq python3 python3-venv python3-pip git curl > /dev/null 2>&1
echo "   ✅ System packages ready"

# ── Step 2: Clone or update repo ──
if [ -d "$APP_DIR/.git" ]; then
    echo "📥 Updating existing installation..."
    cd "$APP_DIR"
    git pull --ff-only
else
    echo "📥 Cloning repository..."
    git clone "$REPO_URL" "$APP_DIR"
    cd "$APP_DIR"
fi
echo "   ✅ Code ready"

# ── Step 3: Python virtual environment ──
echo "🐍 Setting up Python environment..."
python3 -m venv "$APP_DIR/venv"
source "$APP_DIR/venv/bin/activate"
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "   ✅ Python packages installed"

# ── Step 4: Create .env if missing ──
if [ ! -f "$APP_DIR/.env" ]; then
    cp "$APP_DIR/.env.example" "$APP_DIR/.env"
    echo "   📝 Created .env from template — edit it with: nano $APP_DIR/.env"
else
    echo "   ✅ .env already exists (keeping your keys)"
fi

# ── Step 5: Create systemd service ──
echo "⚙️  Setting up auto-start service..."
SERVICE_FILE="/etc/systemd/system/my-agent.service"

sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=My Agent Mini — autonomous AI Slack agent
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$APP_DIR
Environment=PATH=$APP_DIR/venv/bin:/usr/bin:/bin
EnvironmentFile=$APP_DIR/.env
ExecStart=$APP_DIR/venv/bin/python bot.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable my-agent
echo "   ✅ Service configured (auto-starts on boot)"

# ── Done ──
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ✅ My-Agent-Mini installed successfully!                ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║                                                          ║"
echo "║  Next steps:                                             ║"
echo "║  1. Edit your API keys:                                  ║"
echo "║     nano $APP_DIR/.env                                   ║"
echo "║                                                          ║"
echo "║  2. Start the bot:                                       ║"
echo "║     sudo systemctl start my-agent                        ║"
echo "║                                                          ║"
echo "║  3. Check status:                                        ║"
echo "║     sudo systemctl status my-agent                       ║"
echo "║                                                          ║"
echo "║  4. Watch logs:                                          ║"
echo "║     sudo journalctl -u my-agent -f                       ║"
echo "║                                                          ║"
echo "║  Autonomy: ⏰ Schedules • 🏃 Durable background runs     ║"
echo "║            🔍 Critic gate • 🖐️  Approval queue           ║"
echo "║  Slack: /runs /schedules /approvals /costs /health       ║"
echo "╚══════════════════════════════════════════════════════════╝"
