#!/bin/bash
# ╔══════════════════════════════════════════════════════════╗
# ║    My-Agent-Mini — One-Click Server Setup Script         ║
# ║    Run this on your Oracle Cloud / VPS instance          ║
# ╚══════════════════════════════════════════════════════════╝

set -e
echo "🤖 My-Agent-Mini Setup"
echo "======================"

# Update system
echo "📦 Updating system..."
sudo apt update && sudo apt upgrade -y

# Install Python 3.11+ and pip
echo "🐍 Installing Python..."
sudo apt install -y python3 python3-pip python3-venv git

# Clone repo
echo "📥 Cloning repository..."
cd ~
if [ -d "my-agent-mini" ]; then
    echo "   Repository already exists, pulling latest..."
    cd my-agent-mini && git pull
else
    git clone https://github.com/denis94uk-coder/my-agent-mini.git
    cd my-agent-mini
fi

# Create virtual environment
echo "🔧 Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Create .env if doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo ""
    echo "⚠️  IMPORTANT: Edit .env with your API keys!"
    echo "   Run: nano .env"
    echo ""
fi

# Create systemd service for auto-start
echo "🔄 Creating systemd service..."
sudo tee /etc/systemd/system/my-agent.service > /dev/null << 'EOF'
[Unit]
Description=My Agent Mini - Slack AI Bot
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/my-agent-mini
Environment=PATH=/home/ubuntu/my-agent-mini/venv/bin:/usr/bin
EnvironmentFile=/home/ubuntu/my-agent-mini/.env
ExecStart=/home/ubuntu/my-agent-mini/venv/bin/python bot.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable my-agent

echo ""
echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Edit your API keys:  nano .env"
echo "   2. Start the bot:       sudo systemctl start my-agent"
echo "   3. Check status:        sudo systemctl status my-agent"
echo "   4. View logs:           sudo journalctl -u my-agent -f"
echo ""
echo "🔄 To restart after changes:  sudo systemctl restart my-agent"
