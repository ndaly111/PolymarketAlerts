#!/bin/bash
# Setup script for Oracle Cloud / Ubuntu server
# Run this after SSH-ing into your new VM

set -e

echo "==================================="
echo "BTC 15-Min Collector Setup"
echo "==================================="

# Update system
echo "[1/6] Updating system..."
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
echo "[2/6] Installing Python..."
sudo apt install -y python3 python3-pip python3-venv git

# Clone the repo
echo "[3/6] Cloning repository..."
cd ~
if [ -d "PolymarketAlerts" ]; then
    cd PolymarketAlerts && git pull
else
    git clone https://github.com/ndaly111/PolymarketAlerts.git
    cd PolymarketAlerts
fi

# Create virtual environment
echo "[4/6] Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install requests websocket-client

# Create environment file
echo "[5/6] Creating environment file..."
cat > ~/btc15m.env << 'ENVFILE'
# Kalshi API credentials
KALSHI_API_KEY_ID=your_key_id_here
KALSHI_PRIVATE_KEY=your_private_key_here

# Discord webhook for alerts
BTC_15MIN_DISCORD=your_discord_webhook_here
ENVFILE

echo ""
echo "==================================="
echo "Setup complete!"
echo "==================================="
echo ""
echo "Next steps:"
echo "1. Edit your credentials:"
echo "   nano ~/btc15m.env"
echo ""
echo "2. Test the collector:"
echo "   source ~/PolymarketAlerts/venv/bin/activate"
echo "   source ~/btc15m.env"
echo "   python3 ~/PolymarketAlerts/btc15m/scripts/collect_realtime.py --interval 5"
echo ""
echo "3. Set up auto-start (run after testing):"
echo "   sudo cp ~/PolymarketAlerts/btc15m/scripts/btc-collector.service /etc/systemd/system/"
echo "   sudo systemctl daemon-reload"
echo "   sudo systemctl enable btc-collector"
echo "   sudo systemctl start btc-collector"
echo ""
echo "4. Check status:"
echo "   sudo systemctl status btc-collector"
echo "   sudo journalctl -u btc-collector -f"
echo ""
