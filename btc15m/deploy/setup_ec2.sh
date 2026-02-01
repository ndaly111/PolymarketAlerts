#!/bin/bash
# EC2 Setup Script for BTC 15-min Scanner
# Run this on a fresh Ubuntu 22.04 EC2 instance

set -e

echo "=== BTC 15-min Scanner Setup ==="

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11
sudo apt install -y python3.11 python3.11-venv python3-pip git

# Clone repo (or copy files)
cd ~
if [ ! -d "btc15m" ]; then
    echo "Creating btc15m directory..."
    mkdir -p btc15m
fi

cd btc15m

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install flask requests numpy pandas scikit-learn lightgbm python-dotenv

# Create .env file (you'll need to fill this in)
if [ ! -f ".env" ]; then
    cat > .env << 'EOF'
KALSHI_API_KEY_ID=your_key_here
KALSHI_PRIVATE_KEY=your_private_key_here
BTC_15MIN_DISCORD=your_discord_webhook_here
EOF
    echo "Created .env - EDIT THIS FILE with your credentials!"
fi

# Create systemd service for scanner
sudo tee /etc/systemd/system/btc-scanner.service > /dev/null << 'EOF'
[Unit]
Description=BTC 15-min Scanner
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/btc15m
Environment=PATH=/home/ubuntu/btc15m/venv/bin
ExecStart=/home/ubuntu/btc15m/venv/bin/python scripts/local_scanner.py --min-prob 0.80 --contracts 1
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Create systemd service for dashboard
sudo tee /etc/systemd/system/btc-dashboard.service > /dev/null << 'EOF'
[Unit]
Description=BTC Scanner Dashboard
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/btc15m
Environment=PATH=/home/ubuntu/btc15m/venv/bin
ExecStart=/home/ubuntu/btc15m/venv/bin/python dashboard.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Create systemd service for online learning
sudo tee /etc/systemd/system/btc-learner.service > /dev/null << 'EOF'
[Unit]
Description=BTC ML Online Learner
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/btc15m
Environment=PATH=/home/ubuntu/btc15m/venv/bin
ExecStart=/home/ubuntu/btc15m/venv/bin/python scripts/online_learn.py --interval 300
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Edit .env with your credentials: nano .env"
echo "2. Copy your code files to ~/btc15m/"
echo "3. Enable services:"
echo "   sudo systemctl enable btc-scanner btc-dashboard btc-learner"
echo "   sudo systemctl start btc-scanner btc-dashboard btc-learner"
echo ""
echo "4. Check status:"
echo "   sudo systemctl status btc-scanner"
echo "   sudo journalctl -u btc-scanner -f"
echo ""
echo "Dashboard will be at http://YOUR_EC2_IP:5050"
