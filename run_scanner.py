#!/usr/bin/env python3
"""Wrapper to run the scanner with credentials."""

import os
import sys

# Set credentials
os.environ["KALSHI_API_KEY_ID"] = "4678a6f9-1b0e-40ed-9300-fef58667f5cf"
os.environ["BTC_15MIN_DISCORD"] = "https://discord.com/api/webhooks/1467050444197986369/OucySgBGVci00cEbcNtrgeC8KHjc5KlYrqXtUgH7RyOrWPae-99c1TCSfCva7MhDGwWK"
os.environ["KALSHI_PRIVATE_KEY"] = """-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEAs1BvNKDYsr+S6MMBnYT6e1IluQIfA1fC7x2QAWqesDxYDQju
xxleFNQysESrlhYA+GoWs1t5dfSG5GcLaliTmjOfGvIPT9r11+iZLcj+ZI4CrdDQ
0IcCkqat3x5O0dPldRl6M3RgEcDYLG0gtSoyyjBncvwWi0i+gA8x3TpbKICoqhNF
7LDOnoqMyZDg51Pht+b5uB0EjRWP4xBvGO7aeOMba/SL9FYIZUu/B+y6FlYzjggx
++YT5eFBDeIf1JaY8Q8XV3THNQG+Lnnraq3q+ljjZskhwFCqsX8OKW6BpU2YbE6T
tZtqAzPlLldbkHVxg7N98TQ1jv5ZPQEenWIDwwIDAQABAoIBACLaJWMzSCsl9SXs
kSnOqOjDRLW65dA+hJL5Sf4cfQ+ACxUtgUuNcK3XuKpuDnOzfyjJZfQieW8lwfou
1RcsdDPqiKgyGCvLQTFf8jXl9RwSRg8GoS+GrkdWwkC1oVhDOTwTYZ8SpYGN2/Z2
l37e3tRRqCm/OZqE3xIA4g4/w2wRkknMvBdRtyylZpXZRAMtokfg2+T9muHAe3Pl
F32BuB/4CHto8QnvzaF3LEOSVlUTcF+R6+UBCIzEFelwadoNWDHl6jLAR97VPIl2
LXHUX7erx825tj58h/K74vK1v4AJAHR4owNh6F4YtyolK9hkHC/OGiqCvcsGDZsQ
VHf1ORECgYEAyj+JbC5vDaYfqwuaXcHViVvWoGwZbewd8o90YJjyF1/KdWEjbJfT
psjnpn1GFP9X+rQYzXXIwiZerg/3PGkEURVD6oTRNXZOzAXU9Hudxe0H+IzdKHVW
P36EpayBu+WqD8Kq8QleffeSR1/8ArozM9drwHtZK/Mq8MYiFO8XHv8CgYEA4viG
mNPirvS1QecgE46cGaAhefAjlf+jdn2uuCZUMWYGl5VW4aUVgKfpJU79bVoAkH7j
Yp26+M0fj6AEJRbG2bM8KRJB7IW4i7A/Nq63Cs3RjQv5Ua1RrDVu8F/PWID3Frim
a1MHBqd8DJsFfcjf9Am6In9dD67NASQ7dG6hXz0CgYEAifEZ7qIg9mM2uDe6unXJ
Cd2Mnm/8TB++MUEss/G9NLoN4U82lQtcvSXL8Z8B3LJntEN/oyrRmbIH45paG2bZ
HeBuKRcbheZCSh80JuflLSjExf51nPGiuL23BTSKa7hx7Dvu0WV9gFcJ+wp5XPbY
k2Q7RtYadMJuqDfK6RJFZ8sCgYEA4sbgbQigIKD+Dffpc8D0tSHq8NQAKlo/MCAJ
l5MsYiMHiSM8qL3ySxb5+z3+NulLURSHCzx/2SHSXAXMvmYQjunnvT2xwqolJuUV
dASDfPcCXgRNus8KoJ7O1rtUB6DrwLcECI3vojVr24h0pyYypbmmUduh1w6XZIGY
KhBzih0CgYASg+/7GbhY9qFsLh1ETpvA/xB6EyU8WF0rJb4lRa5VVfRCkJmsLGFj
Ro6HOKeegcMdBUdh6RVG0NbSbTvEEWryhgLvjM++7vryJ0weQfmb6ghkJRuCDG5P
bWtav/64BS04BOUDIAKDxgtBxxKSKwPzYshlP0SQFWPVdx5VNr02ow==
-----END RSA PRIVATE KEY-----"""

# Add paths
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "btc15m"))
sys.path.insert(0, str(SCRIPT_DIR))

# Import and run the scanner main
from btc15m.scripts.local_scanner import main
sys.exit(main())
