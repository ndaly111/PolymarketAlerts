"""Binance WebSocket client for real-time BTC price updates.

Provides sub-second price updates instead of polling every 15 seconds.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

try:
    import websocket
    HAS_WEBSOCKET = True
except ImportError:
    HAS_WEBSOCKET = False


# Use Binance.US websocket (btcusd not btcusdt)
BINANCE_WS_URL = "wss://stream.binance.us:9443/ws/btcusd@trade"


@dataclass
class Trade:
    """A single trade from Binance."""
    price: float
    quantity: float
    timestamp: int  # Unix ms
    is_buyer_maker: bool


class BinanceTradeStream:
    """WebSocket stream for real-time BTC trades.

    Usage:
        stream = BinanceTradeStream(on_trade=my_callback)
        stream.start()
        # ... later ...
        stream.stop()
    """

    def __init__(
        self,
        on_trade: Optional[Callable[[Trade], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        reconnect_delay: float = 1.0,
        max_reconnect_delay: float = 30.0,
    ):
        if not HAS_WEBSOCKET:
            raise ImportError("websocket-client is required: pip install websocket-client")

        self.on_trade = on_trade
        self.on_error = on_error
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay

        self._ws: Optional[websocket.WebSocketApp] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._current_delay = reconnect_delay

        # Latest state
        self.last_price: Optional[float] = None
        self.last_trade_time: Optional[datetime] = None
        self.trade_count = 0

    def start(self) -> None:
        """Start the WebSocket stream in a background thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the WebSocket stream."""
        self._running = False
        if self._ws:
            self._ws.close()

    def _run_loop(self) -> None:
        """Main loop with reconnection logic."""
        while self._running:
            try:
                self._connect()
            except Exception as e:
                if self.on_error:
                    self.on_error(e)

            if self._running:
                time.sleep(self._current_delay)
                self._current_delay = min(
                    self._current_delay * 2,
                    self.max_reconnect_delay,
                )

    def _connect(self) -> None:
        """Establish WebSocket connection."""
        self._ws = websocket.WebSocketApp(
            BINANCE_WS_URL,
            on_message=self._on_message,
            on_error=self._on_ws_error,
            on_close=self._on_close,
            on_open=self._on_open,
        )
        self._ws.run_forever()

    def _on_open(self, ws) -> None:
        """Called when connection opens."""
        print("[binance_ws] Connected to Binance trade stream")
        self._current_delay = self.reconnect_delay

    def _on_message(self, ws, message: str) -> None:
        """Handle incoming trade message."""
        try:
            data = json.loads(message)

            trade = Trade(
                price=float(data["p"]),
                quantity=float(data["q"]),
                timestamp=int(data["T"]),
                is_buyer_maker=data["m"],
            )

            self.last_price = trade.price
            self.last_trade_time = datetime.fromtimestamp(
                trade.timestamp / 1000, tz=timezone.utc
            )
            self.trade_count += 1

            if self.on_trade:
                self.on_trade(trade)

        except Exception as e:
            if self.on_error:
                self.on_error(e)

    def _on_ws_error(self, ws, error) -> None:
        """Handle WebSocket error."""
        if self.on_error:
            self.on_error(error)

    def _on_close(self, ws, close_status_code, close_msg) -> None:
        """Handle WebSocket close."""
        print(f"[binance_ws] Connection closed: {close_status_code} {close_msg}")


# Singleton for easy access
_stream: Optional[BinanceTradeStream] = None


def get_realtime_price() -> Optional[float]:
    """Get the latest real-time price from WebSocket stream."""
    global _stream
    if _stream is None:
        return None
    return _stream.last_price


def start_price_stream(
    on_trade: Optional[Callable[[Trade], None]] = None,
) -> BinanceTradeStream:
    """Start the global price stream."""
    global _stream
    if _stream is None:
        _stream = BinanceTradeStream(on_trade=on_trade)
        _stream.start()
    elif on_trade is not None:
        # Update callback on existing stream
        _stream.on_trade = on_trade
    return _stream


def stop_price_stream() -> None:
    """Stop the global price stream."""
    global _stream
    if _stream:
        _stream.stop()
        _stream = None
