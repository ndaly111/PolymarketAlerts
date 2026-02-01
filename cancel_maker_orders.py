#!/usr/bin/env python3
"""Cancel all open maker orders."""

import sqlite3
from pathlib import Path
from kalshi_auth_client import KalshiAuthClient

DB_PATH = Path("props_trades.db")

def main():
    print("Cancelling all open maker orders...")

    client = KalshiAuthClient.from_env()

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT order_id, player_name, stat_type, line, side FROM maker_orders WHERE status = 'open'")
    orders = cur.fetchall()
    conn.close()

    if not orders:
        print("No open orders to cancel.")
        return

    print(f"Found {len(orders)} open orders")

    for order in orders:
        order_id = order["order_id"]
        print(f"  Cancelling: {order['player_name']} {order['stat_type']} {order['line']}+ {order['side']}")
        try:
            client.cancel_order(order_id)
            print(f"    Cancelled: {order_id}")

            # Update database
            conn = sqlite3.connect(str(DB_PATH))
            cur = conn.cursor()
            cur.execute("UPDATE maker_orders SET status = 'cancelled' WHERE order_id = ?", (order_id,))
            conn.commit()
            conn.close()

        except Exception as e:
            print(f"    Error: {e}")

    print("Done.")

if __name__ == "__main__":
    main()
