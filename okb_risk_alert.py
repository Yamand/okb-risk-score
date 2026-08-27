"""
OKB Risk Score — daily Telegram alert.

Reads the latest row from data/okb_risk_history.json (written by
okb_risk_score.py) and pushes a formatted summary to Telegram. Meant to run
right after `okb_risk_score.py --update` in the daily GitHub Actions
workflow, so it always reflects the freshly computed score.

Sends every day (not just on zone change) — this is a daily summary, not a
change-detection alert. Same pattern and same Telegram bot as
btc_risk_alert.py / eth_risk_alert.py / gold_risk_alert.py — one bot, one
more daily message.

Requires two env vars (set as GitHub Actions secrets — the SAME ones used
by the other risk-score repos, no separate bot needed):
    TELEGRAM_BOT_TOKEN  — from @BotFather
    TELEGRAM_CHAT_ID    — your user/chat id

Exits non-zero on any failure (missing secrets, bad history file, failed
send) so a broken alert shows up as a red run in Actions instead of failing
silently.

Usage:
    python okb_risk_alert.py
"""

import json
import os
import sys
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent / "data"
HISTORY_FILE = DATA_DIR / "okb_risk_history.json"

TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"

# Sell zones start here — used only to decide whether to show the $500
# holdings-gate reminder line. Mirrors the tier field written by
# okb_risk_score.py (values: "sell1", "sell2", "sell3").
SELL_TIERS = {"sell1", "sell2", "sell3"}

COMPONENT_LABELS = [
    ("log_regression", "Log-regression"),
    ("ma200_multiple", "200d MA multiple"),
    ("rsi14", "RSI-14"),
    ("vol_adj_momentum", "Vol-adj momentum"),
]


def load_latest() -> dict:
    if not HISTORY_FILE.exists():
        print(f"ERROR: {HISTORY_FILE} not found — run okb_risk_score.py first.", file=sys.stderr)
        sys.exit(1)
    data = json.loads(HISTORY_FILE.read_text())
    if not data:
        print(f"ERROR: {HISTORY_FILE} is empty.", file=sys.stderr)
        sys.exit(1)
    return data[-1]


def fmt_usd(v) -> str:
    return f"${v:,.0f}" if v >= 1000 else f"${v:,.2f}"


def build_message(row: dict) -> str:
    score = row["composite_score"]
    zone = row["zone"]
    action = row["action"]
    price = row["close"]
    tier = row.get("tier")

    lines = [
        f"\U0001fa99 OKB Risk Score — {row['date']}",
        "",
        f"Score:  {score:.2f}  [{zone}]",
        f"Price:  {fmt_usd(price)}  (OKB/USDT)",
        "",
        f"\u2192 {action}",
    ]

    if tier in SELL_TIERS:
        lines.append("   \u26a0\ufe0f Only if OKB holdings \u2265 $500 — check before acting")
    elif row.get("size_usd") is not None:
        mult = row.get("multiplier")
        mult_note = f" ({mult}\u00d7 base)" if mult is not None else ""
        lines.append(f"   DCA this week: ${row['size_usd']:.2f}{mult_note}")

    lines.append("")
    lines.append("Components")
    comps = row.get("components", {})
    for key, label in COMPONENT_LABELS:
        v = comps.get(key)
        v_str = f"{v:.2f}" if v is not None else "\u2014"
        lines.append(f"  {label:<17}{v_str}")

    return "\n".join(lines)


def send_telegram(message: str, token: str, chat_id: str) -> None:
    url = TELEGRAM_API_URL.format(token=token)
    resp = requests.post(
        url,
        data={"chat_id": chat_id, "text": message},
        timeout=30,
    )
    if not resp.ok:
        print(f"ERROR: Telegram send failed ({resp.status_code}): {resp.text}", file=sys.stderr)
        sys.exit(1)


def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("ERROR: TELEGRAM_BOT_TOKEN and/or TELEGRAM_CHAT_ID env vars not set.", file=sys.stderr)
        sys.exit(1)

    row = load_latest()
    message = build_message(row)
    send_telegram(message, token, chat_id)
    print(f"Sent Telegram alert for {row['date']} — score={row['composite_score']} [{row['zone']}]")


if __name__ == "__main__":
    main()
