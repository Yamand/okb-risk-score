"""
OKB Risk Score — daily composite 0-1 score from OKX public candlestick data
for OKB-USDT (OKX's own exchange/utility token).

DATA SOURCE (why OKX, not Binance):
  - OKB is NOT listed on Binance at all (OKX's native token isn't tradable
    there), so the Binance-klines pattern used by the BTC/ETH scripts is not
    an option for this asset.
  - OKX itself is the obvious and best source: OKB-USDT is one of the
    exchange's own oldest pairs, so it should have the deepest available
    history for this token anywhere. Same endpoint and pagination pattern
    already proven working for XAUT-USDT in the gold risk score repo.
  - No live run against OKX has happened yet from the environment that built
    this — verify the first local run's printed date range before relying
    on it, same caveat as the gold repo.

Components (all normalized 0-1 via expanding historical percentile rank,
so the score self-calibrates over time without hardcoded thresholds):

  1. Log-regression band position   (35%) — price vs. long-term log-log growth curve
  2. 200-day MA multiple            (25%) — price stretch vs. long-term trend
  3. RSI-14 (daily)                 (20%) — short-term overbought/oversold
  4. Volatility-adjusted momentum   (20%) — 30d return / 30d realized vol,
                                     3-day EMA smoothed pre-rank to dampen
                                     30d rolling-window edge effects

0 = cheap / accumulate harder.  1 = expensive / reduce or take profit.

OKX API NOTES:
  - Endpoint: GET /api/v5/market/history-candles — OKX's endpoint for older
    historical data (their /market/candles endpoint only returns a limited
    recent window). Docs:
    https://www.okx.com/docs-v5/en/#order-book-trading-market-data-get-candlesticks-history
  - Paginate backwards in time using the "after" param (returns candles with
    timestamp strictly earlier than "after"), starting from now and walking
    back until an empty page signals the true start of history — no
    known/hardcoded listing date is assumed (OKB's exact OKX listing date
    isn't reliably documented, unlike e.g. ETHUSDT's Binance listing date).
  - Response rows are ordered NEWEST-first: [ts, open, high, low, close,
    vol, volCcy, volCcyQuote, confirm], all values as strings, ts in ms.
  - No API key needed for public market data.

Usage:
    python okb_risk_score.py            # fetch full history, recompute, write data/okb_risk_history.json
    python okb_risk_score.py --update   # fetch only recent candles and merge (fast daily run)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

OKX_HISTORY_URL = "https://www.okx.com/api/v5/market/history-candles"
INST_ID = "OKB-USDT"
BAR = "1D"
DATA_DIR = Path(__file__).parent / "data"
HISTORY_FILE = DATA_DIR / "okb_risk_history.json"
# Raw close-price cache: EVERY fetched date/close, including the pre-warmup
# rows that never get a composite_score (dropped from HISTORY_FILE by
# build_output). --update merges from THIS file, not from HISTORY_FILE, so
# the regression fit and expanding percentile ranks always see the full
# price series and match a full recompute exactly. Reconstructing prices
# from the scored output alone silently drops the earliest ~200-260 days,
# which skews the global log-regression fit (those rows anchor the low end
# of the log-days range) and can shift the composite score enough to flip
# a DCA zone near a boundary.
PRICES_FILE = DATA_DIR / "okb_prices_raw.json"

WEIGHTS = {
    "log_regression": 0.35,
    "ma200_multiple": 0.25,
    "rsi14": 0.20,
    "vol_adj_momentum": 0.20,
}

# OKB was launched by the OK Blockchain Foundation in March 2018; no exact
# public day is consistently documented (unlike BTC's genesis block or
# ETHUSDT's Binance listing date), so this uses the first of that month as
# a reasonable anchor. Unlike gold's Nixon Shock anchor (a "no growth-curve"
# asset used only for structural parity), OKB is a real launched token with
# adoption dynamics, so this plays the same role here as ETH's genesis date.
# Must never change once scores have been computed and stored, or historical
# composite_score values become inconsistent with new ones.
REGRESSION_ANCHOR = pd.Timestamp("2018-03-01")


def fetch_okx_history(stop_before: pd.Timestamp = None, limit=100):
    """
    Page backwards through OKX's history-candles endpoint from now until
    either the true start of history (an empty page) or, if stop_before is
    given, until candles older than that date have been reached (used for
    fast --update runs that only need a recent tail).
    """
    all_rows = []
    after_cursor = None  # None = start from the most recent candle
    while True:
        params = {"instId": INST_ID, "bar": BAR, "limit": limit}
        if after_cursor is not None:
            params["after"] = after_cursor
        resp = requests.get(OKX_HISTORY_URL, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("code") != "0":
            raise RuntimeError(f"OKX API error (code={payload.get('code')}): {payload.get('msg')}")
        rows = payload.get("data", [])
        if not rows:
            break
        all_rows.extend(rows)
        oldest_ts_ms = int(rows[-1][0])  # rows are newest-first
        after_cursor = oldest_ts_ms
        if stop_before is not None and oldest_ts_ms < int(stop_before.timestamp() * 1000):
            break
        if len(rows) < limit:
            break
        time.sleep(0.15)  # be polite to the public endpoint
    return all_rows


def okx_rows_to_df(rows):
    df = pd.DataFrame(rows, columns=[
        "ts", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm",
    ])
    df["date"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms").dt.normalize()
    df["close"] = df["close"].astype(float)
    df = df[["date", "close"]].drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    return df


def percentile_rank_expanding(series: pd.Series, min_periods=60) -> pd.Series:
    """
    For each point, rank it against all prior history (inclusive), scaled 0-1.
    This is what makes each component self-calibrating: no hardcoded bounds,
    the definition of 'cheap' vs 'expensive' adapts as more history accumulates.
    """
    def rank_last(window):
        if len(window) < min_periods:
            return np.nan
        return (window <= window[-1]).sum() / len(window)

    return series.expanding(min_periods=min_periods).apply(rank_last, raw=True)


def compute_components(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_price"] = np.log(df["close"])
    df["days_since_anchor"] = (df["date"] - REGRESSION_ANCHOR).dt.days
    df["log_days"] = np.log(df["days_since_anchor"])

    # --- 1. Log-regression band position ---
    # Fit log(price) ~ a * log(days) + b using all available history (refit each run).
    # NOTE: OKX's OKB-USDT history almost certainly doesn't reach back to the token's
    # actual March 2018 launch, so this regression is fit on a shorter window than
    # OKB's full life — it will be less stable in the first year or two of computed
    # scores, same caveat as the ETH script re: Binance's ETHUSDT listing gap.
    coeffs = np.polyfit(df["log_days"], df["log_price"], 1)
    df["log_price_fit"] = np.polyval(coeffs, df["log_days"])
    df["regression_residual"] = df["log_price"] - df["log_price_fit"]
    df["log_regression"] = percentile_rank_expanding(df["regression_residual"])

    # --- 2. 200-day MA multiple ---
    df["ma200"] = df["close"].rolling(200, min_periods=200).mean()
    df["ma200_ratio"] = df["close"] / df["ma200"]
    df["ma200_multiple"] = percentile_rank_expanding(df["ma200_ratio"])

    # --- 3. RSI-14 ---
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    df["rsi14"] = (rsi / 100).clip(0, 1)

    # --- 4. Volatility-adjusted momentum ---
    df["ret"] = df["close"].pct_change()
    df["roc_30d"] = df["close"].pct_change(30)
    df["vol_30d"] = df["ret"].rolling(30, min_periods=30).std()
    df["vol_adj_mom_raw"] = df["roc_30d"] / df["vol_30d"].replace(0, np.nan)
    # 3-day EMA on the raw ratio before ranking — same smoothing as the ETH
    # script, to cut day-to-day composite noise from 30d rolling-window edge
    # effects without adding meaningful lag.
    df["vol_adj_mom_smoothed"] = df["vol_adj_mom_raw"].ewm(span=3, min_periods=1, adjust=False).mean()
    df["vol_adj_momentum"] = percentile_rank_expanding(df["vol_adj_mom_smoothed"])

    return df


def compute_composite(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["composite_score"] = (
        df["log_regression"] * WEIGHTS["log_regression"]
        + df["ma200_multiple"] * WEIGHTS["ma200_multiple"]
        + df["rsi14"] * WEIGHTS["rsi14"]
        + df["vol_adj_momentum"] * WEIGHTS["vol_adj_momentum"]
    )
    return df


# Base weekly DCA size. Zone sizes below are BASE_WEEKLY_USD * multiplier.
BASE_WEEKLY_USD = 10

# Same zone table shape as the ETH script (OKB behaves like a volatile
# altcoin/exchange token, closer to ETH than to gold) — adjust to taste once
# you've seen how OKB's actual score distribution looks.
ZONES = [
    # (upper_bound_exclusive, zone, tier, multiplier, action)
    (0.10, "Extreme Buy",   "buy",   3.0, "Max accumulate"),
    (0.20, "Strong Buy",    "buy",   1.5, "Accumulate"),
    (0.25, "Buy",           "buy",   1.0, "Normal DCA"),
    (0.35, "Reduced Buy",   "buy",   0.5, "Slow down"),
    (0.60, "Stop — Hold",   "hold",  0.0, "Accumulation done"),
    (0.70, "Sell Tier 1",   "sell1", None, "Exit 5% of holdings"),
    (0.80, "Sell Tier 2",   "sell2", None, "Exit 10% of holdings"),
    (1.01, "Sell Tier 3 / Exit", "sell3", None, "Exit 20% or full position"),
]


def zone_for_score(score):
    if pd.isna(score):
        return {"zone": "Insufficient history", "tier": "none", "multiplier": None,
                "size_usd": None, "action": "—"}
    for upper, zone, tier, mult, action in ZONES:
        if score < upper:
            size = round(BASE_WEEKLY_USD * mult, 2) if mult is not None else None
            return {"zone": zone, "tier": tier, "multiplier": mult, "size_usd": size, "action": action}
    # score == 1.0 edge case, falls into last zone above via < 1.01
    upper, zone, tier, mult, action = ZONES[-1]
    return {"zone": zone, "tier": tier, "multiplier": mult, "size_usd": None, "action": action}


def build_output(df: pd.DataFrame) -> list:
    out = []
    for _, row in df.iterrows():
        if pd.isna(row["composite_score"]):
            continue
        z = zone_for_score(row["composite_score"])
        out.append({
            "date": row["date"].strftime("%Y-%m-%d"),
            "close": round(row["close"], 4),
            "composite_score": round(row["composite_score"], 4),
            "zone": z["zone"],
            "tier": z["tier"],
            "multiplier": z["multiplier"],
            "size_usd": z["size_usd"],
            "action": z["action"],
            "components": {
                "log_regression": round(row["log_regression"], 4) if not pd.isna(row["log_regression"]) else None,
                "ma200_multiple": round(row["ma200_multiple"], 4) if not pd.isna(row["ma200_multiple"]) else None,
                "rsi14": round(row["rsi14"], 4) if not pd.isna(row["rsi14"]) else None,
                "vol_adj_momentum": round(row["vol_adj_momentum"], 4) if not pd.isna(row["vol_adj_momentum"]) else None,
            },
        })
    return out


def load_existing_closes() -> pd.DataFrame:
    """
    Load the FULL raw close-price cache (not the scored HISTORY_FILE, which is
    missing the pre-warmup rows — see PRICES_FILE comment above for why that
    distinction matters).
    """
    if not PRICES_FILE.exists():
        return pd.DataFrame(columns=["date", "close"])
    existing = json.loads(PRICES_FILE.read_text())
    if not existing:
        return pd.DataFrame(columns=["date", "close"])
    df = pd.DataFrame({
        "date": pd.to_datetime([r["date"] for r in existing]),
        "close": [r["close"] for r in existing],
    })
    return df


def save_prices_raw(df: pd.DataFrame) -> None:
    """Persist the full date/close series (including pre-warmup rows) for future merges."""
    rows = [{"date": d.strftime("%Y-%m-%d"), "close": round(c, 4)}
            for d, c in zip(df["date"], df["close"])]
    PRICES_FILE.write_text(json.dumps(rows, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true",
                         help="Only fetch recent candles (last 400 days) instead of the full "
                              "history from OKX. Indicators are still recomputed over the "
                              "FULL closing-price series (raw price cache + freshly fetched "
                              "tail merged) so results match a full recompute exactly — this "
                              "flag only speeds up the network fetch, not the math.")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    existing_df = load_existing_closes() if args.update else pd.DataFrame(columns=["date", "close"])

    stop_before = None
    if args.update and not existing_df.empty:
        stop_before = pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(days=400)

    print(f"Fetching {INST_ID} daily candles from OKX"
          f"{f' (stopping once older than {stop_before.date()})' if stop_before is not None else ' (full history)'}...")
    rows = fetch_okx_history(stop_before=stop_before)

    if not rows:
        raise RuntimeError(
            f"OKX returned zero candles for {INST_ID}. Either the instId is wrong (check it "
            f"exists at https://www.okx.com/trade-spot/okb-usdt) or the API is unreachable "
            "from this network — this is why you'd see an empty data/ folder: DATA_DIR.mkdir() "
            "runs before this check, but nothing gets written after it."
        )

    fetched_df = okx_rows_to_df(rows)
    print(f"Fetched {len(fetched_df)} daily candles, {fetched_df['date'].min().date()} to {fetched_df['date'].max().date()}")

    if not existing_df.empty:
        df = (
            pd.concat([existing_df, fetched_df], ignore_index=True)
            .drop_duplicates(subset="date", keep="last")
            .sort_values("date")
            .reset_index(drop=True)
        )
        print(f"Merged with existing history: {len(df)} total daily candles, "
              f"{df['date'].min().date()} to {df['date'].max().date()}")
    else:
        df = fetched_df

    df = compute_components(df)
    df = compute_composite(df)
    output = build_output(df)

    if not output:
        raise RuntimeError(
            f"Fetched {len(df)} candles but none produced a composite_score — likely fewer "
            "than ~200-260 days of history exist yet for this pair (the 200-day MA and "
            "percentile-rank windows need that much warm-up). Check the fetched date range "
            "printed above; if it's genuinely that short, this script can't produce scores "
            "yet and needs more history to accumulate first."
        )

    save_prices_raw(df[["date", "close"]])
    HISTORY_FILE.write_text(json.dumps(output, indent=2))
    print(f"Wrote {HISTORY_FILE}, {len(output)} rows ({PRICES_FILE.name} raw cache also updated)")

    if output:
        latest = output[-1]
        size = f"${latest['size_usd']}" if latest['size_usd'] is not None else "—"
        print(f"\nLatest ({latest['date']}): score={latest['composite_score']} "
              f"[{latest['zone']}] size={size}/wk — {latest['action']}")


if __name__ == "__main__":
    main()
