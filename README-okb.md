# OKB Risk Score

A daily composite 0–1 risk score for OKB (OKX's own exchange/utility token),
built from OKX's free public candlestick data for OKB-USDT. Static site +
GitHub Actions, same pattern as the BTC/ETH/gold risk score repos.

**Live idea:** `0` = cheap, accumulate harder. `1` = expensive, reduce buys /
start distributing once holdings clear your $500 sell-tier threshold.

## Data source: why OKX (there's no alternative here)

Unlike BTC, ETH, and even XAUT, **OKB isn't listed on Binance at all** — it's
OKX's own native token and doesn't trade anywhere OKX doesn't control the
listing. So this isn't a "which source is more reliable" choice like the
gold repo went through — OKX is simply the only real option, and also the
best one: OKB-USDT should be one of the exchange's oldest and deepest pairs
for this specific token.

Uses the same `/api/v5/market/history-candles` endpoint and backward-paging
pattern already proven for XAUT-USDT in the gold repo (`instId=OKB-USDT`
instead of `XAUT-USDT`). No known/hardcoded start date is assumed — the
script pages back until OKX returns an empty page, and whatever date range
that produces on the first run is the real answer for how much history
exists.

**Caveat: this hasn't been run against live OKX data from the environment
that built it** — same caveat as the gold repo. Run it locally once and
sanity-check the printed date range and first few output rows before
trusting it for real use.

## How the score is built

Same four-component methodology as the BTC/ETH versions, each normalized to
0–1 via **expanding historical percentile rank** (today's raw value ranked
against every prior day back to the start of data):

| Component | Weight | What it captures |
|---|---|---|
| Log-regression band position | 35% | Price vs. long-term log-log growth curve, refit each run |
| 200-day MA multiple | 25% | Price stretch vs. long-term trend (price ÷ 200d MA) |
| RSI-14 (daily) | 20% | Short-term overbought/oversold |
| Volatility-adjusted momentum | 20% | 30d return ÷ 30d realized volatility, 3-day EMA smoothed |

Composite = weighted sum of the four, clipped to [0, 1].

**Log-regression anchor.** OKB is a real launched token (OK Blockchain
Foundation, March 2018) with actual adoption dynamics, so this plays the
same role here as ETH's genesis-date anchor — *not* gold's "no growth curve,
just for structural parity" Nixon Shock anchor. The exact launch day isn't
reliably documented anywhere (unlike BTC's genesis block or ETHUSDT's
Binance listing date), so `2018-03-01` is used as a reasonable approximation.
**This must never change once scores have been computed and stored**, or
historical `composite_score` values become inconsistent with new ones.

**Caveat on history depth.** OKX's own OKB-USDT candle history almost
certainly doesn't reach back to OKB's actual 2018 launch (whatever date
range the first fetch prints is the real constraint), so the log-regression
fit will be on a shorter window than OKB's full life — least meaningful in
roughly the first year of computed scores, same caveat as the ETH script.
The first ~200-260 days of any fetched series also won't produce a score at
all (warm-up period for the 200-day MA and percentile-rank windows).

**Zone table.** Uses the same buy/sell zone thresholds as the ETH script
(OKB behaves like a volatile altcoin/exchange token, closer to ETH than to
gold's structurally different rule table) — adjust in `okb_risk_score.py`
and mirror the change in `index.html`'s `ZONES` once you've seen how OKB's
actual score distribution looks.

## Repo structure

```
okb_risk_score.py                   # fetch + compute + write data/okb_risk_history.json
okb_risk_alert.py                   # reads latest row, sends Telegram summary
data/okb_risk_history.json          # generated — one row per day, public/scored output
data/okb_prices_raw.json            # generated — full raw close-price cache (internal use)
index.html                          # static site, reads data/ directly, Chart.js
.github/workflows/daily-update-okb.yml   # cron job, runs okb_risk_score.py --update + alert daily
```

**Why two data files?** `okb_risk_history.json` only contains rows once all
four components have enough history to compute (~200-260 day warm-up), so
the earliest fetched days never appear there. `okb_prices_raw.json` keeps
*every* fetched date/close, including those warm-up days — `--update` merges
from that file so the regression fit always sees the true full price series,
not a truncated one. (This distinction was a real bug caught while building
the BTC script: reconstructing prices from the scored file alone silently
drops the earliest data and can shift the regression fit enough to flip a
DCA zone near a boundary.)

## Setup

1. Push this repo to GitHub, enable **GitHub Pages** (Settings → Pages →
   Deploy from branch → `main` / root).
2. Run the full backfill once, locally or via Actions "Run workflow" with
   the `--update` flag removed, so `data/okb_risk_history.json` and
   `data/okb_prices_raw.json` both exist before the site goes live. Check
   the printed date range on this first run — that's your real confirmation
   of how much history OKX actually has for this pair.
3. The daily workflow (`daily-update-okb.yml`) runs automatically at 00:25
   UTC, pulls the last ~400 days from OKX, recomputes over the full merged
   price series, commits both updated JSON files, and sends a Telegram
   summary. GitHub Pages redeploys automatically on push.
4. Set the same `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` repo secrets you
   already use for the BTC/ETH/gold repos — no new bot needed.

### Local run

```bash
pip install pandas numpy requests
python okb_risk_score.py             # full history backfill (first run)
python okb_risk_score.py --update    # fast daily run (last ~400 days only)
python -m http.server 8000           # then open localhost:8000/index.html
```

## Notes

- No API key required — OKX's `/api/v5/market/history-candles` endpoint is
  public.
- The score is descriptive, not a signal to auto-trade on. Same discipline
  as the other repos: it scales buy size, doesn't override the plan.
- OKB is an exchange token, not a "neutral" asset like BTC/ETH/gold — its
  price is tied to OKX's own business (fee discounts, buyback-and-burn
  program, exchange volume/health). That's a different risk profile than
  the other three assets in this family; keep that in mind when weighting
  how much you lean on this score.
- Not financial advice.
