<p align="center">
  <img src="assets/banner.svg" alt="Falsify: test trading claims before you trade them" width="100%"/>
</p>

<p align="center"><b>Paste a trading tip, a chart screenshot or an influencer clip. Get back whether it survives contact with real market history.</b></p>

<p align="center">
  <a href="https://falsify.baserep.xyz"><img src="https://img.shields.io/badge/Live-falsify.baserep.xyz-13223a?style=for-the-badge" alt="Live app"/></a>
  <img src="https://img.shields.io/badge/Meta%20Global%20AI%20Hackathon-Indie%20track-0866FF?style=for-the-badge" alt="Meta Global AI Developer Hackathon, Indie track"/>
  <img src="https://img.shields.io/badge/Muse%20Spark-compiler%20%2B%20auditor-0866FF?style=for-the-badge" alt="Muse Spark"/>
  <img src="https://img.shields.io/badge/license-MIT-black?style=for-the-badge" alt="MIT license"/>
</p>
<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?style=flat-square" alt="Python 3.10+"/>
  <img src="https://img.shields.io/badge/data-Binance%20spot%202017--2026-F0B90B?style=flat-square" alt="Binance spot data"/>
  <img src="https://img.shields.io/badge/judge%20demo-offline%2C%20no%20key-2ea44f?style=flat-square" alt="Offline judge demo"/>
</p>

> Retail traders lose real money on strategies nobody ever tested: an RSI rule with a "80% win rate", a weekend pump, a breakout that "works every time". Testing one properly takes a quant an afternoon. Falsify does it in seconds, and its default answer is that the claim is wrong until the data says otherwise.

## Try it

**Live: [falsify.baserep.xyz](https://falsify.baserep.xyz).** Paste a post or drop a screenshot or clip, and get the verdict with the claimant's words highlighted next to the rules they became. The 35 public claims in [`corpus/`](corpus/) are one click each and don't call the model.

## Run it in one command

```bash
git clone https://github.com/Makabeez/falsify && cd falsify
make judge-demo          # creates a venv, verifies data hashes, runs every claim in claims/
```

No API key, no network after install. The demo checks every bundled price file against `data/manifest.json` (SHA-256) and refuses to run if one was altered.

## Why the model doesn't write the backtest

LLM-written backtests lie in a specific way: a signal computed on bar *t* filled at bar *t*'s close, an indicator that peeks one bar ahead, a stop that fills at the level after a gap. The output looks great and is fiction.

So Falsify splits the job:

```
  claim (text | screenshot | .mp4)
        │
        ▼
  ┌────────────────────────┐   screenshots and clips only
  │ Muse Spark: TRANSCRIBE │── verbatim visible text + speech: the record quotes are checked against
  └────────────────────────┘
        │
        ▼
  ┌────────────────────────┐   structured output, constrained decoding
  │ Muse Spark: COMPILE    │── ClaimSpec JSON (typed DSL, no code) + one evidence row per rule element
  └────────────────────────┘
        │                    ▲
        ▼                    │ schema error, failed quote or dead rule → sent back, max 2 repairs
  ┌────────────────────────┐ │
  │ Validate + QUOTE CHECK │─┘ deterministic: every rule element quoted verbatim from the source,
  │ + DOES IT EVER FIRE?   │   or declared assumed (paraphrases are refused); and the entry must trigger
  └────────────────────────┘   at least once on the full history (contradictory conditions are refused).
        │
        ▼
  ┌────────────────────────┐
  │ Muse Spark: AUDIT      │── source + spec → every place the spec departs from what was said
  └────────────────────────┘
        │  major issue? → findings go back to the compiler for one repair, then re-audit
        ▼  a part of the claim the DSL can't express → verdict is labelled an approximation
        │  audit or quote check still failing after the repairs → NEEDS A HUMAN CHECK
        │     (visitors flag misread rules → `falsify review queue` → a trader signs off on the permalink)
        │
        ▼
  ┌────────────────────────┐   deterministic, no model in the P&L path
  │ Engine                 │── signal at close t → fill at open t+1, pessimistic stops, real costs
  └────────────────────────┘
        │
        ▼
  ┌────────────────────────┐
  │ Falsification battery  │── costs · random-entry null · post-claim window · halves · 11 other assets · every reading of vague params
  └────────────────────────┘
        │
        ▼
     SURVIVES | DIES | INCONCLUSIVE | UNTESTABLE   + exact / approximation  + "4/5 rule elements quoted, 1 assumed"
```

The model's job is translation, which it is good at. It never gets the last word on either side of it:

- **What the claimant said** is checked by `falsify/grounding.py`. Every entry and exit condition, stop, target, time exit, asset, timeframe and direction needs an evidence row with the claimant's exact words, or an explicit `assumed: true`. A quote that isn't literally in the post (or in the transcript of the screenshot or clip) is refused and sent back. The verdict then tells you how much of what was tested is the claimant's rule and how much is our reading.
- **Whether we tested their rule or a cousin of it** is decided by the audit. Muse Spark re-reads the source next to the compiled spec and lists every departure (it caught WMA silently compiled as SMA, and a DCA ladder quietly dropped). Major findings go back to the compiler once. If something still can't be expressed, the verdict leads with "Tested an approximation of the claim", in the headline, not a footnote.
- **When the checks still disagree, a person decides.** Retries are bounded: two repairs for schema and quote errors, one for the audit. After that the result is marked *Needs a human check* rather than presented as settled. Anyone can say the rules misread a claim; those flags and the escalated results form a queue (`falsify review queue`), and a trader's sign-off (`falsify review sign <key>`) is shown next to the verdict on the result's permalink.
- **How often the compiler reads claims the way a trader does** is measured, not asserted. `corpus/ground_truth.yaml` holds reference rules for all 35 corpus claims, drafted from the claim text by a separate agent that never saw the compiler's output, then reviewed by a trader (`reviewed: true`). `falsify groundtruth` scores routing, asset, timeframe, direction, entry, exit, stop, target and time exit against it ([docs/ground_truth.md](docs/ground_truth.md)); reviewed claims show "Rules checked by a trader" in the app, or where the trader's reading differs.
- **What verification costs** is logged per claim (seconds and tokens for compile, audit, and the repair round) and reported in the [corpus scoreboard](corpus/results/scoreboard.md).
- **What the rule does** is computed by code whose timing contract is pinned by tests (`tests/test_engine.py`), including a check that truncating future data never changes a past trade.

## The battery

| Test | Question it asks | Effect on verdict |
|---|---|---|
| Costs | Does it make money after 10 bps fee + 2 bps slippage per side? | Fatal if not |
| Random-entry null | Do random entries with the same number of trades, holding times and side do as well? Strips out market drift: a long-only rule in a bull market beats zero but not random longs. 2,000 simulations. | SURVIVES needs p < 0.05. Fatal if p ≥ 0.5 (worse than the median coin flip). In between: INCONCLUSIVE |
| Post-claim window | If the claim shows a period, what happened after it? The only truly out-of-sample data. | Fatal if it loses (≥ 10 trades) |
| Sample size | At least 30 trades? | INCONCLUSIVE below |
| Halves | Is the edge positive in both halves of history? | Warning |
| Cross-asset | Does it work on the other 11 bundled assets? | Warning |
| Vague parameters | "Oversold" could mean 20, 25, 30 or 35. Every reading is run; picking the best one pays a Bonferroni penalty. | Warning |
| Claimed vs measured | Claimed 80% win rate, measured 61%. | Warning |
| Buy-and-hold | Would just holding have done better? | Warning |
| Open at the end | A no-stop rule's last position is often a loser that never closed. It is marked at the last close and counted, never dropped. | Counted |
| History coverage | Does the claim's period start before our data? | Warning |

**Verdicts.** **DIES** means evidence against the claim: it loses money after costs, loses after the post date, or random timing does at least as well more often than not. **INCONCLUSIVE** means the data can't tell it from luck: it makes money, but not convincingly enough (0.05 ≤ p < 0.5), or there are fewer than 30 trades. **SURVIVES** needs p < 0.05 and no fatal test. **UNTESTABLE** means no mechanical rule could be recovered, or the asset isn't in the data.

Thresholds are fixed statistical conventions (p < 0.05, n ≥ 30, the median at p = 0.5), stated before any claim is run and never tuned per claim. One change is on the record: until 27 Oct, any p ≥ 0.05 was DIES. A forum claim that beat buy-and-hold after costs came back DIES at p = 0.098, and "can't tell from luck" isn't "refuted". So the split above was made before the full corpus run, and `git log` shows when.

## Is it rigged to kill everything?

No. That's measured, not asserted. `falsify calibrate` runs 300 coin-flip strategies per setup through the battery's own thresholds, then plants a true edge on top and checks whether it gets through. Full table: [docs/calibration.md](docs/calibration.md).

| Setup (bundled data, costs included) | Coin flips a naive backtest calls profitable | Coin flips that pass Falsify | Edge caught 80% of the time |
|---|---|---|---|
| BTC daily, 7-day holds, 30 trades | 58% | 5.3% | 4.5% per trade |
| BTC daily, 7-day holds, 100 trades | 69% | 3.3% | 2.6% per trade |
| BTC 4h, 24h holds, 300 trades | 22% | 6.0% | 0.50% per trade |
| ETH 1h, 12h holds, 1000 trades | 1% | 1.0% | 0.34% per trade |

**On BTC daily, a naive backtest calls most strategies that pick entries by coin flip profitable, because BTC went up. Falsify passes about 5%, exactly what a p < 0.05 threshold promises (300 runs per setup, so ±1.3 points of simulation noise).** Plant a real edge and it gets through: how big it has to be depends on how many trades the claim has. That's why a 30-trade daily setup usually can't prove what it claims, however confident the post sounds.

## What it says about some classic setups

From `make judge-demo` (Binance spot, Aug 2017 to Oct 2026, including the 2018 and 2022 bear markets, costs included):

| Claim (hand-compiled, generic) | Verdict | Why |
|---|---|---|
| ETH 4h: buy RSI oversold, sell overbought, "80% win rate" | **DIES** | 75 trades, 56% win rate, −3.19% per trade, −98% total: no stop, so the 2018 and 2022 bear markets were held all the way down; random timing beat it 99% of the time |
| Buy BTC after a 10% weekly drop, hold 7 days | **DIES** | 58% win rate, but −24% total after costs; random 7-day longs did as well 75% of the time |
| Buy SOL Friday close, sell Monday open | **INCONCLUSIVE** | +0.77% per trade over 324 weekends, but random 2-day holds did as well 15% of the time |
| BTC 4h breakout above 20-bar high on 2× volume, 3% stop / 6% target | **INCONCLUSIVE** | 39% win rate, +0.27% per trade, random entries did as well 33% of the time |
| BTC daily golden cross / death cross | **INCONCLUSIVE** | 9 trades since 2017; buy-and-hold made +1781% vs +635% |
| "Accumulate strong projects when smart money buys" | **UNTESTABLE** | No mechanical trigger |

## Web app

```bash
make web            # http://127.0.0.1:8097, live checks need MODEL_API_KEY in .env
```

`falsify/web.py` (FastAPI) runs live checks in a small worker pool and streams each stage to the page: transcribe, compile, quote check, audit, repair, backtest. Results are cached by content hash, so every result is a permalink (`/r/<key>`) and the same claim never costs twice. Live checks are capped per IP and per day, files are sniffed by content (not by extension) and size-limited, and `FALSIFY_LIVE=0` pauses them while keeping the examples up. The page is one static HTML file with no build step and no third-party JavaScript.

## With Muse Spark

```bash
cp .env.example .env          # add MODEL_API_KEY from dev.meta.ai
set -a; . ./.env; set +a
.venv/bin/falsify ping                  # key works, models listed, one structured call
.venv/bin/falsify check --text "Buy ETH when RSI drops under 30 on the 4h, sell above 70. 80% win rate."
.venv/bin/falsify check screenshot.png --note "posted on X, Oct 2026"
.venv/bin/falsify check clip.mp4
.venv/bin/falsify extract screenshot.png -o claims/my_claim.json   # compile + audit only
```

| Meta model | Role |
|---|---|
| `muse-spark-1.3` | Transcribes screenshots; compiles text and screenshots into a quote-backed `ClaimSpec` with structured output; audits the spec against the source |
| `muse-spark-1.2` | Same three passes for video clips (the 1.3 docs flag audio understanding as degraded; pitches live in the speech) |

## Real vs. simulated

| Component | Status |
|---|---|
| Price history | **Real.** Binance spot klines from the public market-data API, 12 assets × 1h/4h/1d from each asset's listing (BTC/ETH from Aug 2017), pinned snapshot + SHA-256 manifest |
| Costs | **Modeled.** 10 bps fee + 2 bps slippage per side (Binance spot taker), configurable |
| Shorts | **Modeled.** Priced as spot short with no borrow or funding cost |
| Example claims in `claims/` | **Hand-compiled** paraphrases of common retail setups. No real posts, no authors |
| Battery calibration | **Measured** on the bundled data with coin-flip strategies and planted edges (`falsify calibrate`) |
| Muse Spark transcribe/compile/audit | **Live** on Meta Model API (`muse-spark-1.3`). Repair loops (schema, quote check, audit) also pinned against a mock client in `tests/`. Corpus results: `corpus/results/scoreboard.md` |

## Known limits

- **Power.** Measured in [docs/calibration.md](docs/calibration.md): a daily rule with ~30 trades needs about 4.5% per trade of real edge to be caught 80% of the time. Low-frequency claims will mostly come back INCONCLUSIVE or DIES on the null, and that's the honest answer for that sample size, not a verdict that the idea can never work.
- **Rule language.** Indicators: sma, ema, wma, rsi, atr, adx, Bollinger, MACD, Supertrend, breakouts, % change, volume, day-of-week, hour, plus rolling min/max of any of them and previous-bar values ("RSI was under 30 on the last bar", "three higher closes"). Exits: % or ATR stops, % / ATR / R-multiple targets, % or ATR trailing stops, time exits, signal exits. Ichimoku, VWAP, DCA ladders, position sizing, partial exits, breakeven moves and multi-timeframe rules are not expressible yet; claims that need them are tested on their core and labelled approximations, or come back UNTESTABLE.
- **Bar-close logic only.** Intrabar claims ("buy the wick") are approximated at the next open.
- **Spot data.** No funding rates, order flow, sentiment or news; claims that need them are reported UNTESTABLE rather than approximated.
- Not financial advice. A claim that SURVIVES has passed tests, not earned your money.

## Built during the hackathon

Everything in this repository was written during the Meta Global AI Developer Hackathon (26 Oct – 4 Nov 2026). The first commit is the empty kickoff commit of 26 Oct; the git history is the build log. No code from earlier projects is reused.

## Data attribution

Market data: Binance public market-data API (`data-api.binance.vision`). Rebuild with `make data`.

## License

MIT
