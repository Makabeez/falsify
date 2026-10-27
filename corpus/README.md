# Falsify claim corpus

`claims.jsonl` holds 35 public, dated trading claims taken from the open web. Each line is one JSON object. They are inputs for testing the Falsify compiler, which turns a claim into mechanical rules and backtests them on Binance spot data.

## How it was collected

- Web searches across TradingView public ideas and strategy scripts, Substack and other blogs, forums (Gainium community, BabyPips, Elite Trader) and research/backtest sites. Queries used a range of terms, such as RSI, golden cross, EMA crossover, Donchian/Turtle, Bollinger, MACD, Supertrend, buy-the-dip, day-of-week and the 200-week MA.
- Every excerpt comes from a fetch of the page itself, never from a search snippet. The fetch asked for the author's words verbatim. An excerpt is the passage that states the rules and claims, up to about 700 characters, and separate passages are joined with ` … `. Formatting such as bold markers and code backticks was removed. Otherwise the text is as returned.
- `verbatim_confidence` is `low` when the fetched text may not be complete or exact (1 record).
- Mirror domains of TradingView (tr./in./my./…) were treated as one page and stored under `www.tradingview.com`. The corpus holds no more than 12 TradingView records.

## Inclusion criteria

- **rule / backtest_claim**: a crypto trading rule with at least one concrete number, such as an indicator setting or threshold, a percentage move, a breakout length or a holding period. The asset is one of BTC, ETH, SOL, BNB, XRP, DOGE, ADA, AVAX, LINK, LTC, DOT or TRX. The timeframe is 1h, 4h, daily or unstated. A record is `backtest_claim` when the source also states a backtest or performance result.
- **vague**: a discretionary or non-mechanical claim, such as a pattern read, buy-the-fear advice, an undefined proprietary indicator or a system with no parameters.
- **other_asset**: a mechanical rule on an asset outside the universe above (gold, S&P 500, Nikkei 225 stocks, forex).
- **Skipped**: single price predictions, signal-group ads with no rule, 1m–30m timeframes, and anything that depends on leverage, funding, order flow, on-chain data, sentiment or news.

## Fields

`id, url, source, posted (YYYY-MM-DD or null), asset_hint, timeframe_hint, category, excerpt, performance_claim, verbatim_confidence, notes`

When a page showed no year or no date, `notes` says how the date was inferred, for example from a stated test period or a backtest run date, or that it is missing.

## Use and attribution

The excerpts are short quotations used for analysis and commentary, and each one links to its original source. No author names, usernames or handles are stored. Any that appeared inside a quoted passage were left out. The `url` field is kept for citation only. Inclusion does not mean a claim is endorsed. The point of the corpus is to test the claims.

## Caveat on "verbatim"

Pages were read through a web-fetch tool that returns the passages it quotes from the page, so an excerpt is the text as that tool returned it. Spot-check any record against its `url` before citing it. Falsify's quote check runs against the stored excerpt, so a record whose excerpt drifted from the page would show up as a mismatch only when compared with the original.
