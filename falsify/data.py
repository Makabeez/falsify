"""Bundled price history.

Source: Binance public market-data mirror (data-api.binance.vision), spot
klines, no API key. History ships in the repo as Parquet with a SHA-256
manifest so `make judge-demo` runs offline and fails closed if a file was
altered. `falsify data fetch` rebuilds it.

Binance's main API geo-blocks some regions (HTTP 451); the market-data mirror
is the documented public alternative and is what we use here.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://data-api.binance.vision/api/v3/klines"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MANIFEST = DATA_DIR / "manifest.json"

UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT",
    "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT", "DOTUSDT", "TRXUSDT",
]
TIMEFRAMES = ["1h", "4h", "1d"]
START = "2017-01-01"
_BAR_MS = {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}


def _path(symbol: str, timeframe: str) -> Path:
    return DATA_DIR / f"{symbol}_{timeframe}.parquet"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(symbol: str, timeframe: str, start: str = START, end: str | None = None,
          session: requests.Session | None = None) -> pd.DataFrame:
    """Download closed klines [start, end) and return a UTC-indexed OHLCV frame."""
    s = session or requests.Session()
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000) if end else int(time.time() * 1000)
    step = _BAR_MS[timeframe]
    rows: list[list] = []
    cursor = start_ms
    while cursor < end_ms:
        for attempt in range(5):
            r = s.get(BASE_URL, params={"symbol": symbol, "interval": timeframe,
                                        "startTime": cursor, "endTime": end_ms - 1, "limit": 1000}, timeout=20)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            break
        else:
            raise RuntimeError(f"giving up on {symbol} {timeframe} at {cursor}")
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1][0] + step
    if not rows:
        raise RuntimeError(f"no data for {symbol} {timeframe}")
    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_base", "taker_quote", "ignore"])
    now_ms = int(time.time() * 1000)
    df = df[df["close_time"] < now_ms]  # drop the still-forming bar
    df = df.drop_duplicates("open_time").sort_values("open_time")
    out = pd.DataFrame({
        "open": df["open"].astype(float).values,
        "high": df["high"].astype(float).values,
        "low": df["low"].astype(float).values,
        "close": df["close"].astype(float).values,
        "volume": df["volume"].astype(float).values,
    }, index=pd.to_datetime(df["open_time"].values, unit="ms", utc=True))
    out.index.name = "time"
    return out


def build(symbols: list[str] = UNIVERSE, timeframes: list[str] = TIMEFRAMES,
          start: str = START, end: str | None = None) -> dict:
    """Fetch everything, write Parquet + manifest. `end` pins the snapshot."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    end = end or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    session = requests.Session()
    files = {}
    for sym in symbols:
        for tf in timeframes:
            df = fetch(sym, tf, start, end, session)
            p = _path(sym, tf)
            df.to_parquet(p, compression="zstd")
            files[p.name] = {"sha256": _sha256(p), "rows": len(df),
                             "first": df.index[0].isoformat(), "last": df.index[-1].isoformat()}
            print(f"  {p.name:<22} {len(df):>7} bars  {df.index[0].date()} -> {df.index[-1].date()}")
    manifest = {"source": BASE_URL, "start": start, "end_exclusive": end,
                "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "files": files}
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify() -> list[str]:
    """Return a list of problems; empty means every file matches the manifest."""
    if not MANIFEST.exists():
        return ["data/manifest.json missing -- run `falsify data fetch`"]
    manifest = json.loads(MANIFEST.read_text())
    problems = []
    for name, meta in manifest["files"].items():
        p = DATA_DIR / name
        if not p.exists():
            problems.append(f"{name}: missing")
        elif _sha256(p) != meta["sha256"]:
            problems.append(f"{name}: sha256 mismatch")
    return problems


_cache: dict[tuple[str, str], pd.DataFrame] = {}


def available() -> set[tuple[str, str]]:
    if not MANIFEST.exists():
        return set()
    names = json.loads(MANIFEST.read_text())["files"].keys()
    out = set()
    for n in names:
        sym, tf = n.removesuffix(".parquet").split("_")
        out.add((sym, tf))
    return out


def load(symbol: str, timeframe: str) -> pd.DataFrame:
    key = (symbol.upper(), timeframe)
    if key not in _cache:
        p = _path(*key)
        if not p.exists():
            raise FileNotFoundError(f"no bundled data for {symbol} {timeframe}; "
                                    f"bundled universe: {', '.join(UNIVERSE)}")
        _cache[key] = pd.read_parquet(p)
    return _cache[key]
