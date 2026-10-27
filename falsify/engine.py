"""Deterministic backtest engine for a ClaimSpec.

Timing contract (the part LLM-written backtests get wrong):
  * Every indicator at bar t uses data up to and including the CLOSE of t.
  * A condition true at the close of t fills at the OPEN of t+1.
  * Stops and targets are checked intrabar from the fill bar onward; a gap
    through the level fills at the open, and if stop and target are both
    inside one bar we assume the stop hit first (pessimistic).
  * One position at a time, no pyramiding, no compounding of size.
tests/test_engine.py pins each of these with hand-built price paths.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .spec import ClaimSpec, Condition, Direction, Op, Operand, OperandKind


@dataclass(frozen=True)
class Costs:
    fee_bps: float = 10.0       # Binance spot taker, per side
    slippage_bps: float = 2.0   # per side

    @property
    def per_side(self) -> float:
        return (self.fee_bps + self.slippage_bps) / 1e4


# ---------------------------------------------------------------- indicators

def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss
    rsi = 100 - 100 / (1 + rs)
    return rsi.where(loss != 0, 100.0).where(gain.notna())


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev_close).abs(),
                    (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _wma(close: pd.Series, period: int) -> pd.Series:
    w = np.arange(1, period + 1, dtype=float)
    return close.rolling(period, min_periods=period).apply(lambda x: float(np.dot(x, w) / w.sum()), raw=True)


def _macd(close: pd.Series, fast: int, slow: int, signal: int = 9) -> tuple[pd.Series, pd.Series]:
    line = (close.ewm(span=fast, adjust=False, min_periods=fast).mean()
            - close.ewm(span=slow, adjust=False, min_periods=slow).mean())
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig


def _supertrend(df: pd.DataFrame, period: int, mult: float) -> pd.Series:
    """Standard supertrend line (ATR with Wilder smoothing). Uses only data up to each close."""
    h, l, c = (df[x].to_numpy(dtype=float) for x in ("high", "low", "close"))
    atr = _atr(df, period).to_numpy()
    mid = (h + l) / 2
    upper, lower = mid + mult * atr, mid - mult * atr
    n = len(c)
    line = np.full(n, np.nan)
    fu, fl = np.nan, np.nan
    up = True
    for t in range(n):
        if np.isnan(atr[t]):
            continue
        if np.isnan(fu):
            fu, fl = upper[t], lower[t]
            up = c[t] >= mid[t]
        else:
            fu = upper[t] if (upper[t] < fu or c[t - 1] > fu) else fu
            fl = lower[t] if (lower[t] > fl or c[t - 1] < fl) else fl
            if up and c[t] < fl:
                up = False
            elif not up and c[t] > fu:
                up = True
        line[t] = fl if up else fu
    return pd.Series(line, index=df.index)


def _adx(df: pd.DataFrame, period: int) -> pd.Series:
    h, l = df["high"], df["low"]
    up, down = h.diff(), -l.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr = _atr(df, period)
    a = 1 / period
    plus_di = 100 * plus_dm.ewm(alpha=a, adjust=False, min_periods=period).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=a, adjust=False, min_periods=period).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=a, adjust=False, min_periods=period).mean()


def operand_series(op: Operand, df: pd.DataFrame, cache: dict | None = None) -> pd.Series:
    cache = {} if cache is None else cache
    key = (op.kind, op.period, op.value, op.param)
    if key not in cache:
        k, p = op.kind, op.period
        c = df["close"]
        if k == OperandKind.CONST:
            s = pd.Series(float(op.value), index=df.index)
        elif k in (OperandKind.CLOSE, OperandKind.OPEN, OperandKind.HIGH, OperandKind.LOW, OperandKind.VOLUME):
            s = df[k.value]
        elif k == OperandKind.SMA:
            s = c.rolling(p, min_periods=p).mean()
        elif k == OperandKind.EMA:
            s = c.ewm(span=p, adjust=False, min_periods=p).mean()
        elif k == OperandKind.RSI:
            s = _rsi(c, p)
        elif k == OperandKind.PCT_CHANGE:
            s = c.pct_change(p) * 100
        elif k == OperandKind.HIGHEST_HIGH:
            s = df["high"].rolling(p, min_periods=p).max().shift(1)
        elif k == OperandKind.LOWEST_LOW:
            s = df["low"].rolling(p, min_periods=p).min().shift(1)
        elif k == OperandKind.VOLUME_SMA:
            s = df["volume"].rolling(p, min_periods=p).mean()
        elif k == OperandKind.ATR:
            s = _atr(df, p)
        elif k == OperandKind.DAY_OF_WEEK:
            s = pd.Series(df.index.day_of_week, index=df.index, dtype=float)
        elif k == OperandKind.HOUR_UTC:
            s = pd.Series(df.index.hour, index=df.index, dtype=float)
        elif k == OperandKind.WMA:
            s = _wma(c, p)
        elif k in (OperandKind.BB_UPPER, OperandKind.BB_LOWER):
            mid = c.rolling(p, min_periods=p).mean()
            sd = c.rolling(p, min_periods=p).std(ddof=0)  # population std, as TradingView
            s = mid + op.param * sd if k == OperandKind.BB_UPPER else mid - op.param * sd
        elif k in (OperandKind.MACD_LINE, OperandKind.MACD_SIGNAL, OperandKind.MACD_HIST):
            line, sig = _macd(c, p, int(op.param))
            s = line if k == OperandKind.MACD_LINE else sig if k == OperandKind.MACD_SIGNAL else line - sig
        elif k == OperandKind.SUPERTREND:
            s = _supertrend(df, p, op.param)
        elif k == OperandKind.ADX:
            s = _adx(df, p)
        else:  # pragma: no cover - enum is closed
            raise ValueError(k)
        cache[key] = s
    s = cache[key]
    return s * op.multiplier if op.multiplier != 1.0 else s


def condition_mask(cond: Condition, df: pd.DataFrame, cache: dict | None = None) -> np.ndarray:
    left = operand_series(cond.left, df, cache)
    right = operand_series(cond.right, df, cache)
    if cond.op == Op.GT:
        m = left > right
    elif cond.op == Op.LT:
        m = left < right
    elif cond.op == Op.CROSSES_ABOVE:
        m = (left > right) & (left.shift(1) <= right.shift(1))
    else:
        m = (left < right) & (left.shift(1) >= right.shift(1))
    return m.fillna(False).to_numpy(dtype=bool)  # NaN comparisons are False


def signals(spec: ClaimSpec, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    cache: dict = {}
    entry = np.ones(len(df), dtype=bool)
    for c in spec.entry:
        entry &= condition_mask(c, df, cache)
    exit_ = np.zeros(len(df), dtype=bool)
    for c in spec.exit:
        exit_ |= condition_mask(c, df, cache)
    return entry, exit_


# ---------------------------------------------------------------- simulation

TRADE_COLUMNS = ["entry_time", "exit_time", "entry_px", "exit_px", "bars", "gross_pct", "net_pct", "reason"]


def run(spec: ClaimSpec, df: pd.DataFrame, costs: Costs = Costs()) -> pd.DataFrame:
    """Simulate the spec on one OHLCV frame. Returns one row per closed trade."""
    entry_sig, exit_sig = signals(spec, df)
    o, h, l, c = (df[x].to_numpy(dtype=float) for x in ("open", "high", "low", "close"))
    idx = df.index
    n = len(df)
    long = spec.direction == Direction.LONG
    sl = spec.stop_loss_pct / 100 if spec.stop_loss_pct else None
    tp = spec.take_profit_pct / 100 if spec.take_profit_pct else None
    max_hold = spec.max_hold_bars

    trades: list[tuple] = []
    in_pos = False
    entry_t = 0
    entry_px = stop_px = tp_px = 0.0

    def close_trade(t_exit: int, px: float, reason: str) -> None:
        gross = (px / entry_px - 1) if long else (entry_px / px - 1)
        net = (1 + gross) * (1 - costs.per_side) / (1 + costs.per_side) - 1
        trades.append((idx[entry_t], idx[t_exit], entry_px, px, t_exit - entry_t,
                       gross * 100, net * 100, reason))

    for t in range(n):
        exited_on_signal = False
        if in_pos:
            stop_hit = tp_hit = False
            if sl is not None:
                stop_hit = l[t] <= stop_px if long else h[t] >= stop_px
            if tp is not None:
                tp_hit = h[t] >= tp_px if long else l[t] <= tp_px
            if stop_hit:  # pessimistic: stop wins a same-bar tie
                gapped = (o[t] <= stop_px) if long else (o[t] >= stop_px)
                close_trade(t, o[t] if gapped else stop_px, "stop")
                in_pos = False
            elif tp_hit:
                gapped = (o[t] >= tp_px) if long else (o[t] <= tp_px)
                close_trade(t, o[t] if gapped else tp_px, "target")
                in_pos = False
            elif t + 1 < n:
                held = t - entry_t + 1
                if exit_sig[t]:
                    close_trade(t + 1, o[t + 1], "signal")
                    in_pos, exited_on_signal = False, True
                elif max_hold is not None and held >= max_hold:
                    close_trade(t + 1, o[t + 1], "time")
                    in_pos, exited_on_signal = False, True
        # Entry decided at the close of t, filled at the open of t+1. After a signal
        # exit the exit fill already sits on t+1, so entering there too would be a
        # free round trip on one open -- skip and re-evaluate from t+1.
        if not in_pos and not exited_on_signal and entry_sig[t] and t + 1 < n:
            entry_t = t + 1
            entry_px = o[t + 1]
            if sl is not None:
                stop_px = entry_px * (1 - sl) if long else entry_px * (1 + sl)
            if tp is not None:
                tp_px = entry_px * (1 + tp) if long else entry_px * (1 - tp)
            in_pos = True
    # A position still open at the end of the data is marked at the last close and kept,
    # flagged reason="open". Dropping it would hide exactly the losers a no-stop rule creates.
    if in_pos:
        close_trade(n - 1, c[n - 1], "open")
    return pd.DataFrame(trades, columns=TRADE_COLUMNS)


# ---------------------------------------------------------------- metrics

def summarize(trades: pd.DataFrame, df: pd.DataFrame | None = None) -> dict:
    n = len(trades)
    out = {"trades": n}
    if n == 0:
        out.update(win_rate_pct=None, avg_net_pct=None, total_net_pct=0.0, total_gross_pct=0.0,
                   max_drawdown_pct=0.0, avg_bars=None)
    else:
        net = trades["net_pct"].to_numpy() / 100
        gross = trades["gross_pct"].to_numpy() / 100
        equity = np.cumprod(1 + net)
        peak = np.maximum.accumulate(np.concatenate([[1.0], equity]))[1:]
        out.update(
            win_rate_pct=float((net > 0).mean() * 100),
            avg_net_pct=float(net.mean() * 100),
            total_net_pct=float((equity[-1] - 1) * 100),
            total_gross_pct=float((np.prod(1 + gross) - 1) * 100),
            max_drawdown_pct=float(((equity / peak) - 1).min() * 100),
            avg_bars=float(trades["bars"].mean()),
            open_at_end=int((trades["reason"] == "open").sum()),
        )
    if df is not None and len(df) > 1:
        out["buy_hold_pct"] = float((df["close"].iloc[-1] / df["open"].iloc[0] - 1) * 100)
        out["window"] = [df.index[0].date().isoformat(), df.index[-1].date().isoformat()]
        if n:
            out["exposure_pct"] = float(min(trades["bars"].sum() / len(df), 1.0) * 100)
    return out


# ---------------------------------------------------------------- overrides

_PATH = re.compile(r"(\w+)(?:\[(\d+)\])?")
_INT_FIELDS = {"period", "max_hold_bars"}


def with_override(spec: ClaimSpec, target: str, value: float) -> ClaimSpec:
    """Return a copy of `spec` with the field at `target` set to `value`."""
    d = spec.model_dump(mode="json")
    parts = target.split(".")
    node = d
    for i, part in enumerate(parts):
        m = _PATH.fullmatch(part)
        if not m:
            raise ValueError(f"bad target path {target!r}")
        key, index = m.group(1), m.group(2)
        last = i == len(parts) - 1
        if index is not None:
            node = node[key][int(index)]
            if last:
                raise ValueError(f"target {target!r} must end on a field")
        elif last:
            if key not in node:
                raise ValueError(f"target {target!r}: no field {key!r}")
            node[key] = int(round(value)) if key in _INT_FIELDS else float(value)
        else:
            node = node[key]
    return ClaimSpec.model_validate(d)
