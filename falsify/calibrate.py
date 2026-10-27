"""Measure the battery instead of asserting it: false-positive rate and power on real data.

A test that kills every claim is as useless as one that passes every claim. This module
answers two questions on the bundled Binance history, with the battery's own thresholds:

1. False positives. Coin-flip strategies have no timing skill by construction. How often
   do they pass the fatal tests (profitable after costs AND beat random timing at p < 0.05)?
   It should be about 5% or less. For contrast we also report how often they pass the naive
   test most backtests stop at: "made money after costs".

2. Power. Add a true edge of d bps per trade on top of the coin-flip trades. How often does
   the battery let it through? The smallest d detected 80% of the time is the minimum
   detectable edge for that trade count and holding period.

Coin-flip strategies are simulated with the engine's own semantics: flat-only entries,
fixed holding period, fill at the next open, exit at an open, costs on both sides, no
overlapping trades. That differs from the null's sampler (which allows overlap), so the
false-positive rate is a real check, not true by construction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import data as data_mod
from .battery import ALPHA, NULL_SIMS, null_distribution, round_trip_log_cost
from .engine import Costs
from .spec import Direction

EDGES_BPS = [0, 25, 50, 100, 200, 400, 800]


@dataclass(frozen=True)
class Setup:
    asset: str
    timeframe: str
    hold: int
    trades: int
    direction: Direction = Direction.LONG

    @property
    def label(self) -> str:
        unit = {"1h": "h", "4h": "x4h", "1d": "d"}[self.timeframe]
        return f"{self.asset} {self.timeframe}, hold {self.hold}{unit}, ~{self.trades} trades, {self.direction.value}"


DEFAULT_SETUPS = [
    Setup("BTCUSDT", "1d", 7, 30),
    Setup("BTCUSDT", "1d", 7, 100),
    Setup("BTCUSDT", "4h", 6, 100),
    Setup("BTCUSDT", "4h", 6, 300),
    Setup("ETHUSDT", "1h", 12, 300),
    Setup("ETHUSDT", "1h", 12, 1000),
    Setup("SOLUSDT", "1d", 2, 300, Direction.SHORT),
]


def coin_flip_trades(log_open: np.ndarray, hold: int, n_target: int, sign: float,
                     rng: np.random.Generator) -> np.ndarray:
    """Gross log-returns of one non-overlapping random strategy with ~n_target trades."""
    n_bars = len(log_open)
    free = n_bars - 2 - n_target * hold
    if free <= n_target:
        raise ValueError("too many trades for this holding period and history")
    q = n_target / free  # per-bar entry probability while flat
    out = []
    t = 0
    while t + 1 + hold < n_bars:
        if rng.random() < q:
            e = t + 1                      # signal at close t, fill at open t+1
            out.append(sign * (log_open[e + hold] - log_open[e]))
            t = e + hold                   # exit fills at this open; re-evaluate from here
        else:
            t += 1
    return np.asarray(out)


def run_setup(setup: Setup, costs: Costs = Costs(), reps: int = 300, sims: int = 1000,
              edges_bps: list[int] = EDGES_BPS, seed: int = 11) -> dict:
    df = data_mod.load(setup.asset, setup.timeframe)
    log_open = np.log(df["open"].to_numpy(dtype=float))
    sign = 1.0 if setup.direction == Direction.LONG else -1.0
    cost = round_trip_log_cost(costs)
    rng = np.random.default_rng(seed)

    passed = {d: 0 for d in edges_bps}
    naive_pass = 0
    sizes = []
    for r in range(reps):
        gross = coin_flip_trades(log_open, setup.hold, setup.trades, sign, rng)
        n = len(gross)
        sizes.append(n)
        null = null_distribution(df, np.full(n, setup.hold), setup.direction, costs, sims, seed=1000 + r)
        net = gross - cost
        if net.mean() > 0:
            naive_pass += 1
        for d in edges_bps:
            m = net.mean() + math.log1p(d / 1e4)
            p = (np.sum(null >= m) + 1) / (sims + 1)
            if m > 0 and p < ALPHA:
                passed[d] += 1

    power = {d: passed[d] / reps for d in edges_bps}
    return {
        "setup": setup.label,
        "reps": reps,
        "null_sims": sims,
        "trades_median": int(np.median(sizes)),
        "false_positive_pct": power[0] * 100,
        "naive_profitable_pct": naive_pass / reps * 100,
        "power_pct": {str(d): power[d] * 100 for d in edges_bps},
        "min_detectable_edge_bps": _mde(power, edges_bps),
    }


def _mde(power: dict, edges: list[int], target: float = 0.8) -> float | None:
    """Smallest edge with >= target power, linearly interpolated between grid points."""
    for lo, hi in zip(edges, edges[1:]):
        if power[hi] >= target:
            if power[lo] >= target:
                return float(lo)
            frac = (target - power[lo]) / (power[hi] - power[lo])
            return float(lo + frac * (hi - lo))
    return None


def run_all(setups: list[Setup] = DEFAULT_SETUPS, reps: int = 300, sims: int = 1000,
            costs: Costs = Costs()) -> list[dict]:
    return [run_setup(s, costs, reps, sims) for s in setups]


def to_markdown(results: list[dict], costs: Costs = Costs()) -> str:
    head = ("| Setup | Trades | Coin flips that pass Falsify | Coin flips a naive backtest calls profitable | "
            + " | ".join(f"+{d} bps" for d in EDGES_BPS[1:]) + " | Min. detectable edge (80% power) |")
    sep = "|" + "---|" * (5 + len(EDGES_BPS) - 1)
    rows = []
    for r in results:
        mde = r["min_detectable_edge_bps"]
        rows.append(
            f"| {r['setup']} | {r['trades_median']} | {r['false_positive_pct']:.1f}% | {r['naive_profitable_pct']:.0f}% | "
            + " | ".join(f"{r['power_pct'][str(d)]:.0f}%" for d in EDGES_BPS[1:])
            + f" | {'> 800 bps' if mde is None else f'{mde:.0f} bps'} |")
    first = results[0]
    return "\n".join([
        f"Bundled Binance spot data, costs {costs.fee_bps:g}+{costs.slippage_bps:g} bps per side, "
        f"{first['reps']} coin-flip strategies per setup, {first['null_sims']} null simulations each "
        f"(the battery itself uses {NULL_SIMS}). Pass = profitable after costs AND p < {ALPHA} against random timing.",
        "", head, sep, *rows, ""])
