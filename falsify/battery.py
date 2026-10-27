"""The falsification battery and the verdict.

Every test is something a claimant could have run before posting and usually
didn't. Thresholds are statistical conventions stated up front (p < 0.05,
>= 30 trades), never tuned per claim; `falsify corpus` measures how they
behave across a whole set of real claims.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from . import data as data_mod
from .engine import Costs, run, summarize, with_override
from .spec import ClaimSpec, Direction

ALPHA = 0.05            # significance level for the random-entry null
MIN_TRADES = 30         # below this, a win rate is noise
MIN_OOS_TRADES = 10     # post-claim window needs this many trades to count
NULL_SIMS = 2000
MAX_SWEEP = 64          # cap on parameter combinations


class Verdict(str, Enum):
    SURVIVES = "SURVIVES"
    DIES = "DIES"
    INCONCLUSIVE = "INCONCLUSIVE"
    UNTESTABLE = "UNTESTABLE"


@dataclass
class Report:
    verdict: Verdict
    headline: str
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    asset: str | None = None
    timeframe: str | None = None
    primary: dict = field(default_factory=dict)
    null_test: dict = field(default_factory=dict)
    out_of_sample: dict = field(default_factory=dict)
    halves: dict = field(default_factory=dict)
    cross_asset: dict = field(default_factory=dict)
    sweep: dict = field(default_factory=dict)
    claimed_vs_measured: list[dict] = field(default_factory=list)
    costs: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        return d


# ---------------------------------------------------------------- tests

def random_entry_null(trades: pd.DataFrame, df: pd.DataFrame, direction: Direction,
                      costs: Costs, sims: int = NULL_SIMS, seed: int = 7) -> dict:
    """Could random timing with the same trade count, holding times and side do as well?

    Statistic: mean net log-return per trade. Null trades enter at a uniformly
    random open and exit at the open `bars` later, paying the same costs. This
    strips out drift: a long-only rule in a bull market beats zero but not
    random longs.
    """
    n = len(trades)
    opens = df["open"].to_numpy(dtype=float)
    log_open = np.log(opens)
    hold = np.maximum(trades["bars"].to_numpy(dtype=int), 1)
    sign = 1.0 if direction == Direction.LONG else -1.0
    cost = 2 * math.log1p(costs.per_side)

    strat = np.log1p(trades["net_pct"].to_numpy() / 100).mean()

    rng = np.random.default_rng(seed)
    max_entry = len(opens) - 1 - hold  # per-trade upper bound so exit stays in range
    if (max_entry < 1).any():
        return {"p_value": None, "note": "holding periods longer than the data"}
    entries = (rng.random((sims, n)) * max_entry).astype(int)
    exits = entries + hold
    null_means = (sign * (log_open[exits] - log_open[entries]) - cost).mean(axis=1)
    p = (np.sum(null_means >= strat) + 1) / (sims + 1)
    return {
        "p_value": float(p),
        "strategy_mean_log_pct": float(strat * 100),
        "null_median_log_pct": float(np.median(null_means) * 100),
        "null_p95_log_pct": float(np.quantile(null_means, 0.95) * 100),
        "beat_by_random_pct": float(np.mean(null_means >= strat) * 100),
        "sims": sims,
    }


def _mean_net(trades: pd.DataFrame) -> float | None:
    return float(trades["net_pct"].mean()) if len(trades) else None


def _parse_date(s: str | None) -> pd.Timestamp | None:
    if not s:
        return None
    try:
        return pd.Timestamp(s, tz="UTC")
    except (ValueError, TypeError):
        return None


def out_of_sample(spec: ClaimSpec, trades: pd.DataFrame) -> dict:
    """Trades after the period the claimant showed are the only true out-of-sample."""
    end = _parse_date(spec.claimed.period_end)
    if end is None:
        return {"available": False, "note": "claim names no period; nothing is strictly out-of-sample"}
    start = _parse_date(spec.claimed.period_start)
    after = trades[trades["entry_time"] > end]
    during = trades[(trades["entry_time"] <= end) & ((trades["entry_time"] >= start) if start is not None else True)]
    return {
        "available": True,
        "claim_window": [spec.claimed.period_start, spec.claimed.period_end],
        "in_window": summarize(during),
        "after_window": summarize(after),
        "counts": len(after) >= MIN_OOS_TRADES,
    }


def halves(trades: pd.DataFrame, df: pd.DataFrame) -> dict:
    mid = df.index[len(df) // 2]
    first, second = trades[trades["entry_time"] < mid], trades[trades["entry_time"] >= mid]
    m1, m2 = _mean_net(first), _mean_net(second)
    return {"split": mid.date().isoformat(), "first": summarize(first), "second": summarize(second),
            "consistent": bool(m1 is not None and m2 is not None and m1 > 0 and m2 > 0)}


def cross_asset(spec: ClaimSpec, primary: str, costs: Costs) -> dict:
    """Same rules on the rest of the bundled universe. A real effect rarely lives on one ticker."""
    tf = spec.timeframe.value
    have = data_mod.available()
    rows = []
    for sym in data_mod.UNIVERSE:
        if sym == primary or (sym, tf) not in have:
            continue
        t = run(spec, data_mod.load(sym, tf), costs)
        s = summarize(t)
        rows.append({"asset": sym, "trades": s["trades"], "total_net_pct": s["total_net_pct"],
                     "avg_net_pct": s["avg_net_pct"]})
    tested = [r for r in rows if r["trades"] >= MIN_TRADES]
    pos = [r for r in tested if r["avg_net_pct"] > 0]
    return {"assets": rows, "tested": len(tested), "positive": len(pos),
            "positive_share_pct": (len(pos) / len(tested) * 100) if tested else None}


def sweep(spec: ClaimSpec, df: pd.DataFrame, costs: Costs) -> dict:
    """Run every reading of the vague parameters. Picking the best one is a search; it gets charged."""
    fps = spec.free_parameters
    if not fps:
        return {"readings": 1, "note": "claim fixes every parameter"}
    grids = [sorted(set(fp.candidates)) for fp in fps]
    combos = list(itertools.product(*grids))
    truncated = len(combos) > MAX_SWEEP
    if truncated:
        step = len(combos) / MAX_SWEEP
        combos = [combos[int(i * step)] for i in range(MAX_SWEEP)]
    results = []
    for combo in combos:
        s = spec
        try:
            for fp, v in zip(fps, combo):
                s = with_override(s, fp.target, v)
        except (ValueError, KeyError, IndexError) as e:
            results.append({"params": dict(zip([f.name for f in fps], combo)), "error": str(e)})
            continue
        t = run(s, df, costs)
        summ = summarize(t)
        nt = random_entry_null(t, df, s.direction, costs, sims=500) if len(t) >= MIN_TRADES else {"p_value": None}
        results.append({"params": dict(zip([f.name for f in fps], combo)), "trades": summ["trades"],
                        "avg_net_pct": summ["avg_net_pct"], "total_net_pct": summ["total_net_pct"],
                        "p_value": nt["p_value"]})
    ok = [r for r in results if "error" not in r and r["trades"] >= MIN_TRADES]
    positive = [r for r in ok if r["avg_net_pct"] > 0]
    best = min((r for r in ok if r["p_value"] is not None), key=lambda r: r["p_value"], default=None)
    k = len(results)
    return {
        "readings": k,
        "truncated": truncated,
        "positive_share_pct": (len(positive) / len(ok) * 100) if ok else None,
        "best": best,
        "best_p_bonferroni": min(1.0, best["p_value"] * k) if best else None,
        "results": results,
    }


def compare_claim(spec: ClaimSpec, primary: dict) -> list[dict]:
    rows = []
    c = spec.claimed
    pairs = [("win rate %", c.win_rate_pct, primary.get("win_rate_pct")),
             ("avg return / trade %", c.avg_return_per_trade_pct, primary.get("avg_net_pct"))]
    for label, claimed, measured in pairs:
        if claimed is not None and measured is not None:
            rows.append({"metric": label, "claimed": claimed, "measured": round(measured, 2),
                         "gap": round(measured - claimed, 2)})
    return rows


def _buy_hold_warning(spec: ClaimSpec, rep: Report) -> None:
    p = rep.primary
    bh = p.get("buy_hold_pct")
    if spec.direction == Direction.LONG and bh is not None and p.get("total_net_pct", 0) < bh:
        rep.warnings.append(f"just holding did better: {p['total_net_pct']:+.0f}% vs buy-and-hold {bh:+.0f}%")


# ---------------------------------------------------------------- verdict

def falsify(spec: ClaimSpec, costs: Costs = Costs(), with_cross_asset: bool = True) -> Report:
    if not spec.testable:
        return Report(Verdict.UNTESTABLE, "This claim can't be turned into rules anyone could check.",
                      reasons=[spec.untestable_reason or "no mechanical rules"])

    tf = spec.timeframe.value
    have = data_mod.available()
    primary_asset = next((a for a in spec.assets if (a, tf) in have), None)
    if primary_asset is None:
        return Report(Verdict.UNTESTABLE, "No bundled history for the asset/timeframe this claim needs.",
                      reasons=[f"wanted {spec.assets} on {tf}; bundled universe is {data_mod.UNIVERSE}"],
                      timeframe=tf)

    df = data_mod.load(primary_asset, tf)
    trades = run(spec, df, costs)
    primary = summarize(trades, df)
    rep = Report(Verdict.SURVIVES, "", asset=primary_asset, timeframe=tf, primary=primary,
                 costs={"fee_bps_per_side": costs.fee_bps, "slippage_bps_per_side": costs.slippage_bps})
    rep.claimed_vs_measured = compare_claim(spec, primary)

    if primary["trades"] < MIN_TRADES:
        w = primary.get("window", ["?", "?"])
        rep.verdict = Verdict.INCONCLUSIVE
        rep.headline = (f"Only {primary['trades']} trades from {w[0]} to {w[1]} -- "
                        f"too few to tell skill from luck.")
        rep.reasons.append(f"{primary['trades']} trades < {MIN_TRADES} minimum")
        rep.sweep = sweep(spec, df, costs)
        if with_cross_asset:
            rep.cross_asset = cross_asset(spec, primary_asset, costs)
        _buy_hold_warning(spec, rep)
        return rep

    rep.null_test = random_entry_null(trades, df, spec.direction, costs)
    rep.out_of_sample = out_of_sample(spec, trades)
    rep.halves = halves(trades, df)
    rep.sweep = sweep(spec, df, costs)
    if with_cross_asset:
        rep.cross_asset = cross_asset(spec, primary_asset, costs)

    # --- hard failures: any one kills the claim
    if primary["avg_net_pct"] <= 0:
        rep.reasons.append(f"loses money after costs: {primary['avg_net_pct']:+.2f}% per trade, "
                           f"{primary['total_net_pct']:+.1f}% total "
                           f"(before costs {primary['total_gross_pct']:+.1f}%)")
    p = rep.null_test.get("p_value")
    if p is not None and p >= ALPHA:
        rep.reasons.append(f"random entries with the same holding times did as well {rep.null_test['beat_by_random_pct']:.0f}% "
                           f"of the time (p = {p:.3f})")
    oos = rep.out_of_sample
    if oos.get("available") and oos.get("counts"):
        after = oos["after_window"]
        if after["avg_net_pct"] is not None and after["avg_net_pct"] <= 0:
            rep.reasons.append(f"after the period the claim showed, it lost {after['avg_net_pct']:+.2f}% per trade "
                               f"over {after['trades']} trades")

    # --- warnings: reported, not fatal
    if not rep.halves["consistent"]:
        rep.warnings.append(f"not consistent over time: edge is not positive in both halves (split {rep.halves['split']})")
    ca = rep.cross_asset
    if ca and ca.get("positive_share_pct") is not None and ca["positive_share_pct"] < 50:
        rep.warnings.append(f"only works on {ca['positive']}/{ca['tested']} other assets")
    sw = rep.sweep
    if sw.get("readings", 1) > 1 and sw.get("positive_share_pct") is not None and sw["positive_share_pct"] < 50:
        rep.warnings.append(f"fragile: only {sw['positive_share_pct']:.0f}% of reasonable readings of the vague "
                            f"parameters make money")
    if oos.get("available") and not oos.get("counts"):
        rep.warnings.append("too few trades after the claim's period to judge it out-of-sample")
    for row in rep.claimed_vs_measured:
        if row["gap"] < -10 and row["metric"].startswith("win rate"):
            rep.warnings.append(f"claimed {row['claimed']:.0f}% win rate, measured {row['measured']:.0f}%")

    _buy_hold_warning(spec, rep)

    if rep.reasons:
        rep.verdict = Verdict.DIES
        rep.headline = rep.reasons[0][0].upper() + rep.reasons[0][1:] + "."
    else:
        rep.headline = (f"Holds up: {primary['avg_net_pct']:+.2f}% per trade after costs over {primary['trades']} "
                        f"trades, beats random timing (p = {p:.3f}).")
    return rep
