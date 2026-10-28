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

ALPHA = 0.05            # SURVIVES needs p below this against random timing
WORSE_THAN_RANDOM = 0.5 # DIES when random timing does at least as well this often (worse than the median coin flip)
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
    grounding: dict = field(default_factory=dict)
    fidelity: dict = field(default_factory=dict)
    series: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        return d


# ---------------------------------------------------------------- tests

def round_trip_log_cost(costs: Costs) -> float:
    """Exact log cost of one round trip under the engine's cost model."""
    return math.log1p(costs.per_side) - math.log1p(-costs.per_side)


def null_distribution(df: pd.DataFrame, hold: np.ndarray, direction: Direction, costs: Costs,
                      sims: int = NULL_SIMS, seed: int = 7) -> np.ndarray | None:
    """Mean net log-return per trade for `sims` random-timing strategies.

    Each simulated strategy has len(hold) trades with exactly these holding times and side:
    entry at a uniformly random open, exit at the open `hold` bars later, same costs.
    """
    log_open = np.log(df["open"].to_numpy(dtype=float))
    hold = np.maximum(np.asarray(hold, dtype=int), 1)
    max_entry = len(log_open) - 1 - hold  # per-trade upper bound so exit stays in range
    if (max_entry < 1).any():
        return None
    sign = 1.0 if direction == Direction.LONG else -1.0
    rng = np.random.default_rng(seed)
    entries = (rng.random((sims, len(hold))) * max_entry).astype(int)
    return (sign * (log_open[entries + hold] - log_open[entries]) - round_trip_log_cost(costs)).mean(axis=1)


def random_entry_null(trades: pd.DataFrame, df: pd.DataFrame, direction: Direction,
                      costs: Costs, sims: int = NULL_SIMS, seed: int = 7) -> dict:
    """Could random timing with the same trade count, holding times and side do as well?

    Statistic: mean net log-return per trade. This strips out drift: a long-only rule in
    a bull market beats zero but not random longs. Calibration and power of this test on
    the bundled data are measured by `falsify calibrate` (docs/calibration.md).
    """
    null_means = null_distribution(df, trades["bars"].to_numpy(dtype=int), direction, costs, sims, seed)
    if null_means is None:
        return {"p_value": None, "note": "holding periods longer than the data"}
    strat = np.log1p(trades["net_pct"].to_numpy() / 100).mean()
    p = (np.sum(null_means >= strat) + 1) / (sims + 1)
    return {
        "p_value": float(p),
        "strategy_mean_log_pct": float(strat * 100),
        "null_median_log_pct": float(np.median(null_means) * 100),
        "null_p95_log_pct": float(np.quantile(null_means, 0.95) * 100),
        "beat_by_random_pct": float(np.mean(null_means >= strat) * 100),
        "sims": sims,
        "hist": _histogram(null_means * 100, strat * 100),
    }


def _histogram(values: np.ndarray, marker: float, bins: int = 40) -> dict:
    """Null distribution for the UI: counts per bin plus where the claim landed (% per trade, log)."""
    lo, hi = float(min(values.min(), marker)), float(max(values.max(), marker))
    pad = (hi - lo) * 0.05 or 1e-6
    counts, edges = np.histogram(values, bins=bins, range=(lo - pad, hi + pad))
    return {"edges": [round(float(e), 4) for e in edges], "counts": counts.astype(int).tolist(),
            "strategy": round(float(marker), 4)}


def _mean_net(trades: pd.DataFrame) -> float | None:
    return float(trades["net_pct"].mean()) if len(trades) else None


def _parse_date(s: str | None) -> pd.Timestamp | None:
    if not s:
        return None
    try:
        return pd.Timestamp(s, tz="UTC")
    except (ValueError, TypeError):
        return None


def out_of_sample(spec: ClaimSpec, trades: pd.DataFrame, posted: str | None = None) -> dict:
    """Trades the claimant could not have seen are the only true out-of-sample.

    The boundary is the later of the claim's stated period end and the post date: the
    author can have fitted the rule to anything up to the day they posted it.
    """
    candidates = [d for d in (_parse_date(spec.claimed.period_end), _parse_date(posted)) if d is not None]
    if not candidates:
        return {"available": False, "note": "claim names no period and has no post date; nothing is strictly out-of-sample"}
    end = max(candidates)
    start = _parse_date(spec.claimed.period_start)
    after = trades[trades["entry_time"] > end]
    during = trades[(trades["entry_time"] <= end) & ((trades["entry_time"] >= start) if start is not None else True)]
    return {
        "available": True,
        "boundary": end.date().isoformat(),
        "boundary_source": "post date" if posted and _parse_date(posted) == end else "claimed period end",
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


def equity_series(trades: pd.DataFrame, df: pd.DataFrame, points: int = 260) -> dict:
    """Growth of 1 for the rule (stepping at each trade exit) and for buy-and-hold, on a common date grid."""
    idx = df.index
    step = max(1, len(idx) // points)
    grid = idx[::step]
    if grid[-1] != idx[-1]:
        grid = grid.append(idx[-1:])
    bh = (df["close"] / df["open"].iloc[0]).reindex(grid)
    eq = pd.Series(1.0, index=grid)
    if len(trades):
        growth = (1 + trades["net_pct"].to_numpy() / 100).cumprod()
        exits = pd.DatetimeIndex(trades["exit_time"])
        pos = np.searchsorted(exits.values, grid.values, side="right")
        eq = pd.Series(np.where(pos > 0, growth[np.maximum(pos - 1, 0)], 1.0), index=grid)
    return {"t": [d.date().isoformat() for d in grid],
            "rule": [round(float(v), 5) for v in eq.to_numpy()],
            "buy_hold": [round(float(v), 5) for v in bh.to_numpy()]}


def first_signal_possible(spec: ClaimSpec, df: pd.DataFrame):
    """First bar on which every entry operand has a value (after indicator warm-up)."""
    from .engine import operand_series
    ok = pd.Series(True, index=df.index)
    for c in spec.entry:
        for op in (c.left, c.right):
            ok &= operand_series(op, df).notna()
    valid = ok[ok]
    return valid.index[0] if len(valid) else None


def _buy_hold_warning(spec: ClaimSpec, rep: Report) -> None:
    p = rep.primary
    bh = p.get("buy_hold_pct")
    if spec.direction == Direction.LONG and bh is not None and p.get("total_net_pct", 0) < bh:
        rep.warnings.append(f"just holding did better: {p['total_net_pct']:+.0f}% vs buy-and-hold {bh:+.0f}%")


# ---------------------------------------------------------------- verdict

def falsify(spec: ClaimSpec, costs: Costs = Costs(), with_cross_asset: bool = True,
            posted: str | None = None) -> Report:
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
    claim_start = _parse_date(spec.claimed.period_start)
    rep = Report(Verdict.SURVIVES, "", asset=primary_asset, timeframe=tf, primary=primary,
                 costs={"fee_bps_per_side": costs.fee_bps, "slippage_bps_per_side": costs.slippage_bps})
    rep.claimed_vs_measured = compare_claim(spec, primary)
    rep.series = equity_series(trades, df)
    if claim_start is not None and claim_start < df.index[0]:
        rep.warnings.append(f"the claim's results start {claim_start.date()}; Binance history for {primary_asset} "
                            f"starts {df.index[0].date()}, so the test covers less history than the claim")
    first = first_signal_possible(spec, df)
    if first is not None and (first - df.index[0]) > (df.index[-1] - df.index[0]) * 0.25:
        rep.warnings.append(f"indicator warm-up: the longest lookback needs data until {first.date()} before the first "
                            f"signal is possible, so only part of the history is really tested")
    if primary.get("open_at_end"):
        rep.warnings.append("the last trade is still open at the end of the data; it is marked at the last close")

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
    rep.out_of_sample = out_of_sample(spec, trades, posted)
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
    if p is not None and p >= WORSE_THAN_RANDOM:
        rep.reasons.append(f"no better than random timing: random entries with the same holding times did as well "
                           f"{rep.null_test['beat_by_random_pct']:.0f}% of the time (p = {p:.3f})")
    oos = rep.out_of_sample
    if oos.get("available") and oos.get("counts"):
        after = oos["after_window"]
        if after["avg_net_pct"] is not None and after["avg_net_pct"] <= 0:
            rep.reasons.append(f"after {oos['boundary']} ({oos['boundary_source']}), it lost "
                               f"{after['avg_net_pct']:+.2f}% per trade over {after['trades']} trades")

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
    elif p is None or p >= ALPHA:
        # Makes money after costs, but the data cannot tell it from luck. Not refuted, not supported.
        rep.verdict = Verdict.INCONCLUSIVE
        beat = rep.null_test.get("beat_by_random_pct")
        rep.reasons.append(f"makes money after costs ({primary['avg_net_pct']:+.2f}% per trade), but random timing did as "
                           f"well {beat:.0f}% of the time (p = {p:.3f}); needs p < {ALPHA} to count as an edge"
                           if p is not None else "the random-timing test could not be run")
        rep.headline = (f"Makes money ({primary['avg_net_pct']:+.2f}% per trade after costs, {primary['trades']} trades) "
                        f"but can't be told from luck: random timing did as well {beat:.0f}% of the time."
                        if p is not None else "Could not run the random-timing test.")
    else:
        rep.headline = (f"Holds up: {primary['avg_net_pct']:+.2f}% per trade after costs over {primary['trades']} "
                        f"trades, beats random timing (p = {p:.3f}).")
    return rep


def attach_grounding(rep: Report, summary: dict | None) -> Report:
    """Record how much of the tested rule is the claimant's words vs. our reading."""
    if not summary or not summary.get("rule_elements"):
        return rep
    rep.grounding = summary
    n, q = summary["rule_elements"], summary["quoted"]
    if summary.get("problems"):
        rep.warnings.append(f"quote check failed on {len(summary['problems'])} item(s) after repairs -- "
                            f"verdict tests our reading, review the evidence")
    if q * 2 < n:
        rep.warnings.append(f"mostly our reading: only {q}/{n} rule elements are quoted from the claim")
    return rep


def attach_fidelity(rep: Report, fid: dict | None) -> Report:
    """If what we tested is not exactly what was claimed, say so in the headline, not in a footnote."""
    if not fid:
        return rep
    rep.fidelity = fid
    if fid.get("level") == "approximation" and rep.verdict != Verdict.UNTESTABLE:
        rep.headline = "Tested an approximation of the claim. " + rep.headline
        for g in fid.get("gaps", []):
            rep.warnings.append(f"left out: {g.split(':', 1)[-1].strip()}")
        for m in fid.get("audit_majors", []):
            rep.warnings.append(f"audit: {m}")
    return rep
