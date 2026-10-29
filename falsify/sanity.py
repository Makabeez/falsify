"""Deterministic sanity check on a compiled spec, run inside the compile loop: does the entry ever fire?

A spec can be valid, quote-clean and still impossible: "RSI at its 50-bar low" AND "RSI rising for
3 bars" on the same bar never happen together, so the backtest finds zero trades and the verdict
says "too few trades" about a rule nobody wrote. This check runs before the expensive audit and
sends the spec back with per-condition counts, the same way a schema error is sent back.
"""

from __future__ import annotations

import numpy as np

from . import data as data_mod
from .engine import condition_mask
from .spec import ClaimSpec


def entry_problems(spec: ClaimSpec) -> list[str]:
    if not spec.testable or not spec.entry or not spec.assets:
        return []
    key = (spec.assets[0], spec.timeframe.value)
    if key not in data_mod.available():
        return []  # no data: the verdict will say so; nothing to check here
    df = data_mod.load(*key)
    cache: dict = {}
    masks = [np.nan_to_num(condition_mask(c, df, cache), nan=0).astype(bool) for c in spec.entry]
    if np.logical_and.reduce(masks).any():
        return []
    n = len(df)
    counts = ", ".join(f"entry[{i}] ({c.left.label()} {c.op.value} {c.right.label()}) holds on {int(m.sum())} bars"
                       for i, (c, m) in enumerate(zip(spec.entry, masks)))
    never = [i for i, m in enumerate(masks) if not m.any()]
    if never:
        why = (f"entry[{never[0]}] is never true on its own" + ("" if len(never) == 1 else f" (also {never[1:]})")
               + "; check the operand, threshold and units")
    else:
        why = ("each condition holds sometimes but never all on the same bar, so two of them contradict each "
               "other; if the claim describes a sequence, put the earlier event in the past with offset")
    return [f"the entry never fires on {n} bars of {key[0]} {key[1]} history: {why}. Counts: {counts}."]
