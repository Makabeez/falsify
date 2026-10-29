"""Quote-checked grounding: every rule element is either quoted from the source or declared assumed.

The model compiles the claim, but it does not get to assert what the claimant said. Each
entry/exit condition, stop, target, time exit, asset, timeframe and direction needs an
evidence row:

  {"field": "entry[0]", "quote": "rsi drops under 30", "assumed": false}
  {"field": "max_hold_bars", "quote": null, "assumed": true}

This module checks those rows deterministically, outside the model's control:
  * every rule element has a row,
  * every non-assumed row quotes words that are actually in the source (for text) or in
    the transcript of the media (for screenshots and clips),
  * field paths point at something that exists.
Problems go back to the model as a repair request, like a schema error. What survives is
reported as "k of n rule elements quoted from the source".
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from .spec import ClaimSpec

EXIT_FIELDS = ("stop_loss_pct", "stop_atr_mult", "take_profit_pct", "take_profit_atr_mult", "take_profit_r",
               "trailing_stop_pct", "trailing_stop_atr_mult", "max_hold_bars")
_FIELD = re.compile(r"^(entry|exit)\[(\d+)\]$|^(" + "|".join(EXIT_FIELDS) + r"|assets|timeframe|direction)$"
                    r"|^claimed\.(win_rate_pct|total_return_pct|avg_return_per_trade_pct|period_start|period_end|other)$")
_TRANSLATE = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-",
                            "−": "-", " ": " "})
MAX_QUOTE = 240


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).translate(_TRANSLATE).lower()
    t = re.sub(r"[^\w%$.\-+<>=/: ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def required_fields(spec: ClaimSpec) -> list[str]:
    req = [f"entry[{i}]" for i in range(len(spec.entry))] + [f"exit[{i}]" for i in range(len(spec.exit))]
    for name in EXIT_FIELDS:
        if getattr(spec, name) is not None:
            req.append(name)
    return req + ["assets", "timeframe", "direction"]


@dataclass
class Grounding:
    required: list[str]
    quoted: list[str] = field(default_factory=list)
    assumed: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def summary(self) -> dict:
        return {"rule_elements": len(self.required), "quoted": len(self.quoted), "assumed": len(self.assumed),
                "quoted_fields": self.quoted, "assumed_fields": self.assumed, "problems": self.problems}


def check(spec: ClaimSpec, source_text: str) -> Grounding:
    """Deterministic check of spec.evidence against the source text (or media transcript)."""
    g = Grounding(required=required_fields(spec) if spec.testable else [])
    if not spec.testable:
        return g
    haystack = normalize(source_text or "")
    seen: dict[str, bool] = {}
    for row in spec.evidence:
        m = _FIELD.match(row.field)
        if not m:
            g.problems.append(f"evidence field {row.field!r} is not a valid path")
            continue
        if m.group(1):
            n = len(spec.entry) if m.group(1) == "entry" else len(spec.exit)
            if int(m.group(2)) >= n:
                g.problems.append(f"evidence field {row.field!r} points past the end of {m.group(1)}")
                continue
        if row.assumed:
            seen.setdefault(row.field, False)
            continue
        q = normalize(row.quote or "")
        if len(q) < 2:
            g.problems.append(f"{row.field}: quote is empty; quote the claimant's exact words or mark assumed=true")
        elif len(row.quote or "") > MAX_QUOTE:
            g.problems.append(f"{row.field}: quote is longer than {MAX_QUOTE} characters; quote only the words that state it")
        elif q not in haystack:
            g.problems.append(f"{row.field}: quote {row.quote!r} is not in the source; copy the exact words, or mark assumed=true")
        else:
            seen[row.field] = True
    for f in g.required:
        if f not in seen:
            g.problems.append(f"{f}: no evidence row; quote where the claim states it, or add it with assumed=true")
        elif seen[f]:
            g.quoted.append(f)
        else:
            g.assumed.append(f)
    return g
