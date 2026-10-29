"""Compare compiled ClaimSpecs with a trader-reviewed ground truth (corpus/ground_truth.yaml).

Meta's verifiable-agents pattern: check the agent's output against an expert reference and
report coverage. The reference was drafted from the claim text by a separate agent that never
saw the compiler's output, then reviewed and signed off by a trader (`reviewed: true`).

Scoring, per claim:
  routing    testable (full/partial) vs not, as the compiler decided
  asset / timeframe / direction
  entry / exit  every reference condition matched by a compiled one and nothing extra.
             `exact` needs the same comparison; `lenient` also accepts a state reading for
             an event reading (close > sma vs close crosses above sma) and mirrored sides.
  stop / target / time_exit
Only fields where the reference states something are scored.

    falsify groundtruth                 # all claims, reviewed + drafts, shown separately
    falsify groundtruth --reviewed-only
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GT_PATH = ROOT / "corpus" / "ground_truth.yaml"
RESULTS = ROOT / "corpus" / "results"

_NUM = r"-?\d+(?:\.\d+)?"
_OPS = {">": "gt", "<": "lt", ">=": "ge", "<=": "le", "crosses_above": "crosses_above", "crosses_below": "crosses_below"}
_MIRROR = {"gt": "lt", "lt": "gt", "ge": "le", "le": "ge", "crosses_above": "crosses_below", "crosses_below": "crosses_above"}
_LENIENT = {"crosses_above": "gt", "crosses_below": "lt", "ge": "gt", "le": "lt"}
_DEFAULTS = {"rsi": (14, None), "atr": (14, None), "adx": (14, None), "macd_line": (12, 26.0), "macd_signal": (12, 26.0),
             "macd_hist": (12, 26.0), "bb_upper": (20, 2.0), "bb_lower": (20, 2.0), "supertrend": (10, 3.0)}


@dataclass(frozen=True)
class Opnd:
    kind: str
    period: int | None = None
    param: float | None = None
    value: float | None = None
    mult: float = 1.0
    window_fn: str | None = None
    window: int | None = None
    offset: int = 0


def _round(x):
    return None if x is None else round(float(x), 6)


def parse_operand(text: str) -> Opnd:
    t = text.strip().replace(" ", "")
    mult = 1.0
    m = re.fullmatch(rf"({_NUM})\*(.+)", t)
    if m and not re.fullmatch(_NUM, t):
        mult, t = float(m.group(1)), m.group(2)
    offset = 0
    m = re.fullmatch(r"(.+)\[(\d+)\]", t)
    if m:
        t, offset = m.group(1), int(m.group(2))
    window_fn = window = None
    m = re.fullmatch(r"(min|max)\((.+),(\d+)\)", t)
    if m:
        window_fn, t, window = m.group(1), m.group(2), int(m.group(3))
    if re.fullmatch(_NUM, t):
        return Opnd("const", value=_round(t))
    m = re.fullmatch(r"([a-z_]+)(?:\(([^)]*)\))?", t)
    if not m:
        raise ValueError(f"can't parse operand {text!r}")
    kind, args = m.group(1), [a for a in (m.group(2) or "").split(",") if a]
    period = int(float(args[0])) if args else None
    param = _round(args[1]) if len(args) > 1 else None
    if kind in _DEFAULTS:
        dp, dq = _DEFAULTS[kind]
        period = period or dp
        param = param if param is not None else _round(dq)
    return Opnd(kind, period, param, None, _round(mult), window_fn, window, offset)


def parse_condition(text: str) -> tuple[Opnd, str, Opnd]:
    for sym in ("crosses_above", "crosses_below", ">=", "<=", ">", "<"):
        parts = re.split(rf"\s+{re.escape(sym)}\s+", text.strip(), maxsplit=1)
        if len(parts) == 2:
            return parse_operand(parts[0]), _OPS[sym], parse_operand(parts[1])
    raise ValueError(f"can't parse condition {text!r}")


def from_spec_operand(o: dict) -> Opnd:
    kind = o["kind"]
    if kind == "const":
        return Opnd("const", value=_round(o["value"]))
    return Opnd(kind, o.get("period"), _round(o.get("param")), None, _round(o.get("multiplier", 1.0)),
                o.get("window_fn"), o.get("window"), o.get("offset") or 0)


def from_spec_condition(c: dict) -> tuple[Opnd, str, Opnd]:
    return from_spec_operand(c["left"]), c["op"], from_spec_operand(c["right"])


def _canon(cond, lenient: bool):
    a, op, b = cond
    if lenient:
        op = _LENIENT.get(op, op)
    # mirror so the left side sorts first: a > b == b < a
    if (b.kind, str(b)) < (a.kind, str(a)):
        a, b, op = b, a, _MIRROR.get(op, op)
        if lenient:
            op = _LENIENT.get(op, op)
    return (a, op, b)


def conditions_match(ref: list, got: list, lenient: bool) -> bool:
    r = sorted(map(str, (_canon(c, lenient) for c in ref)))
    g = sorted(map(str, (_canon(c, lenient) for c in got)))
    return r == g


def parse_exit(text: str | None, kind: str) -> dict:
    """'3%' | '2*atr(14)' | 'trailing 5%' | 'trailing 3*atr(14)' | '1.8R' -> spec fields."""
    if text is None:
        return {}
    t = str(text).strip().lower().replace(" ", "")
    trailing = t.startswith("trailing")
    t = t.removeprefix("trailing")
    if m := re.fullmatch(rf"({_NUM})%", t):
        name = "trailing_stop_pct" if trailing else ("stop_loss_pct" if kind == "stop" else "take_profit_pct")
        return {name: _round(m.group(1))}
    if m := re.fullmatch(rf"({_NUM})\*?atr(?:\((\d+)\))?", t):
        name = "trailing_stop_atr_mult" if trailing else ("stop_atr_mult" if kind == "stop" else "take_profit_atr_mult")
        return {name: _round(m.group(1)), "stop_atr_period": int(m.group(2) or 14)}
    if kind == "target" and (m := re.fullmatch(rf"({_NUM})r", t)):
        return {"take_profit_r": _round(m.group(1))}
    raise ValueError(f"can't parse {kind} {text!r}")


STOP_FIELDS = ("stop_loss_pct", "stop_atr_mult", "trailing_stop_pct", "trailing_stop_atr_mult")
TARGET_FIELDS = ("take_profit_pct", "take_profit_atr_mult", "take_profit_r")


def _exit_match(ref: dict, spec: dict, names: tuple) -> bool:
    got = {k: _round(spec.get(k)) for k in names if spec.get(k) is not None}
    want = {k: v for k, v in ref.items() if k in names}
    if got != want:
        return False
    if any(k.endswith("atr_mult") for k in want):
        return spec.get("stop_atr_period", 14) == ref.get("stop_atr_period", 14)
    return True


@dataclass
class ClaimScore:
    id: str
    reviewed: bool
    fields: dict = field(default_factory=dict)   # name -> True / False
    lenient: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def score_claim(cid: str, ref: dict, spec: dict, verdict: str | None = None) -> ClaimScore:
    """`verdict` is the pipeline's final verdict: a rule on an asset we have no data for compiles fine
    but ends UNTESTABLE, which is what the reference's testable: no means."""
    s = ClaimScore(cid, bool(ref.get("reviewed")))
    ref_testable = str(ref.get("testable")) in ("full", "partial")
    got_testable = bool(spec.get("testable")) and verdict != "UNTESTABLE"
    s.fields["routing"] = ref_testable == got_testable
    if not (ref_testable and got_testable):
        if not s.fields["routing"]:
            s.notes.append(f"reference says testable: {ref.get('testable')}, pipeline said "
                           f"{'testable' if got_testable else 'UNTESTABLE'}")
        return s
    if ref.get("asset"):
        want = str(ref["asset"]).upper().replace("USDT", "")
        s.fields["asset"] = any(a.upper().replace("USDT", "") == want for a in spec.get("assets", []))
    if ref.get("timeframe") in ("1h", "4h", "1d"):
        s.fields["timeframe"] = spec.get("timeframe") == ref["timeframe"]
    if ref.get("direction") in ("long", "short"):
        s.fields["direction"] = spec.get("direction") == ref["direction"]
    for part in ("entry", "exit"):
        ref_conds = ref.get(part) or []
        if not ref_conds:
            continue  # the reference states no rule here; nothing to score
        try:
            rc = [parse_condition(x) for x in ref_conds]
        except ValueError as e:
            s.notes.append(f"reference {part} unparsable: {e}")
            continue
        gc = [from_spec_condition(c) for c in spec.get(part, [])]
        s.fields[part] = conditions_match(rc, gc, lenient=False)
        s.lenient[part] = conditions_match(rc, gc, lenient=True)
    try:
        ref_stop, ref_tgt = parse_exit(ref.get("stop"), "stop"), parse_exit(ref.get("target"), "target")
    except ValueError as e:
        s.notes.append(str(e))
        ref_stop = ref_tgt = None
    if ref_stop:
        s.fields["stop"] = _exit_match(ref_stop, spec, STOP_FIELDS)
    if ref_tgt:
        s.fields["target"] = _exit_match(ref_tgt, spec, TARGET_FIELDS)
    if ref.get("time_exit"):
        s.fields["time_exit"] = spec.get("max_hold_bars") == int(ref["time_exit"])
    return s


def load_reference(path: Path = GT_PATH) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def evaluate(results_dir: Path = RESULTS, gt_path: Path = GT_PATH, reviewed_only: bool = False) -> dict:
    ref = load_reference(gt_path)
    scores = []
    for cid, r in sorted(ref.items()):
        f = results_dir / f"{cid}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        if "error" in d or (reviewed_only and not r.get("reviewed")):
            continue
        scores.append(score_claim(cid, r, d["spec"], (d.get("report") or {}).get("verdict")))

    def agg(group):
        out = {}
        names = ["routing", "asset", "timeframe", "direction", "entry", "exit", "stop", "target", "time_exit"]
        for name in names:
            vals = [s.fields[name] for s in group if name in s.fields]
            if vals:
                out[name] = (sum(vals), len(vals))
        for name in ("entry", "exit"):
            vals = [s.lenient[name] for s in group if name in s.lenient]
            if vals:
                out[name + "_lenient"] = (sum(vals), len(vals))
        rule_vals = [v for s in group for k, v in s.fields.items() if k != "routing"]
        out["rule_elements"] = (sum(rule_vals), len(rule_vals))
        both = [s for s in group if len(s.fields) > 1]
        out["claims_fully_right"] = (sum(all(s.fields.values()) for s in both), len(both))
        return out

    reviewed = [s for s in scores if s.reviewed]
    return {"all": agg(scores), "reviewed": agg(reviewed) if reviewed else None,
            "n_reviewed": len(reviewed), "n": len(scores), "claims": scores}


def to_markdown(ev: dict) -> str:
    def pct(t):
        return f"{t[0]}/{t[1]} ({t[0] / t[1] * 100:.0f}%)" if t and t[1] else "n/a"

    rows = [("Testable or not (routing)", "routing"), ("Asset", "asset"), ("Timeframe", "timeframe"),
            ("Long or short", "direction"), ("Entry rules, exact", "entry"), ("Entry rules, same meaning", "entry_lenient"),
            ("Exit rules, exact", "exit"), ("Exit rules, same meaning", "exit_lenient"), ("Stop", "stop"),
            ("Target", "target"), ("Time exit", "time_exit"), ("All rule elements", "rule_elements"),
            ("Claims with every element right", "claims_fully_right")]
    has_rev = ev["reviewed"] is not None
    head = "| Check | Trader-reviewed | All drafts |" if has_rev else "| Check | Drafted reference (not yet reviewed) |"
    lines = ["# Compiler vs. ground truth", "",
             "Generated by `falsify groundtruth`. The reference rules in `corpus/ground_truth.yaml` were drafted from "
             "the claim text alone by a separate agent that never saw the compiler's output, then reviewed by a "
             "trader (`reviewed: true`). This measures how often Muse Spark's compiled rules say the same thing.",
             "", f"{ev['n_reviewed']} of {ev['n']} claims reviewed.", "", head,
             "|---|---|---|" if has_rev else "|---|---|"]
    for label, key in rows:
        if has_rev:
            lines.append(f"| {label} | {pct(ev['reviewed'].get(key))} | {pct(ev['all'].get(key))} |")
        else:
            lines.append(f"| {label} | {pct(ev['all'].get(key))} |")
    misses = [s for s in ev["claims"] if not all(s.fields.values())]
    if misses:
        lines += ["", "## Where they differ", ""]
        for s in misses:
            bad = [k for k, v in s.fields.items() if not v]
            soft = [k for k in ("entry", "exit") if k in bad and s.lenient.get(k)]
            tag = "" if s.reviewed else " (draft reference)"
            extra = f"; same meaning on {', '.join(soft)}" if soft else ""
            lines.append(f"- **{s.id}**{tag}: differs on {', '.join(bad)}{extra}"
                         + (f". {' '.join(s.notes)}" if s.notes else ""))
    return "\n".join(lines) + "\n"
