"""Human sign-off: the last step of the verification chain.

    deterministic checks (schema, quote check)  ->  LLM judge (independent audit)
    -> bounded repairs  ->  a person, when the machines still disagree

Three sources of human judgement, all shown next to the verdict:

* Corpus claims: a trader reviewed the reference rules in corpus/ground_truth.yaml. If the
  compiled rules match the trader's reading, the result says so. If not, it shows where they
  differ and the trader's note.
* Live checks: anyone can say whether the compiled rules match the claim. Those flags go to
  a queue (web_cache/reviews.jsonl). They never change a verdict on their own.
* The reviewer works through the queue with `falsify review queue` and signs off with
  `falsify review sign <key> --agree|--disagree --note ...`. The sign-off is stored with the
  result and shown on its permalink.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path(os.environ.get("FALSIFY_CACHE", ROOT / "web_cache"))
_lock = threading.Lock()
_gt_cache: dict = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def review_state(report: dict) -> dict:
    """Escalation flag, always recomputed from the report so stored results follow the current rule.

    An UNTESTABLE verdict the auditor disagrees with is escalated too: "can't be tested" can be the misreading."""
    reasons = [f"audit still disagrees: {m}" for m in (report.get("fidelity") or {}).get("audit_majors", [])]
    reasons += [f"quote check: {p}" for p in (report.get("grounding") or {}).get("problems", [])]
    if (report.get("primary") or {}).get("trades") == 0:
        reasons.append("the rules never trigger on the full price history, which usually means they misread the claim")
    return {"needs_human": bool(reasons), "reasons": reasons}


# ---------------------------------------------------------------- corpus: trader-reviewed ground truth

def _reference() -> dict:
    from . import groundtruth
    path = groundtruth.GT_PATH
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}
    if _gt_cache.get("mtime") != mtime:
        try:
            _gt_cache.update(mtime=mtime, ref=groundtruth.load_reference(path))
        except Exception:  # a broken YAML must not take the gallery down
            _gt_cache.update(mtime=mtime, ref={})
    return _gt_cache["ref"]


def trader_signoff(cid: str, spec: dict, verdict: str | None) -> dict | None:
    """Compare the compiled rules with the trader's reading of the claim, if the trader reviewed it."""
    from .groundtruth import score_claim
    ref = _reference().get(cid)
    if not ref or not ref.get("reviewed"):
        return None
    s = score_claim(cid, ref, spec, verdict)
    differs = [k for k, ok in s.fields.items() if not ok and not s.lenient.get(k)]
    return {"by": "trader", "agree": not differs, "differs_on": differs,
            "note": (ref.get("reviewer_note") or "").strip() or None}


# ---------------------------------------------------------------- live: flags from visitors, sign-off by the reviewer

def _flags_path(cache: Path) -> Path:
    return cache / "reviews.jsonl"


def add_flag(key: str, agree: bool, note: str | None, ip: str, cache: Path = CACHE) -> None:
    row = {"at": _now(), "key": key, "agree": bool(agree), "note": (note or "").strip()[:500] or None,
           "ip": hashlib.sha256(ip.encode()).hexdigest()[:12]}  # enough to spot one person spamming, not to identify them
    cache.mkdir(parents=True, exist_ok=True)
    with _lock, _flags_path(cache).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def flags(key: str | None = None, cache: Path = CACHE) -> list[dict]:
    p = _flags_path(cache)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if key is None or row.get("key") == key:
            out.append(row)
    return out


def flag_counts(key: str, cache: Path = CACHE) -> dict:
    rows = {r["ip"]: r for r in flags(key, cache)}.values()  # one vote per visitor, latest wins
    return {"agree": sum(r["agree"] for r in rows), "disagree": sum(not r["agree"] for r in rows)}


def sign(key: str, agree: bool, note: str | None, by: str = "trader", cache: Path = CACHE) -> dict:
    p = cache / f"{key}.json"
    if not p.exists():
        raise FileNotFoundError(f"no stored result {key}")
    with _lock:
        d = json.loads(p.read_text(encoding="utf-8"))
        d["signoff"] = {"by": by, "agree": bool(agree), "note": (note or "").strip() or None, "at": _now()}
        p.write_text(json.dumps(d, default=str), encoding="utf-8")
    return d["signoff"]


def queue(cache: Path = CACHE) -> list[dict]:
    """Stored live results that need a person: escalated by the pipeline or flagged by a visitor, not yet signed off."""
    flagged: dict[str, list] = {}
    for r in flags(cache=cache):
        flagged.setdefault(r["key"], []).append(r)
    out = []
    for p in sorted(cache.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if "report" not in d or d.get("signoff"):
            continue
        st = review_state(d["report"])
        fl = flagged.get(p.stem, [])
        if not (st["needs_human"] or any(not r["agree"] for r in fl)):
            continue
        out.append({"key": p.stem, "created": d.get("created"), "verdict": d["report"]["verdict"],
                    "summary": (d.get("spec") or {}).get("claim_summary"), "reasons": st["reasons"],
                    "flags": [{"agree": r["agree"], "note": r["note"]} for r in fl]})
    return out
