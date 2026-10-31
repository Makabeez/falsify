"""The dial: run the corpus at several reasoning efforts and score each against the trader's ground truth.

Fireworks' Muse workshop measured Muse Spark across its effort levels on simple benchmarks and found
the cheap end often ties or beats the expensive end. This does the same measurement on Falsify's own
task, with Falsify's own grader: the trader-reviewed reference rules in corpus/ground_truth.yaml,
scored deterministically by falsify.groundtruth (no LLM judge).

    falsify sweep run --efforts low,medium,high --workers 4     # resumable; one JSON per claim per effort
    falsify sweep report --out docs/effort_sweep.md

Only the model side runs here (compile, quote check, fires check, audit, repair). Routing is scored
from the compiled spec and the bundled data, so the backtest battery is not needed.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import data as data_mod
from .corpus import _note, load
from .groundtruth import GT_PATH, load_reference, score_claim

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "corpus" / "sweep"
CLAIMS = ROOT / "corpus" / "claims.jsonl"


def _verdict(spec) -> str:
    """Enough of a verdict to score routing: untestable, or testable on data we have."""
    if not spec.testable:
        return "UNTESTABLE"
    have = data_mod.available()
    return "TESTED" if any((a, spec.timeframe.value) in have for a in spec.assets) else "UNTESTABLE"


def run_one(rec: dict, effort: str, model: str | None = None) -> dict:
    from .extract import Source, compile_and_audit, fidelity
    t0 = time.time()
    out: dict = {"id": rec["id"], "effort": effort}
    try:
        spec, audit, trace = compile_and_audit(Source.from_text(rec["excerpt"], note=_note(rec)),
                                               model=model, effort=effort)
        fid = fidelity(spec, audit)
        out.update(spec=spec.model_dump(mode="json"), verdict=_verdict(spec), cost=trace.get("cost", []),
                   audit_majors=fid["audit_majors"], grounding_problems=(trace.get("grounding") or {}).get("problems", []),
                   audit_repair=bool(trace.get("audit_repair")))
    except Exception as e:  # one bad claim must not stop the sweep
        out["error"] = f"{type(e).__name__}: {e}"[:500]
    out["seconds"] = round(time.time() - t0, 1)
    return out


def run(efforts: list[str], ids: list[str] | None = None, workers: int = 4, model: str | None = None,
        out_dir: Path = OUT, claims: Path = CLAIMS) -> None:
    recs = [r for r in load(claims) if not ids or r["id"] in ids]
    todo = []
    for e in efforts:
        (out_dir / e).mkdir(parents=True, exist_ok=True)
        for r in recs:
            f = out_dir / e / f"{r['id']}.json"
            if not f.exists() or "error" in json.loads(f.read_text(encoding="utf-8")):
                todo.append((r, e))
    print(f"sweep    {len(recs)} claims x {len(efforts)} efforts, {len(todo)} to run, {workers} workers")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(run_one, r, e, model): (r["id"], e) for r, e in todo}
        for fut in as_completed(futs):
            res = fut.result()
            (out_dir / res["effort"] / f"{res['id']}.json").write_text(json.dumps(res, indent=1, default=str),
                                                                     encoding="utf-8")
            tag = res.get("verdict") or ("ERROR " + res["error"][:80])
            print(f"  {res['effort']:<8} {res['id']}  {tag}  ({res['seconds']}s)")


def evaluate(out_dir: Path = OUT, gt_path: Path = GT_PATH) -> list[dict]:
    ref = load_reference(gt_path)
    rows = []
    order = {e: i for i, e in enumerate(("minimal", "low", "medium", "high", "xhigh", "max"))}
    for d in sorted((p for p in out_dir.iterdir() if p.is_dir()), key=lambda p: order.get(p.name, 99)):
        results = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(d.glob("c*.json"))]
        ok = [r for r in results if "error" not in r and r["id"] in ref]
        scores = [score_claim(r["id"], ref[r["id"]], r["spec"], r["verdict"]) for r in ok]
        routing = [s.fields["routing"] for s in scores]
        elems = [v for s in scores for k, v in s.fields.items() if k != "routing"]
        full = [s for s in scores if len(s.fields) > 1]
        tok = [sum(c.get("tokens", 0) for c in r["cost"]) for r in ok]
        secs = [r["seconds"] for r in ok]
        rows.append({
            "effort": d.name, "claims": len(results), "errors": len(results) - len(ok),
            "routing": (sum(routing), len(routing)), "rule_elements": (sum(elems), len(elems)),
            "fully_right": (sum(all(s.fields.values()) for s in full), len(full)),
            "escalated": sum(bool(r["audit_majors"] or r["grounding_problems"]) for r in ok),
            "repaired": sum(r["audit_repair"] for r in ok),
            "tokens_mean": round(sum(tok) / len(tok)) if tok else 0,
            "seconds_mean": round(sum(secs) / len(secs), 1) if secs else 0,
            "differs": {s.id: [k for k, v in s.fields.items() if not v] for s in scores if not all(s.fields.values())},
        })
    return rows


def to_markdown(rows: list[dict]) -> str:
    def frac(t):
        return f"{t[0]}/{t[1]} ({t[0] / t[1] * 100:.0f}%)" if t[1] else "n/a"

    lines = ["# The dial: reasoning effort vs. accuracy on Falsify's own task", "",
             "Generated by `falsify sweep report`. Each row runs the whole model side of the pipeline (compile, quote "
             "check, fires check, audit, one repair) on the corpus at one Muse Spark reasoning effort, then scores the "
             "compiled rules against the trader-reviewed reference in `corpus/ground_truth.yaml`. The grader is "
             "deterministic Python, not a model.", "",
             "| Effort | Testable or not | Rule elements right | Claims fully right | Sent to a human | Audit repairs "
             "| Mean tokens / claim | Mean seconds / claim | Errors |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        n = r["claims"] - r["errors"]
        lines.append(f"| {r['effort']} | {frac(r['routing'])} | {frac(r['rule_elements'])} | {frac(r['fully_right'])} | "
                     f"{r['escalated']}/{n} | {r['repaired']}/{n} | {r['tokens_mean']:,} | {r['seconds_mean']} | {r['errors']} |")
    lines += ["", "## Where each effort differs from the trader", ""]
    for r in rows:
        bits = [f"{cid} ({', '.join(v)})" for cid, v in r["differs"].items()]
        lines.append(f"- **{r['effort']}**: " + ("; ".join(bits) if bits else "none"))
    return "\n".join(lines) + "\n"
