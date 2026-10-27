"""Run a whole corpus of real claims through Muse Spark + the battery, and score the pipeline.

    falsify corpus run corpus/claims.jsonl --out corpus/results     # resumable, parallel
    falsify corpus report --out corpus/results                     # rebuild scoreboard.md

Each record is compiled from its excerpt (the claimant's words; the only quotable text),
audited, quote-checked and falsified with its post date as the out-of-sample boundary.
One JSON per claim lands in the results folder; finished claims are skipped on re-run.

The scoreboard also measures the compiler where the corpus labels allow it without hand
labelling: claims collected as "vague" or "other_asset" should come back UNTESTABLE, and
claims collected as rules should come back testable.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .battery import attach_fidelity, attach_grounding, falsify
from .engine import Costs


def load(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _note(rec: dict) -> str:
    bits = [f"Source type: {rec.get('source')}."]
    if rec.get("posted"):
        bits.append(f"Posted {rec['posted']}.")
    if rec.get("asset_hint"):
        bits.append(f"Page is about: {rec['asset_hint']}.")
    if rec.get("timeframe_hint"):
        bits.append(f"Chart timeframe on the page: {rec['timeframe_hint']}.")
    return " ".join(bits)


def run_one(rec: dict, costs: Costs, model: str | None = None) -> dict:
    from .extract import Source, compile_and_audit, fidelity
    t0 = time.time()
    src = Source.from_text(rec["excerpt"], note=_note(rec))
    out: dict = {"id": rec["id"], "record": rec}
    try:
        spec, audit, trace = compile_and_audit(src, model=model)
        rep = falsify(spec, costs, posted=rec.get("posted"))
        attach_grounding(rep, trace.get("grounding"))
        attach_fidelity(rep, fidelity(spec, audit))
        out.update(spec=spec.model_dump(mode="json"), audit=audit.model_dump(mode="json"), trace=trace,
                   report=rep.to_dict())
    except Exception as e:  # one bad claim must not stop the corpus
        out["error"] = f"{type(e).__name__}: {e}"
    out["seconds"] = round(time.time() - t0, 1)
    return out


def run(path: str | Path, out_dir: str | Path, costs: Costs = Costs(), ids: list[str] | None = None,
        limit: int | None = None, workers: int = 4, model: str | None = None, retry_errors: bool = False) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    recs = load(path)
    if ids:
        recs = [r for r in recs if r["id"] in set(ids)]
    todo = []
    for r in recs:
        f = out / f"{r['id']}.json"
        if f.exists():
            prev = json.loads(f.read_text(encoding="utf-8"))
            if not (retry_errors and "error" in prev):
                continue
        todo.append(r)
    if limit:
        todo = todo[:limit]
    print(f"corpus   {len(recs)} records, {len(todo)} to run, {workers} workers")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(run_one, r, costs, model): r["id"] for r in todo}
        for fut in as_completed(futs):
            res = fut.result()
            (out / f"{res['id']}.json").write_text(json.dumps(res, indent=2, default=str) + "\n", encoding="utf-8")
            if "error" in res:
                print(f"  {res['id']}  ERROR  {res['error'][:120]}")
            else:
                rp = res["report"]
                print(f"  {res['id']}  {rp['verdict']:<12} {rp['headline'][:90]}  ({res['seconds']}s)")
    report(out)


def _tokens(res: dict) -> int:
    total = 0
    for a in (res.get("trace") or {}).get("attempts", []):
        total += (a.get("usage") or {}).get("total_tokens") or 0
    return total


def report(out_dir: str | Path) -> str:
    out = Path(out_dir)
    results = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(out.glob("c*.json"))]
    ok = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]
    verdicts = Counter(r["report"]["verdict"] for r in ok)

    def cat(r):
        return r["record"]["category"]

    routed = {
        "vague -> UNTESTABLE": [r for r in ok if cat(r) == "vague"],
        "other_asset -> UNTESTABLE": [r for r in ok if cat(r) == "other_asset"],
        "rule/backtest -> testable (the rest are refused with a stated reason)": [r for r in ok if cat(r) in ("rule", "backtest_claim")],
    }

    def correct(name, r):
        v = r["report"]["verdict"]
        return v == "UNTESTABLE" if "UNTESTABLE" in name else v != "UNTESTABLE"

    tested = [r for r in ok if r["report"]["verdict"] != "UNTESTABLE"]
    q = sum(r["report"]["grounding"].get("quoted", 0) for r in tested if r["report"].get("grounding"))
    n = sum(r["report"]["grounding"].get("rule_elements", 0) for r in tested if r["report"].get("grounding"))
    first_pass = sum(1 for r in ok if len(r["trace"]["attempts"]) == 1 and not r["trace"]["attempts"][0].get("error")
                     and not r["trace"]["attempts"][0].get("grounding_problems"))
    unfaithful = [r for r in ok if not r["audit"]["faithful"]]
    approx = [r for r in ok if (r["report"].get("fidelity") or {}).get("level") == "approximation"
              and r["report"]["verdict"] != "UNTESTABLE"]
    repaired = [r for r in ok if r["trace"].get("audit_repair")]

    lines = ["# Corpus scoreboard", "",
             f"{len(results)} real public claims run through Muse Spark (compile, quote check, audit) and the "
             f"falsification battery. Generated by `falsify corpus report`.", "",
             "## Verdicts", "",
             "| Verdict | Claims |", "|---|---|"]
    lines += [f"| {v} | {verdicts.get(v, 0)} |" for v in ("SURVIVES", "DIES", "INCONCLUSIVE", "UNTESTABLE")]
    if errors:
        lines.append(f"| pipeline error | {len(errors)} |")
    lines += ["", "## How well the compiler did", "",
              "| Check | Result |", "|---|---|"]
    for name, rs in routed.items():
        if rs:
            c = sum(correct(name, r) for r in rs)
            lines.append(f"| {name} | {c}/{len(rs)} |")
    if n:
        lines.append(f"| Rule elements quoted verbatim from the claim (tested claims) | {q}/{n} ({q / n * 100:.0f}%) |")
    lines.append(f"| Compiled valid and quote-clean on the first attempt | {first_pass}/{len(ok)} |")
    lines.append(f"| Audit found major issues and sent the spec back for one repair | {len(repaired)}/{len(ok)} |")
    lines.append(f"| Still flagged unfaithful after repair | {len(unfaithful)}/{len(ok)} |")
    lines.append(f"| Verdicts on an approximation (disclosed gap or unresolved audit issue) | {len(approx)}/{len(tested)} |")
    tok = sum(_tokens(r) for r in ok)
    if ok:
        lines.append(f"| Mean tokens per claim (transcribe + compile, excl. audit) | {tok // len(ok):,} |")

    # Corpus-level multiple testing: Benjamini-Hochberg at 5% FDR over every claim that got a null p-value.
    ps = sorted(((r["report"]["null_test"]["p_value"], r["id"]) for r in ok
                 if (r["report"].get("null_test") or {}).get("p_value") is not None), key=lambda x: x[0])
    m = len(ps)
    cutoff_rank = 0
    for k, (pv, _) in enumerate(ps, start=1):
        if pv <= 0.05 * k / m:
            cutoff_rank = k
    bh_pass = {cid for _, cid in ps[:cutoff_rank]}
    single = [r["id"] for r in ok if r["report"]["verdict"] == "SURVIVES"]
    lines += ["", "## Correcting for testing many claims at once", "",
              f"{m} claims got a random-timing p-value. At p < 0.05 each, about {m * 0.05:.1f} would pass by luck alone. "
              f"Benjamini-Hochberg at a 5% false-discovery rate keeps the ones that still stand when the whole corpus "
              f"is taken into account.", "",
              f"- SURVIVES on its own: {len(single)} ({', '.join(sorted(single)) or 'none'})",
              f"- Still survives after the corpus-level correction: {len([c for c in single if c in bh_pass])} "
              f"({', '.join(sorted(c for c in single if c in bh_pass)) or 'none'})", ""]
    for r in ok:
        r["_bh"] = r["id"] in bh_pass

    lines += ["", "## Claim by claim", "",
              "| id | Kind | Claim (neutral summary) | Claimed | Measured | Verdict | Why | Quoted |",
              "|---|---|---|---|---|---|---|---|"]
    for r in results:
        rec = r["record"]
        if "error" in r:
            lines.append(f"| [{rec['id']}]({rec['url']}) | {rec['category']} | | | | ERROR | {r['error'][:80]} | |")
            continue
        rp, sp = r["report"], r["spec"]
        claimed = rec.get("performance_claim") or ""
        p = rp.get("primary") or {}
        measured = ""
        if p.get("trades"):
            measured = f"{p['trades']} trades, win {p['win_rate_pct']:.0f}%, {p['avg_net_pct']:+.2f}%/trade"
        g = rp.get("grounding") or {}
        quoted = f"{g['quoted']}/{g['rule_elements']}" if g else ""
        flag = " (approx.)" if (rp.get("fidelity") or {}).get("level") == "approximation" \
            and rp["verdict"] != "UNTESTABLE" else ""
        why = rp["headline"].replace("Tested an approximation of the claim. ", "").replace("|", "/")
        bh = " ✓ corpus-corrected" if r.get("_bh") and rp["verdict"] == "SURVIVES" else ""
        lines.append(f"| [{rec['id']}]({rec['url']}) | {rec['category']} | {sp['claim_summary'].replace('|', '/')} | "
                     f"{claimed.replace('|', '/')} | {measured} | **{rp['verdict']}**{flag}{bh} | {why} | {quoted} |")
    md = "\n".join(lines) + "\n"
    (out / "scoreboard.md").write_text(md, encoding="utf-8")
    print(f"\nwrote {out / 'scoreboard.md'}  " + "  ".join(f"{k}={v}" for k, v in verdicts.items())
          + (f"  errors={len(errors)}" if errors else ""))
    return md
