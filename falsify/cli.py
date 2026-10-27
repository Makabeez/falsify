"""falsify CLI.

  falsify judge-demo              offline: verify data, run every claim in claims/, no key needed
  falsify run claims/x.json       run the battery on a compiled spec
  falsify check <file|--text>     Muse Spark compiles + audits the claim, then the battery runs it
  falsify extract <file|--text>   compile only, write the spec JSON
  falsify data fetch|verify       rebuild or verify bundled history
  falsify corpus run|report       run real claims end to end, build corpus/results/scoreboard.md
  falsify calibrate               false-positive rate + power of the battery on bundled data
  falsify ping                    check MODEL_API_KEY, list models, one structured-output call
  falsify schema                  print the ClaimSpec JSON schema sent to Muse Spark
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import data as data_mod
from .battery import Report, Verdict, attach_fidelity, attach_grounding, falsify
from .engine import Costs
from .spec import model_json_schema

ROOT = Path(__file__).resolve().parent.parent
CLAIMS = ROOT / "claims"

_COLOR = {"SURVIVES": "\033[32m", "DIES": "\033[31m", "INCONCLUSIVE": "\033[33m", "UNTESTABLE": "\033[90m"}
_RESET = "\033[0m"


def _c(verdict: str) -> str:
    if not sys.stdout.isatty():
        return verdict
    return f"{_COLOR.get(verdict, '')}{verdict}{_RESET}"


def _fmt(x, spec=".2f", suffix=""):
    return "n/a" if x is None else f"{x:{spec}}{suffix}"


def print_report(rep: Report, summary: str, audit: dict | None = None) -> None:
    print(f"\n{'=' * 78}\n{summary}\n{'-' * 78}")
    print(f"VERDICT  {_c(rep.verdict.value)}   {rep.headline}")
    if rep.asset:
        p = rep.primary
        w = p.get("window", ["?", "?"])
        print(f"tested   {rep.asset} {rep.timeframe}  {w[0]} -> {w[1]}   costs {rep.costs.get('fee_bps_per_side')}+"
              f"{rep.costs.get('slippage_bps_per_side')} bps/side")
        print(f"trades   {p['trades']}   win {_fmt(p.get('win_rate_pct'), '.0f', '%')}   "
              f"avg {_fmt(p.get('avg_net_pct'), '+.2f', '%')}/trade   total {_fmt(p.get('total_net_pct'), '+.1f', '%')} "
              f"(gross {_fmt(p.get('total_gross_pct'), '+.1f', '%')})   buy&hold {_fmt(p.get('buy_hold_pct'), '+.0f', '%')}   "
              f"maxDD {_fmt(p.get('max_drawdown_pct'), '.1f', '%')}")
    nt = rep.null_test
    if nt.get("p_value") is not None:
        print(f"null     random timing beat it {nt['beat_by_random_pct']:.1f}% of {nt['sims']} runs  (p = {nt['p_value']:.3f})")
    oos = rep.out_of_sample
    if oos.get("available"):
        a = oos["after_window"]
        print(f"OOS      after {oos['boundary']} ({oos['boundary_source']}): {a['trades']} trades, "
              f"avg {_fmt(a.get('avg_net_pct'), '+.2f', '%')}")
    h = rep.halves
    if h:
        print(f"halves   before {h['split']}: {_fmt(h['first'].get('avg_net_pct'), '+.2f', '%')}  "
              f"after: {_fmt(h['second'].get('avg_net_pct'), '+.2f', '%')}")
    ca = rep.cross_asset
    if ca and ca.get("tested"):
        print(f"assets   positive on {ca['positive']}/{ca['tested']} other assets")
    sw = rep.sweep
    if sw.get("readings", 1) > 1:
        print(f"readings {sw['readings']} readings of the vague parameters, "
              f"{_fmt(sw.get('positive_share_pct'), '.0f', '%')} profitable; best p after search penalty "
              f"{_fmt(sw.get('best_p_bonferroni'), '.3f')}")
    for row in rep.claimed_vs_measured:
        print(f"claimed  {row['metric']}: claimed {row['claimed']}, measured {row['measured']}")
    for r in rep.reasons:
        print(f"  x {r}")
    for w in rep.warnings:
        print(f"  ! {w}")
    fid = rep.fidelity
    if fid:
        print(f"fidelity {fid['level']}" + ("" if fid["level"] == "exact" else
              f"  ({len(fid.get('gaps', []))} gap(s) disclosed, {len(fid.get('audit_majors', []))} major audit issue(s))"))
    g = rep.grounding
    if g:
        print(f"grounding {g['quoted']}/{g['rule_elements']} rule elements quoted from the source, "
              f"{g['assumed']} assumed ({', '.join(g['assumed_fields']) or 'none'})")
        for pr in g.get("problems", []):
            print(f"  ? {pr}")
    if audit:
        tag = "faithful" if audit.get("faithful") else "NOT faithful"
        print(f"audit    Muse Spark audit: spec is {tag} to the source")
        for i in audit.get("issues", []):
            print(f"  ? [{i['severity']}] {i['field']}: {i['problem']}")


def _source(args):
    from .extract import Source
    if args.text:
        return Source.from_text(args.text, note=args.note)
    if not args.input:
        sys.exit("give a file (text, image, .mp4) or --text")
    return Source.from_path(args.input, note=args.note)


def cmd_extract(args) -> int:
    from .extract import compile_and_audit, compile_claim, save
    src = _source(args)
    if args.no_audit:
        (spec, trace), audit = compile_claim(src, model=args.model), None
    else:
        spec, audit, trace = compile_and_audit(src, model=args.model)
    out = args.out or "spec.json"
    save(spec, out, audit, trace)
    print(f"wrote {out}  ({spec.claim_summary})")
    if audit and not audit.faithful:
        print("audit flagged issues -- review before trusting the verdict")
    return 0


def _costs(args) -> Costs:
    return Costs(fee_bps=args.fee_bps, slippage_bps=args.slippage_bps)


def cmd_run(args) -> int:
    from .extract import load_spec
    spec, raw = load_spec(args.spec)
    rep = falsify(spec, _costs(args), with_cross_asset=not args.fast, posted=args.posted or raw.get("posted"))
    attach_grounding(rep, (raw.get("trace") or {}).get("grounding"))
    if raw.get("audit"):
        from .extract import fidelity
        attach_fidelity(rep, fidelity(spec, raw["audit"]))
    print_report(rep, spec.claim_summary, raw.get("audit"))
    if args.json:
        Path(args.json).write_text(json.dumps({"spec": spec.model_dump(mode="json"), "report": rep.to_dict()},
                                              indent=2, default=str) + "\n")
    return 0


def cmd_check(args) -> int:
    from .extract import compile_and_audit, compile_claim, fidelity
    src = _source(args)
    if args.no_audit:
        (spec, trace), audit = compile_claim(src, model=args.model), None
    else:
        spec, audit, trace = compile_and_audit(src, model=args.model)
    rep = falsify(spec, _costs(args), with_cross_asset=not args.fast, posted=args.posted)
    attach_grounding(rep, trace.get("grounding"))
    attach_fidelity(rep, fidelity(spec, audit))
    print_report(rep, spec.claim_summary, audit.model_dump() if audit else None)
    if args.json:
        Path(args.json).write_text(json.dumps({"spec": spec.model_dump(mode="json"),
                                               "audit": audit.model_dump() if audit else None,
                                               "trace": trace, "report": rep.to_dict()},
                                              indent=2, default=str) + "\n")
    return 0


def cmd_judge_demo(args) -> int:
    from .extract import load_spec
    t0 = time.time()
    problems = data_mod.verify()
    if problems:
        print("data manifest check FAILED:\n  " + "\n  ".join(problems))
        return 2
    print(f"data     {len(data_mod.available())} files match data/manifest.json (sha256)")
    files = sorted(CLAIMS.glob("*.json"))
    if not files:
        print("no claims/*.json found")
        return 1
    rows = []
    for f in files:
        spec, raw = load_spec(f)
        rep = falsify(spec, _costs(args), with_cross_asset=not args.fast)
        attach_grounding(rep, (raw.get("trace") or {}).get("grounding"))
        print_report(rep, f"[{f.stem}] {spec.claim_summary}", raw.get("audit"))
        rows.append((f.stem, rep.verdict.value, rep.headline))
    print(f"\n{'=' * 78}\nSUMMARY  ({time.time() - t0:.1f}s, no API key, no network)")
    for stem, v, head in rows:
        print(f"  {v:<13} {stem:<28} {head[:70]}")
    counts = {v.value: sum(1 for r in rows if r[1] == v.value) for v in Verdict}
    print("  " + "  ".join(f"{k}={n}" for k, n in counts.items()))
    return 0


def cmd_ping(args) -> int:
    """Check the key, list Meta models, and make one tiny structured-output call."""
    from .extract import MODEL, VIDEO_MODEL, _client, _structured
    client = _client()
    ids = sorted(m.id for m in client.models.list().data)
    print("models   " + ", ".join(ids))
    for want in {MODEL, VIDEO_MODEL}:
        print(f"         {want}: {'available' if want in ids else 'NOT LISTED'}")
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
    t0 = time.time()
    raw, usage = _structured(client, MODEL, [{"role": "user", "content": "Return ok=true."}], "Ping", schema)
    print(f"call     {MODEL} structured output -> {raw.strip()}  ({time.time() - t0:.1f}s, {usage})")
    return 0


def cmd_calibrate(args) -> int:
    from . import calibrate
    costs = _costs(args)
    t0 = time.time()
    results = calibrate.run_all(reps=args.reps, sims=args.sims, costs=costs)
    md = calibrate.to_markdown(results, costs)
    print(md)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(CALIBRATION_HEADER + md + CALIBRATION_FOOTER, encoding="utf-8")
        json_path = out.with_suffix(".json")
        json_path.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {out} and {json_path}  ({time.time() - t0:.0f}s)")
    return 0


CALIBRATION_HEADER = """# Calibration: does the battery reject the right things?

Generated by `falsify calibrate --out docs/calibration.md`. Re-run it to reproduce.

A strategy that picks entries by coin flip has no timing skill by construction. A good test
passes it rarely; a naive backtest ("did it make money after costs?") passes it whenever the
market went up. Then we add a true edge of +d bps per trade and ask how often the test lets
it through: that is the power, and the edge it catches 80% of the time is the minimum
detectable edge.

"""

CALIBRATION_FOOTER = """
## Reading it

- **False positives sit at the 5% the threshold promises**, within simulation error (with 300
  runs per setup, one standard error is about 1.3 points), while a naive backtest calls most
  coin-flip strategies on BTC daily profitable, because BTC went up. That gap is the drift the
  random-entry null removes.
- **Power depends on trade count and holding period, not on the claim's confidence.** A daily
  rule with 30 trades needs an edge of about 4.5% per trade before the data can tell it from luck.
  An hourly rule with hundreds of trades is caught at well under 1%. This is why many claims come
  back INCONCLUSIVE or DIES on the null: their sample cannot prove what they assert.
- Coin-flip strategies are simulated with the engine's own semantics (flat-only, fill at the next
  open, no overlapping trades); the null sampler allows overlap, so the false-positive rate is a real
  check and not true by construction.
"""


def cmd_corpus(args) -> int:
    from . import corpus
    if args.action == "run":
        ids = args.ids.split(",") if args.ids else None
        corpus.run(args.path, args.out, _costs(args), ids=ids, limit=args.limit, workers=args.workers,
                   model=args.model, retry_errors=args.retry_errors)
    else:
        corpus.report(args.out)
    return 0


def cmd_data(args) -> int:
    if args.action == "fetch":
        data_mod.build(end=args.end)
        return 0
    problems = data_mod.verify()
    print("ok" if not problems else "\n".join(problems))
    return 0 if not problems else 2


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="falsify", description="Test trading claims before you trade them.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_costs(p):
        p.add_argument("--fee-bps", type=float, default=10.0, help="fee per side in bps (default 10, Binance spot taker)")
        p.add_argument("--slippage-bps", type=float, default=2.0, help="slippage per side in bps (default 2)")
        p.add_argument("--fast", action="store_true", help="skip the cross-asset test")

    def add_source(p):
        p.add_argument("input", nargs="?", help="claim file: .txt/.md, image, or .mp4")
        p.add_argument("--text", help="claim as text instead of a file")
        p.add_argument("--note", help="your context (where/when posted); sent to the model, never quotable")
        p.add_argument("--model", help="override the Muse Spark model id")
        p.add_argument("--no-audit", action="store_true", help="skip the audit pass")

    p = sub.add_parser("judge-demo", help="offline demo on bundled claims and data")
    add_costs(p)
    p.set_defaults(fn=cmd_judge_demo)

    p = sub.add_parser("run", help="run the battery on a spec JSON")
    p.add_argument("spec")
    p.add_argument("--posted", help="post date YYYY-MM-DD; trades after it are out-of-sample")
    p.add_argument("--json", help="write the full report here")
    add_costs(p)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("check", help="compile with Muse Spark, audit, then run")
    add_source(p)
    p.add_argument("--posted", help="post date YYYY-MM-DD; trades after it are out-of-sample")
    p.add_argument("--json", help="write spec + audit + report here")
    add_costs(p)
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("extract", help="compile with Muse Spark and write the spec")
    add_source(p)
    p.add_argument("-o", "--out", help="output path (default spec.json)")
    p.set_defaults(fn=cmd_extract)

    p = sub.add_parser("corpus", help="run a corpus of real claims end to end and build the scoreboard")
    p.add_argument("action", choices=["run", "report"])
    p.add_argument("path", nargs="?", default="corpus/claims.jsonl")
    p.add_argument("--out", default="corpus/results")
    p.add_argument("--ids", help="comma-separated ids to run, e.g. c003,c005")
    p.add_argument("--limit", type=int, help="run at most this many new claims")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--model", help="override the Muse Spark model id")
    p.add_argument("--retry-errors", action="store_true", help="re-run claims that errored last time")
    add_costs(p)
    p.set_defaults(fn=cmd_corpus)

    p = sub.add_parser("calibrate", help="measure false-positive rate and power on the bundled data")
    p.add_argument("--reps", type=int, default=300, help="coin-flip strategies per setup")
    p.add_argument("--sims", type=int, default=1000, help="null simulations per strategy")
    p.add_argument("--out", help="write markdown (and .json next to it), e.g. docs/calibration.md")
    add_costs(p)
    p.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser("ping", help="check MODEL_API_KEY and Muse Spark access")
    p.set_defaults(fn=cmd_ping)

    p = sub.add_parser("data", help="bundled price history")
    p.add_argument("action", choices=["fetch", "verify"])
    p.add_argument("--end", help="exclusive end date YYYY-MM-DD for a pinned snapshot")
    p.set_defaults(fn=cmd_data)

    p = sub.add_parser("schema", help="print the ClaimSpec JSON schema")
    p.set_defaults(fn=lambda a: print(json.dumps(model_json_schema(), indent=2)) or 0)

    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
