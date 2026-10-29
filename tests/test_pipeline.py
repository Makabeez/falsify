"""compile_and_audit with a fake Meta client: bounded repair, escalation, and the cost ledger."""

import json
from types import SimpleNamespace

from falsify.battery import attach_fidelity, attach_grounding, falsify
from falsify.extract import Source, compile_and_audit, fidelity

SPEC = {"claim_summary": "Buy BTC daily when close > 100", "source_type": "text", "testable": True,
        "assets": ["BTCUSDT"], "timeframe": "1d", "direction": "long",
        "entry": [{"left": {"kind": "close"}, "op": "gt", "right": {"kind": "const", "value": 100}}],
        "exit": [], "max_hold_bars": 5,
        "evidence": [{"field": f, "quote": None, "assumed": True}
                     for f in ("entry[0]", "max_hold_bars", "assets", "timeframe", "direction")]}
MAJOR = {"faithful": False, "issues": [{"field": "max_hold_bars", "problem": "post says 7 days", "severity": "major"}]}


class FakeClient:
    def __init__(self, audits):
        self.audits, self.calls = list(audits), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kw):
        name = kw["response_format"]["json_schema"]["name"]
        self.calls.append(name)
        body = SPEC if name == "ClaimSpec" else self.audits.pop(0)
        usage = SimpleNamespace(model_dump=lambda: {"total_tokens": 1000 if name == "ClaimSpec" else 400})
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=json.dumps(body)))],
                               usage=usage)


def _run(audits):
    client = FakeClient(audits)
    spec, audit, trace = compile_and_audit(Source.from_text("btc above 100, hold a week"), model="m", client=client)
    rep = falsify(spec, with_cross_asset=False)
    attach_grounding(rep, trace.get("grounding"))
    attach_fidelity(rep, fidelity(spec, audit))
    return client, trace, rep


def test_clean_audit_no_repair_no_escalation():
    client, trace, rep = _run([{"faithful": True, "issues": []}])
    assert client.calls == ["ClaimSpec", "Audit"]
    assert [c["step"] for c in trace["cost"]] == ["compile", "audit"]
    assert trace["cost"][0]["tokens"] == 1000 and trace["cost"][1]["tokens"] == 400
    assert rep.review == {"needs_human": False, "reasons": []}


def test_one_repair_round_then_escalate():
    client, trace, rep = _run([MAJOR, MAJOR])
    assert client.calls == ["ClaimSpec", "Audit", "ClaimSpec", "Audit"]  # bounded: exactly one audit repair
    assert [c["step"] for c in trace["cost"]] == ["compile", "audit", "audit repair", "audit"]
    assert rep.review["needs_human"] and "7 days" in rep.review["reasons"][0]
    assert rep.warnings[0].startswith("needs a human check")


def test_repair_fixes_it():
    client, trace, rep = _run([MAJOR, {"faithful": True, "issues": []}])
    assert trace["audit_repair"] and not rep.review["needs_human"]


def test_failed_audit_repair_keeps_first_pass_and_escalates():
    """c012/c024 on 29 Oct: the repair round returned an invalid spec 3 times and the whole claim errored."""
    bad = dict(SPEC, assets=[])

    class Flaky(FakeClient):
        def create(self, **kw):
            r = super().create(**kw)
            if kw["response_format"]["json_schema"]["name"] == "ClaimSpec" and self.calls.count("ClaimSpec") > 1:
                r.choices[0].message.content = json.dumps(bad)
            return r

    client = Flaky([MAJOR])
    spec, audit, trace = compile_and_audit(Source.from_text("btc above 100, hold a week"), model="m", client=client)
    assert client.calls == ["ClaimSpec", "Audit", "ClaimSpec", "ClaimSpec", "ClaimSpec"]
    assert spec.assets == ["BTCUSDT"] and trace["audit_repair_failed"]
    assert trace["cost"][-1] == {"step": "audit repair", "seconds": trace["cost"][-1]["seconds"], "tokens": 0, "failed": True}
    rep = falsify(spec, with_cross_asset=False)
    attach_fidelity(rep, fidelity(spec, audit))
    assert rep.review["needs_human"]


def test_entry_that_never_fires_is_sent_back():
    from falsify import data as data_mod
    import pytest
    if ("BTCUSDT", "1d") not in data_mod.available():
        pytest.skip("bundled data missing")
    never = dict(SPEC, entry=[{"left": {"kind": "rsi", "period": 14}, "op": "le",
                                "right": {"kind": "rsi", "period": 14, "window_fn": "min", "window": 50}},
                               {"left": {"kind": "rsi", "period": 14}, "op": "gt",
                                "right": {"kind": "rsi", "period": 14, "offset": 1}}])

    class Seq(FakeClient):
        def create(self, **kw):
            r = super().create(**kw)
            if kw["response_format"]["json_schema"]["name"] == "ClaimSpec" and self.calls.count("ClaimSpec") == 1:
                r.choices[0].message.content = json.dumps(never)
            return r

    client = Seq([{"faithful": True, "issues": []}])
    spec, audit, trace = compile_and_audit(Source.from_text("btc above 100, hold a week"), model="m", client=client)
    first = trace["attempts"][0]["sanity_problems"][0]
    assert "never all on the same bar" in first and "entry[1]" in first
    assert len(trace["attempts"]) == 2 and spec.entry[0].right.kind.value == "const"


def test_zero_trades_escalates():
    from falsify.battery import Report, Verdict, escalate
    rep = escalate(Report(verdict=Verdict.INCONCLUSIVE, headline="x", primary={"trades": 0}))
    assert rep.review["needs_human"] and "never trigger" in rep.review["reasons"][0]
