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
