import json

from falsify import grounding
from falsify.extract import Source, compile_claim
from falsify.spec import ClaimSpec

TEXT = "Buy ETH on the 4H when RSI drops under 30, sell when it’s back above 70. 80% win rate!!"

RSI = {"kind": "rsi", "period": 14}
BASE = {"claim_summary": "RSI swing on ETH", "source_type": "text", "testable": True, "assets": ["ETHUSDT"],
        "timeframe": "4h", "direction": "long",
        "entry": [{"left": RSI, "op": "crosses_below", "right": {"kind": "const", "value": 30}}],
        "exit": [{"left": RSI, "op": "crosses_above", "right": {"kind": "const", "value": 70}}]}
GOOD_EVIDENCE = [
    {"field": "entry[0]", "quote": "RSI drops under 30", "assumed": False},
    {"field": "exit[0]", "quote": "sell when it's back above 70", "assumed": False},  # straight apostrophe vs curly
    {"field": "assets", "quote": "Buy ETH", "assumed": False},
    {"field": "timeframe", "quote": "on the 4H", "assumed": False},
    {"field": "direction", "quote": "Buy", "assumed": False},
]


def spec(evidence):
    return ClaimSpec.model_validate(dict(BASE, evidence=evidence))


def test_exact_quotes_pass_and_normalize_typography():
    g = grounding.check(spec(GOOD_EVIDENCE), TEXT)
    assert g.ok, g.problems
    assert len(g.quoted) == 5 and not g.assumed


def test_paraphrase_is_rejected():
    ev = [dict(GOOD_EVIDENCE[0], quote="RSI falls below 30")] + GOOD_EVIDENCE[1:]
    g = grounding.check(spec(ev), TEXT)
    assert not g.ok and "entry[0]" in g.problems[0] and "not in the source" in g.problems[0]


def test_missing_rows_and_bad_paths_are_reported():
    ev = GOOD_EVIDENCE[:1] + [{"field": "exit[3]", "quote": "70", "assumed": False},
                              {"field": "leverage", "quote": None, "assumed": True}]
    g = grounding.check(spec(ev), TEXT)
    text = " | ".join(g.problems)
    assert "past the end of exit" in text and "'leverage' is not a valid path" in text
    for f in ("exit[0]", "assets", "timeframe", "direction"):
        assert f"{f}: no evidence row" in text


def test_assumed_rows_count_as_assumed_not_quoted():
    ev = GOOD_EVIDENCE[:4] + [{"field": "direction", "quote": None, "assumed": True}]
    g = grounding.check(spec(ev), TEXT)
    assert g.ok and g.assumed == ["direction"]


class _Fake:
    def __init__(self, outputs):
        from types import SimpleNamespace as NS
        self.outputs, self.calls = list(outputs), []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        from types import SimpleNamespace as NS
        self.calls.append(kw)
        return NS(choices=[NS(finish_reason="stop", message=NS(content=self.outputs.pop(0)))], usage=None)


def test_compile_sends_failed_quote_back_for_repair():
    bad = dict(BASE, evidence=[dict(GOOD_EVIDENCE[0], quote="RSI is oversold")] + GOOD_EVIDENCE[1:])
    good = dict(BASE, evidence=GOOD_EVIDENCE)
    fake = _Fake([json.dumps(bad), json.dumps(good)])
    s, trace = compile_claim(Source.from_text(TEXT), client=fake)
    assert trace["grounding"]["quoted"] == 5 and not trace["grounding"]["problems"]
    assert "quote check" in fake.calls[1]["messages"][-1]["content"]


def test_image_quotes_are_checked_against_transcript_and_note_is_not_quotable(tmp_path):
    img = tmp_path / "post.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 16)
    transcript = {"visible_text": "ETH 4H. Buy when RSI drops under 30, sell above 70", "speech": "",
                  "chart": "ETHUSDT 4h candles with RSI(14)."}
    ev = [{"field": "entry[0]", "quote": "RSI drops under 30", "assumed": False},
          {"field": "exit[0]", "quote": "sell above 70", "assumed": False},
          {"field": "assets", "quote": "ETH", "assumed": False},
          {"field": "timeframe", "quote": "posted on X", "assumed": False},  # from the note -> must fail
          {"field": "direction", "quote": "Buy", "assumed": False}]
    out = dict(BASE, source_type="image", evidence=ev)
    fake = _Fake([json.dumps(transcript), json.dumps(out), json.dumps(out), json.dumps(out)])
    s, trace = compile_claim(Source.from_path(img, note="posted on X"), client=fake)
    assert trace["transcript"]["visible_text"].startswith("ETH 4H")
    assert any("timeframe" in p for p in trace["grounding"]["problems"])  # survives repairs -> reported, not hidden
    assert len(fake.calls) == 4  # transcribe + compile + 2 repairs
    first_compile = json.dumps(fake.calls[1]["messages"][0]["content"])
    assert "TRANSCRIPT OF THE MEDIA" in first_compile and "never quote it" in first_compile


def test_corpus_run_with_fake_model(tmp_path, monkeypatch):
    from falsify import corpus, extract
    rec = {"id": "c900", "url": "https://example.com/x", "source": "blog", "posted": "2023-06-01",
           "asset_hint": "ETH", "timeframe_hint": "4h", "category": "backtest_claim", "excerpt": TEXT,
           "performance_claim": "80% win rate", "verbatim_confidence": "high", "notes": ""}
    path = tmp_path / "claims.jsonl"
    path.write_text(json.dumps(rec) + "\n")
    audit = {"faithful": True, "issues": []}
    good = json.dumps(dict(BASE, evidence=GOOD_EVIDENCE, claimed={"win_rate_pct": 80}))
    fake = _Fake([good, json.dumps(audit)])
    monkeypatch.setattr(extract, "_client", lambda: fake)
    out = tmp_path / "results"
    corpus.run(path, out, workers=1)
    res = json.loads((out / "c900.json").read_text())
    assert "error" not in res, res.get("error")
    assert res["report"]["out_of_sample"]["boundary"] == "2023-06-01"
    assert res["report"]["out_of_sample"]["boundary_source"] == "post date"
    md = (out / "scoreboard.md").read_text()
    assert "rule/backtest -> testable | 1/1" in md and "5/5" in md
    assert "never quote it" in json.dumps(fake.calls[0]["messages"])  # hints go in as context, not as claim text
    corpus.run(path, out, workers=1)  # resumable: nothing left to run, no extra model calls
    assert len(fake.calls) == 2
