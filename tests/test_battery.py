import json

import numpy as np
import pandas as pd

from falsify.battery import random_entry_null
from falsify.engine import Costs
from falsify.extract import Source
from falsify.spec import Direction, SourceType, model_json_schema


def _depth(node, d=0):
    if isinstance(node, dict):
        return max([d] + [_depth(v, d + 1) for v in node.values()])
    if isinstance(node, list):
        return max([d] + [_depth(v, d + 1) for v in node])
    return d


def test_schema_fits_meta_structured_output_limits():
    schema = model_json_schema()
    text = json.dumps(schema)
    assert _depth(schema) <= 10 + 4  # JSON nesting incl. wrappers; Meta counts schema levels, not JSON levels
    # no self-reference: every $ref target must not reference itself
    for name, d in schema.get("$defs", {}).items():
        assert f'"#/$defs/{name}"' not in json.dumps(d), f"{name} is recursive"
    assert len(text) < 120_000


def test_null_does_not_reward_drift():
    # pure uptrend: any long wins, so a long "strategy" must NOT look significant vs random longs
    n = 2000
    opens = 100 * np.exp(np.linspace(0, 2, n))
    df = pd.DataFrame({"open": opens}, index=pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC"))
    rng = np.random.default_rng(1)
    entries = np.sort(rng.choice(np.arange(10, n - 20), 60, replace=False))
    hold = 5
    gross = opens[entries + hold] / opens[entries] - 1
    trades = pd.DataFrame({"bars": hold, "net_pct": gross * 100})
    res = random_entry_null(trades, df, Direction.LONG, Costs(0, 0))
    assert res["p_value"] > 0.05


def test_null_flags_real_timing_edge():
    rng = np.random.default_rng(3)
    n = 3000
    rets = rng.normal(0, 0.02, n)
    opens = 100 * np.exp(np.cumsum(rets))
    df = pd.DataFrame({"open": opens}, index=pd.date_range("2015-01-01", periods=n, freq="D", tz="UTC"))
    # cheat: pick entries with the best 5-bar forward return -> must be significant
    fwd = np.log(opens[5:]) - np.log(opens[:-5])
    entries = np.argsort(fwd)[-80:]
    gross = opens[entries + 5] / opens[entries] - 1
    trades = pd.DataFrame({"bars": 5, "net_pct": gross * 100})
    res = random_entry_null(trades, df, Direction.LONG, Costs(0, 0))
    assert res["p_value"] < 0.01


def test_source_builds_openai_compatible_parts(tmp_path):
    img = tmp_path / "c.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 16)
    s = Source.from_path(img, note="from X")
    parts = s.content("instr")
    assert s.kind == SourceType.IMAGE
    assert parts[-1]["type"] == "image_url" and parts[-1]["image_url"]["url"].startswith("data:image/png;base64,")
    t = Source.from_text("buy btc when rsi < 30")
    assert t.content("i")[1]["text"].startswith("CLAIM:")


class _FakeCompletions:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def create(self, **kw):
        from types import SimpleNamespace as NS
        self.calls.append(kw)
        content = self.outputs.pop(0)
        return NS(choices=[NS(finish_reason="stop", message=NS(content=content))], usage=None)


class _FakeClient:
    def __init__(self, outputs):
        from types import SimpleNamespace as NS
        self.chat = NS(completions=_FakeCompletions(outputs))


def test_compile_repairs_invalid_output_once():
    from falsify.extract import compile_claim
    bad = {"claim_summary": "x", "source_type": "text", "testable": True, "assets": ["BTCUSDT"],
           "entry": [{"left": {"kind": "rsi", "period": 14}, "op": "lt", "right": {"kind": "const", "value": 30}}]}
    good = dict(bad, max_hold_bars=5, evidence=[
        {"field": "entry[0]", "quote": "rsi<30", "assumed": False},
        {"field": "assets", "quote": "btc", "assumed": False},
        {"field": "max_hold_bars", "quote": None, "assumed": True},
        {"field": "timeframe", "quote": None, "assumed": True},
        {"field": "direction", "quote": "buy", "assumed": False}])
    client = _FakeClient([json.dumps(bad), json.dumps(good)])
    spec, trace = compile_claim(Source.from_text("buy btc rsi<30"), client=client)
    assert spec.max_hold_bars == 5 and len(trace["attempts"]) == 2
    second = client.chat.completions.calls[1]
    assert second["response_format"]["type"] == "json_schema"
    assert "failed validation" in second["messages"][-1]["content"]
