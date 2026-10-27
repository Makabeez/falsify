import json

import numpy as np
import pandas as pd
import pytest

from falsify.engine import Costs, operand_series, run
from falsify.spec import ClaimSpec, Operand


def ohlc(n=400, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    o = np.concatenate([[100], c[:-1]])
    h = np.maximum(o, c) * (1 + rng.uniform(0, 0.01, n))
    l = np.minimum(o, c) * (1 - rng.uniform(0, 0.01, n))
    idx = pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": 1.0}, index=idx)


def S(kind, **kw):
    return operand_series(Operand(kind=kind, **kw), DF)


DF = ohlc()


def test_wma_matches_definition():
    w = S("wma", period=5)
    c = DF["close"].to_numpy()
    expect = np.dot(c[-5:], np.arange(1, 6)) / 15
    assert w.iloc[-1] == pytest.approx(expect)
    assert w.iloc[:4].isna().all()


def test_bollinger_matches_definition_and_default_width():
    up = S("bb_upper", period=20)
    lo = S("bb_lower", period=20, param=2.5)
    win = DF["close"].iloc[-20:]
    assert up.iloc[-1] == pytest.approx(win.mean() + 2 * win.std(ddof=0))
    assert lo.iloc[-1] == pytest.approx(win.mean() - 2.5 * win.std(ddof=0))


def test_macd_defaults_and_hist_identity():
    op = Operand(kind="macd_line")
    assert op.period == 12 and op.param == 26
    line, sig, hist = S("macd_line"), S("macd_signal"), S("macd_hist")
    assert (hist - (line - sig)).abs().max() < 1e-9
    with pytest.raises(ValueError):
        Operand(kind="macd_line", period=26, param=12)


def test_adx_in_range():
    a = S("adx", period=14).dropna()
    assert len(a) > 300 and a.between(0, 100).all()


@pytest.mark.parametrize("kind,kw", [("supertrend", {}), ("adx", {"period": 14}), ("wma", {"period": 9}),
                                     ("bb_upper", {"period": 20}), ("macd_signal", {})])
def test_new_indicators_have_no_lookahead(kind, kw):
    full = operand_series(Operand(kind=kind, **kw), DF)
    part = operand_series(Operand(kind=kind, **kw), DF.iloc[:250])
    pd.testing.assert_series_equal(full.iloc[:250], part, check_names=False)


def test_supertrend_flips_with_trend():
    n = 200
    c = np.concatenate([np.linspace(100, 200, n // 2), np.linspace(200, 100, n // 2)])
    df = pd.DataFrame({"open": c, "high": c * 1.005, "low": c * 0.995, "close": c, "volume": 1.0},
                      index=pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC"))
    st = operand_series(Operand(kind="supertrend"), df)
    assert (df["close"] > st).iloc[50:95].all()     # uptrend: line below price
    assert (df["close"] < st).iloc[150:].all()      # downtrend: line above price


def test_open_position_at_end_is_marked_not_dropped():
    # enter, then price collapses with no stop and a target that is never reached
    c = [100, 101, 100, 90, 80, 70]
    df = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0},
                      index=pd.date_range("2024-01-01", periods=len(c), freq="D", tz="UTC"))
    spec = ClaimSpec.model_validate(dict(
        claim_summary="t", source_type="text", testable=True, assets=["BTCUSDT"], timeframe="1d",
        entry=[{"left": {"kind": "close"}, "op": "gt", "right": {"kind": "const", "value": 100.5}}],
        exit=[], take_profit_pct=50))
    t = run(spec, df, Costs(0, 0))
    assert len(t) == 1 and t.iloc[0].reason == "open"
    assert t.iloc[0].exit_px == 70 and t.iloc[0].net_pct == pytest.approx(-30.0)


class _Fake:
    def __init__(self, outputs):
        from types import SimpleNamespace as NS
        self.outputs, self.calls = list(outputs), []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        from types import SimpleNamespace as NS
        self.calls.append(kw)
        return NS(choices=[NS(finish_reason="stop", message=NS(content=self.outputs.pop(0)))], usage=None)


TEXT = "Buy SOL when WMA7 crosses up over WMA25, sell when WMA7 crosses down over WMA25."
EV = [{"field": "entry[0]", "quote": "WMA7 crosses up over WMA25", "assumed": False},
      {"field": "exit[0]", "quote": "WMA7 crosses down over WMA25", "assumed": False},
      {"field": "assets", "quote": "SOL", "assumed": False},
      {"field": "timeframe", "quote": None, "assumed": True},
      {"field": "direction", "quote": "Buy", "assumed": False}]


def _spec(kind):
    return dict(claim_summary="WMA cross on SOL", source_type="text", testable=True, assets=["SOLUSDT"],
                timeframe="1d", direction="long",
                entry=[{"left": {"kind": kind, "period": 7}, "op": "crosses_above", "right": {"kind": kind, "period": 25}}],
                exit=[{"left": {"kind": kind, "period": 7}, "op": "crosses_below", "right": {"kind": kind, "period": 25}}],
                evidence=EV)


def test_audit_major_issue_triggers_one_repair_and_clears():
    from falsify.extract import Source, compile_and_audit, fidelity
    bad_audit = {"faithful": False, "issues": [{"field": "entry[0].left.kind", "problem": "WMA is not SMA", "severity": "major"}]}
    ok_audit = {"faithful": True, "issues": []}
    fake = _Fake([json.dumps(_spec("sma")), json.dumps(bad_audit), json.dumps(_spec("wma")), json.dumps(ok_audit)])
    spec, audit, trace = compile_and_audit(Source.from_text(TEXT), client=fake)
    assert spec.entry[0].left.kind.value == "wma" and audit.faithful and trace["audit_repair"]
    assert "WMA is not SMA" in json.dumps(fake.calls[2]["messages"])  # audit findings went back to the compiler
    assert fidelity(spec, audit)["level"] == "exact"


def test_unresolved_gap_is_reported_as_approximation_in_headline():
    from falsify.battery import Report, Verdict, attach_fidelity
    from falsify.extract import Audit, fidelity
    s = ClaimSpec.model_validate(dict(_spec("wma"), assumptions=["NOT EXPRESSIBLE: five averaging orders ignored"]))
    fid = fidelity(s, Audit(faithful=True, issues=[]))
    assert fid["level"] == "approximation"
    rep = attach_fidelity(Report(Verdict.DIES, "Loses money after costs."), fid)
    assert rep.headline.startswith("Tested an approximation") and any("averaging" in w for w in rep.warnings)
