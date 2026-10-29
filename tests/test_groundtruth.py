"""The ground-truth scorer: label grammar and per-claim scoring."""

import pytest

from falsify.groundtruth import Opnd, conditions_match, parse_condition, parse_exit, parse_operand, score_claim


def test_parse_operand_grammar():
    assert parse_operand("50") == Opnd("const", value=50.0)
    assert parse_operand("rsi") == Opnd("rsi", 14, None, None, 1.0)  # default period
    v = parse_operand("2*volume_sma(20)")
    assert (v.kind, v.period, v.mult) == ("volume_sma", 20, 2.0)
    w = parse_operand("min(rsi(14), 50)")
    assert (w.kind, w.period, w.window_fn, w.window) == ("rsi", 14, "min", 50)
    o = parse_operand("close[1]")
    assert (o.kind, o.offset) == ("close", 1)
    assert parse_operand("bb_lower(20)").param == 2.0
    assert parse_operand("rsi(14)[160]+5").add == 5.0 and parse_operand("rsi(14)[160]+5").offset == 160
    with pytest.raises(ValueError):
        parse_operand("close^2")


def test_parse_condition_and_exits():
    a, op, b = parse_condition("close crosses_above sma(50)")
    assert op == "crosses_above" and a.kind == "close" and b.period == 50
    assert parse_condition("rsi(14) >= 70")[1] == "ge"
    assert parse_exit("3%", "stop") == {"stop_loss_pct": 3.0}
    assert parse_exit("trailing 5%", "stop") == {"trailing_stop_pct": 5.0}
    assert parse_exit("2*atr(14)", "stop") == {"stop_atr_mult": 2.0, "stop_atr_period": 14}
    assert parse_exit("1.8R", "target") == {"take_profit_r": 1.8}
    with pytest.raises(ValueError):
        parse_exit("1.8R", "stop")


def test_lenient_accepts_state_for_event_and_mirrored_sides():
    ref = [parse_condition("close crosses_above sma(50)")]
    got = [parse_condition("sma(50) < close")]
    assert not conditions_match(ref, got, lenient=False)
    assert conditions_match(ref, got, lenient=True)
    assert conditions_match([parse_condition("rsi(14) < 30")], [parse_condition("30 > rsi(14)")], lenient=False)


def _spec(**kw):
    base = {"testable": True, "assets": ["BTCUSDT"], "timeframe": "1d", "direction": "long",
            "entry": [{"left": {"kind": "rsi", "period": 14}, "op": "lt", "right": {"kind": "const", "value": 30}}],
            "exit": [], "stop_loss_pct": 3.0, "max_hold_bars": 5}
    base.update(kw)
    return base


def test_score_claim_fields_and_routing():
    ref = {"testable": "full", "asset": "BTC", "timeframe": "1d", "direction": "long",
           "entry": ["rsi(14) < 30"], "stop": "3%", "time_exit": 5, "reviewed": True}
    s = score_claim("c1", ref, _spec(), "INCONCLUSIVE")
    assert s.reviewed and all(s.fields.values())
    assert set(s.fields) == {"routing", "asset", "timeframe", "direction", "entry", "stop", "time_exit"}

    s = score_claim("c1", ref, _spec(stop_loss_pct=None, stop_atr_mult=2.0), "DIES")
    assert s.fields["stop"] is False

    # compiled fine but the asset has no data: final verdict decides routing
    s = score_claim("c2", {"testable": "no"}, _spec(), "UNTESTABLE")
    assert s.fields == {"routing": True}
    s = score_claim("c3", {"testable": "full"}, _spec(), "UNTESTABLE")
    assert s.fields == {"routing": False} and s.notes


def test_integer_calendar_ranges_are_equivalent():
    ref = [parse_condition("day_of_week >= 2"), parse_condition("day_of_week <= 2")]
    got = [parse_condition("day_of_week > 1.5"), parse_condition("day_of_week < 2.5")]
    assert conditions_match(ref, got, lenient=False)
    assert not conditions_match(ref, [parse_condition("day_of_week > 2.5")] + got[1:], lenient=False)
