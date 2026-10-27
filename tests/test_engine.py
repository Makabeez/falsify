"""Pins the timing contract. If any of these fail, every verdict is suspect."""

import json

import numpy as np
import pandas as pd
import pytest

from falsify import data as data_mod
from falsify.engine import Costs, run, with_override
from falsify.spec import ClaimSpec

ZERO = Costs(fee_bps=0, slippage_bps=0)


def frame(opens, highs=None, lows=None, closes=None):
    n = len(opens)
    opens = np.asarray(opens, dtype=float)
    closes = np.asarray(closes if closes is not None else opens, dtype=float)
    highs = np.asarray(highs if highs is not None else np.maximum(opens, closes), dtype=float)
    lows = np.asarray(lows if lows is not None else np.minimum(opens, closes), dtype=float)
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes,
                         "volume": np.ones(n)}, index=idx)


def spec(**kw):
    base = dict(claim_summary="t", source_type="text", testable=True, assets=["BTCUSDT"], timeframe="1d",
                direction="long",
                entry=[{"left": {"kind": "close"}, "op": "gt", "right": {"kind": "const", "value": 100}}],
                exit=[], max_hold_bars=1)
    base.update(kw)
    return ClaimSpec.model_validate(base)


def test_signal_at_close_fills_next_open():
    # close crosses 100 on bar 2; the fill must be bar 3's open, never bar 2's close
    df = frame(opens=[90, 95, 98, 107, 110, 111], closes=[92, 97, 105, 108, 111, 112])
    t = run(spec(), df, ZERO)
    first = t.iloc[0]
    assert first.entry_time == df.index[3]
    assert first.entry_px == 107
    assert first.exit_time == df.index[4] and first.exit_px == 110  # 1-bar time exit at next open


def test_stop_gap_fills_at_open_not_at_stop():
    df = frame(opens=[90, 90, 100, 80, 80], closes=[90, 101, 100, 80, 80],
               highs=[90, 101, 100, 80, 80], lows=[90, 90, 99, 79, 79])
    s = spec(max_hold_bars=None, stop_loss_pct=5)  # stop at 95
    t = run(s, df, ZERO)
    assert t.iloc[0].reason == "stop"
    assert t.iloc[0].exit_px == 80  # gapped through 95, filled at the open


def test_stop_beats_target_inside_one_bar():
    df = frame(opens=[90, 90, 100, 100], closes=[90, 101, 100, 100],
               highs=[90, 101, 120, 100], lows=[90, 90, 80, 100])
    s = spec(max_hold_bars=None, stop_loss_pct=5, take_profit_pct=10)
    t = run(s, df, ZERO)
    assert t.iloc[0].reason == "stop" and t.iloc[0].exit_px == 95


def test_exit_signal_fills_next_open_and_no_same_open_reentry():
    df = frame(opens=[90, 90, 101, 102, 103, 104], closes=[90, 101, 102, 103, 104, 105])
    s = spec(max_hold_bars=None,
             exit=[{"left": {"kind": "close"}, "op": "gt", "right": {"kind": "const", "value": 102.5}}])
    t = run(s, df, ZERO)
    # entry signal bar 1 -> fill bar 2 open 101; exit signal at close of bar 3 (103) -> fill bar 4 open 103
    assert t.iloc[0].entry_px == 101 and t.iloc[0].exit_px == 103
    # bar 3 also had an entry signal; it must not re-enter on the exit open
    if len(t) > 1:
        assert t.iloc[1].entry_time > t.iloc[0].exit_time


def test_short_side_pnl_sign():
    df = frame(opens=[90, 90, 110, 100, 100], closes=[90, 101, 110, 100, 100])
    t = run(spec(direction="short"), df, ZERO)
    assert t.iloc[0].entry_px == 110 and t.iloc[0].exit_px == 100
    assert t.iloc[0].gross_pct == pytest.approx(10.0)


def test_costs_are_charged_both_sides():
    df = frame(opens=[90, 90, 100, 100, 100], closes=[90, 101, 100, 100, 100])
    t = run(spec(), df, Costs(fee_bps=10, slippage_bps=0))
    assert t.iloc[0].gross_pct == pytest.approx(0.0)
    assert t.iloc[0].net_pct == pytest.approx(((1 - 0.001) / (1 + 0.001) - 1) * 100)


@pytest.mark.skipif(("BTCUSDT", "1d") not in data_mod.available(), reason="bundled data missing")
@pytest.mark.parametrize("fixture", ["02_rsi_oversold_eth", "05_breakout_volume_btc4h", "03_buy_10pct_weekly_drop_btc"])
def test_no_lookahead_on_real_data(fixture):
    """Truncating the future must not change any trade that closed before the cut."""
    from pathlib import Path
    raw = json.loads((Path(__file__).parent.parent / "claims" / f"{fixture}.json").read_text())
    s = ClaimSpec.model_validate(raw["spec"])
    df = data_mod.load(s.assets[0], s.timeframe.value)
    full = run(s, df, ZERO)
    for frac in (0.3, 0.55, 0.8):
        k = int(len(df) * frac)
        part = run(s, df.iloc[:k], ZERO)
        cut = df.index[k - 2]
        a = full[full.exit_time <= cut].reset_index(drop=True)
        b = part[part.exit_time <= cut].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)


def test_override_paths():
    s = spec(free_parameters=[{"name": "x", "target": "entry[0].right.value", "candidates": [1, 2]}])
    s2 = with_override(s, "entry[0].right.value", 123)
    assert s2.entry[0].right.value == 123 and s.entry[0].right.value == 100
    s3 = with_override(s, "max_hold_bars", 4.6)
    assert s3.max_hold_bars == 5
    with pytest.raises(ValueError):
        with_override(s, "entry[0].nope", 1)


def test_spec_validation():
    with pytest.raises(ValueError):
        spec(max_hold_bars=None, exit=[])  # no exit rule at all
    with pytest.raises(ValueError):
        ClaimSpec.model_validate({"claim_summary": "x", "source_type": "text", "testable": False})
    with pytest.raises(ValueError):
        spec(entry=[{"left": {"kind": "sma"}, "op": "gt", "right": {"kind": "const", "value": 1}}])  # sma w/o period
