"""ClaimSpec: the typed contract between the model and the engine.

Muse Spark never writes backtest code. It compiles a trading claim (text,
chart screenshot, video clip) into this schema via constrained decoding, and
a deterministic engine runs it. That keeps model output out of the P&L path,
which is where LLM-written backtests usually lie (lookahead, off-by-one fills).

Design constraints from the Meta Model API structured-output docs:
no recursive schemas, nesting depth <= 10. Conditions are therefore a flat
AND-list for entry and an OR-list for exit -- enough for the overwhelming
majority of retail claims, and anything it can't express is flagged
UNTESTABLE with a reason instead of being silently approximated.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class OperandKind(str, Enum):
    CONST = "const"            # a fixed number, e.g. RSI threshold 30
    CLOSE = "close"
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    VOLUME = "volume"
    SMA = "sma"                # simple moving average of close, `period`
    EMA = "ema"                # exponential moving average of close, `period`
    RSI = "rsi"                # Wilder RSI, `period`
    PCT_CHANGE = "pct_change"  # % change of close over `period` bars
    HIGHEST_HIGH = "highest_high"  # highest high of the PREVIOUS `period` bars
    LOWEST_LOW = "lowest_low"      # lowest low of the PREVIOUS `period` bars
    VOLUME_SMA = "volume_sma"
    ATR = "atr"
    DAY_OF_WEEK = "day_of_week"    # 0=Mon .. 6=Sun (UTC)
    HOUR_UTC = "hour_utc"
    WMA = "wma"                    # linearly weighted moving average of close, `period`
    BB_UPPER = "bb_upper"          # Bollinger: sma(period) + param * stdev(period); param default 2
    BB_LOWER = "bb_lower"
    MACD_LINE = "macd_line"        # ema(period, default 12) - ema(param, default 26)
    MACD_SIGNAL = "macd_signal"    # ema(9) of macd_line
    MACD_HIST = "macd_hist"        # macd_line - macd_signal
    SUPERTREND = "supertrend"      # supertrend line, ATR `period` (default 10) x `param` (default 3); close > line = uptrend
    ADX = "adx"                    # Wilder ADX, `period`


PERIOD_KINDS = {
    OperandKind.SMA, OperandKind.EMA, OperandKind.RSI, OperandKind.PCT_CHANGE,
    OperandKind.HIGHEST_HIGH, OperandKind.LOWEST_LOW, OperandKind.VOLUME_SMA,
    OperandKind.ATR, OperandKind.WMA, OperandKind.BB_UPPER, OperandKind.BB_LOWER, OperandKind.ADX,
}
DEFAULT_PERIOD = {OperandKind.MACD_LINE: 12, OperandKind.MACD_SIGNAL: 12, OperandKind.MACD_HIST: 12,
                  OperandKind.SUPERTREND: 10}
DEFAULT_PARAM = {OperandKind.BB_UPPER: 2.0, OperandKind.BB_LOWER: 2.0, OperandKind.MACD_LINE: 26.0,
                 OperandKind.MACD_SIGNAL: 26.0, OperandKind.MACD_HIST: 26.0, OperandKind.SUPERTREND: 3.0}


class Operand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: OperandKind
    period: Optional[int] = Field(
        default=None, description="Lookback in bars. Required for sma/ema/rsi/pct_change/highest_high/lowest_low/volume_sma/atr."
    )
    value: Optional[float] = Field(
        default=None, description="Required when kind=const. Percentages as plain numbers: 5 means 5%."
    )
    multiplier: float = Field(
        default=1.0, description="Scales the operand, e.g. 1.5 x volume_sma for 'volume 50% above average'."
    )
    param: Optional[float] = Field(
        default=None, description="Second setting: Bollinger std-dev width (default 2), MACD slow length "
                                  "(default 26, fast = period default 12, signal fixed 9), Supertrend ATR "
                                  "multiplier (default 3, ATR length = period default 10). Null otherwise."
    )

    @model_validator(mode="after")
    def _check(self) -> "Operand":
        if self.kind == OperandKind.CONST and self.value is None:
            raise ValueError("const operand needs value")
        if self.kind in PERIOD_KINDS and (self.period is None or self.period < 1):
            raise ValueError(f"{self.kind.value} operand needs period >= 1")
        if self.kind in DEFAULT_PERIOD and self.period is None:
            self.period = DEFAULT_PERIOD[self.kind]
        if self.kind in DEFAULT_PARAM and self.param is None:
            self.param = DEFAULT_PARAM[self.kind]
        if self.kind in (OperandKind.MACD_LINE, OperandKind.MACD_SIGNAL, OperandKind.MACD_HIST) \
                and self.param is not None and self.param <= self.period:
            raise ValueError("MACD slow length (param) must exceed fast length (period)")
        if self.period is not None and self.period > 2000:
            raise ValueError("period above 2000 bars is not supported")
        return self

    def label(self) -> str:
        base = self.kind.value
        if self.kind == OperandKind.CONST:
            base = f"{self.value:g}"
        elif self.period is not None:
            base = f"{base}({self.period}{f',{self.param:g}' if self.param is not None else ''})"
        if self.multiplier != 1.0:
            base = f"{self.multiplier:g}*{base}"
        return base


class Op(str, Enum):
    GT = "gt"
    LT = "lt"
    CROSSES_ABOVE = "crosses_above"
    CROSSES_BELOW = "crosses_below"


class Condition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    left: Operand
    op: Op
    right: Operand

    def label(self) -> str:
        sym = {"gt": ">", "lt": "<", "crosses_above": "x^", "crosses_below": "xv"}[self.op.value]
        return f"{self.left.label()} {sym} {self.right.label()}"


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"


class Timeframe(str, Enum):
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"


class SourceType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"


class ClaimedPerformance(BaseModel):
    """What the claimant says the strategy does. Null when not stated."""
    model_config = ConfigDict(extra="forbid")

    win_rate_pct: Optional[float] = None
    total_return_pct: Optional[float] = None
    avg_return_per_trade_pct: Optional[float] = None
    period_start: Optional[str] = Field(default=None, description="ISO date YYYY-MM-DD if the claim names a period")
    period_end: Optional[str] = Field(default=None, description="ISO date YYYY-MM-DD")
    other: Optional[str] = Field(default=None, description="Any other stated performance claim, verbatim-short")


class FreeParameter(BaseModel):
    """A number the claim leaves vague ('a short MA', 'oversold').

    The engine sweeps `candidates` and charges a multiple-testing penalty for
    every extra variant tried -- vagueness is not free.
    """
    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Human label, e.g. 'fast MA length'")
    target: str = Field(
        description="Path into the spec: 'entry[0].left.period', 'exit[1].right.value', 'stop_loss_pct', 'take_profit_pct' or 'max_hold_bars'"
    )
    candidates: list[float] = Field(description="2-6 plausible values a reader could reasonably have meant")


class Evidence(BaseModel):
    """Where a rule element comes from. Checked against the source by falsify.grounding."""
    model_config = ConfigDict(extra="forbid")

    field: str = Field(description="entry[i], exit[i], stop_loss_pct, take_profit_pct, max_hold_bars, assets, "
                                   "timeframe, direction, or claimed.<name>")
    quote: Optional[str] = Field(default=None, description="The claimant's exact words that state this, copied "
                                                           "verbatim from the source or transcript. Null if assumed.")
    assumed: bool = Field(description="True if the claim never states this and you chose it")


class ClaimSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_summary: str = Field(description="One neutral sentence restating the claim. No author handles.")
    source_type: SourceType
    testable: bool = Field(description="False if the claim cannot be turned into mechanical rules")
    untestable_reason: Optional[str] = Field(default=None, description="Required when testable=false")

    assets: list[str] = Field(default_factory=list, description="Binance spot symbols, e.g. BTCUSDT")
    timeframe: Timeframe = Timeframe.D1
    direction: Direction = Direction.LONG

    entry: list[Condition] = Field(default_factory=list, description="ALL must be true on a bar close to enter")
    exit: list[Condition] = Field(default_factory=list, description="ANY true on a bar close exits")
    stop_loss_pct: Optional[float] = Field(default=None, description="e.g. 3 for a 3% stop")
    take_profit_pct: Optional[float] = None
    max_hold_bars: Optional[int] = Field(default=None, description="Time exit in bars")

    free_parameters: list[FreeParameter] = Field(default_factory=list)
    claimed: ClaimedPerformance = Field(default_factory=ClaimedPerformance)
    assumptions: list[str] = Field(
        default_factory=list,
        description="Every interpretation choice made while compiling the claim. Be explicit.",
    )
    evidence: list[Evidence] = Field(
        default_factory=list,
        description="One row per rule element: every entry[i], every exit[i], each of stop_loss_pct / "
                    "take_profit_pct / max_hold_bars that is set, plus assets, timeframe and direction.",
    )

    @model_validator(mode="after")
    def _check(self) -> "ClaimSpec":
        if not self.testable:
            if not self.untestable_reason:
                raise ValueError("untestable_reason required when testable=false")
            return self
        if not self.assets:
            raise ValueError("testable claim needs at least one asset")
        if not self.entry:
            raise ValueError("testable claim needs at least one entry condition")
        if not self.exit and self.stop_loss_pct is None and self.take_profit_pct is None and self.max_hold_bars is None:
            raise ValueError("testable claim needs an exit rule (condition, stop, target or time exit)")
        self.assets = [a.upper().replace("/", "").replace("-", "") for a in self.assets]
        return self


def model_json_schema() -> dict:
    """Schema handed to Meta Model API structured output."""
    return ClaimSpec.model_json_schema()
