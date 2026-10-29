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
    DAY_OF_MONTH = "day_of_month"              # 1..31 (UTC)
    DAYS_LEFT_IN_MONTH = "days_left_in_month"  # 0 on the last day of the month
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


class WindowFn(str, Enum):
    MIN = "min"
    MAX = "max"


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
    add: float = Field(
        default=0.0, description="Constant added after scaling, in the operand's own units: 'RSI of 160 bars ago "
                                 "plus 5 points' is rsi with offset 160 and add 5; 'close 2% above the SMA' is "
                                 "multiplier 1.02, not add."
    )
    param: Optional[float] = Field(
        default=None, description="Second setting: Bollinger std-dev width (default 2), MACD slow length "
                                  "(default 26, fast = period default 12, signal fixed 9), Supertrend ATR "
                                  "multiplier (default 3, ATR length = period default 10). Null otherwise."
    )
    window_fn: Optional[WindowFn] = Field(
        default=None, description="Rolling extreme of this operand: 'min' or 'max' over the last `window` bars "
                                  "INCLUDING the current one. E.g. RSI at its 50-bar low: rsi(14) le min(rsi(14), 50)."
    )
    window: Optional[int] = Field(default=None, description="Bars for window_fn.")
    offset: int = Field(
        default=0, description="Value this many bars ago: 1 = previous bar. 'RSI rising for 3 bars' is "
                               "rsi > rsi[1] AND rsi[1] > rsi[2] AND rsi[2] > rsi[3]."
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
        if (self.window_fn is None) != (self.window is None):
            raise ValueError("window_fn and window go together")
        if self.window is not None and not 1 <= self.window <= 2000:
            raise ValueError("window must be 1..2000 bars")
        if not 0 <= self.offset <= 500:
            raise ValueError("offset must be 0..500 bars")
        if self.kind == OperandKind.CONST and (self.window_fn is not None or self.offset or self.add):
            raise ValueError("a constant has no window, offset or add; put the number in value")
        return self

    def label(self) -> str:
        base = self.kind.value
        if self.kind == OperandKind.CONST:
            base = f"{self.value:g}"
        elif self.period is not None:
            base = f"{base}({self.period}{f',{self.param:g}' if self.param is not None else ''})"
        if self.window_fn is not None:
            base = f"{self.window_fn.value}({base}, {self.window})"
        if self.offset:
            base = f"{base}[{self.offset}]"
        if self.multiplier != 1.0:
            base = f"{self.multiplier:g}*{base}"
        if self.add:
            base = f"{base}{self.add:+g}"
        return base


class Op(str, Enum):
    GT = "gt"
    LT = "lt"
    GE = "ge"
    LE = "le"
    CROSSES_ABOVE = "crosses_above"
    CROSSES_BELOW = "crosses_below"


class Condition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    left: Operand
    op: Op
    right: Operand

    def label(self) -> str:
        sym = {"gt": ">", "lt": "<", "ge": ">=", "le": "<=", "crosses_above": "x^", "crosses_below": "xv"}[self.op.value]
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
        description="Path into the spec: 'entry[0].left.period', 'exit[1].right.value', 'entry[0].left.window', "
                    "'stop_loss_pct', 'stop_atr_mult', 'take_profit_pct', 'take_profit_atr_mult', 'take_profit_r', "
                    "'trailing_stop_pct', 'trailing_stop_atr_mult' or 'max_hold_bars'"
    )
    candidates: list[float] = Field(description="2-6 plausible values a reader could reasonably have meant")


class Evidence(BaseModel):
    """Where a rule element comes from. Checked against the source by falsify.grounding."""
    model_config = ConfigDict(extra="forbid")

    field: str = Field(description="entry[i], exit[i], any exit field that is set (stop_loss_pct, stop_atr_mult, "
                                   "take_profit_pct, take_profit_atr_mult, take_profit_r, trailing_stop_pct, "
                                   "trailing_stop_atr_mult, max_hold_bars), assets, timeframe, direction, or claimed.<name>")
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
    stop_loss_pct: Optional[float] = Field(default=None, description="Fixed stop, % from entry: 3 = 3%")
    stop_atr_mult: Optional[float] = Field(
        default=None, description="Fixed stop at entry minus mult x ATR(stop_atr_period) (plus, for shorts), "
                                  "ATR taken at the signal bar. Use instead of stop_loss_pct, not with it.")
    stop_atr_period: int = Field(default=14, description="ATR length for every ATR-based stop, target or trail")
    take_profit_pct: Optional[float] = Field(default=None, description="Fixed target, % from entry")
    take_profit_atr_mult: Optional[float] = Field(default=None, description="Target at entry + mult x ATR")
    take_profit_r: Optional[float] = Field(
        default=None, description="Target as a multiple of the initial stop distance ('2R', '1.8:1 reward to "
                                  "risk'). Needs stop_loss_pct or stop_atr_mult.")
    trailing_stop_pct: Optional[float] = Field(
        default=None, description="Trailing stop, % below the highest close since entry (above the lowest, for shorts)")
    trailing_stop_atr_mult: Optional[float] = Field(
        default=None, description="Trailing stop, mult x ATR below the highest close since entry")
    max_hold_bars: Optional[int] = Field(default=None, description="Time exit in bars")

    free_parameters: list[FreeParameter] = Field(default_factory=list)
    claimed: ClaimedPerformance = Field(default_factory=ClaimedPerformance)
    assumptions: list[str] = Field(
        default_factory=list,
        description="Every interpretation choice made while compiling the claim. Be explicit.",
    )
    evidence: list[Evidence] = Field(
        default_factory=list,
        description="One row per rule element: every entry[i], every exit[i], every exit field that is set "
                    "(stops, targets, trailing stops, max_hold_bars), plus assets, timeframe and direction.",
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
        exits = (self.stop_loss_pct, self.stop_atr_mult, self.take_profit_pct, self.take_profit_atr_mult,
                 self.take_profit_r, self.trailing_stop_pct, self.trailing_stop_atr_mult, self.max_hold_bars)
        if not self.exit and all(x is None for x in exits):
            raise ValueError("testable claim needs an exit rule (condition, stop, target, trailing stop or time exit)")
        if self.stop_loss_pct is not None and self.stop_atr_mult is not None:
            raise ValueError("use stop_loss_pct or stop_atr_mult, not both")
        if sum(x is not None for x in (self.take_profit_pct, self.take_profit_atr_mult, self.take_profit_r)) > 1:
            raise ValueError("use one of take_profit_pct, take_profit_atr_mult, take_profit_r")
        if self.trailing_stop_pct is not None and self.trailing_stop_atr_mult is not None:
            raise ValueError("use trailing_stop_pct or trailing_stop_atr_mult, not both")
        if self.take_profit_r is not None and self.stop_loss_pct is None and self.stop_atr_mult is None:
            raise ValueError("take_profit_r needs an initial stop (stop_loss_pct or stop_atr_mult)")
        for name in ("stop_loss_pct", "stop_atr_mult", "take_profit_pct", "take_profit_atr_mult", "take_profit_r",
                     "trailing_stop_pct", "trailing_stop_atr_mult"):
            v = getattr(self, name)
            if v is not None and v <= 0:
                raise ValueError(f"{name} must be positive")
        if not 1 <= self.stop_atr_period <= 500:
            raise ValueError("stop_atr_period must be 1..500")
        self.assets = [a.upper().replace("/", "").replace("-", "") for a in self.assets]
        return self


def model_json_schema() -> dict:
    """Schema handed to Meta Model API structured output."""
    return ClaimSpec.model_json_schema()
