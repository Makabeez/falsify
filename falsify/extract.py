"""Muse Spark: compile a trading claim (text, chart screenshot, video clip) into a ClaimSpec.

Two model passes, both schema-constrained:
  1. compile  -- source -> ClaimSpec (structured output). If the result fails
                 our own validators, the error goes back to the model and it
                 retries (max 2 repairs). The model never writes code.
  2. audit    -- the source and the compiled spec go back to Muse Spark,
                 which lists every place the spec departs from what the
                 claimant actually said. Findings ship with the verdict.

Meta Model API is OpenAI-compatible; we use the OpenAI SDK with a different
base_url. Key: MODEL_API_KEY. Nothing else in the product path calls a
non-Meta model.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from . import grounding, sanity
from .data import UNIVERSE
from .spec import ClaimSpec, SourceType, model_json_schema

BASE_URL = os.environ.get("MODEL_API_BASE", "https://api.meta.ai/v1")
MODEL = os.environ.get("FALSIFY_MODEL", "muse-spark-1.3")
# Docs flag audio understanding as degraded on 1.3; clips carry the pitch in speech.
VIDEO_MODEL = os.environ.get("FALSIFY_VIDEO_MODEL", "muse-spark-1.2")
MAX_REPAIRS = 2

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
VIDEO_EXT = {".mp4"}

COMPILE_PROMPT = f"""You compile trading claims into mechanical rules so they can be tested.

You receive a claim as text, a screenshot (tweet, chart with annotations, Discord post) or a short video.
Return a ClaimSpec. You are a compiler, not a critic: represent what the claimant said, as faithfully as
the schema allows. A separate engine runs the test.

Rules:
- Rules are evaluated on bar CLOSES; trades fill at the next bar's open. Do not try to encode intrabar logic.
- Assets: Binance spot USDT symbols. Bundled universe: {", ".join(UNIVERSE)}. If the claim is about another
  asset (stocks, indices, forex, a small-cap token), still name it as SYMBOLUSDT or the ticker; the engine
  will report it as untestable for lack of data. Never substitute a different asset.
- If the claim names no asset or timeframe, choose the most natural reading and record it in assumptions.
- Vague numbers ("oversold", "a short moving average", "a big drop", "tight stop") must still get a concrete
  value in the rules, AND be listed in free_parameters with 3-5 candidates a reasonable reader might mean,
  including the value you used. target paths look like entry[0].right.value, entry[1].left.period,
  exit[0].right.period, stop_loss_pct, take_profit_pct, max_hold_bars.
- Percentages are plain numbers: 3 means 3%. pct_change is in percent.
- day_of_week: 0=Monday ... 6=Sunday, UTC. Express "on Friday" as day_of_week gt 3.5 AND day_of_week lt 4.5.
  "On the 4th of the month" is day_of_month gt 3.5 AND day_of_month lt 4.5; "on the last day of the month" is
  days_left_in_month lt 0.5. Calendar rules need daily or shorter bars.
- Do not correct the claimant's text. If a word looks like a typo (e.g. "4th" where "4h" would make sense), compile
  what is written, and say in assumptions which other reading was possible.
- "Price above X" uses close. Breakouts use highest_high / lowest_low, which already look at the PREVIOUS N bars
  (the current bar excluded): "close above the 20-day high" is close gt highest_high(20) with offset 0. Adding
  offset 1 would lag it a second time.
- Indicators available: sma, ema, wma, rsi, atr, adx, pct_change, highest_high, lowest_low, volume_sma,
  bb_upper / bb_lower (period, param = std width), macd_line / macd_signal / macd_hist (period = fast,
  param = slow, signal 9), supertrend (period = ATR length, param = multiplier; close above it = uptrend),
  day_of_week, hour_utc, day_of_month (1-31), days_left_in_month (0 = last day of the month). Use the claim's settings; when it gives none, use the indicator's usual default and
  list that setting in free_parameters. Never swap one indicator for another (WMA is not SMA).
- Exits: stop_loss_pct (fixed %), stop_atr_mult (k x ATR(stop_atr_period) at the signal bar), take_profit_pct,
  take_profit_atr_mult, take_profit_r ("2R", "1:2 risk-reward": a multiple of the initial stop distance, needs a
  stop), trailing_stop_pct / trailing_stop_atr_mult (from the best close since entry), max_hold_bars. Use one stop
  type and one target type.
- Operand extras: offset = value N bars ago ("previous RSI" is offset 1; "rising for 3 bars" is x > x[1] AND
  x[1] > x[2] AND x[2] > x[3]); window_fn min/max with window = lowest/highest of that operand over the last N bars
  including the current one ("RSI at its 50-bar low": rsi le min(rsi, 50)); add = a constant added after the
  multiplier ("RSI 160 bars ago plus 5": rsi, offset 160, add 5). Comparisons: gt, lt, ge, le, crosses_above,
  crosses_below.
- All entry conditions must be able to hold on the SAME bar. A sequence ("hit its 50-bar low, then rose for 3
  bars") needs offsets: the low is 3 bars back, rsi[3] le min(rsi, 50)[3], plus the three rises. "At its low"
  and "rising" on the same bar contradict each other and the rule never fires. Every entry is checked against
  the full price history; a rule that never fires is sent back.
- If the claim uses something the schema cannot express (an averaging / DCA ladder, position sizing, partial
  exits, moving a stop to breakeven, pyramiding, multi-timeframe conditions, an indicator not listed above, a
  proprietary score), compile the closest testable core
  and add an assumption that starts with "NOT EXPRESSIBLE:" naming exactly what was left out. If what is left
  out is the heart of the strategy, set testable=false instead.
- Every testable spec needs an exit: a condition, a stop, a target, a trailing stop or max_hold_bars. If the
  claim gives no exit at all, use the most common reading for that setup and say so in assumptions.
- Every testable spec needs assets, even when the claim is about something outside the bundled universe:
  name it (EURUSD, SPX, XAUUSD) and keep testable=true; the engine reports it as lacking data.
- The engine always tests the rule over ALL available history, starting flat, one position at a time. The test
  window and the starting position are not part of the rule: do not mark them NOT EXPRESSIBLE, and put any
  period the claimant cites in `claimed`.
- Record any performance the claimant states (win rate, returns, period) in `claimed`. Dates as YYYY-MM-DD.
- testable=false only when no mechanical rule can be recovered (e.g. "buy strong projects", "follow smart
  money", discretionary chart reading with no stated trigger, or rules that need data we don't have such as
  funding rates, order flow, sentiment or news). Give the reason in one sentence.
- claim_summary: one neutral sentence. Never include usernames, handles or names of the author.
- assumptions: list every interpretation you made. The test is only as honest as this list.
- evidence: one row for EVERY rule element: each entry[i], each exit[i], each stop, target, trailing stop and
  max_hold_bars field that you set (field = its name, e.g. stop_atr_mult), plus assets, timeframe and direction. If the claimant states it, copy their exact
  words into quote (short: only the words that state it, copied verbatim from the CLAIM or TRANSCRIPT, never from
  CONTEXT) and set assumed=false. If you chose it, set quote=null and assumed=true. Quotes are checked
  mechanically against the source; a paraphrase is rejected.
"""

TRANSCRIBE_PROMPT = """Transcribe this media exactly, for an audit trail. Do not interpret or summarize.
visible_text: every piece of text visible in the image or video frames (post text, chart labels, annotations,
indicator names and settings, numbers, captions), verbatim, in reading order. Leave out usernames, handles,
profile names and avatars.
speech: every spoken word in the video, verbatim. Empty for images or silent clips.
chart: one or two neutral sentences on what any chart shows (asset, timeframe, indicators, marked levels or
arrows), only what is literally visible."""

AUDIT_PROMPT = """You audit a compiled trading rule against its source.

You get the original claim (text, image or video) and a JSON ClaimSpec that another pass compiled from it.
List every place where the spec departs from what the claimant actually said or showed: wrong asset,
wrong timeframe, wrong threshold, missing condition, invented condition, wrong exit, wrong direction,
misread chart annotation, one indicator swapped for another. The engine always tests the rule over all
available history starting flat; the test window, the starting position, compounding and position size are
NOT issues (other checks cover the claimant's period). Interpretation choices that the spec already
lists in `assumptions` are not issues unless the choice is clearly not what the claimant meant. A part of the
claim marked "NOT EXPRESSIBLE" is still an issue if leaving it out changes what is being tested. severity is
"major" when the difference could change the verdict, else "minor". faithful=true only if a neutral reader
would accept the spec as a fair encoding of the claim."""


class Transcript(BaseModel):
    model_config = ConfigDict(extra="forbid")
    visible_text: str
    speech: str
    chart: str

    def as_text(self) -> str:
        parts = [self.visible_text.strip()]
        if self.speech.strip():
            parts.append(self.speech.strip())
        return "\n".join(p for p in parts if p)


class AuditIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: str = Field(description="Spec field, e.g. entry[0].right.value")
    problem: str
    severity: str = Field(description="'major' if it could flip the verdict, else 'minor'")


class Audit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    faithful: bool
    issues: list[AuditIssue] = Field(default_factory=list)


@dataclass
class Source:
    kind: SourceType
    text: str | None = None       # the claimant's words (text sources only)
    data_url: str | None = None   # image/video as a data URL
    name: str = "claim"
    note: str | None = None       # operator context ("posted on X, Oct 2026"); never quotable
    transcript: "Transcript | None" = None  # filled by the first compile of a media source

    @classmethod
    def from_text(cls, text: str, note: str | None = None) -> "Source":
        return cls(SourceType.TEXT, text=text, note=note)

    @classmethod
    def from_path(cls, path: str | Path, note: str | None = None) -> "Source":
        p = Path(path)
        ext = p.suffix.lower()
        if ext in {".txt", ".md"}:
            return cls(SourceType.TEXT, text=p.read_text(encoding="utf-8"), name=p.name, note=note)
        if ext in IMAGE_EXT:
            kind = SourceType.IMAGE
        elif ext in VIDEO_EXT:
            kind = SourceType.VIDEO
        else:
            raise ValueError(f"unsupported file type {ext}: use text, {sorted(IMAGE_EXT)} or .mp4")
        mime = mimetypes.guess_type(p.name)[0] or ("video/mp4" if kind == SourceType.VIDEO else "image/png")
        b64 = base64.b64encode(p.read_bytes()).decode()
        return cls(kind, data_url=f"data:{mime};base64,{b64}", name=p.name, note=note)

    def content(self, instruction: str, transcript: str | None = None) -> list[dict]:
        parts: list[dict] = [{"type": "text", "text": instruction}]
        if self.text:
            parts.append({"type": "text", "text": f"CLAIM:\n{self.text}"})
        if self.note:
            parts.append({"type": "text", "text": f"CONTEXT FROM THE USER (not the claimant's words, never quote it):\n{self.note}"})
        if transcript:
            parts.append({"type": "text", "text": f"TRANSCRIPT OF THE MEDIA (quote evidence from this, verbatim):\n{transcript}"})
        if self.kind == SourceType.IMAGE:
            parts.append({"type": "image_url", "image_url": {"url": self.data_url}})
        elif self.kind == SourceType.VIDEO:
            parts.append({"type": "video_url", "video_url": {"url": self.data_url}})
        return parts


def _client():
    try:
        from openai import OpenAI
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("pip install openai") from e
    key = os.environ.get("MODEL_API_KEY")
    if not key:
        raise RuntimeError("MODEL_API_KEY is not set (Meta Model API key, see .env.example)")
    return OpenAI(base_url=BASE_URL, api_key=key)


MAX_TOKENS = int(os.environ.get("FALSIFY_MAX_TOKENS", "32000"))
REASONING = os.environ.get("FALSIFY_REASONING", "medium")


def _structured(client, model: str, messages: list[dict], name: str, schema: dict) -> tuple[str, dict]:
    """One schema-constrained call. Reasoning shares max_tokens with the JSON, so if the
    budget runs out we retry once with reasoning dialled down before giving up."""
    usage_all: list[dict] = []
    for effort in (REASONING, "low"):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=MAX_TOKENS,
            reasoning_effort=effort,
            response_format={"type": "json_schema", "json_schema": {"name": name, "schema": schema}},
        )
        choice = resp.choices[0]
        usage = resp.usage.model_dump() if resp.usage else {}
        usage_all.append(dict(usage, reasoning_effort=effort))
        if choice.finish_reason != "length":
            merged = dict(usage_all[-1])
            merged["total_tokens"] = sum((u.get("total_tokens") or 0) for u in usage_all)
            merged["calls"] = len(usage_all)
            return choice.message.content or "", merged
    raise RuntimeError(f"model ran out of output budget ({MAX_TOKENS} tokens) twice, even at low reasoning effort")


def transcribe(source: Source, model: str | None = None, client=None, ledger: list | None = None) -> Transcript:
    """Media -> verbatim transcript. The quotes in a compiled spec are checked against this."""
    client = client or _client()
    model = model or (VIDEO_MODEL if source.kind == SourceType.VIDEO else MODEL)
    t0 = time.time()
    raw, usage = _structured(client, model, [{"role": "user", "content": source.content(TRANSCRIBE_PROMPT)}],
                         "Transcript", Transcript.model_json_schema())
    if ledger is not None:
        ledger.append({"step": "transcribe", "seconds": round(time.time() - t0, 1),
                       "tokens": (usage or {}).get("total_tokens") or 0})
    return Transcript.model_validate_json(raw)


def _noop(stage: str) -> None:
    pass


def compile_claim(source: Source, model: str | None = None, client=None,
                  prior: tuple[str, str] | None = None, on_stage=_noop) -> tuple[ClaimSpec, dict]:
    """Source -> validated, quote-checked ClaimSpec. Returns (spec, trace).

    Each attempt is checked twice outside the model: pydantic validation, then the
    deterministic quote check in falsify.grounding. Either failure goes back to the model
    as a repair request. Grounding problems that survive the last repair are not fatal:
    they ship in trace["grounding"] and are printed with the verdict.
    """
    client = client or _client()
    model = model or (VIDEO_MODEL if source.kind == SourceType.VIDEO else MODEL)
    trace: dict = {"model": model, "attempts": []}
    transcript_text = None
    if source.kind == SourceType.TEXT:
        ground_text = source.text or ""
    else:
        if source.transcript is None:
            on_stage("transcribe")
            source.transcript = transcribe(source, model=model, client=client)
        trace["transcript"] = source.transcript.model_dump()
        transcript_text = source.transcript.as_text()
        ground_text = transcript_text
    messages: list[dict] = [{"role": "user", "content": source.content(COMPILE_PROMPT, transcript_text)}]
    if prior is not None:  # (previous ClaimSpec JSON, feedback on it)
        messages += [{"role": "assistant", "content": prior[0]}, {"role": "user", "content": prior[1]}]
    last_spec: ClaimSpec | None = None
    last_grounding = None
    for attempt in range(MAX_REPAIRS + 1):
        on_stage("compile" if attempt == 0 and prior is None else "repair")
        raw, usage = _structured(client, model, messages, "ClaimSpec", model_json_schema())
        trace["attempts"].append({"usage": usage})
        try:
            spec = ClaimSpec.model_validate_json(raw)
        except ValidationError as e:
            trace["attempts"][-1]["error"] = str(e)
            if attempt == MAX_REPAIRS:
                if last_spec is not None:
                    break
                raise RuntimeError(f"model output failed validation {MAX_REPAIRS + 1} times: {e}") from e
            feedback = f"That ClaimSpec failed validation:\n{e}\nReturn a corrected ClaimSpec."
        else:
            spec.source_type = source.kind
            on_stage("quote_check")
            g = grounding.check(spec, ground_text)
            fires = sanity.entry_problems(spec)
            last_spec, last_grounding = spec, g
            trace["attempts"][-1]["grounding_problems"] = g.problems
            if fires:
                trace["attempts"][-1]["sanity_problems"] = fires
            if (g.ok and not fires) or attempt == MAX_REPAIRS:
                break
            parts = []
            if g.problems:
                parts.append("its evidence failed the quote check:\n- " + "\n- ".join(g.problems))
            if fires:
                parts.append("it was run against the price history:\n- " + "\n- ".join(fires))
            feedback = ("The ClaimSpec is valid but " + "\nAlso, ".join(parts)
                        + "\nReturn a corrected ClaimSpec. Change only what these problems require.")
        messages += [{"role": "assistant", "content": raw}, {"role": "user", "content": feedback}]
    assert last_spec is not None and last_grounding is not None
    trace["grounding"] = last_grounding.summary()
    return last_spec, trace


def audit_claim(source: Source, spec: ClaimSpec, model: str | None = None, client=None,
                ledger: list | None = None) -> Audit:
    client = client or _client()
    t0 = time.time()
    model = model or (VIDEO_MODEL if source.kind == SourceType.VIDEO else MODEL)
    content = source.content(AUDIT_PROMPT)
    content.append({"type": "text", "text": "COMPILED SPEC:\n" + spec.model_dump_json(indent=2)})
    raw, usage = _structured(client, model, [{"role": "user", "content": content}], "Audit", Audit.model_json_schema())
    if ledger is not None:
        ledger.append({"step": "audit", "seconds": round(time.time() - t0, 1), "tokens": (usage or {}).get("total_tokens") or 0})
    return Audit.model_validate_json(raw)


def _compile_cost(step: str, trace: dict, t0: float) -> dict:
    tokens = sum(((a.get("usage") or {}).get("total_tokens") or 0) for a in trace.get("attempts", []))
    return {"step": step, "seconds": round(time.time() - t0, 1), "tokens": tokens,
            "quote_repairs": max(0, len(trace.get("attempts", [])) - 1)}


def major_issues(audit: Audit) -> list[AuditIssue]:
    return [i for i in audit.issues if i.severity.strip().lower() == "major"]


def compile_and_audit(source: Source, model: str | None = None, client=None,
                      on_stage=_noop) -> tuple[ClaimSpec, Audit, dict]:
    """Compile, audit, and give the compiler one chance to fix what the auditor found.

    The verifier runs outside the compiler's control. If the audit still finds major
    issues after one repair round, the spec is kept but reported as an approximation.
    """
    client = client or _client()
    ledger: list[dict] = []  # what verification costs: seconds and tokens per step
    if source.kind != SourceType.TEXT and source.transcript is None:
        on_stage("transcribe")
        source.transcript = transcribe(source, model=model, client=client, ledger=ledger)
    t0 = time.time()
    spec, trace = compile_claim(source, model=model, client=client, on_stage=on_stage)
    ledger.append(_compile_cost("compile", trace, t0))
    on_stage("audit")
    audit = audit_claim(source, spec, model=model, client=client, ledger=ledger)
    trace["audits"] = [audit.model_dump()]
    majors = major_issues(audit)
    if majors:
        feedback = ("An independent audit compared your ClaimSpec with the source and found these major issues:\n- "
                    + "\n- ".join(f"{i.field}: {i.problem}" for i in majors)
                    + "\nFix every issue the schema can express. For anything it cannot express, keep the closest "
                      "testable core and say so in an assumption starting with \"NOT EXPRESSIBLE:\" -- or set "
                      "testable=false if that part is the heart of the strategy. Return the full corrected ClaimSpec.")
        t1 = time.time()
        try:
            spec2, trace2 = compile_claim(source, model=model, client=client,
                                          prior=(spec.model_dump_json(), feedback), on_stage=on_stage)
        except RuntimeError as e:
            # The repair itself failed validation every time. Keep the first-pass spec and its audit: its major
            # issues stay on the record, so the result is escalated to a human instead of lost.
            ledger.append({"step": "audit repair", "seconds": round(time.time() - t1, 1), "tokens": 0, "failed": True})
            trace["audit_repair"] = True
            trace["audit_repair_failed"] = str(e)[:500]
            trace["cost"] = ledger
            return spec, audit, trace
        ledger.append(_compile_cost("audit repair", trace2, t1))
        on_stage("audit")
        audit2 = audit_claim(source, spec2, model=model, client=client, ledger=ledger)
        trace2["first_pass"] = {"attempts": trace["attempts"], "grounding": trace.get("grounding")}
        trace2["audits"] = trace["audits"] + [audit2.model_dump()]
        trace2["audit_repair"] = True
        spec, audit, trace = spec2, audit2, trace2
    trace["cost"] = ledger
    return spec, audit, trace


def fidelity(spec: ClaimSpec, audit: Audit | dict | None) -> dict:
    """exact | approximation | unknown, with the reasons."""
    gaps = [a for a in spec.assumptions if a.strip().upper().startswith("NOT EXPRESSIBLE")]
    if audit is None:
        return {"level": "unknown" if not gaps else "approximation", "gaps": gaps, "audit_majors": []}
    issues = audit.issues if isinstance(audit, Audit) else [AuditIssue(**i) for i in audit.get("issues", [])]
    majors = [f"{i.field}: {i.problem}" for i in issues if i.severity.strip().lower() == "major"]
    level = "approximation" if (gaps or majors) else "exact"
    return {"level": level, "gaps": gaps, "audit_majors": majors}


def save(spec: ClaimSpec, path: str | Path, audit: Audit | None = None, trace: dict | None = None) -> None:
    out = {"spec": spec.model_dump(mode="json")}
    if audit is not None:
        out["audit"] = audit.model_dump(mode="json")
    if trace is not None:
        out["trace"] = trace
    Path(path).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


def load_spec(path: str | Path) -> tuple[ClaimSpec, dict]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if "spec" in d:
        return ClaimSpec.model_validate(d["spec"]), d
    return ClaimSpec.model_validate(d), {"spec": d}
