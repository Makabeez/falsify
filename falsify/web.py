"""Falsify web app: paste a post, drop a screenshot or a clip, get a verdict.

    uvicorn falsify.web:app --host 127.0.0.1 --port 8095

Live checks call Meta Model API with the server's MODEL_API_KEY, so they are guarded:
per-IP and daily caps, size limits, a small worker pool, and FALSIFY_LIVE=0 as a kill
switch. Results are cached on disk by content hash, which also makes every result a
shareable permalink (/r/<key>). The corpus gallery re-runs the deterministic battery
on stored specs and never calls the model.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response

from . import data as data_mod
from .battery import attach_fidelity, attach_grounding, falsify
from .spec import ClaimSpec

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
CACHE = Path(os.environ.get("FALSIFY_CACHE", ROOT / "web_cache"))
CORPUS = ROOT / "corpus"
LIVE = os.environ.get("FALSIFY_LIVE", "1") != "0"
PER_IP_PER_HOUR = int(os.environ.get("FALSIFY_PER_IP_PER_HOUR", "6"))
DAILY_LIMIT = int(os.environ.get("FALSIFY_DAILY_LIMIT", "150"))
MAX_TEXT = 4000
MAX_IMAGE = 8 * 1024 * 1024
MAX_VIDEO = 25 * 1024 * 1024
WORKERS = int(os.environ.get("FALSIFY_WORKERS", "2"))
MAX_QUEUE = 8
KEY_RE = re.compile(r"^[0-9a-f]{20}$")
STAGES = ["transcribe", "compile", "quote_check", "audit", "repair", "backtest"]

app = FastAPI(title="Falsify", docs_url=None, redoc_url=None)
_pool = ThreadPoolExecutor(max_workers=WORKERS)
_lock = threading.Lock()
_jobs: dict[str, dict] = {}
_ip_hits: dict[str, deque] = defaultdict(deque)
_day = {"date": date.today().isoformat(), "count": 0}
_corpus_cache: dict[str, dict] = {}


# ---------------------------------------------------------------- helpers

def _client_ip(request: Request) -> str:
    return request.headers.get("cf-connecting-ip") or (request.client.host if request.client else "unknown")


def _sniff(blob: bytes) -> str | None:
    if blob[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if blob[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if blob[:4] == b"RIFF" and blob[8:12] == b"WEBP":
        return ".webp"
    if blob[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif"
    if blob[4:8] == b"ftyp":
        return ".mp4"
    return None


def _clean_date(s: str | None) -> str | None:
    if not s:
        return None
    s = s.strip()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        raise HTTPException(400, "Post date must look like 2026-03-14.")
    return s


def _check_quota(ip: str) -> None:
    now = time.time()
    with _lock:
        today = date.today().isoformat()
        if _day["date"] != today:
            _day.update(date=today, count=0)
        if _day["count"] >= DAILY_LIMIT:
            raise HTTPException(429, "Today's live checks are used up. The examples below still work; try again tomorrow.")
        hits = _ip_hits[ip]
        while hits and now - hits[0] > 3600:
            hits.popleft()
        if len(hits) >= PER_IP_PER_HOUR:
            raise HTTPException(429, f"That's {PER_IP_PER_HOUR} checks from you this hour. Try again later, "
                                     f"or open one of the examples.")
        for k in [k for k, j in _jobs.items() if j["status"] in ("done", "error") and now - j["created"] > 3600]:
            del _jobs[k]  # finished results live on disk; keep memory bounded
        running = sum(1 for j in _jobs.values() if j["status"] in ("queued", "running"))
        if running >= MAX_QUEUE:
            raise HTTPException(503, "A lot of claims are being tested right now. Try again in a minute.")
        hits.append(now)
        _day["count"] += 1


def _result_path(key: str) -> Path:
    return CACHE / f"{key}.json"


def _report_payload(spec: ClaimSpec, audit: dict | None, trace: dict | None, posted: str | None) -> dict:
    from .extract import fidelity
    rep = falsify(spec, posted=posted)
    attach_grounding(rep, (trace or {}).get("grounding"))
    if audit is not None:
        attach_fidelity(rep, fidelity(spec, audit))
    return rep.to_dict()


# ---------------------------------------------------------------- live checks

def _run_job(key: str, kind: str, text: str | None, blob: bytes | None, ext: str | None,
             note: str | None, posted: str | None) -> None:
    from .extract import Source, compile_and_audit

    job = _jobs[key]
    job["status"] = "running"

    def stage(name: str) -> None:
        job["stage"] = name
        if name not in job["stages"]:
            job["stages"].append(name)

    tmp = None
    try:
        if kind == "text":
            src = Source.from_text(text or "", note=note)
        else:
            tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
            tmp.write(blob or b"")
            tmp.close()
            src = Source.from_path(tmp.name, note=note)
        t0 = time.time()
        spec, audit, trace = compile_and_audit(src, on_stage=stage)
        stage("backtest")
        report = _report_payload(spec, audit.model_dump(), trace, posted)
        result = {
            "key": key,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seconds": round(time.time() - t0, 1),
            "source": {"kind": kind, "text": text, "note": note, "posted": posted,
                       "transcript": trace.get("transcript"), "media": f"/api/media/{key}" if kind == "image" else None},
            "spec": spec.model_dump(mode="json"),
            "audit": audit.model_dump(),
            "trace": {k: trace.get(k) for k in ("model", "grounding", "audit_repair", "audits")},
            "report": report,
        }
        CACHE.mkdir(parents=True, exist_ok=True)
        _result_path(key).write_text(json.dumps(result, default=str), encoding="utf-8")
        job.update(status="done", result=result)
    except Exception as e:  # report, don't crash the worker
        job.update(status="error", error=f"{type(e).__name__}: {str(e)[:300]}")
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass


@app.post("/api/check")
async def check(request: Request, text: str | None = Form(None), note: str | None = Form(None),
                posted: str | None = Form(None), file: UploadFile | None = File(None)) -> JSONResponse:
    posted = _clean_date(posted)
    note = (note or "").strip()[:300] or None
    text = (text or "").strip() or None
    blob, ext, kind = None, None, "text"
    if file is not None and file.filename:
        blob = await file.read(MAX_VIDEO + 1)
        ext = _sniff(blob)
        if ext is None:
            raise HTTPException(400, "Use a PNG, JPG, WEBP or GIF screenshot, or an MP4 clip.")
        kind = "video" if ext == ".mp4" else "image"
        limit = MAX_VIDEO if kind == "video" else MAX_IMAGE
        if len(blob) > limit:
            raise HTTPException(413, f"That file is over {limit // (1024 * 1024)} MB. Trim it and try again.")
        if text:
            note = ((note or "") + " | Text pasted with the file: " + text[:500]).strip(" |")
            text = None
    elif not text:
        raise HTTPException(400, "Paste a claim or drop a screenshot or clip first.")
    elif len(text) > MAX_TEXT:
        raise HTTPException(413, f"Keep the claim under {MAX_TEXT} characters; paste the part with the rules.")

    h = hashlib.sha256()
    for part in (kind, text or "", note or "", posted or ""):
        h.update(part.encode() + b"\0")
    if blob:
        h.update(blob)
    key = h.hexdigest()[:20]

    if _result_path(key).exists():
        return JSONResponse({"key": key, "status": "done", "cached": True})
    with _lock:
        job = _jobs.get(key)
    if job and job["status"] in ("queued", "running"):
        return JSONResponse({"key": key, "status": job["status"]})
    if not LIVE:
        raise HTTPException(503, "Live checks are paused right now. The examples below still work.")
    if not os.environ.get("MODEL_API_KEY"):
        raise HTTPException(503, "Live checks aren't configured on this server. The examples below still work.")
    _check_quota(_client_ip(request))
    if kind == "image":
        CACHE.mkdir(parents=True, exist_ok=True)
        (CACHE / f"{key}{ext}").write_bytes(blob)
    with _lock:
        _jobs[key] = {"status": "queued", "stage": None, "stages": [], "kind": kind, "created": time.time()}
    _pool.submit(_run_job, key, kind, text, blob, ext, note, posted)
    return JSONResponse({"key": key, "status": "queued"})


@app.get("/api/result/{key}")
def result(key: str) -> JSONResponse:
    if not KEY_RE.match(key):
        raise HTTPException(404, "No result with that link.")
    p = _result_path(key)
    if p.exists():
        return JSONResponse({"status": "done", "result": json.loads(p.read_text(encoding="utf-8"))})
    job = _jobs.get(key)
    if not job:
        raise HTTPException(404, "No result with that link. It may have expired; test the claim again.")
    out = {k: job.get(k) for k in ("status", "stage", "stages", "kind", "error")}
    if job["status"] == "done":
        out["result"] = job["result"]
    return JSONResponse(out)


@app.get("/api/media/{key}")
def media(key: str) -> Response:
    if not KEY_RE.match(key):
        raise HTTPException(404)
    for ext in (".png", ".jpg", ".webp", ".gif"):
        p = CACHE / f"{key}{ext}"
        if p.exists():
            return FileResponse(p, headers={"Cache-Control": "public, max-age=86400"})
    raise HTTPException(404)


# ---------------------------------------------------------------- corpus gallery

def _corpus_files() -> list[Path]:
    return sorted((CORPUS / "results").glob("c*.json"))


@app.get("/api/corpus")
def corpus_list() -> JSONResponse:
    items = []
    for f in _corpus_files():
        d = json.loads(f.read_text(encoding="utf-8"))
        if "error" in d:
            continue
        rec, rp = d["record"], d["report"]
        items.append({
            "id": d["id"], "url": rec["url"], "source": rec["source"], "posted": rec.get("posted"),
            "category": rec["category"], "summary": d["spec"]["claim_summary"],
            "claimed": rec.get("performance_claim"), "verdict": rp["verdict"],
            "headline": rp["headline"].replace("Tested an approximation of the claim. ", ""),
            "approximation": (rp.get("fidelity") or {}).get("level") == "approximation" and rp["verdict"] != "UNTESTABLE",
        })
    md = (CORPUS / "results" / "scoreboard.md")
    bh = []
    if md.exists():
        m = re.search(r"Still survives after the corpus-level correction: \d+ \(([^)]*)\)", md.read_text(encoding="utf-8"))
        if m and m.group(1) != "none":
            bh = [x.strip() for x in m.group(1).split(",")]
    return JSONResponse({"items": items, "corpus_corrected": bh})


@app.get("/api/corpus/{cid}")
def corpus_item(cid: str) -> JSONResponse:
    if not re.fullmatch(r"c\d{3}", cid):
        raise HTTPException(404)
    if cid not in _corpus_cache:
        f = CORPUS / "results" / f"{cid}.json"
        if not f.exists():
            raise HTTPException(404, "No such example.")
        d = json.loads(f.read_text(encoding="utf-8"))
        if "error" in d:
            raise HTTPException(404, "That example didn't compile.")
        spec = ClaimSpec.model_validate(d["spec"])
        rec = d["record"]
        report = _report_payload(spec, d.get("audit"), d.get("trace"), rec.get("posted"))
        _corpus_cache[cid] = {
            "key": cid,
            "source": {"kind": "text", "text": rec["excerpt"], "note": None, "posted": rec.get("posted"),
                       "url": rec["url"], "origin": rec["source"], "transcript": None, "media": None},
            "spec": d["spec"], "audit": d.get("audit"),
            "trace": {k: (d.get("trace") or {}).get(k) for k in ("model", "grounding", "audit_repair", "audits")},
            "report": report,
        }
    return JSONResponse({"status": "done", "result": _corpus_cache[cid]})


# ---------------------------------------------------------------- meta + pages

@app.get("/api/meta")
def meta() -> JSONResponse:
    calib = ROOT / "docs" / "calibration.json"
    return JSONResponse({
        "live": LIVE and bool(os.environ.get("MODEL_API_KEY")),
        "universe": data_mod.UNIVERSE,
        "per_ip_per_hour": PER_IP_PER_HOUR,
        "calibration": json.loads(calib.read_text()) if calib.exists() else None,
    })


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "data_files": len(data_mod.available())}


@app.get("/")
@app.get("/r/{key}")
@app.get("/example/{cid}")
def index(key: str | None = None, cid: str | None = None) -> FileResponse:
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/favicon.svg")
def favicon() -> FileResponse:
    return FileResponse(STATIC / "favicon.svg", media_type="image/svg+xml")
