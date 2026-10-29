import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from falsify import web  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "CACHE", tmp_path)
    monkeypatch.setattr(web, "LIVE", True)
    monkeypatch.setenv("MODEL_API_KEY", "test-key-not-real")
    web._jobs.clear()
    web._ip_hits.clear()
    web._day.update(count=0)
    return TestClient(web.app)


def test_pages_and_corpus(client):
    assert client.get("/").status_code == 200
    assert client.get("/example/c033").status_code == 200
    items = client.get("/api/corpus").json()["items"]
    assert len(items) >= 30 and {"verdict", "summary", "url"} <= set(items[0])
    r = client.get("/api/corpus/c033").json()["result"]
    assert r["report"]["series"]["t"] and r["report"]["null_test"]["hist"]["counts"]
    assert client.get("/api/corpus/../../etc").status_code == 404
    assert client.get("/api/result/not-a-key").status_code == 404


def test_check_validation(client):
    assert client.post("/api/check", data={}).status_code == 400
    assert client.post("/api/check", data={"text": "x" * 5000}).status_code == 413
    assert client.post("/api/check", data={"text": "buy btc", "posted": "last week"}).status_code == 400
    r = client.post("/api/check", files={"file": ("evil.png", b"<script>alert(1)</script>", "image/png")})
    assert r.status_code == 400  # content is sniffed, the extension and MIME type are not trusted


def test_live_check_runs_job_and_caches(client, monkeypatch):
    from falsify import extract
    from falsify.extract import Audit
    from falsify.spec import ClaimSpec
    spec = ClaimSpec.model_validate(json.loads(open("claims/02_rsi_oversold_eth.json").read())["spec"])

    def fake(src, on_stage=lambda s: None, **kw):
        for s in ("compile", "quote_check", "audit"):
            on_stage(s)
        return spec, Audit(faithful=True, issues=[]), {"grounding": None, "audits": []}

    monkeypatch.setattr(extract, "compile_and_audit", fake)
    monkeypatch.setattr(web, "_pool", type("P", (), {"submit": staticmethod(lambda f, *a: f(*a))})())
    r = client.post("/api/check", data={"text": "Buy ETH when RSI < 30 on the 4h", "posted": "2021-01-01"}).json()
    res = client.get(f"/api/result/{r['key']}").json()
    assert res["status"] == "done" and res["result"]["report"]["verdict"] == "DIES"
    assert res["result"]["report"]["out_of_sample"]["boundary"] == "2021-01-01"
    again = client.post("/api/check", data={"text": "Buy ETH when RSI < 30 on the 4h", "posted": "2021-01-01"}).json()
    assert again == {"key": r["key"], "status": "done", "cached": True}  # no second model call, no quota spent


def test_quota_per_ip(client, monkeypatch):
    monkeypatch.setattr(web, "PER_IP_PER_HOUR", 2)
    monkeypatch.setattr(web, "_pool", type("P", (), {"submit": staticmethod(lambda f, *a: None)})())
    codes = [client.post("/api/check", data={"text": f"claim {i}"}).status_code for i in range(3)]
    assert codes == [200, 200, 429]


def test_escalation_flag_and_signoff(client, monkeypatch):
    """Machines still disagree after the repairs -> needs a human; a visitor flags it; the reviewer signs off."""
    from falsify import extract, review
    from falsify.extract import Audit, AuditIssue
    from falsify.spec import ClaimSpec
    spec = ClaimSpec.model_validate(json.loads(open("claims/02_rsi_oversold_eth.json").read())["spec"])
    audit = Audit(faithful=False, issues=[AuditIssue(field="exit[0]", problem="claim exits at RSI 50, spec at 70",
                                                     severity="major")])

    def fake(src, on_stage=lambda s: None, **kw):
        return spec, audit, {"grounding": None, "audits": [audit.model_dump()] * 2, "audit_repair": True}

    monkeypatch.setattr(extract, "compile_and_audit", fake)
    monkeypatch.setattr(web, "_pool", type("P", (), {"submit": staticmethod(lambda f, *a: f(*a))})())
    web._flag_hits.clear()
    key = client.post("/api/check", data={"text": "ETH RSI dip buy, out at 50"}).json()["key"]
    res = client.get(f"/api/result/{key}").json()["result"]
    assert res["review"]["needs_human"] and "RSI 50" in res["review"]["reasons"][0]
    assert res["report"]["warnings"][0].startswith("needs a human check")

    assert client.post(f"/api/review/{key}", json={"agree": "yes"}).status_code == 400
    assert client.post("/api/review/0123456789abcdef0123", json={"agree": True}).status_code == 404
    r = client.post(f"/api/review/{key}", json={"agree": False, "note": "exit is RSI 50"}).json()
    assert r["flags"] == {"agree": 0, "disagree": 1}
    q = review.queue(web.CACHE)
    assert [i["key"] for i in q] == [key] and q[0]["flags"][0]["note"] == "exit is RSI 50"

    review.sign(key, False, "Exit should be RSI(14) > 50.", cache=web.CACHE)
    res = client.get(f"/api/result/{key}").json()["result"]
    assert res["signoff"]["agree"] is False and res["signoff"]["by"] == "trader"
    assert review.queue(web.CACHE) == []  # signed off leaves the queue


def test_corpus_trader_signoff(client, monkeypatch, tmp_path):
    from falsify import groundtruth
    gt = tmp_path / "gt.yaml"
    d = json.loads(open("corpus/results/c033.json").read())
    sp = d["spec"]
    gt.write_text(f"c033:\n  testable: full\n  timeframe: {sp['timeframe']}\n  reviewed: true\n"
                  f"  reviewer_note: Exit was not in the post.\n", encoding="utf-8")
    monkeypatch.setattr(groundtruth, "GT_PATH", gt)
    r = client.get("/api/corpus/c033").json()["result"]
    assert r["signoff"] == {"by": "trader", "agree": True, "differs_on": [], "note": "Exit was not in the post."}
    gt.write_text("c033:\n  testable: no\n  reviewed: true\n", encoding="utf-8")
    import os
    os.utime(gt, (1, 1))  # new mtime -> reloaded
    r = client.get("/api/corpus/c033").json()["result"]
    assert r["signoff"]["agree"] is False and r["signoff"]["differs_on"] == ["routing"]
