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
