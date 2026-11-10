"""HTTP contract: create/get/list/cancel and their status codes."""
import os

import pytest
from fastapi.testclient import TestClient

from app import repo


@pytest.fixture()
def client(db_url, conn, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", db_url)
    import importlib
    from app import api, config
    importlib.reload(config)
    importlib.reload(api)
    with TestClient(api.app) as c:
        yield c


def post(client, key, **body):
    return client.post("/jobs", json=body, headers={"Idempotency-Key": key})


def test_create_get_and_idempotent_retry(client):
    r1 = post(client, "a", duration_ms=10)
    assert r1.status_code == 201 and r1.json()["status"] == "queued"
    r2 = post(client, "a", duration_ms=10)
    assert r2.status_code == 200 and r2.json()["id"] == r1.json()["id"]
    assert post(client, "a", duration_ms=11).status_code == 409
    assert client.get(f"/jobs/{r1.json()['id']}").json()["idempotency_key"] == "a"
    assert "fencing_token" not in r1.json()


def test_missing_idempotency_key_is_rejected(client):
    assert client.post("/jobs", json={}).status_code == 422


def test_pagination_is_complete_and_ordered(client):
    ids = [post(client, f"p{i}").json()["id"] for i in range(123)]
    seen, cursor = [], None
    while True:
        page = client.get("/jobs", params={"limit": 50, **({"cursor": cursor} if cursor else {})}).json()
        seen += [j["id"] for j in page["items"]]
        cursor = page["next_cursor"]
        if not cursor:
            break
    assert seen == ids
    assert client.get("/jobs", params={"cursor": "garbage"}).status_code == 400


def test_cancel_codes(client, conn):
    queued = post(client, "q").json()["id"]
    assert client.post(f"/jobs/{queued}/cancel").status_code == 200
    assert client.post(f"/jobs/{queued}/cancel").status_code == 200   # idempotent
    running = post(client, "r").json()["id"]
    claim = repo.claim_next(conn, "w", 30)
    assert str(claim.job_id) == running
    r = client.post(f"/jobs/{running}/cancel")
    assert r.status_code == 202 and r.json()["cancel_requested"] is True
    done = post(client, "d").json()["id"]
    c = repo.claim_next(conn, "w", 30)
    repo.finish(conn, c, "w", "succeeded", result={})
    r = client.post(f"/jobs/{done}/cancel")
    assert r.status_code == 409 and r.json()["status"] == "succeeded"
    assert client.post("/jobs/00000000-0000-0000-0000-000000000000/cancel").status_code == 404
