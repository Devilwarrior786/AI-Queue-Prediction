"""
End-to-end API tests for the AI Queue Prediction backend.

Runs against the real app (lifespan included: DB init + model load) using
FastAPI's TestClient. Tests only add tokens/logs and never mutate the trained
model files.

Run with:  pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# System / health
# ---------------------------------------------------------------------------

def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_ready"] is True  # pretrained model ships with the repo


def test_ui_pages_served(client):
    for path, marker in [
        ("/", "Queue Status"),
        ("/tokens", "Smart Token Kiosk"),
        ("/staff", "Staff Panel"),
    ]:
        r = client.get(path)
        assert r.status_code == 200
        assert marker in r.text


# ---------------------------------------------------------------------------
# Prediction / queue
# ---------------------------------------------------------------------------

def test_status_prediction(client):
    r = client.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_wait_mins"] >= 0.5
    assert body["confidence"] in ("model", "formula")
    assert body["status_label"] in ("short", "moderate", "long")


def test_queue_update_returns_prediction_and_logs(client):
    r = client.post("/api/queue/update", json={"queue_depth": 20, "active_counters": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["queue_depth"] == 20
    assert body["active_counters"] == 3
    # Heavier queue must be predicted as 'long' or at least not 'short'
    assert body["predicted_wait_mins"] > 5


def test_queue_update_validation(client):
    r = client.post("/api/queue/update", json={"queue_depth": -1, "active_counters": 3})
    assert r.status_code == 422


def test_history_endpoint(client):
    client.post("/api/queue/update", json={"queue_depth": 8, "active_counters": 4})
    r = client.get("/api/history?limit=5")
    assert r.status_code == 200
    rows = r.json()
    assert 1 <= len(rows) <= 5
    assert "queue_depth" in rows[0]


def test_history_limit_bounds(client):
    assert client.get("/api/history?limit=0").status_code == 400
    assert client.get("/api/history?limit=999").status_code == 400


def test_service_log_updates_average(client):
    r1 = client.get("/api/status").json()
    r = client.post("/api/service/log", json={"service_time_mins": 9.0})
    assert r.status_code == 200
    r2 = client.get("/api/status").json()
    # A slow 9-min service must push the rolling average up
    assert r2["avg_service_time_mins"] >= r1["avg_service_time_mins"]


# ---------------------------------------------------------------------------
# Token lifecycle
# ---------------------------------------------------------------------------

def test_token_full_lifecycle(client):
    # Issue
    r = client.post(
        "/api/tokens/issue",
        json={"service_type": "Cash Deposit / Withdrawal", "customer_name": "Test User"},
    )
    assert r.status_code == 200
    tok = r.json()
    assert tok["token_number"].startswith("C-")
    assert tok["status"] == "waiting"
    assert tok["estimated_wait_mins"] >= 0.5
    assert ":" in tok["estimated_time"]  # e.g. "07:35 PM"

    number = tok["token_number"]

    # Lookup while waiting
    r = client.get(f"/api/tokens/{number}")
    assert r.status_code == 200
    assert r.json()["people_ahead"] >= 0

    # Active list includes it with people_ahead
    active = client.get("/api/tokens/active").json()
    match = [t for t in active if t["token_number"] == number]
    assert match and "people_ahead" in match[0]

    # Call to counter (this also writes the realised wait into queue_logs)
    r = client.patch(f"/api/tokens/{number}/status", json={"status": "serving", "counter_id": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "serving"
    assert body["counter_id"] == 2
    assert body["called_at"] is not None
    assert body["people_ahead"] == 0

    # Complete (auto-logs the service duration)
    r = client.patch(f"/api/tokens/{number}/status", json={"status": "completed"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "completed"
    assert body["completed_at"] is not None

    # Summary counts move
    summary = client.get("/api/tokens/summary").json()
    assert summary["completed"] >= 1

    # The realised wait must be recorded for the retraining feedback loop
    history = client.get("/api/history?limit=10").json()
    assert any(row["actual_wait_mins"] is not None for row in history), (
        "Serving a token must write actual_wait_mins to queue_logs"
    )


def test_token_numbers_never_collide_across_days(client):
    """Regression: numbering reset daily and hit the UNIQUE constraint."""
    seen = set()
    for svc in ("General Banking", "Loans & Mortgages", "Account Services",
                "Govt Documentation / Passbook", "General Banking"):
        r = client.post("/api/tokens/issue", json={"service_type": svc})
        assert r.status_code == 200, r.text
        num = r.json()["token_number"]
        assert num not in seen
        seen.add(num)


def test_token_lookup_unknown(client):
    assert client.get("/api/tokens/Z-99999").status_code == 404


def test_token_status_unknown(client):
    r = client.patch("/api/tokens/Z-99999/status", json={"status": "completed"})
    assert r.status_code == 404


def test_token_status_invalid_value(client):
    r = client.patch("/api/tokens/C-99999/status", json={"status": "flying"})
    assert r.status_code == 422


def test_cancel_token(client):
    r = client.post("/api/tokens/issue", json={"service_type": "General Banking"})
    number = r.json()["token_number"]
    r = client.patch(f"/api/tokens/{number}/status", json={"status": "cancelled"})
    assert r.status_code == 200
    assert r.json()["status"] == "cancelled"
    active = client.get("/api/tokens/active").json()
    assert number not in [t["token_number"] for t in active]
