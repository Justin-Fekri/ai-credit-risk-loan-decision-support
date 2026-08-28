from pathlib import Path
import os

import pytest
from fastapi.testclient import TestClient

from credit_risk.config import ALL_FEATURES
from credit_risk.train import train


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    artifacts = tmp_path_factory.mktemp("artifacts")
    train(n_samples=600, output_dir=Path(artifacts), random_state=7)

    os.environ["CREDIT_RISK_ARTIFACTS"] = str(artifacts)
    import api.main as main

    main.load_runtime()
    return TestClient(main.app, raise_server_exceptions=True)


def test_health_and_queue(client):
    assert client.get("/health").json()["status"] == "ok"
    queue = client.get("/queue").json()
    assert queue["summary"]["pending"] >= 1
    assert len(queue["items"]) >= 1
    first = queue["items"][0]
    detail = client.get(f"/applications/{first['application_id']}").json()
    assert "shap" in detail
    assert detail["pd_default"] >= 0


def test_score_and_decision(client):
    queue = client.get("/queue").json()["items"]
    features = queue[0]["features"]
    scored = client.post("/score", json=features).json()
    assert 0 <= scored["pd_default"] <= 1
    assert scored["policy_recommendation"] in {"approve", "refer", "decline"}
    assert scored["shap"][0]["feature"] in ALL_FEATURES

    pending = next(item for item in queue if item["status"] == "pending")
    decided = client.post(
        f"/applications/{pending['application_id']}/decision",
        json={"decision": "refer", "notes": "Need employer verification."},
    ).json()
    assert decided["decision"] == "refer"
    assert decided["status"] == "reviewed"
