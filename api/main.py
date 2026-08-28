from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import BorrowerFeatures, DecisionRequest, reference_values
from credit_risk.config import ALL_FEATURES, FEATURE_LABELS, PURPOSE_LABELS, get_artifacts_dir
from credit_risk.explain import CreditExplainer
from credit_risk.pipeline import predict_pd
from credit_risk.policy import eligibility_label, recommend, risk_band
from credit_risk.train import load_bundle, train


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_runtime()
    yield


app = FastAPI(
    title="Northbridge Credit — Decision Support API",
    description=(
        "Default-risk scores and TreeSHAP explanations for human underwriters. "
        "This service recommends; it does not originate a loan."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATE: dict = {
    "bundle": None,
    "explainer": None,
    "queue": [],
    "metrics": {},
    "importance": [],
    "model_card": {},
    "ready": False,
}


def _ensure_artifacts():
    artifacts = get_artifacts_dir()
    model_path = artifacts / "model.joblib"
    if not model_path.exists():
        train(output_dir=artifacts)
    return artifacts


def load_runtime() -> None:
    artifacts = _ensure_artifacts()
    bundle = load_bundle(artifacts)
    background = pd.read_csv(artifacts / "shap_background.csv")
    STATE["bundle"] = bundle
    STATE["explainer"] = CreditExplainer(bundle, background)
    STATE["queue"] = json.loads((artifacts / "queue.json").read_text())
    STATE["metrics"] = json.loads((artifacts / "metrics.json").read_text())
    STATE["importance"] = json.loads((artifacts / "global_importance.json").read_text())
    STATE["model_card"] = json.loads((artifacts / "model_card.json").read_text())
    STATE["ready"] = True


def _score_frame(frame: pd.DataFrame) -> dict:
    bundle = STATE["bundle"]
    explainer: CreditExplainer = STATE["explainer"]
    pd_hat = float(predict_pd(bundle.pipeline, frame)[0])
    shap_rows = explainer.explain_frame(frame)[0]
    row = frame.iloc[0]
    rec, rationale = recommend(pd_hat, float(row["credit_score"]), float(row["debt_to_income"]))
    return {
        "features": {name: _jsonable(row[name]) for name in ALL_FEATURES},
        "pd_default": round(pd_hat, 4),
        "risk_band": risk_band(pd_hat),
        "policy_recommendation": rec,
        "policy_rationale": rationale,
        "eligibility": eligibility_label(rec),
        "shap": shap_rows,
        "top_drivers": shap_rows[:5],
        "expected_pd": round(explainer.expected_value, 4),
    }


def _jsonable(value):
    if hasattr(value, "item"):
        return value.item()
    return value


def _find(application_id: str) -> dict:
    for item in STATE["queue"]:
        if item["application_id"] == application_id:
            return item
    raise HTTPException(status_code=404, detail="Application not found")


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if STATE["ready"] else "loading"}


@app.get("/reference")
def reference() -> dict:
    return {
        "features": FEATURE_LABELS,
        "purpose_labels": PURPOSE_LABELS,
        "enums": reference_values(),
    }


@app.get("/model")
def model_card() -> dict:
    return {
        "card": STATE["model_card"],
        "metrics": STATE["metrics"],
        "global_importance": STATE["importance"],
        "expected_pd": round(STATE["explainer"].expected_value, 4) if STATE["explainer"] else None,
    }


@app.get("/queue")
def queue(status: str | None = None) -> dict:
    items = STATE["queue"]
    if status:
        items = [item for item in items if item["status"] == status]
    summary = {
        "pending": sum(1 for item in STATE["queue"] if item["status"] == "pending"),
        "approved": sum(1 for item in STATE["queue"] if item.get("decision") == "approve"),
        "referred": sum(1 for item in STATE["queue"] if item.get("decision") == "refer"),
        "declined": sum(1 for item in STATE["queue"] if item.get("decision") == "decline"),
        "high_risk": sum(
            1
            for item in STATE["queue"]
            if item["risk_band"] in {"elevated", "severe"} and item["status"] == "pending"
        ),
    }
    return {"items": items, "summary": summary}


@app.get("/applications/{application_id}")
def get_application(application_id: str) -> dict:
    item = deepcopy(_find(application_id))
    if "shap" not in item or not item.get("expected_pd"):
        frame = pd.DataFrame([item["features"]])
        scored = _score_frame(frame)
        item.update({k: scored[k] for k in ("shap", "top_drivers", "expected_pd")})
    return item


@app.post("/score")
def score(payload: BorrowerFeatures) -> dict:
    frame = pd.DataFrame([payload.model_dump()])
    result = _score_frame(frame)
    result["application_id"] = None
    result["borrower_name"] = "Ad-hoc applicant"
    result["status"] = "scored"
    result["decision"] = None
    result["decision_notes"] = None
    return result


@app.post("/applications/{application_id}/decision")
def decide(application_id: str, payload: DecisionRequest) -> dict:
    item = _find(application_id)
    if item["status"] != "pending":
        raise HTTPException(status_code=409, detail="This file already has a recorded decision.")
    item["status"] = "reviewed"
    item["decision"] = payload.decision
    item["decision_notes"] = payload.notes
    item["underwriter"] = payload.underwriter
    item["decided_at"] = datetime.now(timezone.utc).isoformat()
    return item
