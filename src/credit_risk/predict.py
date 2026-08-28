"""Scoring applications and rendering reviewer-facing adjudication summaries.

This module is the "decision support, not decision" boundary of the project.
Nothing here approves or declines a loan. It produces a risk estimate, a
recommendation band, and the reasons behind them, formatted for a person who
has to make -- and be accountable for -- the actual call.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEATURES, DecisionPolicy
from .explain import explain_application

RISK_BANDS = (
    (0.10, "Very low"),
    (0.25, "Low"),
    (0.45, "Moderate"),
    (0.65, "Elevated"),
    (1.01, "High"),
)

# Isotonic calibration is a step function: it maps everything in its top bin to
# exactly 1.0 and its bottom bin to 0.0. Reporting "100% chance of default" to a
# loan officer would be indefensible -- no model earns that certainty from a
# finite sample -- so scores are clipped to an honest interval.
PROBABILITY_BOUNDS = (0.005, 0.995)

DISCLAIMER = (
    "DECISION SUPPORT ONLY -- this is a statistical estimate, not an approval "
    "or a denial. The final lending determination rests with the human reviewer, "
    "who should weigh information not captured by the model."
)


def risk_band(probability: float) -> str:
    for ceiling, label in RISK_BANDS:
        if probability < ceiling:
            return label
    return "High"


def _frame_from(application: dict | pd.Series | pd.DataFrame) -> pd.DataFrame:
    if isinstance(application, pd.DataFrame):
        frame = application.copy()
    elif isinstance(application, pd.Series):
        frame = application.to_frame().T
    else:
        frame = pd.DataFrame([application])

    missing = [c for c in FEATURES if c not in frame.columns]
    if missing:
        raise ValueError(f"Application is missing required fields: {missing}")
    return frame


def score_application(
    bundle: dict,
    application: dict | pd.Series | pd.DataFrame,
    background: pd.DataFrame | None = None,
    top_n: int = 5,
) -> dict:
    """Score one application and explain it.

    The calibrated model produces the probability (so the number is meaningful);
    the underlying pipeline produces the explanation (SHAP needs the raw
    estimator, which calibration wraps out of reach).
    """
    frame = _frame_from(application)
    features = frame[FEATURES]

    raw = float(bundle["model"].predict_proba(features)[0, 1])
    probability = float(np.clip(raw, *PROBABILITY_BOUNDS))
    policy: DecisionPolicy = bundle.get("policy", DecisionPolicy())

    reasons = []
    explainer_pipeline = bundle.get("uncalibrated_model")
    if explainer_pipeline is not None:
        reference = background if background is not None else features
        try:
            reasons = explain_application(
                explainer_pipeline, features, reference, top_n=top_n
            )
        except Exception as exc:  # explanation must never break scoring
            reasons = [{"error": f"explanation unavailable ({type(exc).__name__}: {exc})"}]

    return {
        "application_id": frame.get("application_id", pd.Series(["(unnamed)"])).iloc[0],
        "default_probability": round(probability, 4),
        "risk_band": risk_band(probability),
        "recommendation": policy.band(probability),
        "model_threshold": bundle.get("threshold"),
        "model_name": bundle.get("model_name"),
        "reasons": reasons,
    }


def score_batch(bundle: dict, df: pd.DataFrame, explain: bool = False) -> pd.DataFrame:
    """Score a whole CSV of applications. Explanations are opt-in (they cost time)."""
    features = df[FEATURES]
    probabilities = np.clip(
        bundle["model"].predict_proba(features)[:, 1], *PROBABILITY_BOUNDS
    )
    policy: DecisionPolicy = bundle.get("policy", DecisionPolicy())

    out = df.copy()
    out["default_probability"] = probabilities.round(4)
    out["risk_band"] = [risk_band(p) for p in probabilities]
    out["recommendation"] = [policy.band(p) for p in probabilities]

    if explain:
        top_reasons = []
        for i in range(len(features)):
            result = score_application(bundle, features.iloc[[i]], background=features, top_n=3)
            top_reasons.append(
                "; ".join(
                    f"{r['label']} {r['direction']}"
                    for r in result["reasons"]
                    if "label" in r
                )
            )
        out["top_reasons"] = top_reasons

    return out


def render_adjudication(result: dict) -> str:
    """The block a loan officer actually reads."""
    width = 74
    lines = [
        "=" * width,
        f"  LOAN APPLICATION REVIEW -- {result['application_id']}",
        "=" * width,
        "",
        f"  Estimated default probability : {result['default_probability']:.1%}",
        f"  Risk band                     : {result['risk_band']}",
        f"  System recommendation         : {result['recommendation'].replace('_', ' ')}",
        "",
        "  Why the model scored it this way",
        "  " + "-" * (width - 4),
    ]

    if not result["reasons"] or "error" in result["reasons"][0]:
        detail = result["reasons"][0]["error"] if result["reasons"] else "not available"
        lines.append(f"  (explanation {detail})")
    else:
        for i, reason in enumerate(result["reasons"], 1):
            value = reason["value"]
            shown = f"{value:,.2f}" if isinstance(value, float) else str(value)
            arrow = "^" if "increases" in reason["direction"] else "v"
            lines.append(
                f"  {i}. {arrow} {reason['label']:<28} = {shown:<14}"
                f"({reason['direction']})"
            )

    action = {
        "RECOMMEND_APPROVE": "Low estimated risk. Reviewer to confirm and proceed.",
        "MANUAL_REVIEW": "Borderline. Requires a reviewer's judgement before any decision.",
        "RECOMMEND_DECLINE": "High estimated risk. Reviewer to verify before declining.",
    }[result["recommendation"]]

    lines += [
        "",
        f"  Next step: {action}",
        "",
        "  " + "-" * (width - 4),
    ]
    # Wrap the disclaimer to the box width.
    words, line = DISCLAIMER.split(), "  "
    for word in words:
        if len(line) + len(word) + 1 > width - 2:
            lines.append(line)
            line = "  " + word
        else:
            line = f"{line} {word}" if line.strip() else f"  {word}"
    lines.append(line)
    lines.append("=" * width)
    return "\n".join(lines)
