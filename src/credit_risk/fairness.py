"""Fairness auditing for lending decisions.

Lending is a regulated decision. A model can hit a strong ROC-AUC overall and
still approve one group at a far lower rate than another. This module measures
that gap so a human can act on it.

Two things this module deliberately does *not* do:
  * it does not feed the audit segments into the model -- they are never inputs
  * it does not silently "correct" a disparity -- it reports it for review

Reference point: the four-fifths (80%) rule used by US regulators, where a
selection rate below 80% of the best-performing group's rate is treated as
evidence of adverse impact worth investigating.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

FOUR_FIFTHS = 0.80


def add_audit_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the segments we audit on. These are never model features."""
    out = df.copy()
    out["age_group"] = pd.cut(
        out["age"],
        bins=[0, 30, 45, 60, 200],
        labels=["under_30", "30_to_44", "45_to_59", "60_plus"],
    ).astype("string")
    return out


def _segment_stats(y_true: np.ndarray, approved: np.ndarray) -> dict:
    """Approval rate plus error rates for one segment."""
    n = len(y_true)
    if n == 0:
        return {}
    defaulted = y_true == 1
    repaid = y_true == 0
    # A "positive" prediction here is flagging default risk (i.e. NOT approving).
    flagged = ~approved
    return {
        "applications": int(n),
        "approval_rate": round(float(approved.mean()), 4),
        "actual_default_rate": round(float(defaulted.mean()), 4),
        # Of borrowers who really defaulted, how many did we flag?
        "true_positive_rate": round(float(flagged[defaulted].mean()), 4)
        if defaulted.any()
        else None,
        # Of borrowers who really repaid, how many did we wrongly flag?
        "false_positive_rate": round(float(flagged[repaid].mean()), 4)
        if repaid.any()
        else None,
    }


def audit(
    df: pd.DataFrame,
    y_true,
    y_prob,
    threshold: float,
    segments: tuple[str, ...] = ("age_group", "employment_status", "home_ownership"),
    min_segment_size: int = 30,
) -> dict:
    """Per-segment approval and error rates, plus disparate impact ratios."""
    frame = add_audit_segments(df).reset_index(drop=True)
    y_true = np.asarray(y_true)
    approved = np.asarray(y_prob) < threshold  # below the risk cut-off = approve

    report: dict = {
        "threshold": float(threshold),
        "overall": _segment_stats(y_true, approved),
        "segments": {},
        "notes": [
            "Audit attributes are never used as model inputs.",
            "Disparate impact ratio = segment approval rate / highest segment "
            "approval rate; below 0.80 flags the segment for human review.",
            f"Segments with fewer than {min_segment_size} applications are "
            "reported but not flagged, since the estimate is too noisy.",
        ],
    }

    for column in segments:
        if column not in frame.columns:
            continue
        stats: dict = {}
        for value, index in frame.groupby(column, observed=True).groups.items():
            position = frame.index.get_indexer(index)
            stats[str(value)] = _segment_stats(y_true[position], approved[position])

        rates = {
            k: v["approval_rate"]
            for k, v in stats.items()
            if v.get("applications", 0) >= min_segment_size
        }
        reference = max(rates.values()) if rates else None

        for value in stats.values():
            if reference and value.get("applications", 0) >= min_segment_size:
                ratio = round(value["approval_rate"] / reference, 4) if reference else None
                value["disparate_impact_ratio"] = ratio
                value["flagged_for_review"] = bool(ratio is not None and ratio < FOUR_FIFTHS)
            else:
                value["disparate_impact_ratio"] = None
                value["flagged_for_review"] = False

        report["segments"][column] = stats

    report["flagged_segments"] = [
        f"{column}={key}"
        for column, stats in report["segments"].items()
        for key, value in stats.items()
        if value.get("flagged_for_review")
    ]
    return report


def format_report(report: dict) -> str:
    """Console-friendly rendering of the audit."""
    lines = [
        "FAIRNESS AUDIT",
        "-" * 72,
        f"Overall approval rate: {report['overall']['approval_rate']:.1%}"
        f"   (threshold {report['threshold']:.2f})",
        "",
    ]
    for column, stats in report["segments"].items():
        lines.append(f"{column}")
        lines.append(
            f"  {'segment':<18}{'n':>7}{'approve':>10}{'default':>10}"
            f"{'DI ratio':>11}{'flag':>7}"
        )
        for key, value in sorted(stats.items()):
            ratio = value.get("disparate_impact_ratio")
            lines.append(
                f"  {key:<18}{value['applications']:>7}"
                f"{value['approval_rate']:>10.1%}"
                f"{value['actual_default_rate']:>10.1%}"
                f"{(f'{ratio:.2f}' if ratio is not None else '-'):>11}"
                f"{('YES' if value.get('flagged_for_review') else ''):>7}"
            )
        lines.append("")

    if report["flagged_segments"]:
        lines.append("Flagged for human review: " + ", ".join(report["flagged_segments"]))
    else:
        lines.append("No segment fell below the four-fifths threshold.")
    return "\n".join(lines)


def save_report(report: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    return path
