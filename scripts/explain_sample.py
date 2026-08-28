#!/usr/bin/env python3
"""Print TreeSHAP drivers for one queued application (CLI companion to the workbench)."""

import argparse
import json
from pathlib import Path

import pandas as pd

from credit_risk.config import get_artifacts_dir
from credit_risk.explain import CreditExplainer
from credit_risk.pipeline import predict_pd
from credit_risk.train import load_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", help="Application id, e.g. NB-26-10012")
    args = parser.parse_args()

    artifacts = get_artifacts_dir()
    bundle = load_bundle(artifacts)
    background = pd.read_csv(artifacts / "shap_background.csv")
    explainer = CreditExplainer(bundle, background)
    queue = json.loads((artifacts / "queue.json").read_text())
    item = queue[0]
    if args.id:
        item = next(row for row in queue if row["application_id"] == args.id)

    frame = pd.DataFrame([item["features"]])
    pd_hat = float(predict_pd(bundle.pipeline, frame)[0])
    shap_rows = explainer.explain_frame(frame)[0]
    print(f"{item['application_id']}  {item.get('borrower_name', '')}")
    print(f"P(default) = {pd_hat:.3f}")
    print("Top influential features:")
    for row in shap_rows[:8]:
        sign = "+" if row["shap"] > 0 else ""
        print(f"  {row['label']:28} {row['value']:>12}  SHAP {sign}{row['shap']:.4f}")


if __name__ == "__main__":
    main()
