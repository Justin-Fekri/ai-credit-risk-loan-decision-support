#!/usr/bin/env python3
"""Train the XGBoost credit-risk pipeline and write artifacts/."""

from credit_risk.train import train

if __name__ == "__main__":
    metrics = train()
    print("Trained default-risk model")
    for key in ("roc_auc", "pr_auc", "ks", "brier", "default_rate", "n_test"):
        print(f"  {key}: {metrics[key]}")
