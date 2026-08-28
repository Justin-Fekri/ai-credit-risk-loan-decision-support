# AI Credit Risk & Loan Decision Support System

Python · scikit-learn · XGBoost · Explainable AI (TreeSHAP) · MIT License

A **decision-support** workbench for personal-loan underwriters. An XGBoost classifier estimates 12-month default risk from credit history and financial attributes. TreeSHAP surfaces the borrower features that moved the score. A separate policy layer maps that probability onto approve / refer / decline. **A person still records the decision.**

This is not an automated origination engine.

## What it does

- **Binary classification pipeline** — scikit-learn `ColumnTransformer` + `XGBClassifier` on tabular borrower attributes (income, DTI, credit score, utilization, delinquencies, housing, purpose, and related fields).
- **Default risk & eligibility** — the model outputs P(default); policy rules turn that into a recommendation the underwriter can override.
- **Feature transparency** — interventional TreeSHAP in probability units, rolled back from one-hot columns to original fields so a reviewer can see *why* a file scored the way it did.
- **Human in the loop** — the Risk Desk queue lets an underwriter open a file, read the drivers, and approve, refer, or decline with a written rationale.

The training book is **fully synthetic**. No real consumer records are used.

## Architecture

```
credit_risk/          data generation, sklearn pipeline, SHAP, policy overlay
api/                  FastAPI: queue, scoring, explanations, decisions
scripts/train.py      fit the model and write artifacts/
scripts/explain_sample.py
web/                  Next.js underwriter workbench
artifacts/            fitted pipeline, metrics, sample queue
```

## Run locally

Python 3.11+ and Node 20+ .

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export PYTHONPATH=.
python scripts/train.py     # skip if artifacts/ already present

python -m uvicorn api.main:app --host 0.0.0.0 --port 45232
```

In another terminal:

```bash
cd web
npm install
npm run dev
```

Open [http://localhost:45231](http://localhost:45231). The UI proxies `/backend/*` to the API.

```bash
pytest
python scripts/explain_sample.py
```

## Model notes

| Item | Detail |
| --- | --- |
| Task | 12-month default (binary) |
| Algorithm | XGBoost (`hist`) in a scikit-learn pipeline |
| Categorical handling | One-hot inside `ColumnTransformer` |
| Explainability | `shap.TreeExplainer`, interventional, probability output |
| Policy overlay | Score + credit-score + DTI gates → approve / refer / decline |
| Decisioning | Human underwriter; model never books a loan |

Holdout metrics are written to `artifacts/metrics.json` at train time. Retrain after changing the data generator or hyperparameters.

## Why decision support

Credit files need an accountable reviewer. The desk is built so the model is evidence — a probability, a policy band, and the features that mattered — not a silent yes/no. That matches how a model-risk and fair-lending process actually uses machine learning: as an input to a documented human decision.
