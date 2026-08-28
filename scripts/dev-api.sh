#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export CREDIT_RISK_ARTIFACTS="${CREDIT_RISK_ARTIFACTS:-$ROOT/artifacts}"

if [[ ! -f "$CREDIT_RISK_ARTIFACTS/model.joblib" ]]; then
  python scripts/train.py
fi

exec python -m uvicorn api.main:app --host 0.0.0.0 --port 45232
