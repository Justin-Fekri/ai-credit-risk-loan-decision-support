"""Command-line interface: generate-data, train, evaluate, explain, predict."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import __version__
from .config import (
    FEATURES,
    METRICS_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    RAW_DATA_PATH,
    REPORT_DIR,
    TARGET,
)


def _banner(text: str) -> None:
    print(f"\n{text}\n{'=' * len(text)}")


# --- commands --------------------------------------------------------------


def cmd_generate_data(args) -> None:
    from .data import generate_dataset, save_dataset

    _banner("Generating synthetic loan applications")
    df = generate_dataset(n_rows=args.rows, seed=args.seed)
    path = save_dataset(df, Path(args.output))
    print(f"Rows          : {len(df):,}")
    print(f"Default rate  : {df[TARGET].mean():.2%}")
    print(f"Columns       : {len(df.columns)}")
    print(f"Written to    : {path}")


def cmd_train(args) -> None:
    from .train import train

    _banner("Training credit risk model")
    train(
        data_path=Path(args.data) if args.data else None,
        model_path=Path(args.model),
        cost_false_negative=args.cost_fn,
        cost_false_positive=args.cost_fp,
    )


def cmd_evaluate(args) -> None:
    from .data import load_dataset
    from .evaluate import (
        classification_metrics,
        cost_analysis,
        plot_diagnostics,
        review_workload,
        save_metrics,
    )
    from .fairness import audit, format_report, save_report
    from .train import load_bundle, split_data

    _banner("Evaluating model on held-out test data")
    bundle = load_bundle(Path(args.model))
    df = load_dataset(Path(args.data)) if args.data else load_dataset()
    X_train, X_test, y_train, y_test = split_data(df)

    y_prob = bundle["model"].predict_proba(X_test[FEATURES])[:, 1]
    threshold = bundle["threshold"]

    metrics = classification_metrics(y_test, y_prob, threshold)
    metrics["model_name"] = bundle["model_name"]
    metrics["cost_analysis"] = cost_analysis(y_test, y_prob, threshold)
    metrics["decision_bands"] = review_workload(y_test, y_prob, bundle["policy"])
    metrics["training_metadata"] = bundle["training_metadata"]
    metrics["leaderboard"] = bundle["leaderboard"]

    print(f"Model            : {bundle['model_name']}")
    print(f"ROC-AUC          : {metrics['roc_auc']:.4f}")
    print(f"PR-AUC           : {metrics['pr_auc']:.4f}")
    print(f"Brier score      : {metrics['brier_score']:.4f}  (lower = better calibrated)")
    print(f"Threshold        : {threshold:.2f}")
    print(
        f"Precision/Recall : {metrics['precision']:.3f} / {metrics['recall']:.3f}"
        f"   F1 {metrics['f1']:.3f}"
    )

    print("\nHow the portfolio splits across decision bands:")
    print(f"  {'band':<20}{'apps':>7}{'share':>9}{'actual default':>17}")
    for band, stats in metrics["decision_bands"].items():
        rate = stats["actual_default_rate"]
        print(
            f"  {band.replace('_', ' '):<20}{stats['applications']:>7}"
            f"{stats['share_of_portfolio']:>9.1%}"
            f"{(f'{rate:.1%}' if rate is not None else '-'):>17}"
        )

    print()
    report = audit(X_test, y_test, y_prob, threshold)
    print(format_report(report))
    save_report(report, REPORT_DIR / "fairness.json")

    save_metrics(metrics, Path(args.metrics))
    paths = plot_diagnostics(y_test, y_prob)
    print(f"\nMetrics  -> {args.metrics}")
    print(f"Fairness -> {REPORT_DIR / 'fairness.json'}")
    for path in paths:
        print(f"Plot     -> {path}")


def cmd_explain(args) -> None:
    from .data import load_dataset
    from .explain import (
        aggregate_shap,
        global_importance,
        plot_global_importance,
        shap_values,
    )
    from .train import load_bundle, split_data

    _banner("Explaining model behaviour")
    bundle = load_bundle(Path(args.model))
    df = load_dataset(Path(args.data)) if args.data else load_dataset()
    X_train, X_test, y_train, y_test = split_data(df)
    sample = X_test.sample(min(args.sample, len(X_test)), random_state=RANDOM_STATE)
    y_sample = y_test.loc[sample.index]

    print("Permutation importance (drop in ROC-AUC when a feature is shuffled):")
    importance = global_importance(bundle["model"], sample, y_sample, n_repeats=args.repeats)
    for _, row in importance.head(12).iterrows():
        bar = "#" * max(int(row["importance"] * 400), 0)
        print(f"  {row['feature']:<26}{row['importance']:>8.4f}  {bar}")

    path = plot_global_importance(importance)
    print(f"\nChart -> {path}")

    print("\nSHAP mean |contribution| (pipeline-level):")
    values, encoded = shap_values(bundle["uncalibrated_model"], sample)
    if values is None:
        print("  shap not installed -- skipped")
    else:
        summary = aggregate_shap(values, encoded)
        for _, row in summary.head(12).iterrows():
            print(f"  {row['feature']:<26}{row['mean_abs_shap']:>8.4f}")
        summary.to_csv(REPORT_DIR / "shap_importance.csv", index=False)
        print(f"\nSHAP table -> {REPORT_DIR / 'shap_importance.csv'}")

    importance.to_csv(REPORT_DIR / "permutation_importance.csv", index=False)


def cmd_predict(args) -> None:
    from .data import load_dataset
    from .predict import render_adjudication, score_application, score_batch
    from .train import load_bundle, split_data

    bundle = load_bundle(Path(args.model))

    if args.batch:
        _banner(f"Scoring batch: {args.batch}")
        df = pd.read_csv(args.batch)
        scored = score_batch(bundle, df, explain=args.explain)
        output = Path(args.output or REPORT_DIR / "scored_applications.csv")
        output.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(output, index=False)
        print(scored["recommendation"].value_counts().to_string())
        print(f"\nScored {len(scored):,} applications -> {output}")
        return

    if args.application:
        payload = json.loads(Path(args.application).read_text())
        applications = payload if isinstance(payload, list) else [payload]
    else:
        # No input given: demonstrate on a couple of held-out applications.
        df = load_dataset()
        _, X_test, _, _ = split_data(df)
        # Carry the application_id through so the summary is identifiable.
        sample = df.loc[X_test.index[: args.samples]]
        applications = sample.drop(columns=[TARGET]).to_dict(orient="records")
        print(f"(no --application given; showing {len(applications)} held-out cases)")

    df = load_dataset()
    background = df[FEATURES]

    for application in applications:
        result = score_application(bundle, application, background=background)
        print()
        print(render_adjudication(result))


# --- parser ----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="credit-risk",
        description="AI Credit Risk & Loan Decision Support System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate-data", help="create the synthetic dataset")
    p.add_argument("--rows", type=int, default=12_000)
    p.add_argument("--seed", type=int, default=RANDOM_STATE)
    p.add_argument("--output", default=str(RAW_DATA_PATH))
    p.set_defaults(func=cmd_generate_data)

    p = sub.add_parser("train", help="cross-validate, select, calibrate, and save a model")
    p.add_argument("--data", default=None, help="path to a CSV (defaults to the bundled data)")
    p.add_argument("--model", default=str(MODEL_PATH))
    p.add_argument("--cost-fn", type=float, default=5.0,
                   help="relative cost of approving a borrower who defaults")
    p.add_argument("--cost-fp", type=float, default=1.0,
                   help="relative cost of declining a borrower who would have repaid")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="metrics, decision bands, fairness audit, plots")
    p.add_argument("--data", default=None)
    p.add_argument("--model", default=str(MODEL_PATH))
    p.add_argument("--metrics", default=str(METRICS_PATH))
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("explain", help="global feature importance and SHAP summary")
    p.add_argument("--data", default=None)
    p.add_argument("--model", default=str(MODEL_PATH))
    p.add_argument("--sample", type=int, default=800)
    p.add_argument("--repeats", type=int, default=8)
    p.set_defaults(func=cmd_explain)

    p = sub.add_parser("predict", help="score applications and print reviewer summaries")
    p.add_argument("--model", default=str(MODEL_PATH))
    p.add_argument("--application", help="path to a JSON file (object or list of objects)")
    p.add_argument("--batch", help="path to a CSV of applications")
    p.add_argument("--output", help="where to write batch results")
    p.add_argument("--explain", action="store_true", help="add reason codes to batch output")
    p.add_argument("--samples", type=int, default=2)
    p.set_defaults(func=cmd_predict)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0
