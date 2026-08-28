import { serverApi } from "@/lib/server-api";
import { pct } from "@/lib/format";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export const dynamic = "force-dynamic";

export default async function ModelPage() {
  let data;
  try {
    data = await serverApi.model();
  } catch (err) {
    return (
      <div className="rounded-xl border border-dashed p-8">
        <h2 className="font-heading text-2xl">Model card unavailable</h2>
        <p className="mt-2 text-sm text-muted-foreground">
          {err instanceof Error ? err.message : "Could not reach the scoring service."}
        </p>
      </div>
    );
  }

  const maxImp = Math.max(...data.global_importance.map((row) => row.mean_abs_shap), 0.001);
  const { confusion } = data.metrics;

  return (
    <div className="mx-auto max-w-6xl space-y-8">
      <div>
        <p className="text-[11px] tracking-[0.2em] uppercase text-primary">
          Transparency
        </p>
        <h2 className="mt-1 font-heading text-4xl">Model card</h2>
        <p className="mt-2 max-w-2xl text-sm leading-relaxed text-muted-foreground">
          Binary XGBoost classifier inside a scikit-learn pipeline. It estimates
          12-month default risk from credit and financial attributes, then a
          separate policy layer maps that probability to approve / refer /
          decline. A person still records the decision.
        </p>
      </div>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Metric label="ROC-AUC" value={data.metrics.roc_auc.toFixed(3)} />
        <Metric label="Gini" value={data.metrics.gini.toFixed(3)} />
        <Metric label="KS" value={data.metrics.ks.toFixed(3)} />
        <Metric label="PR-AUC" value={data.metrics.pr_auc.toFixed(3)} />
        <Metric label="Brier" value={data.metrics.brier.toFixed(3)} />
        <Metric label="Precision" value={data.metrics.precision.toFixed(3)} />
        <Metric label="Recall" value={data.metrics.recall.toFixed(3)} />
        <Metric label="Book default rate" value={pct(data.metrics.default_rate)} />
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Global drivers</CardTitle>
            <p className="text-sm text-muted-foreground">
              Mean |TreeSHAP| on a held-out sample. These are the borrower
              fields the model uses most, not a causal ranking.
            </p>
          </CardHeader>
          <CardContent className="space-y-2">
            {data.global_importance.slice(0, 10).map((row) => (
              <div key={row.feature}>
                <div className="mb-1 flex justify-between text-xs">
                  <span>{row.label}</span>
                  <span className="font-mono">{row.mean_abs_shap.toFixed(3)}</span>
                </div>
                <div className="h-2 rounded-full bg-muted">
                  <div
                    className="h-2 rounded-full bg-primary"
                    style={{ width: `${(row.mean_abs_shap / maxImp) * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Holdout confusion</CardTitle>
            <p className="text-sm text-muted-foreground">
              Operating point 0.50 on the test fold ({data.metrics.n_test} files).
              Used for monitoring, not as an origination cutoff.
            </p>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-2 text-center">
              <Conf cell="True negative" value={confusion.tn} />
              <Conf cell="False positive" value={confusion.fp} />
              <Conf cell="False negative" value={confusion.fn} />
              <Conf cell="True positive" value={confusion.tp} />
            </div>
            <div className="mt-6">
              <p className="mb-2 text-xs uppercase tracking-[0.14em] text-muted-foreground">
                ROC
              </p>
              <svg viewBox="0 0 100 100" className="h-48 w-full rounded-lg bg-muted/40">
                <line x1="0" y1="100" x2="100" y2="0" stroke="currentColor" strokeOpacity="0.15" />
                <polyline
                  fill="none"
                  stroke="oklch(0.38 0.055 168)"
                  strokeWidth="2"
                  points={data.metrics.roc_curve
                    .map((pt) => `${pt.fpr * 100},${100 - pt.tpr * 100}`)
                    .join(" ")}
                />
              </svg>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Limits of this system</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm leading-relaxed text-muted-foreground">
          <p>
            Training data is fully synthetic. It is designed to look like a
            consumer-loan book so the pipeline, explanations, and workbench can
            be demonstrated without using real applicant records.
          </p>
          <p>
            The score is not a credit decision. Adverse-action, fair-lending,
            and model-risk rules in a live lender would require independent
            validation, monitoring, and a documented policy overlay. This desk
            is built so a person remains accountable for every file.
          </p>
          <p>
            Algorithm: {String(data.card.algorithm)}. Explainability:{" "}
            {String(data.card.explainability)}.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <Card>
      <CardContent>
        <p className="text-[11px] tracking-[0.14em] uppercase text-muted-foreground">
          {label}
        </p>
        <p className="mt-1 font-heading text-3xl tabular-nums">{value}</p>
      </CardContent>
    </Card>
  );
}

function Conf({ cell, value }: { cell: string; value: number }) {
  return (
    <div className="rounded-lg bg-muted/60 p-3">
      <p className="text-[11px] text-muted-foreground">{cell}</p>
      <p className="font-heading text-2xl">{value}</p>
    </div>
  );
}
