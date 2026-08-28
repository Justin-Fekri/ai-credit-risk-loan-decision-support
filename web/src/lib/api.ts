export type ShapRow = {
  feature: string;
  label: string;
  value: string;
  raw_value: string | number;
  shap: number;
  direction: "raises_risk" | "lowers_risk";
};

export type Application = {
  application_id: string | null;
  borrower_name: string | null;
  features: Record<string, string | number>;
  pd_default: number;
  risk_band: "low" | "moderate" | "elevated" | "severe" | string;
  policy_recommendation: "approve" | "refer" | "decline" | string;
  policy_rationale: string;
  eligibility: string;
  shap?: ShapRow[];
  top_drivers?: ShapRow[];
  expected_pd?: number;
  status: string;
  decision: string | null;
  decision_notes: string | null;
  underwriter?: string;
  decided_at?: string;
};

export type QueueResponse = {
  items: Application[];
  summary: {
    pending: number;
    approved: number;
    referred: number;
    declined: number;
    high_risk: number;
  };
};

export type ModelResponse = {
  card: Record<string, string | number | boolean>;
  metrics: {
    roc_auc: number;
    gini: number;
    pr_auc: number;
    brier: number;
    ks: number;
    precision: number;
    recall: number;
    f1: number;
    default_rate: number;
    n_test: number;
    confusion: { tn: number; fp: number; fn: number; tp: number };
    roc_curve: { fpr: number; tpr: number }[];
  };
  global_importance: { feature: string; label: string; mean_abs_shap: number }[];
  expected_pd: number;
};

export type ReferenceResponse = {
  features: Record<string, string>;
  purpose_labels: Record<string, string>;
  enums: Record<string, string[]>;
};

async function parse<T>(res: Response | Promise<Response>): Promise<T> {
  const resolved = await res;
  if (!resolved.ok) {
    const text = await resolved.text();
    throw new Error(text || `Request failed (${resolved.status})`);
  }
  return resolved.json() as Promise<T>;
}

export const api = {
  queue: () => parse<QueueResponse>(fetch("/backend/queue")),
  application: (id: string) =>
    parse<Application>(fetch(`/backend/applications/${id}`)),
  model: () => parse<ModelResponse>(fetch("/backend/model")),
  reference: () => parse<ReferenceResponse>(fetch("/backend/reference")),
  score: (features: Record<string, string | number>) =>
    parse<Application>(
      fetch("/backend/score", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(features),
      })
    ),
  decide: (id: string, decision: string, notes: string) =>
    parse<Application>(
      fetch(`/backend/applications/${id}/decision`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ decision, notes, underwriter: "A. Chen" }),
      })
    ),
};
