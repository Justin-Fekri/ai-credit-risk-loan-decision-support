"use client";

import { useMemo, useState } from "react";
import { buttonVariants } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { api, type Application } from "@/lib/api";
import { PdMeter } from "@/components/pd-meter";
import { RecBadge, RiskBadge } from "@/components/risk-badge";
import { ShapChart } from "@/components/shap-chart";
import { PURPOSE_LABELS } from "@/lib/copy";

const DEFAULTS: Record<string, string | number> = {
  annual_income: 72000,
  loan_amount: 14000,
  debt_to_income: 0.31,
  credit_score: 694,
  employment_length_years: 4.5,
  credit_history_length_years: 9,
  revolving_utilization: 0.41,
  num_open_accounts: 8,
  num_delinquencies_24m: 0,
  num_inquiries_6m: 1,
  public_records: 0,
  monthly_debt: 1480,
  age: 34,
  home_ownership: "RENT",
  loan_purpose: "debt_consolidation",
  employment_status: "employed",
  verification_status: "source_verified",
};

const SELECTS: Record<string, string[]> = {
  home_ownership: ["MORTGAGE", "OWN", "RENT", "OTHER"],
  loan_purpose: Object.keys(PURPOSE_LABELS),
  employment_status: ["employed", "self_employed", "retired", "unemployed"],
  verification_status: ["verified", "source_verified", "not_verified"],
};

const FIELDS: { key: string; label: string; step?: string }[] = [
  { key: "annual_income", label: "Annual income" },
  { key: "loan_amount", label: "Requested amount" },
  { key: "monthly_debt", label: "Monthly debt service" },
  { key: "debt_to_income", label: "DTI (0–1)", step: "0.01" },
  { key: "credit_score", label: "Credit score" },
  { key: "revolving_utilization", label: "Utilization (0–1)", step: "0.01" },
  { key: "employment_length_years", label: "Years in role", step: "0.1" },
  { key: "credit_history_length_years", label: "Credit history (years)", step: "0.1" },
  { key: "num_open_accounts", label: "Open accounts" },
  { key: "num_delinquencies_24m", label: "Delinquencies (24 mo)" },
  { key: "num_inquiries_6m", label: "Inquiries (6 mo)" },
  { key: "public_records", label: "Public records" },
  { key: "age", label: "Age" },
];

export default function ScorePage() {
  const [form, setForm] = useState(DEFAULTS);
  const [result, setResult] = useState<Application | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const payload = useMemo(() => {
    const next: Record<string, string | number> = { ...form };
    for (const key of Object.keys(next)) {
      if (!(key in SELECTS) && typeof next[key] === "string") {
        next[key] = Number(next[key]);
      }
    }
    return next;
  }, [form]);

  async function onSubmit(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      setResult(await api.score(payload));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Scoring failed.");
      setResult(null);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto grid max-w-6xl gap-8 lg:grid-cols-[minmax(0,0.95fr)_minmax(0,1.05fr)]">
      <div>
        <p className="text-[11px] tracking-[0.2em] uppercase text-primary">
          Ad-hoc scoring
        </p>
        <h2 className="mt-1 font-heading text-4xl">Score a file</h2>
        <p className="mt-2 text-sm text-muted-foreground">
          Enter borrower attributes to estimate default risk and see which
          fields the model leans on. Nothing here originates a loan.
        </p>
        <form onSubmit={onSubmit} className="mt-6 space-y-4">
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            {FIELDS.map((field) => (
              <label key={field.key} className="block text-xs">
                <span className="mb-1 block text-muted-foreground">{field.label}</span>
                <input
                  type="number"
                  step={field.step ?? "1"}
                  value={form[field.key]}
                  onChange={(event) =>
                    setForm((prev) => ({ ...prev, [field.key]: event.target.value }))
                  }
                  required
                  className="h-8 w-full rounded-lg border border-input bg-card px-2.5 text-sm outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"
                />
              </label>
            ))}
            {Object.entries(SELECTS).map(([key, options]) => (
              <label key={key} className="block text-xs">
                <span className="mb-1 block text-muted-foreground">
                  {key.replaceAll("_", " ")}
                </span>
                <select
                  className="h-8 w-full rounded-lg border border-input bg-card px-2 text-sm"
                  value={String(form[key])}
                  onChange={(event) =>
                    setForm((prev) => ({ ...prev, [key]: event.target.value }))
                  }
                >
                  {options.map((option) => (
                    <option key={option} value={option}>
                      {PURPOSE_LABELS[option] ?? option.replaceAll("_", " ")}
                    </option>
                  ))}
                </select>
              </label>
            ))}
          </div>
          {error ? <p className="text-sm text-destructive">{error}</p> : null}
          <button
            type="submit"
            disabled={busy}
            className={cn(buttonVariants({ size: "lg" }))}
          >
            {busy ? "Scoring…" : "Estimate default risk"}
          </button>
        </form>
      </div>
      <div>
        {!result ? (
          <div className="rounded-xl border border-dashed border-border bg-card/50 p-8 text-sm text-muted-foreground">
            Results land here: probability of default, policy band, and the
            borrower features that most influenced the score.
          </div>
        ) : (
          <div className="space-y-5">
            <div className="flex flex-wrap gap-2">
              <RiskBadge band={result.risk_band} />
              <RecBadge rec={result.policy_recommendation} />
            </div>
            <div className="rounded-xl border border-border bg-card p-5">
              <PdMeter value={result.pd_default} band={result.risk_band} />
              <p className="mt-4 text-sm leading-relaxed">{result.policy_rationale}</p>
            </div>
            <div className="rounded-xl border border-border bg-card p-5">
              <ShapChart rows={result.shap ?? []} expectedPd={result.expected_pd} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
