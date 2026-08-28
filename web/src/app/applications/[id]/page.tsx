"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { api, type Application } from "@/lib/api";
import { BorrowerFacts } from "@/components/borrower-facts";
import { DecisionPanel } from "@/components/decision-panel";
import { PdMeter } from "@/components/pd-meter";
import { RecBadge, RiskBadge } from "@/components/risk-badge";
import { ShapChart } from "@/components/shap-chart";
import { money } from "@/lib/format";

export default function ApplicationPage() {
  const params = useParams<{ id: string }>();
  const [app, setApp] = useState<Application | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!params.id) return;
    api
      .application(params.id)
      .then(setApp)
      .catch((err) =>
        setError(err instanceof Error ? err.message : "File could not be opened.")
      );
  }, [params.id]);

  if (error) {
    return (
      <div className="rounded-xl border border-dashed border-border p-8">
        <h2 className="font-heading text-2xl">File not found</h2>
        <p className="mt-2 text-sm text-muted-foreground">{error}</p>
        <Link href="/" className="mt-4 inline-flex h-8 items-center rounded-lg bg-primary px-3 text-sm text-primary-foreground">
          Back to queue
        </Link>
      </div>
    );
  }

  if (!app) {
    return <p className="text-sm text-muted-foreground">Opening file…</p>;
  }

  return (
    <div className="mx-auto max-w-6xl space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <Link href="/" className="text-xs text-muted-foreground hover:underline">
            ← Queue
          </Link>
          <p className="mt-2 font-mono text-xs text-muted-foreground">
            {app.application_id}
          </p>
          <h2 className="font-heading text-4xl">{app.borrower_name}</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            {money(Number(app.features.loan_amount))} ·{" "}
            {String(app.features.loan_purpose).replaceAll("_", " ")}
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <RiskBadge band={app.risk_band} />
          <RecBadge rec={app.policy_recommendation} />
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1.1fr)_minmax(0,0.9fr)]">
        <div className="space-y-6">
          <div className="rounded-xl border border-border bg-card p-5">
            <PdMeter value={app.pd_default} band={app.risk_band} />
            <p className="mt-5 border-t border-border pt-4 text-sm leading-relaxed">
              <span className="font-medium">{app.eligibility}.</span>{" "}
              {app.policy_rationale}
            </p>
          </div>
          <div className="rounded-xl border border-border bg-card p-5">
            <ShapChart rows={app.shap ?? []} expectedPd={app.expected_pd} />
          </div>
        </div>
        <div className="space-y-6">
          <BorrowerFacts app={app} />
          <DecisionPanel app={app} />
        </div>
      </div>
    </div>
  );
}
