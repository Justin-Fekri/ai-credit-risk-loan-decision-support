import type { Application } from "@/lib/api";
import { formatFeature, featureLabel } from "@/lib/format";

const GROUPS = [
  {
    title: "Capacity",
    keys: ["annual_income", "monthly_debt", "debt_to_income", "loan_amount"],
  },
  {
    title: "Credit file",
    keys: [
      "credit_score",
      "credit_history_length_years",
      "revolving_utilization",
      "num_open_accounts",
      "num_delinquencies_24m",
      "num_inquiries_6m",
      "public_records",
    ],
  },
  {
    title: "Profile",
    keys: [
      "age",
      "employment_status",
      "employment_length_years",
      "home_ownership",
      "loan_purpose",
      "verification_status",
    ],
  },
];

export function BorrowerFacts({ app }: { app: Application }) {
  return (
    <div className="space-y-5">
      {GROUPS.map((group) => (
        <div key={group.title}>
          <p className="mb-2 text-[11px] tracking-[0.16em] uppercase text-muted-foreground">
            {group.title}
          </p>
          <dl className="divide-y divide-border rounded-xl border border-border bg-card">
            {group.keys.map((key) => (
              <div
                key={key}
                className="flex items-baseline justify-between gap-4 px-3 py-2"
              >
                <dt className="text-xs text-muted-foreground">
                  {featureLabel(key)}
                </dt>
                <dd className="font-mono text-sm">
                  {formatFeature(key, app.features[key])}
                </dd>
              </div>
            ))}
          </dl>
        </div>
      ))}
    </div>
  );
}
