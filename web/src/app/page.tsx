import Link from "next/link";
import { serverApi } from "@/lib/server-api";
import type { QueueResponse } from "@/lib/api";
import { RecBadge, RiskBadge } from "@/components/risk-badge";
import { money, pct } from "@/lib/format";
import { Card, CardContent } from "@/components/ui/card";

export const dynamic = "force-dynamic";

export default async function QueuePage() {
  let data: QueueResponse;
  try {
    data = await serverApi.queue();
  } catch (err) {
    return (
      <EmptyState
        title="Queue unavailable"
        body="Start the Python API on port 45232, then refresh. The workbench reads the scoring service directly."
        detail={err instanceof Error ? err.message : String(err)}
      />
    );
  }

  const pending = data.items.filter((item) => item.status === "pending");
  const reviewed = data.items.filter((item) => item.status !== "pending");

  return (
    <div className="mx-auto max-w-6xl space-y-8">
      <div>
        <p className="text-[11px] tracking-[0.2em] uppercase text-primary">
          Personal loans
        </p>
        <h2 className="mt-1 font-heading text-4xl">Review queue</h2>
        <p className="mt-2 max-w-2xl text-sm leading-relaxed text-muted-foreground">
          Each file has a modelled 12-month default probability and a policy
          recommendation. Open a file to see which borrower attributes moved
          the score, then record your own decision.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
        <Stat label="Pending" value={String(data.summary.pending)} />
        <Stat label="High risk pending" value={String(data.summary.high_risk)} />
        <Stat label="Approved today" value={String(data.summary.approved)} />
        <Stat label="Referred" value={String(data.summary.referred)} />
        <Stat label="Declined" value={String(data.summary.declined)} />
      </div>

      {pending.length === 0 ? (
        <EmptyState
          title="Nothing waiting"
          body="Every file in this session has an underwriter decision. Score a new applicant, or review the model card."
        />
      ) : (
        <FileTable items={pending} caption="Awaiting decision" />
      )}

      {reviewed.length > 0 ? (
        <FileTable items={reviewed} caption="Already reviewed" />
      ) : null}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <Card>
      <CardContent className="pt-1">
        <p className="text-[11px] tracking-[0.14em] uppercase text-muted-foreground">
          {label}
        </p>
        <p className="mt-1 font-heading text-3xl">{value}</p>
      </CardContent>
    </Card>
  );
}

function FileTable({
  items,
  caption,
}: {
  items: QueueResponse["items"];
  caption: string;
}) {
  return (
    <section>
      <h3 className="mb-3 font-heading text-2xl">{caption}</h3>
      <div className="overflow-x-auto rounded-xl border border-border bg-card">
        <table className="w-full text-sm">
          <thead className="bg-muted/60 text-left text-[11px] tracking-[0.12em] uppercase text-muted-foreground">
            <tr>
              <th className="px-4 py-3">File</th>
              <th className="px-4 py-3">Borrower</th>
              <th className="hidden px-4 py-3 md:table-cell">Amount</th>
              <th className="hidden px-4 py-3 lg:table-cell">Score</th>
              <th className="px-4 py-3">P(default)</th>
              <th className="hidden px-4 py-3 sm:table-cell">Policy</th>
            </tr>
          </thead>
          <tbody>
            {items.map((item) => (
              <tr key={item.application_id} className="border-t border-border">
                <td className="px-4 py-3 font-mono text-xs">
                  <Link
                    href={`/applications/${item.application_id}`}
                    className="underline-offset-2 hover:underline"
                  >
                    {item.application_id}
                  </Link>
                </td>
                <td className="px-4 py-3">
                  <Link href={`/applications/${item.application_id}`} className="font-medium">
                    {item.borrower_name}
                  </Link>
                </td>
                <td className="hidden px-4 py-3 md:table-cell">
                  {money(Number(item.features.loan_amount))}
                </td>
                <td className="hidden px-4 py-3 lg:table-cell">
                  {item.features.credit_score}
                </td>
                <td className="px-4 py-3">
                  <div className="flex flex-col items-start gap-1">
                    <span className="font-mono">{pct(item.pd_default)}</span>
                    <RiskBadge band={item.risk_band} />
                  </div>
                </td>
                <td className="hidden px-4 py-3 sm:table-cell">
                  <RecBadge rec={item.policy_recommendation} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function EmptyState({
  title,
  body,
  detail,
}: {
  title: string;
  body: string;
  detail?: string;
}) {
  return (
    <div className="rounded-xl border border-dashed border-border bg-card/60 p-8">
      <h3 className="font-heading text-2xl">{title}</h3>
      <p className="mt-2 max-w-xl text-sm text-muted-foreground">{body}</p>
      {detail ? (
        <pre className="mt-4 overflow-x-auto text-xs text-destructive">{detail}</pre>
      ) : null}
    </div>
  );
}
