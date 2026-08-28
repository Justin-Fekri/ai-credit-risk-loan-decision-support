"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Textarea } from "@/components/ui/textarea";
import { buttonVariants } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { api, type Application } from "@/lib/api";
import { RecBadge } from "@/components/risk-badge";

export function DecisionPanel({ app }: { app: Application }) {
  const router = useRouter();
  const [notes, setNotes] = useState(app.decision_notes ?? "");
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState<string | null>(null);

  if (app.status === "reviewed" || app.decision) {
    return (
      <div className="rounded-xl border border-border bg-card p-5">
        <p className="text-[11px] tracking-[0.16em] uppercase text-muted-foreground">
          Recorded decision
        </p>
        <div className="mt-3">
          <RecBadge rec={app.decision ?? ""} />
        </div>
        <p className="mt-3 text-sm leading-relaxed">{app.decision_notes}</p>
        <p className="mt-2 text-xs text-muted-foreground">
          {app.underwriter}
          {app.decided_at ? ` · ${new Date(app.decided_at).toLocaleString()}` : ""}
        </p>
      </div>
    );
  }

  async function submit(decision: "approve" | "refer" | "decline") {
    if (!app.application_id) return;
    if (notes.trim().length < 8) {
      setError("Add a short rationale — this file stays in the audit trail.");
      return;
    }
    setError(null);
    setPending(decision);
    try {
      await api.decide(app.application_id, decision, notes.trim());
      router.refresh();
      router.push("/");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not record the decision.");
    } finally {
      setPending(null);
    }
  }

  return (
    <div className="rounded-xl border border-border bg-card p-5">
      <p className="text-[11px] tracking-[0.16em] uppercase text-muted-foreground">
        Underwriter action
      </p>
      <h3 className="mt-1 font-heading text-xl">Your decision</h3>
      <p className="mt-2 text-sm text-muted-foreground">
        Policy suggestion: {app.eligibility}. Override it if the file warrants
        it, and say why.
      </p>
      <Textarea
        className="mt-4 min-h-28"
        placeholder="e.g. Verified overtime history offsets DTI. Residual income covers the payment with a 1.4x cushion."
        value={notes}
        onChange={(event) => setNotes(event.target.value)}
      />
      {error ? <p className="mt-2 text-sm text-destructive">{error}</p> : null}
      <div className="mt-4 grid grid-cols-1 gap-2 sm:grid-cols-3">
        <button
          type="button"
          disabled={pending !== null}
          onClick={() => submit("approve")}
          className={cn(
            buttonVariants(),
            "bg-[oklch(0.4_0.08_155)] hover:bg-[oklch(0.35_0.08_155)]"
          )}
        >
          {pending === "approve" ? "Saving…" : "Approve"}
        </button>
        <button
          type="button"
          disabled={pending !== null}
          onClick={() => submit("refer")}
          className={cn(buttonVariants({ variant: "outline" }))}
        >
          {pending === "refer" ? "Saving…" : "Refer"}
        </button>
        <button
          type="button"
          disabled={pending !== null}
          onClick={() => submit("decline")}
          className={cn(buttonVariants({ variant: "destructive" }))}
        >
          {pending === "decline" ? "Saving…" : "Decline"}
        </button>
      </div>
    </div>
  );
}
