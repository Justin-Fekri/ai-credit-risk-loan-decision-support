import { cn } from "@/lib/utils";
import { recCopy, riskCopy } from "@/lib/format";

const RISK: Record<string, string> = {
  low: "bg-[oklch(0.93_0.04_155)] text-[oklch(0.32_0.08_155)]",
  moderate: "bg-[oklch(0.95_0.04_85)] text-[oklch(0.42_0.1_70)]",
  elevated: "bg-[oklch(0.94_0.05_55)] text-[oklch(0.42_0.12_45)]",
  severe: "bg-[oklch(0.94_0.04_28)] text-[oklch(0.42_0.14_28)]",
};

const REC: Record<string, string> = {
  approve: "bg-[oklch(0.93_0.04_155)] text-[oklch(0.32_0.08_155)]",
  refer: "bg-[oklch(0.94_0.04_80)] text-[oklch(0.4_0.1_70)]",
  decline: "bg-[oklch(0.94_0.04_28)] text-[oklch(0.42_0.14_28)]",
};

export function RiskBadge({ band }: { band: string }) {
  return (
    <span
      className={cn(
        "inline-flex h-6 items-center rounded-full px-2.5 text-xs font-medium",
        RISK[band] ?? "bg-muted text-muted-foreground"
      )}
    >
      {riskCopy(band)} risk
    </span>
  );
}

export function RecBadge({ rec }: { rec: string }) {
  return (
    <span
      className={cn(
        "inline-flex h-6 items-center rounded-full px-2.5 text-xs font-medium",
        REC[rec] ?? "bg-muted"
      )}
    >
      {recCopy(rec)}
    </span>
  );
}
