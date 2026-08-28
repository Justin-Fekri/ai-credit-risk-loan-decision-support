import { pct } from "@/lib/format";
import { cn } from "@/lib/utils";

export function PdMeter({
  value,
  band,
}: {
  value: number;
  band: string;
}) {
  const color = {
    low: "text-[oklch(0.4_0.1_155)]",
    moderate: "text-[oklch(0.5_0.12_75)]",
    elevated: "text-[oklch(0.5_0.14_45)]",
    severe: "text-[oklch(0.48_0.16_28)]",
  }[band] ?? "text-foreground";

  const ticks = [0, 0.1, 0.18, 0.32, 0.5];

  return (
    <div>
      <p className="text-xs tracking-[0.16em] uppercase text-muted-foreground">
        Estimated 12-month default
      </p>
      <p className={cn("mt-1 font-heading text-5xl tabular-nums", color)}>
        {pct(value)}
      </p>
      <div className="relative mt-4 h-2.5 overflow-hidden rounded-full bg-muted">
        <div
          className="absolute inset-y-0 left-0 rounded-full bg-primary"
          style={{ width: `${Math.min(value / 0.6, 1) * 100}%` }}
        />
      </div>
      <div className="mt-1 flex justify-between font-mono text-[10px] text-muted-foreground">
        {ticks.map((tick) => (
          <span key={tick}>{pct(tick, 0)}</span>
        ))}
      </div>
    </div>
  );
}
