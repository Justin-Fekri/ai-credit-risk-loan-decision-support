import type { ShapRow } from "@/lib/api";
import { pct } from "@/lib/format";

export function ShapChart({
  rows,
  expectedPd,
}: {
  rows: ShapRow[];
  expectedPd?: number;
}) {
  const visible = rows.slice(0, 10);
  const maxAbs = Math.max(...visible.map((row) => Math.abs(row.shap)), 0.01);

  return (
    <div>
      <div className="mb-4 flex items-baseline justify-between gap-4">
        <div>
          <h3 className="font-heading text-xl">Why this score</h3>
          <p className="mt-1 text-sm text-muted-foreground">
            TreeSHAP contribution to P(default). Bars to the right raise risk;
            bars to the left lower it versus the book average
            {expectedPd != null ? ` (${pct(expectedPd)}).` : "."}
          </p>
        </div>
      </div>
      {visible.length === 0 ? (
        <p className="rounded-lg border border-dashed border-border p-6 text-sm text-muted-foreground">
          Explanations are still loading for this file.
        </p>
      ) : (
        <ul className="space-y-2.5">
          {visible.map((row) => {
            const width = (Math.abs(row.shap) / maxAbs) * 50;
            const raises = row.shap > 0;
            return (
              <li key={row.feature} className="grid grid-cols-[minmax(0,1fr)_120px_minmax(0,1fr)] items-center gap-2 text-sm">
                <div className="flex justify-end">
                  {!raises && (
                    <div
                      className="h-6 rounded-l-md bg-[oklch(0.62_0.11_155)]"
                      style={{ width: `${width}%` }}
                    />
                  )}
                </div>
                <div className="text-center">
                  <p className="truncate text-xs font-medium">{row.label}</p>
                  <p className="truncate font-mono text-[11px] text-muted-foreground">
                    {row.value}
                  </p>
                </div>
                <div className="flex justify-start">
                  {raises && (
                    <div
                      className="h-6 rounded-r-md bg-[oklch(0.62_0.14_28)]"
                      style={{ width: `${width}%` }}
                    />
                  )}
                </div>
              </li>
            );
          })}
        </ul>
      )}
      <div className="mt-4 flex justify-center gap-6 text-xs text-muted-foreground">
        <span className="flex items-center gap-1.5">
          <span className="size-2 rounded-sm bg-[oklch(0.62_0.11_155)]" />
          Lowers default risk
        </span>
        <span className="flex items-center gap-1.5">
          <span className="size-2 rounded-sm bg-[oklch(0.62_0.14_28)]" />
          Raises default risk
        </span>
      </div>
    </div>
  );
}
