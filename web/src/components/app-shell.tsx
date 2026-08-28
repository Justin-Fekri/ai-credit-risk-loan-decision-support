"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ClipboardList, Gauge, Scale, Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/", label: "Review queue", icon: ClipboardList },
  { href: "/score", label: "Score a file", icon: Gauge },
  { href: "/model", label: "Model card", icon: Scale },
];

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();

  return (
    <div className="flex min-h-screen">
      <aside className="hidden w-64 shrink-0 flex-col bg-sidebar text-sidebar-foreground md:flex">
        <div className="px-5 pt-6 pb-5">
          <p className="text-[11px] tracking-[0.22em] uppercase text-sidebar-primary">
            Northbridge Credit
          </p>
          <h1 className="mt-2 font-heading text-2xl leading-tight text-white">
            Risk desk
          </h1>
          <p className="mt-2 text-xs leading-relaxed text-sidebar-foreground/70">
            Decision support for personal-loan underwriters. The model recommends.
            You decide.
          </p>
        </div>
        <nav className="flex flex-1 flex-col gap-1 px-3">
          {NAV.map((item) => {
            const active =
              item.href === "/"
                ? pathname === "/" || pathname.startsWith("/applications")
                : pathname.startsWith(item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "flex items-center gap-2 rounded-lg px-3 py-2 text-sm transition-colors",
                  active
                    ? "bg-sidebar-accent text-sidebar-accent-foreground"
                    : "text-sidebar-foreground/75 hover:bg-sidebar-accent/70 hover:text-white"
                )}
              >
                <item.icon className="size-4" />
                {item.label}
              </Link>
            );
          })}
        </nav>
        <div className="m-3 rounded-xl border border-sidebar-border bg-sidebar-accent/50 p-3 text-xs leading-relaxed text-sidebar-foreground/75">
          <Sparkles className="mb-2 size-4 text-sidebar-primary" />
          TreeSHAP explanations are probability contributions, grouped back to
          original borrower fields. Not an automated origination engine.
        </div>
      </aside>
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center justify-between border-b border-border bg-card/80 px-4 py-3 backdrop-blur md:px-8">
          <div className="md:hidden">
            <p className="text-[10px] tracking-[0.2em] uppercase text-primary">
              Northbridge
            </p>
            <p className="font-heading text-lg">Risk desk</p>
          </div>
          <p className="hidden text-sm text-muted-foreground md:block">
            Human-in-the-loop underwriting · book 2026Q3
          </p>
          <div className="text-right">
            <p className="text-sm font-medium">A. Chen</p>
            <p className="text-xs text-muted-foreground">Senior underwriter</p>
          </div>
        </header>
        <div className="flex gap-2 overflow-x-auto border-b border-border px-4 py-2 md:hidden">
          {NAV.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className="rounded-full border border-border bg-card px-3 py-1 text-xs"
            >
              {item.label}
            </Link>
          ))}
        </div>
        <main className="flex-1 px-4 py-6 md:px-8 md:py-8">{children}</main>
      </div>
    </div>
  );
}
