import Link from "next/link";

export default function NotFound() {
  return (
    <div className="rounded-xl border border-dashed border-border p-8">
      <h2 className="font-heading text-2xl">File not found</h2>
      <p className="mt-2 text-sm text-muted-foreground">
        That application is not in the current queue.
      </p>
      <Link
        href="/"
        className="mt-4 inline-flex h-8 items-center rounded-lg bg-primary px-3 text-sm text-primary-foreground"
      >
        Back to queue
      </Link>
    </div>
  );
}
