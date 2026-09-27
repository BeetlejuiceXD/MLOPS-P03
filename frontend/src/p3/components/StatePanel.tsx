import type { ReactNode } from "react";

/**
 * D01-05 — Estado vacío o bloqueado de las páginas P3. `blocked` significa que
 * una regla del protocolo (#33) impide la acción, no que haya fallado la red:
 * los errores HTTP y las respuestas fuera de contrato usan `ErrorState`.
 */
export function StatePanel({
  variant,
  title,
  children,
}: Readonly<{
  variant: "empty" | "blocked";
  title: string;
  children?: ReactNode;
}>) {
  return (
    <div
      data-testid={`p3-state-${variant}`}
      className={`flex flex-col items-center justify-center gap-2 rounded-2xl border px-6 py-12 text-center ${
        variant === "blocked"
          ? "border-status-pending/40 bg-status-pending-soft"
          : "border-dashed border-border-strong bg-surface"
      }`}
    >
      {variant === "blocked" && (
        <span className="text-xs font-semibold uppercase tracking-wide text-status-pending">
          Bloqueado
        </span>
      )}
      <p className="text-sm font-medium text-ink">{title}</p>
      {children && <div className="max-w-md text-sm text-ink-muted">{children}</div>}
    </div>
  );
}
