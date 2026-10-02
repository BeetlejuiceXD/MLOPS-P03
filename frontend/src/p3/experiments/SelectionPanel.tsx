import { ErrorState } from "@/components/ui/ErrorState";
import { Skeleton } from "@/components/ui/Skeleton";
import type { ExperimentRun, SelectionState } from "../contracts";
import { dateTime, percent } from "../format";
import { selectionCrossCheck } from "../selection";

type SelectionFetch =
  | { status: "loading"; reload: () => void }
  | { status: "error"; message: string; reload: () => void }
  | { status: "success"; data: SelectionState; reload: () => void };

/**
 * D05-03 — Candidato y campaña aceptada tal como los entrega la selección (D04-04/D05-02).
 * Experiments no ordena ni elige: muestra el candidato de la API, lo coteja con el run de
 * MLflow y lo rotula como propuesta hasta el cierre formal (D05-08). Un error de la
 * selección se queda en este panel; la lista de runs sigue visible.
 */
export function SelectionPanel({
  selection,
  runs,
}: Readonly<{ selection: SelectionFetch; runs: readonly ExperimentRun[] }>) {
  return (
    <section
      data-testid="experiments-selection"
      aria-label="Selección del candidato"
      className="flex flex-col gap-2 rounded-2xl border border-border bg-surface px-5 py-4 text-sm"
    >
      <h2 className="text-sm font-semibold text-ink">
        Selección por validation (D05-02) — solo lectura
      </h2>
      {selection.status === "loading" && <Skeleton className="h-10" />}
      {selection.status === "error" && (
        <ErrorState
          title="No se pudo leer la selección."
          message={selection.message}
          onRetry={selection.reload}
        />
      )}
      {selection.status === "success" && <SelectionBody state={selection.data} runs={runs} />}
    </section>
  );
}

function SelectionBody({
  state,
  runs,
}: Readonly<{ state: SelectionState; runs: readonly ExperimentRun[] }>) {
  // El contrato garantiza: `open` ⇔ sin candidato.
  if (state.candidate === null) {
    return (
      <p className="text-ink-muted">
        Todavía no hay candidato propuesto: MODEL SELECTION sigue abierto.
      </p>
    );
  }
  const candidate = state.candidate;
  const check = selectionCrossCheck(runs, state);
  // El job sale del run de MLflow del candidato (tags.job_id), no de la selección.
  const candidateJob = check.candidate?.tags.job_id;
  const rows = `Filas de campaña comparables: ${state.campaign_rows.length} (mínimo ${state.min_comparable_runs})`;

  if (check.problems.length > 0) {
    return (
      <div
        data-testid="selection-mismatch"
        className="rounded-xl border border-status-pending/40 bg-status-pending-soft px-4 py-3"
      >
        <p className="font-medium">
          La selección no coincide con lo que Experiments recibe de MLflow: no se presenta como
          aceptada.
        </p>
        <ul className="mt-1 list-disc pl-5 text-xs">
          {check.problems.map((problem) => (
            <li key={problem}>{problem}</li>
          ))}
        </ul>
        <p className="mt-1 text-xs text-ink-muted">{rows}</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-1">
      <p>
        {state.status === "closed" ? "Candidato seleccionado" : "Candidato"}:{" "}
        <span className="font-mono text-xs">{candidate.run_id}</span> · fila OFAT{" "}
        {candidate.campaign_row}
        {candidateJob === undefined ? "" : ` · job ${candidateJob}`} · mejor época{" "}
        {candidate.best_epoch} · val accuracy {percent(candidate.val_accuracy)} · macro-F1{" "}
        {candidate.val_macro_f1.toFixed(4)} · val loss {candidate.val_loss.toFixed(4)}
      </p>
      {state.status === "closed" ? (
        <p className="text-status-done">MODEL SELECTION CLOSED el {dateTime(state.closed_at)}.</p>
      ) : (
        <p className="text-status-pending">
          Propuesta pendiente de cierre (D05-08), propuesta el {dateTime(state.proposed_at)}. No es
          un cierre formal.
        </p>
      )}
      <p className="text-xs text-ink-muted">
        {rows}
        {state.ready_to_close ? "" : " — todavía no se puede cerrar"}. Ranking solo con métricas de
        validation del mejor checkpoint; Experiments no lo recalcula.
      </p>
    </div>
  );
}
