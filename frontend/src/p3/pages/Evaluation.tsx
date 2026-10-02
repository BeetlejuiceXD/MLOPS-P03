import { PageHeader } from "@/pipeline/components/PageHeader";
import { useEvaluation } from "../api";
import { FetchBoundary } from "../components/FetchBoundary";
import { StatePanel } from "../components/StatePanel";
import { dateTime, percent, shortHash } from "../format";

/**
 * D01-05 — Evaluation: frozen test. La API responde `blocked` hasta MODEL
 * SELECTION CLOSED y la página no muestra nada del test en ese estado.
 *
 * D05-05 — Cuatro estados distintos, todos decididos por la API (la página no guarda
 * nada): `blocked` (selección abierta), `pending` (cerrada, sin evaluación todavía),
 * `ready` con su namespace visible (`synthetic` se rotula como recorrido de prueba) y el
 * error de la API con su motivo (fallo del servicio o datos incoherentes, 503).
 */
function NamespaceBadge({ namespace }: Readonly<{ namespace: "official" | "synthetic" }>) {
  return (
    <span
      data-testid="evaluation-namespace"
      className={`rounded-full px-2 py-0.5 font-mono text-xs ${
        namespace === "official"
          ? "bg-status-done-soft text-status-done"
          : "bg-status-pending-soft text-status-pending"
      }`}
    >
      {namespace}
    </span>
  );
}

export function EvaluationPage() {
  const evaluation = useEvaluation();

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Evaluation"
        subtitle="Evaluación final del candidato elegido por validation sobre el test congelado."
      />
      <FetchBoundary results={[evaluation]}>
        {(data) => {
          if (data.state === "blocked") {
            return (
              <StatePanel variant="blocked" title="El test sigue cerrado.">
                No existe MODEL SELECTION CLOSED. {data.detail}
              </StatePanel>
            );
          }
          if (data.state === "pending") {
            return (
              <div
                data-testid="evaluation-pending"
                className="flex flex-col items-center gap-2 rounded-2xl border border-dashed border-border-strong bg-surface px-6 py-12 text-center"
              >
                <NamespaceBadge namespace={data.namespace} />
                <p className="text-sm font-medium text-ink">
                  La selección está cerrada; la evaluación oficial aún no existe.
                </p>
                <p className="max-w-md text-sm text-ink-muted">{data.detail}</p>
                <p className="text-xs text-ink-muted">
                  Candidato{" "}
                  <span className="font-mono">{shortHash(data.selection.candidate_run_id)}</span>
                  {" · "}cerrada {dateTime(data.selection.closed_at)}
                  {" · "}manifest <span className="font-mono">{shortHash(data.manifest_hash)}</span>
                </p>
              </div>
            );
          }
          const { labels, rows } = data.confusion_matrix;
          const hits = labels.reduce((sum, _label, i) => sum + (rows[i]?.[i] ?? 0), 0);
          return (
            <div data-testid="p3-content" className="flex flex-col gap-6">
              {data.namespace === "synthetic" && (
                <div
                  data-testid="evaluation-synthetic-warning"
                  role="note"
                  className="rounded-2xl border border-status-pending/40 bg-status-pending-soft px-5 py-3 text-sm text-ink"
                >
                  Recorrido de prueba con predicciones sintéticas: no es la evaluación oficial del
                  frozen test.
                </div>
              )}
              <p
                data-testid="evaluation-provenance"
                className="flex flex-wrap items-center gap-2 text-xs text-ink-muted"
              >
                <NamespaceBadge namespace={data.namespace} />
                <span>
                  Manifest <span className="font-mono">{shortHash(data.manifest_hash)}</span>
                </span>
                <span>· Evaluada {dateTime(data.evaluated_at)}</span>
              </p>
              <section className="grid gap-4 rounded-2xl border border-border bg-surface p-5 sm:grid-cols-3">
                <div>
                  <h2 className="text-xs text-ink-muted">Candidato</h2>
                  <p className="font-mono text-sm">{shortHash(data.selection.candidate_run_id)}</p>
                  <p className="text-xs text-ink-muted">
                    Selección cerrada {dateTime(data.selection.closed_at)}
                  </p>
                </div>
                <div>
                  <h2 className="text-xs text-ink-muted">Accuracy top-1 (test)</h2>
                  <p className="text-sm font-semibold text-ink">{`${hits} / ${data.n_test}`}</p>
                  <p className="text-xs text-ink-muted">{percent(data.metrics.accuracy)}</p>
                </div>
                <div>
                  <h2 className="text-xs text-ink-muted">Macro F1 · baseline mayoritario</h2>
                  <p className="text-sm text-ink">
                    {percent(data.metrics.macro_f1)} · {percent(data.majority_baseline_accuracy)}
                  </p>
                </div>
              </section>
              <table
                data-testid="confusion-matrix"
                className="w-fit border-collapse text-sm"
                aria-label="Matriz de confusión: filas reales, columnas predichas"
              >
                <thead>
                  <tr>
                    <th className="px-3 py-2 text-xs text-ink-muted">Real \ Predicha</th>
                    {labels.map((label) => (
                      <th key={label} className="px-3 py-2">
                        {label}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {labels.map((label, i) => (
                    <tr key={label}>
                      <th className="px-3 py-2 text-left">{label}</th>
                      {labels.map((column, j) => (
                        <td key={column} className="border border-border px-3 py-2 text-center">
                          {rows[i]?.[j] ?? 0}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          );
        }}
      </FetchBoundary>
    </div>
  );
}
