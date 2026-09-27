import { PageHeader } from "@/pipeline/components/PageHeader";
import { useEvaluation } from "../api";
import { FetchBoundary } from "../components/FetchBoundary";
import { StatePanel } from "../components/StatePanel";
import { dateTime, percent, shortHash } from "../format";

/**
 * D01-05 — Evaluation: frozen test. La API responde `blocked` hasta MODEL
 * SELECTION CLOSED y la página no muestra nada del test en ese estado.
 */
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
          const { labels, rows } = data.confusion_matrix;
          const hits = labels.reduce((sum, _label, i) => sum + (rows[i]?.[i] ?? 0), 0);
          return (
            <div data-testid="p3-content" className="flex flex-col gap-6">
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
