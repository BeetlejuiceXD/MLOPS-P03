import { PageHeader } from "@/pipeline/components/PageHeader";
import { useExperimentRuns } from "../api";
import { FetchBoundary } from "../components/FetchBoundary";
import { StatePanel } from "../components/StatePanel";
import { percent, shortHash } from "../format";

/** D01-05 — Experiments: corridas del experimento MLflow `p3-cnn-classifier`. */
export function ExperimentsPage() {
  const runs = useExperimentRuns();

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Experiments"
        subtitle="Corridas registradas en MLflow; la selección usa solo métricas de validation."
      />
      <FetchBoundary results={[runs]}>
        {(data) =>
          data.runs.length === 0 ? (
            <StatePanel variant="empty" title="Todavía no hay corridas en MLflow.">
              La campaña de al menos diez corridas válidas aparecerá aquí.
            </StatePanel>
          ) : (
            <div
              data-testid="p3-content"
              className="overflow-x-auto rounded-2xl border border-border"
            >
              <table className="w-full text-left text-sm">
                <thead className="bg-surface text-xs text-ink-muted">
                  <tr>
                    <th className="px-3 py-2">Run</th>
                    <th className="px-3 py-2">Estado</th>
                    <th className="px-3 py-2">Seed</th>
                    <th className="px-3 py-2">Capas</th>
                    <th className="px-3 py-2">LR</th>
                    <th className="px-3 py-2">Optimizador</th>
                    <th className="px-3 py-2">Batch</th>
                    <th className="px-3 py-2">Mejor val acc.</th>
                  </tr>
                </thead>
                <tbody>
                  {data.runs.map((run) => (
                    <tr key={run.run_id} className="border-t border-border">
                      <td className="px-3 py-2 font-mono text-xs">{shortHash(run.run_id)}</td>
                      <td className="px-3 py-2">{run.status}</td>
                      <td className="px-3 py-2">{run.params.seed}</td>
                      <td className="px-3 py-2">{run.params.trainable_layers}</td>
                      <td className="px-3 py-2">{run.params.learning_rate}</td>
                      <td className="px-3 py-2">{run.params.optimizer}</td>
                      <td className="px-3 py-2">{run.params.batch_size}</td>
                      <td className="px-3 py-2">
                        {run.summary ? percent(run.summary.best_val_accuracy) : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )
        }
      </FetchBoundary>
    </div>
  );
}
