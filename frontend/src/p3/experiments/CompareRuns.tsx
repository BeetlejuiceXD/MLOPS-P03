import type { ExperimentRun } from "../contracts";
import { compareParams } from "../experiments";
import { percent, shortHash } from "../format";
import { CurveChart } from "./CurveChart";

/**
 * D04-02 — Comparación de runs de training elegidos en la tabla: TrainingConfig lado a
 * lado (marca los campos que difieren), mejores métricas de validation y curvas
 * superpuestas. Solo runs de training: los auxiliares no se pueden elegir.
 */
export function CompareRuns({ runs }: Readonly<{ runs: readonly ExperimentRun[] }>) {
  const params = compareParams(runs);
  return (
    <section
      data-testid="experiments-compare"
      aria-label="Comparación de runs"
      className="flex flex-col gap-4 rounded-2xl border border-border bg-surface p-5"
    >
      <h2 className="text-sm font-semibold text-ink">Comparación ({runs.length} runs)</h2>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead className="text-ink-muted">
            <tr>
              <th className="px-2 py-1">Campo</th>
              {runs.map((run) => (
                <th key={run.run_id} className="px-2 py-1 font-mono">
                  {shortHash(run.run_id)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {params.map((row) => (
              <tr
                key={row.key}
                data-testid={`compare-param-${row.key}`}
                data-differs={String(row.differs)}
                className={`border-t border-border ${row.differs ? "bg-status-pending-soft font-semibold" : ""}`}
              >
                <td className="px-2 py-1">{row.key}</td>
                {row.values.map((value, index) => (
                  <td key={runs[index]?.run_id} className="px-2 py-1 font-mono">
                    {value}
                  </td>
                ))}
              </tr>
            ))}
            {(
              [
                ["Mejor época", (r: ExperimentRun) => String(r.summary?.best_epoch ?? "—")],
                [
                  "Mejor val accuracy",
                  (r: ExperimentRun) => (r.summary ? percent(r.summary.best_val_accuracy) : "—"),
                ],
                [
                  "Mejor macro-F1",
                  (r: ExperimentRun) => (r.summary ? r.summary.best_val_macro_f1.toFixed(4) : "—"),
                ],
                [
                  "Val loss (mejor época)",
                  (r: ExperimentRun) => (r.summary ? r.summary.best_val_loss.toFixed(4) : "—"),
                ],
                ["Elegible", (r: ExperimentRun) => (r.campaign_eligible ? "sí" : "no")],
              ] as const
            ).map(([label, read]) => (
              <tr key={label} className="border-t border-border-strong">
                <td className="px-2 py-1 font-medium">{label}</td>
                {runs.map((run) => (
                  <td key={run.run_id} className="px-2 py-1 font-mono">
                    {read(run)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="grid gap-6 lg:grid-cols-2">
        <CurveChart runs={runs} metric="val_accuracy" showTable />
        <CurveChart runs={runs} metric="val_loss" />
      </div>
    </section>
  );
}
