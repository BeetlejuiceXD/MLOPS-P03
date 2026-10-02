import { apiUrl } from "@/hooks/useValidatedFetch";
import { useExperimentRun } from "../api";
import { FetchBoundary } from "../components/FetchBoundary";
import { CURVE_METRICS } from "../experiments";
import { dateTime, percent } from "../format";
import { CurveChart } from "./CurveChart";

const artifactHref = (runId: string, path: string) =>
  apiUrl(
    `/experiments/runs/${runId}/artifacts/${path.split("/").map(encodeURIComponent).join("/")}`
  );

const formatBytes = (bytes: number) =>
  bytes >= 1024 * 1024 ? `${(bytes / (1024 * 1024)).toFixed(1)} MB` : `${bytes} B`;

/**
 * D04-02 — Detalle de un run desde `GET /api/experiments/runs/:id` (D04-01): provenance,
 * params, resumen, curvas por época y artefactos descargables. Si la API rechaza el run
 * (auxiliar → 409, MLflow caído → 503) se muestra su motivo, nunca datos inventados.
 */
export function RunDetail({ runId, onClose }: Readonly<{ runId: string; onClose: () => void }>) {
  const detail = useExperimentRun(runId);
  return (
    <section
      aria-label="Detalle del run"
      className="flex flex-col gap-4 rounded-2xl border border-border bg-surface p-5"
    >
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-ink">Detalle del run</h2>
        <button type="button" className="text-xs text-ink-muted underline" onClick={onClose}>
          Cerrar detalle
        </button>
      </div>
      <FetchBoundary results={[detail]}>
        {({ run, artifacts }) => (
          <div data-testid="experiments-detail" className="flex flex-col gap-4 text-sm">
            <dl className="grid gap-x-6 gap-y-1 sm:grid-cols-2">
              <dt className="text-ink-muted">Run</dt>
              <dd className="font-mono text-xs">{run.run_id}</dd>
              <dt className="text-ink-muted">Estado</dt>
              <dd>{run.status}</dd>
              <dt className="text-ink-muted">Inicio / fin</dt>
              <dd>
                {dateTime(run.start_time)} → {dateTime(run.end_time)}
              </dd>
              <dt className="text-ink-muted">Job</dt>
              <dd data-testid="detail-job">{run.tags.job_id}</dd>
              <dt className="text-ink-muted">Commit</dt>
              <dd className="font-mono text-xs">{run.tags.git_commit}</dd>
              <dt className="text-ink-muted">Release / manifest</dt>
              <dd>
                {run.tags.dvc_release} · {run.tags.manifest_version}
              </dd>
              <dt className="text-ink-muted">manifest_hash</dt>
              <dd className="font-mono text-xs break-all">{run.tags.manifest_hash}</dd>
              <dt className="text-ink-muted">dvc_release_hash</dt>
              <dd className="font-mono text-xs break-all">{run.tags.dvc_release_hash}</dd>
              <dt className="text-ink-muted">checkpoint_sha256</dt>
              <dd className="font-mono text-xs break-all">{run.checkpoint_sha256 ?? "—"}</dd>
            </dl>

            <div>
              {run.summary ? (
                <p>
                  Mejor época: {run.summary.best_epoch} · val accuracy{" "}
                  {percent(run.summary.best_val_accuracy)} · macro-F1{" "}
                  {run.summary.best_val_macro_f1.toFixed(4)} · val loss{" "}
                  {run.summary.best_val_loss.toFixed(4)}
                </p>
              ) : (
                <p className="text-ink-muted">Sin resumen: el run no ha terminado.</p>
              )}
              <p className="mt-1 text-xs">
                {run.campaign_eligible
                  ? "Elegible para la campaña."
                  : `No elegible: ${run.ineligible_reasons.join("; ")}`}
              </p>
            </div>

            <div className="grid gap-6 lg:grid-cols-2">
              {CURVE_METRICS.map((metric) => (
                <CurveChart key={metric} runs={[run]} metric={metric} />
              ))}
            </div>

            <details open>
              <summary className="cursor-pointer text-xs text-ink-muted">
                Historial por época ({run.history.length})
              </summary>
              <table data-testid="curve-table" className="mt-2 w-full text-left text-xs">
                <thead className="text-ink-muted">
                  <tr>
                    <th className="px-2 py-1">Época</th>
                    <th className="px-2 py-1">train_loss</th>
                    <th className="px-2 py-1">train_accuracy</th>
                    <th className="px-2 py-1">val_loss</th>
                    <th className="px-2 py-1">val_accuracy</th>
                    <th className="px-2 py-1">val_macro_f1</th>
                    <th className="px-2 py-1">learning_rate</th>
                  </tr>
                </thead>
                <tbody>
                  {run.history.map((h) => (
                    <tr key={h.epoch} className="border-t border-border font-mono">
                      <td className="px-2 py-1">{h.epoch}</td>
                      <td className="px-2 py-1">{h.train_loss.toFixed(4)}</td>
                      <td className="px-2 py-1">{h.train_accuracy.toFixed(4)}</td>
                      <td className="px-2 py-1">{h.val_loss.toFixed(4)}</td>
                      <td className="px-2 py-1">{h.val_accuracy.toFixed(4)}</td>
                      <td className="px-2 py-1">{h.val_macro_f1.toFixed(4)}</td>
                      <td className="px-2 py-1">{h.learning_rate}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </details>

            <div>
              <h3 className="text-xs font-semibold text-ink">Artefactos ({artifacts.length})</h3>
              {artifacts.length === 0 ? (
                <p className="text-xs text-ink-muted">El run no tiene artefactos en MLflow.</p>
              ) : (
                <ul className="mt-1 flex flex-col gap-1 text-xs">
                  {artifacts.map((artifact) =>
                    artifact.is_dir ? (
                      <li key={artifact.path} className="text-ink-muted">
                        {artifact.path}/
                      </li>
                    ) : (
                      <li key={artifact.path}>
                        <a
                          className="font-mono text-accent underline"
                          href={artifactHref(run.run_id, artifact.path)}
                          download
                        >
                          {artifact.path}
                        </a>
                        {artifact.size_bytes !== null && (
                          <span className="text-ink-muted">
                            {" "}
                            · {formatBytes(artifact.size_bytes)}
                          </span>
                        )}
                      </li>
                    )
                  )}
                </ul>
              )}
            </div>
          </div>
        )}
      </FetchBoundary>
    </section>
  );
}
