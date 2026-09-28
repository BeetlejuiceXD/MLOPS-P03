import { Fragment, useState } from "react";
import type { TrainingJob } from "../contracts";
import { dateTime, shortHash } from "../format";
import { cancelTrainingJob, fetchJobLogs } from "../mutations";

type Logs =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ok"; lines: string[] };

/** D02-05 — Jobs persistidos: estado, progreso, run de MLflow, error, logs y cancelación. */
export function JobsTable({
  jobs,
  onChanged,
}: Readonly<{ jobs: TrainingJob[]; onChanged: () => void }>) {
  const [openLogs, setOpenLogs] = useState<Record<number, Logs>>({});
  const [actionError, setActionError] = useState<string | null>(null);

  async function toggleLogs(id: number) {
    if (openLogs[id]) {
      setOpenLogs(({ [id]: _closed, ...rest }) => rest);
      return;
    }
    setOpenLogs((current) => ({ ...current, [id]: { status: "loading" } }));
    try {
      const logs = await fetchJobLogs(id);
      setOpenLogs((current) => ({
        ...current,
        [id]: {
          status: "ok",
          lines: logs.lines.map((line) => `${dateTime(line.ts)} [${line.level}] ${line.message}`),
        },
      }));
    } catch (error) {
      setOpenLogs((current) => ({
        ...current,
        [id]: { status: "error", message: error instanceof Error ? error.message : "Error" },
      }));
    }
  }

  async function cancel(id: number) {
    setActionError(null);
    try {
      await cancelTrainingJob(id);
    } catch (error) {
      setActionError(error instanceof Error ? error.message : "No se pudo cancelar el job.");
    }
    onChanged();
  }

  return (
    <div data-testid="p3-content" className="flex flex-col gap-2">
      {actionError && <p className="text-sm text-status-pending">{actionError}</p>}
      <div className="overflow-x-auto rounded-2xl border border-border">
        <table className="w-full text-left text-sm">
          <thead className="bg-surface text-xs text-ink-muted">
            <tr>
              <th className="px-3 py-2">Job</th>
              <th className="px-3 py-2">Tipo</th>
              <th className="px-3 py-2">Estado</th>
              <th className="px-3 py-2">Época</th>
              <th className="px-3 py-2">Run MLflow</th>
              <th className="px-3 py-2">Creado</th>
              <th className="px-3 py-2">Acciones</th>
            </tr>
          </thead>
          <tbody>
            {jobs.map((job) => {
              const logs = openLogs[job.id];
              const cancellable =
                (job.status === "queued" || job.status === "running") && !job.cancel_requested;
              return (
                <Fragment key={job.id}>
                  <tr className="border-t border-border align-top">
                    <td className="px-3 py-2">#{job.id}</td>
                    <td className="px-3 py-2">{job.task}</td>
                    <td className="px-3 py-2">
                      <span>{job.status}</span>
                      {job.cancel_requested && job.status === "running" && (
                        <span className="ml-2 rounded bg-status-pending-soft px-1.5 text-xs">
                          cancelación pedida
                        </span>
                      )}
                      {job.error && (
                        <p className="mt-1 max-w-md text-xs text-ink-muted">{job.error}</p>
                      )}
                    </td>
                    <td className="px-3 py-2">
                      {job.progress ? `${job.progress.epoch} / ${job.progress.total_epochs}` : "—"}
                    </td>
                    <td className="px-3 py-2 font-mono text-xs">
                      {job.mlflow_run_id ? shortHash(job.mlflow_run_id) : "—"}
                    </td>
                    <td className="px-3 py-2">{dateTime(job.created_at)}</td>
                    <td className="flex gap-2 px-3 py-2">
                      <button
                        type="button"
                        aria-label={`Logs del job ${job.id}`}
                        className="text-xs underline"
                        onClick={() => void toggleLogs(job.id)}
                      >
                        {logs ? "Ocultar logs" : "Logs"}
                      </button>
                      {cancellable && (
                        <button
                          type="button"
                          aria-label={`Cancelar job ${job.id}`}
                          className="text-xs text-status-pending underline"
                          onClick={() => void cancel(job.id)}
                        >
                          Cancelar
                        </button>
                      )}
                    </td>
                  </tr>
                  {logs && (
                    <tr className="border-t border-border bg-surface">
                      <td colSpan={7} className="px-3 py-2">
                        {logs.status === "loading" && <p className="text-xs">Cargando logs…</p>}
                        {logs.status === "error" && <p className="text-xs">{logs.message}</p>}
                        {logs.status === "ok" && (
                          <pre className="max-h-64 overflow-auto whitespace-pre-wrap text-xs">
                            {logs.lines.length ? logs.lines.join("\n") : "Sin logs todavía."}
                          </pre>
                        )}
                      </td>
                    </tr>
                  )}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
