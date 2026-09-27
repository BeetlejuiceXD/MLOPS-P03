import { PageHeader } from "@/pipeline/components/PageHeader";
import { useManifest, useReleases, useTrainingJobs } from "../api";
import { FetchBoundary } from "../components/FetchBoundary";
import { StatePanel } from "../components/StatePanel";
import type { TrainingJob } from "../contracts";
import { dateTime, shortHash } from "../format";

/**
 * D01-05 — Training: procedencia (release aprobado + manifest 70/20/10) y jobs
 * persistidos. El formulario de TrainingConfig y el lanzamiento llegan en D02-05;
 * aquí queda la estructura con sus estados vacío, bloqueado y de error.
 */
export function TrainingPage() {
  const releases = useReleases();
  const manifest = useManifest();
  const jobs = useTrainingJobs();

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Training"
        subtitle="Entrenamiento sobre el release aprobado y el manifest P3 70/20/10 congelado."
      />
      <FetchBoundary results={[releases, manifest, jobs]}>
        {(releaseData, manifestData, jobData) => {
          const release = releaseData.approved.at(-1);
          if (!release) {
            return (
              <StatePanel variant="blocked" title="No hay un release aprobado.">
                Training solo usa releases que pasaron la compuerta de calidad de P2.
              </StatePanel>
            );
          }
          if (!manifestData.frozen) {
            return (
              <StatePanel variant="blocked" title="El manifest 70/20/10 no está congelado.">
                No se lanzan entrenamientos hasta congelar el manifest derivado.
              </StatePanel>
            );
          }
          return (
            <div data-testid="p3-content" className="flex flex-col gap-6">
              <section className="grid gap-4 rounded-2xl border border-border bg-surface p-5 sm:grid-cols-2">
                <div>
                  <h2 className="text-sm font-semibold text-ink">Release aprobado</h2>
                  <p className="mt-1 text-sm text-ink">{release.dataset_version}</p>
                  <p className="text-xs text-ink-muted">
                    Calidad: {release.status} · originales:{" "}
                    {Object.entries(release.originals_per_class)
                      .map(([name, count]) => `${name} ${count}`)
                      .join(", ")}
                  </p>
                </div>
                <div>
                  <h2 className="text-sm font-semibold text-ink">Manifest P3</h2>
                  <p className="mt-1 text-sm text-ink">{manifestData.manifest_version}</p>
                  <p className="text-xs text-ink-muted">
                    hash {shortHash(manifestData.manifest_hash)} · seed {manifestData.seed} · crops{" "}
                    train {manifestData.splits.train.crops} / val {manifestData.splits.val.crops} /
                    test {manifestData.splits.test.crops}
                  </p>
                </div>
              </section>
              <section className="flex flex-col gap-3">
                <h2 className="text-sm font-semibold text-ink">Trabajos</h2>
                {jobData.jobs.length === 0 ? (
                  <StatePanel variant="empty" title="Todavía no hay trabajos de entrenamiento.">
                    Cuando se lance un job, su estado, progreso y logs quedarán aquí aunque
                    recargues la página.
                  </StatePanel>
                ) : (
                  <JobsTable jobs={jobData.jobs} />
                )}
              </section>
            </div>
          );
        }}
      </FetchBoundary>
    </div>
  );
}

function JobsTable({ jobs }: Readonly<{ jobs: TrainingJob[] }>) {
  return (
    <div className="overflow-x-auto rounded-2xl border border-border">
      <table className="w-full text-left text-sm">
        <thead className="bg-surface text-xs text-ink-muted">
          <tr>
            <th className="px-3 py-2">Job</th>
            <th className="px-3 py-2">Estado</th>
            <th className="px-3 py-2">Época</th>
            <th className="px-3 py-2">Run MLflow</th>
            <th className="px-3 py-2">Creado</th>
          </tr>
        </thead>
        <tbody>
          {jobs.map((job) => (
            <tr key={job.id} className="border-t border-border">
              <td className="px-3 py-2">#{job.id}</td>
              <td className="px-3 py-2">{job.status}</td>
              <td className="px-3 py-2">
                {job.progress ? `${job.progress.epoch} / ${job.progress.total_epochs}` : "—"}
              </td>
              <td className="px-3 py-2 font-mono text-xs">
                {job.mlflow_run_id ? shortHash(job.mlflow_run_id) : "—"}
              </td>
              <td className="px-3 py-2">{dateTime(job.created_at)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
