import type { z } from "zod";
import { ErrorState } from "@/components/ui/ErrorState";
import { Skeleton } from "@/components/ui/Skeleton";
import { PageHeader } from "@/pipeline/components/PageHeader";
import { useManifest, useReleases } from "../api";
import { StatePanel } from "../components/StatePanel";
import type { manifestSummarySchema, releasesResponseSchema } from "../contracts";
import { shortHash } from "../format";
import { JobsTable } from "../training/JobsTable";
import { type RealTrainingStatus, TrainingForm } from "../training/TrainingForm";
import { useTrainingJobs } from "../training/useTrainingJobs";

type ReleasesResponse = z.infer<typeof releasesResponseSchema>;
type ManifestSummary = z.infer<typeof manifestSummarySchema>;
type Source<T> =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "success"; data: T };

function realTrainingStatus(
  releases: Source<ReleasesResponse>,
  manifest: Source<ManifestSummary>
): RealTrainingStatus | null {
  if (releases.status === "loading" || manifest.status === "loading") return null;
  if (releases.status === "error") return { eligible: false, reason: releases.message };
  if (manifest.status === "error") return { eligible: false, reason: manifest.message };
  if (releases.data.approved.length === 0) {
    return { eligible: false, reason: "No hay un release aprobado." };
  }
  if (!manifest.data.frozen) {
    return { eligible: false, reason: "El manifest 70/20/10 no está congelado." };
  }
  const approved = releases.data.approved.some(
    (release) => release.dataset_version === manifest.data.dataset_version
  );
  if (!approved) {
    return { eligible: false, reason: "El manifest no corresponde a un release aprobado." };
  }
  return { eligible: true, reason: null };
}

/**
 * Training (D01-05 → D02-05): formulario de TrainingConfig, jobs persistentes y
 * procedencia. La tarea controlada no necesita release ni manifest oficiales; el
 * entrenamiento real se bloquea con su motivo hasta que ambos sean elegibles (D03-03).
 */
export function TrainingPage({ pollMs = 3000 }: Readonly<{ pollMs?: number }>) {
  const releases = useReleases();
  const manifest = useManifest();
  const jobs = useTrainingJobs(pollMs);

  const releaseSource = releases as Source<ReleasesResponse>;
  const manifestSource = manifest as Source<ManifestSummary>;
  const real = realTrainingStatus(releaseSource, manifestSource);
  const release =
    releaseSource.status === "success" ? releaseSource.data.approved.at(-1) : undefined;
  const manifestData = manifestSource.status === "success" ? manifestSource.data : undefined;

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Training"
        subtitle="Jobs persistentes: se ejecutan en trainer-worker, fuera del request, y su estado sobrevive a recargas."
      />

      <section className="grid gap-4 rounded-2xl border border-border bg-surface p-5 sm:grid-cols-2">
        <div>
          <h2 className="text-sm font-semibold text-ink">Release aprobado</h2>
          {release ? (
            <>
              <p className="mt-1 text-sm text-ink">{release.dataset_version}</p>
              <p className="text-xs text-ink-muted">
                Calidad: {release.status} · originales:{" "}
                {Object.entries(release.originals_per_class)
                  .map(([name, count]) => `${name} ${count}`)
                  .join(", ")}
              </p>
            </>
          ) : (
            <p className="mt-1 text-xs text-ink-muted">—</p>
          )}
        </div>
        <div>
          <h2 className="text-sm font-semibold text-ink">Manifest P3</h2>
          {manifestData ? (
            <>
              <p className="mt-1 text-sm text-ink">{manifestData.manifest_version}</p>
              <p className="text-xs text-ink-muted">
                hash {shortHash(manifestData.manifest_hash)} · seed {manifestData.seed} · crops
                train {manifestData.splits.train.crops} / val {manifestData.splits.val.crops} / test{" "}
                {manifestData.splits.test.crops}
                {manifestData.frozen ? " · congelado" : " · sin congelar"}
              </p>
            </>
          ) : (
            <p className="mt-1 text-xs text-ink-muted">—</p>
          )}
        </div>
      </section>

      {real && !real.eligible && (
        <StatePanel variant="blocked" title="Entrenamiento real no disponible.">
          {real.reason}
        </StatePanel>
      )}

      <TrainingForm
        realTraining={real}
        defaultDatasetVersion={manifestData?.dataset_version ?? release?.dataset_version ?? ""}
        defaultManifestHash={manifestData?.manifest_hash ?? ""}
        onCreated={() => void jobs.reload()}
      />

      <section className="flex flex-col gap-3">
        <h2 className="text-sm font-semibold text-ink">Trabajos</h2>
        {jobs.state.status === "loading" && <Skeleton className="h-24" />}
        {jobs.state.status === "error" && (
          <ErrorState
            title="No se pudieron cargar los jobs."
            message={jobs.state.message}
            onRetry={() => void jobs.reload()}
          />
        )}
        {jobs.state.status === "success" &&
          (jobs.state.jobs.length === 0 ? (
            <StatePanel variant="empty" title="Todavía no hay trabajos de entrenamiento.">
              Cuando se lance un job, su estado, progreso y logs quedarán aquí aunque recargues la
              página.
            </StatePanel>
          ) : (
            <JobsTable jobs={jobs.state.jobs} onChanged={() => void jobs.reload()} />
          ))}
      </section>
    </div>
  );
}
