/**
 * D05-02 — Recalcula el expediente de campaña y el ranking validation-only desde la
 * evidencia mergeada de D04-03 (`reports/campaign_p3.json`), contra el manifest congelado
 * (`reports/manifest_p3.json`). No lee MLflow, ni el frozen test, ni escribe estado.
 *
 * Reconstruye cada intento con la forma que entrega el adaptador de D04-01 (tags de
 * procedencia, resumen del mejor checkpoint y `checkpoint_sha256`) y lo pasa por los mismos
 * `buildCampaignDossier` y `selectCandidate` que usa `GET /api/selection/campaign`. La
 * respuesta de la API sobre el MLflow real debe dar el mismo ranking y candidato. El
 * `outcome_hash` también cubre los runs excluidos (smoke, short-run), que no están en el
 * reporte, así que solo coincide si MLflow no tiene ninguno.
 *
 * El reporte no trae las curvas por época. Para cumplir el contrato, `history` repite las
 * métricas del mejor checkpoint en las épocas 1..best_epoch: es relleno, no la curva real,
 * y no interviene en la selección (que usa `summary`).
 *
 *   npx tsx src/cli/replay-campaign-report.ts [campaign_p3.json] [manifest_p3.json]
 */
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { buildCampaignDossier } from '../logic/campaign-dossier.js';
import { type SelectionReference, selectCandidate } from '../logic/model-selection.js';
import {
  experimentRunSchema,
  manifestSummarySchema,
  trainingJobSchema,
} from '../logic/p3.contracts.js';

export interface ReportAttempt {
  row: number;
  job: Record<string, unknown> & { config: { learning_rate: number }; finished_at?: unknown };
  report: {
    run_id: string;
    run_status: string;
    best_epoch: number;
    best_val_accuracy: number;
    best_val_macro_f1: number;
    checkpoint_sha256: string | null;
    tags: Record<string, string>;
  };
  details: { start_time: number; best_val_loss: number };
}

export interface CampaignReport {
  results: ReportAttempt[];
}

// Los tags que el adaptador de D04-01 copia al contrato; checkpoint_sha256 va en su campo.
const PROVENANCE_TAGS = [
  'git_commit',
  'dvc_release',
  'dvc_images_md5',
  'dvc_annotations_md5',
  'dvc_release_hash',
  'manifest_version',
  'manifest_hash',
] as const;

const typedTag = (value: string | undefined): unknown =>
  value !== undefined && /^-?\d+$/.test(value) ? Number(value) : value;

function runOf({ report, details, job }: ReportAttempt) {
  const checkpoint = report.checkpoint_sha256 || null;
  const ineligible = [
    ...(report.run_status === 'FINISHED' ? [] : [`estado ${report.run_status}: el run no terminó`]),
    ...(checkpoint === null ? ['sin checkpoint_sha256: no hay checkpoint verificado'] : []),
  ];
  const best = {
    train_loss: details.best_val_loss,
    train_accuracy: report.best_val_accuracy,
    val_loss: details.best_val_loss,
    val_accuracy: report.best_val_accuracy,
    val_macro_f1: report.best_val_macro_f1,
    learning_rate: job.config.learning_rate,
  };
  return experimentRunSchema.parse({
    run_id: report.run_id,
    experiment_name: 'p3-cnn-classifier',
    status: report.run_status,
    start_time: new Date(details.start_time).toISOString(),
    end_time: job.finished_at ?? null,
    params: job.config,
    tags: {
      ...Object.fromEntries(PROVENANCE_TAGS.map((name) => [name, report.tags[name]])),
      classes: (report.tags.classes ?? '').split(','),
      seed: typedTag(report.tags.seed),
      job_id: typedTag(report.tags.job_id),
    },
    summary: {
      best_epoch: report.best_epoch,
      best_val_accuracy: report.best_val_accuracy,
      best_val_macro_f1: report.best_val_macro_f1,
      best_val_loss: details.best_val_loss,
    },
    history: Array.from({ length: report.best_epoch }, (_, i) => ({ epoch: i + 1, ...best })),
    checkpoint_sha256: checkpoint,
    campaign_eligible: ineligible.length === 0,
    ineligible_reasons: ineligible,
  });
}

/** Expediente y selección sobre la evidencia de D04-03, con la referencia del manifest. */
export function replayCampaignReport(evidence: CampaignReport, manifest: unknown) {
  const summary = manifestSummarySchema.parse(manifest);
  if (!summary.frozen) throw new Error('El manifest P3 no está congelado (D03-01).');
  const reference: SelectionReference = {
    dataset_version: summary.dataset_version,
    manifest_hash: summary.manifest_hash,
    dvc_release_hash: summary.dvc_release_hash,
  };
  const runs = evidence.results.map(runOf);
  const jobs = evidence.results.map((attempt) => trainingJobSchema.parse(attempt.job));
  const outcome = selectCandidate(runs, reference);
  const dossier = buildCampaignDossier(reference, { runs }, jobs);
  return {
    reference,
    attempts: runs.length,
    accepted_rows: dossier.accepted_rows,
    close_blockers: dossier.close_blockers,
    ready_to_close: dossier.ready_to_close,
    outcome_hash: dossier.outcome_hash,
    matches_selection: dossier.outcome_hash === outcome.outcome_hash,
    rows: dossier.rows,
    unattributed: dossier.unattributed,
    ranking: dossier.ranking,
    candidate: dossier.candidate,
  };
}

async function main() {
  const [reportPath = '../reports/campaign_p3.json', manifestPath = '../reports/manifest_p3.json'] =
    process.argv.slice(2);
  const read = async (file: string) => JSON.parse(await readFile(file, 'utf8')) as unknown;
  const result = replayCampaignReport(
    (await read(reportPath)) as CampaignReport,
    await read(manifestPath),
  );
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
  if (!result.ready_to_close || !result.matches_selection) process.exitCode = 1;
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  main().catch((error: unknown) => {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  });
}
