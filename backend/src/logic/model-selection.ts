/**
 * D04-04 — Selección del candidato SOLO por validation (protocolo #33).
 *
 * Función pura sobre runs con el contrato `experiment_runs_response` (los del adaptador
 * MLflow de D04-01). Ningún dato de test entra aquí: el contrato rechaza cualquier
 * métrica que no sea de validation, y un run así se excluye.
 *
 * - Solo cuentan runs FINISHED de la matriz OFAT congelada (config completa idéntica a
 *   una fila, seed incluida), del manifest congelado y con procedencia DVC coherente.
 *   Smoke, short-run, tareas controladas o cualquier config fuera de la matriz se
 *   excluyen con su motivo; nada se infiere solo de FINISHED.
 * - Ranking: val_accuracy → val_macro_f1 → menor val_loss → menor run_id, comparando a 4
 *   decimales, con las métricas del mejor checkpoint (`summary`), no de la última época.
 * - Cada fila de la matriz cuenta una sola vez (su corrida terminada más temprana): volver
 *   a correr una configuración no da más oportunidades de ganar.
 *
 * El candidato que sale de aquí es preparatorio: el cierre (MODEL SELECTION CLOSED) lo
 * persiste `model-selection.service.ts` y en la campaña oficial lo declara D05-02.
 */
import { createHash } from 'node:crypto';
import { ValidationError } from './errors.js';
import {
  type ExperimentRun,
  experimentRunSchema,
  TRAINING_CONFIG_DEFAULTS,
  type TrainingConfig,
} from './p3.contracts.js';

/** Mínimo de configuraciones distintas de la matriz para poder cerrar la selección (#33). */
export const MIN_COMPARABLE_RUNS = 10;

export interface CampaignRow {
  row: number;
  change: string;
  config: TrainingConfig;
}

const row = (n: number, change: string, overrides: Partial<TrainingConfig>): CampaignRow => ({
  row: n,
  change,
  config: { ...TRAINING_CONFIG_DEFAULTS, seed: 7, ...overrides },
});

/**
 * Matriz final OFAT de 12 configuraciones (#33, comentario 5850854067). Lo que la
 * matriz no barre queda en los defaults congelados (`TRAINING_CONFIG_DEFAULTS`).
 */
export const CAMPAIGN_MATRIX: readonly CampaignRow[] = [
  row(1, 'baseline', {}),
  row(2, 'trainable_layers', { trainable_layers: 'head_only' }),
  row(3, 'learning_rate', { learning_rate: 3e-4 }),
  row(4, 'optimizer', { optimizer: 'sgd' }),
  row(5, 'batch_size', { batch_size: 32 }),
  row(6, 'max_epochs', { max_epochs: 60 }),
  row(7, 'image_size', { image_size: 160 }),
  row(8, 'hidden_layers', { hidden_layers: 1 }),
  row(9, 'dropout', { dropout: 0.3 }),
  row(10, 'hidden_layers + dropout', { hidden_layers: 1, dropout: 0.3 }),
  row(11, 'réplica de varianza (seed 21)', { seed: 21 }),
  row(12, 'réplica de varianza (seed 77)', { seed: 77 }),
];

export interface SelectionReference {
  dataset_version: string;
  manifest_hash: string;
  dvc_release_hash: string;
}

export const exclusionReasons = [
  'invalid_contract',
  'not_finished',
  'outside_campaign_matrix',
  'manifest_mismatch',
  'release_mismatch',
  'provenance_inconsistent',
  'duplicate_campaign_row',
] as const;
export type ExclusionReason = (typeof exclusionReasons)[number];

export interface RankedRun {
  run_id: string;
  campaign_row: number;
  start_time: string;
  best_epoch: number;
  val_accuracy: number;
  val_macro_f1: number;
  val_loss: number;
}

export interface ExcludedRun {
  run_id: string | null;
  reason: ExclusionReason;
  detail: string;
}

export interface SelectionOutcome {
  reference: SelectionReference;
  ranking: RankedRun[];
  excluded: ExcludedRun[];
  candidate: RankedRun | null;
  campaign_rows: number[];
  ready_to_close: boolean;
  outcome_hash: string;
}

export function campaignRowOf(params: TrainingConfig): number | null {
  const keys = Object.keys(params) as (keyof TrainingConfig)[];
  const match = CAMPAIGN_MATRIX.find(
    (entry) =>
      keys.length === Object.keys(entry.config).length &&
      keys.every((key) => entry.config[key] === params[key]),
  );
  return match?.row ?? null;
}

/** Igualdad a 4 decimales (#33): se compara el valor redondeado a 1e-4 como entero. */
const at4 = (value: number) => Math.round(value * 10_000);

/** Negativo si `a` va antes que `b` en el ranking. */
export function compareRanked(a: RankedRun, b: RankedRun): number {
  return (
    at4(b.val_accuracy) - at4(a.val_accuracy) ||
    at4(b.val_macro_f1) - at4(a.val_macro_f1) ||
    at4(a.val_loss) - at4(b.val_loss) ||
    compareText(a.run_id, b.run_id)
  );
}

function compareText(a: string, b: string): number {
  if (a === b) return 0;
  return a < b ? -1 : 1;
}

const sha256 = (text: string) => createHash('sha256').update(text).digest('hex');

function rawRunId(run: unknown): string | null {
  const value = (run as { run_id?: unknown } | null)?.run_id;
  return typeof value === 'string' ? value : null;
}

function formatIssues(issues: { path: PropertyKey[]; message: string }[]): string {
  return issues
    .slice(0, 3)
    .map((issue) => `${issue.path.map(String).join('.') || '(raíz)'}: ${issue.message}`)
    .join('; ');
}

type Assessment =
  | { eligible: true; run: RankedRun }
  | { eligible: false; reason: ExclusionReason; detail: string };

function assess(raw: unknown, reference: SelectionReference): Assessment {
  const parsed = experimentRunSchema.safeParse(raw);
  if (!parsed.success) {
    return {
      eligible: false,
      reason: 'invalid_contract',
      detail: formatIssues(parsed.error.issues),
    };
  }
  const run: ExperimentRun = parsed.data;
  if (run.status !== 'FINISHED' || run.summary === null) {
    return { eligible: false, reason: 'not_finished', detail: `status=${run.status}` };
  }
  const campaignRow = campaignRowOf(run.params);
  if (campaignRow === null) {
    return {
      eligible: false,
      reason: 'outside_campaign_matrix',
      detail: 'La config no coincide con ninguna de las 12 filas OFAT de #33 (seed incluida).',
    };
  }
  const { tags } = run;
  if (tags.manifest_hash !== reference.manifest_hash) {
    return {
      eligible: false,
      reason: 'manifest_mismatch',
      detail: `manifest_hash ${tags.manifest_hash} ≠ congelado ${reference.manifest_hash}`,
    };
  }
  if (
    tags.dvc_release !== reference.dataset_version ||
    tags.dvc_release_hash !== reference.dvc_release_hash
  ) {
    return {
      eligible: false,
      reason: 'release_mismatch',
      detail: `${tags.dvc_release}/${tags.dvc_release_hash} ≠ ${reference.dataset_version}/${reference.dvc_release_hash}`,
    };
  }
  const expectedHash = sha256(`${tags.dvc_images_md5}:${tags.dvc_annotations_md5}`);
  if (tags.dvc_release_hash !== expectedHash) {
    return {
      eligible: false,
      reason: 'provenance_inconsistent',
      detail: 'dvc_release_hash ≠ sha256(<images_md5>:<annotations_md5>)',
    };
  }
  return {
    eligible: true,
    run: {
      run_id: run.run_id,
      campaign_row: campaignRow,
      start_time: run.start_time,
      best_epoch: run.summary.best_epoch,
      val_accuracy: run.summary.best_val_accuracy,
      val_macro_f1: run.summary.best_val_macro_f1,
      val_loss: run.summary.best_val_loss,
    },
  };
}

export function selectCandidate(
  runs: readonly unknown[],
  reference: SelectionReference,
): SelectionOutcome {
  const seen = new Set<string>();
  for (const raw of runs) {
    const id = rawRunId(raw);
    if (id === null) continue;
    if (seen.has(id)) throw new ValidationError(`run_id repetido en la fuente: ${id}`);
    seen.add(id);
  }

  const excluded: ExcludedRun[] = [];
  const byRow = new Map<number, RankedRun[]>();
  for (const raw of runs) {
    const result = assess(raw, reference);
    if (!result.eligible) {
      excluded.push({ run_id: rawRunId(raw), reason: result.reason, detail: result.detail });
      continue;
    }
    byRow.set(result.run.campaign_row, [...(byRow.get(result.run.campaign_row) ?? []), result.run]);
  }

  const ranking: RankedRun[] = [];
  for (const [campaignRow, candidates] of byRow) {
    const [first, ...repeated] = [...candidates].sort(
      (a, b) =>
        Date.parse(a.start_time) - Date.parse(b.start_time) || compareText(a.run_id, b.run_id),
    );
    if (first) ranking.push(first);
    for (const run of repeated) {
      excluded.push({
        run_id: run.run_id,
        reason: 'duplicate_campaign_row',
        detail: `La fila ${campaignRow} ya cuenta con ${first?.run_id} (corrida más temprana).`,
      });
    }
  }
  ranking.sort(compareRanked);
  excluded.sort(
    (a, b) => compareText(a.run_id ?? '~', b.run_id ?? '~') || compareText(a.reason, b.reason),
  );

  const campaignRows = ranking.map((run) => run.campaign_row).sort((a, b) => a - b);
  return {
    reference,
    ranking,
    excluded,
    candidate: ranking[0] ?? null,
    campaign_rows: campaignRows,
    ready_to_close: campaignRows.length >= MIN_COMPARABLE_RUNS,
    outcome_hash: sha256(JSON.stringify({ reference, ranking, excluded })),
  };
}
