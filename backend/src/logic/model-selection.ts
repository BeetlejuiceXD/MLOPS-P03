/**
 * D04-04 — Selección del candidato SOLO por validation (stub Red).
 */
import { TRAINING_CONFIG_DEFAULTS, type TrainingConfig } from './p3.contracts.js';

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

export function campaignRowOf(_params: TrainingConfig): number | null {
  throw new Error('NotImplemented');
}

export function compareRanked(_a: RankedRun, _b: RankedRun): number {
  throw new Error('NotImplemented');
}

export function selectCandidate(
  _runs: readonly unknown[],
  _reference: SelectionReference,
): SelectionOutcome {
  throw new Error('NotImplemented');
}
