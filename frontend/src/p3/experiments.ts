/**
 * D04-02 — Lógica de la página Experiments sobre el contrato de D04-01: filtros,
 * conteo de campaña, comparación de params y curvas. Solo transforma lo que sirve la API;
 * no rellena ni interpola nada.
 */
import { type ExperimentRun, trainingConfigSchema } from "./contracts";

export type EligibilityFilter = "all" | "eligible" | "not_eligible";
export type CampaignFilter = "all" | "accepted";
export interface RunFilters {
  status: string;
  campaign: CampaignFilter;
  eligibility: EligibilityFilter;
  seed: string;
  trainable_layers: string;
  optimizer: string;
  learning_rate: string;
}
export const ALL = "all";
export const EMPTY_FILTERS: RunFilters = {
  status: ALL,
  campaign: ALL,
  eligibility: ALL,
  seed: ALL,
  trainable_layers: ALL,
  optimizer: ALL,
  learning_rate: ALL,
};

type ParamFilter = "seed" | "trainable_layers" | "optimizer" | "learning_rate";
const PARAM_FILTERS: readonly ParamFilter[] = [
  "seed",
  "trainable_layers",
  "optimizer",
  "learning_rate",
];

/**
 * `accepted`: run_ids de la campaña aceptada según la selección (D05-03); el filtro
 * "Campaña aceptada" no recalcula nada, solo usa esa lista.
 */
export function filterRuns(
  runs: readonly ExperimentRun[],
  filters: RunFilters,
  accepted: ReadonlySet<string> = new Set()
): ExperimentRun[] {
  return runs.filter((run) => {
    if (filters.status !== ALL && run.status !== filters.status) return false;
    if (filters.campaign === "accepted" && !accepted.has(run.run_id)) return false;
    if (filters.eligibility === "eligible" && !run.campaign_eligible) return false;
    if (filters.eligibility === "not_eligible" && run.campaign_eligible) return false;
    return PARAM_FILTERS.every(
      (key) => filters[key] === ALL || String(run.params[key]) === filters[key]
    );
  });
}

const sortValues = (values: Iterable<string>) => {
  const unique = [...new Set(values)];
  const numeric = unique.every((value) => value !== "" && !Number.isNaN(Number(value)));
  return numeric
    ? unique.sort((a, b) => Number(a) - Number(b))
    : unique.sort((a, b) => a.localeCompare(b));
};

/** Valores presentes en los datos para cada filtro (nunca una lista fija). */
export function filterOptions(runs: readonly ExperimentRun[]) {
  return {
    status: sortValues(runs.map((run) => run.status)),
    seed: sortValues(runs.map((run) => String(run.params.seed))),
    trainable_layers: sortValues(runs.map((run) => run.params.trainable_layers)),
    optimizer: sortValues(runs.map((run) => run.params.optimizer)),
    learning_rate: sortValues(runs.map((run) => String(run.params.learning_rate))),
  };
}

/**
 * Conteo de campaña: solo runs de training; elegibles según la API (D04-01). Los
 * auxiliares y excluidos se cuentan aparte y nunca suman a la campaña.
 */
export function campaignCounts(response: {
  runs: readonly ExperimentRun[];
  excluded: readonly unknown[];
}) {
  return {
    training: response.runs.length,
    eligible: response.runs.filter((run) => run.campaign_eligible).length,
    excluded: response.excluded.length,
  };
}

/**
 * D05-03 — Con filas aceptadas por la selección (D05-02) solo cuentan esos runs; el resto
 * de runs de training (reintentos, no aceptados) se cuenta aparte. `null` sin aceptación.
 */
export function acceptedCounts(
  runs: readonly ExperimentRun[],
  accepted: ReadonlySet<string>
): { accepted: number; notAccepted: number } | null {
  if (accepted.size === 0) return null;
  const counted = runs.filter((run) => accepted.has(run.run_id)).length;
  return { accepted: counted, notAccepted: runs.length - counted };
}

export const CONFIG_KEYS = Object.keys(
  trainingConfigSchema.shape
) as (keyof ExperimentRun["params"])[];

/** Cada campo de TrainingConfig lado a lado; `differs` si no es igual en todos. */
export function compareParams(runs: readonly ExperimentRun[]) {
  return CONFIG_KEYS.map((key) => {
    const values = runs.map((run) => String(run.params[key]));
    return { key, values, differs: new Set(values).size > 1 };
  });
}

export type CurveMetric = "val_accuracy" | "val_macro_f1" | "val_loss" | "train_loss";
export const CURVE_METRICS: readonly CurveMetric[] = [
  "val_accuracy",
  "val_macro_f1",
  "val_loss",
  "train_loss",
];

/** Filas por época con el valor de cada run; null donde ese run no tiene la época. */
export function curveRows(runs: readonly ExperimentRun[], metric: CurveMetric) {
  const epochs = [...new Set(runs.flatMap((run) => run.history.map((h) => h.epoch)))].sort(
    (a, b) => a - b
  );
  return epochs.map((epoch) => {
    const row: Record<string, number | null> & { epoch: number } = { epoch };
    for (const run of runs) {
      row[run.run_id] = run.history.find((h) => h.epoch === epoch)?.[metric] ?? null;
    }
    return row;
  });
}
