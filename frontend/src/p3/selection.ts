/**
 * D05-03 — Cotejo entre la selección (D04-04/D05-02, `GET /api/selection`) y los runs que
 * Experiments recibe de MLflow (D04-01). No ordena ni elige nada: el ranking y el candidato
 * son los de la API. Solo comprueba que lo que dice la selección existe en Experiments con
 * los mismos valores.
 */
import type { ExperimentRun, RankedRun, SelectionState } from "./contracts";

const METRICS: readonly [keyof RankedRun, (run: ExperimentRun) => number | undefined][] = [
  ["best_epoch", (run) => run.summary?.best_epoch],
  ["val_accuracy", (run) => run.summary?.best_val_accuracy],
  ["val_macro_f1", (run) => run.summary?.best_val_macro_f1],
  ["val_loss", (run) => run.summary?.best_val_loss],
];

/** run_id → fila OFAT, tal como la asignó la selección (no se recalcula aquí). */
export function campaignRows(selection: Pick<SelectionState, "ranking">) {
  return new Map(selection.ranking.map((ranked) => [ranked.run_id, ranked.campaign_row]));
}

/**
 * Problemas que impiden presentar la aceptación: candidato o fila aceptada que no aparece
 * en Experiments, que no es elegible o cuyas métricas del mejor checkpoint no coinciden.
 */
export function selectionCrossCheck(
  runs: readonly ExperimentRun[],
  selection: Pick<SelectionState, "candidate" | "ranking">
): { candidate: ExperimentRun | null; problems: string[] } {
  if (selection.candidate === null) return { candidate: null, problems: [] };
  const byId = new Map(runs.map((run) => [run.run_id, run]));
  const problems: string[] = [];

  const candidateRun = byId.get(selection.candidate.run_id) ?? null;
  if (candidateRun === null) {
    problems.push(
      `El candidato ${selection.candidate.run_id} de la selección no aparece en Experiments (MLflow).`
    );
  }
  for (const ranked of selection.ranking) {
    const run = byId.get(ranked.run_id);
    if (!run) {
      if (ranked.run_id !== selection.candidate.run_id) {
        problems.push(
          `La fila ${ranked.campaign_row} (${ranked.run_id}) no aparece en Experiments.`
        );
      }
      continue;
    }
    if (!run.campaign_eligible) {
      problems.push(
        `La fila ${ranked.campaign_row} (${ranked.run_id}) no es elegible en Experiments: ${run.ineligible_reasons.join("; ")}`
      );
    }
    for (const [key, read] of METRICS) {
      const shown = read(run);
      if (shown !== ranked[key]) {
        problems.push(
          `${ranked.run_id}: ${key} de la selección = ${ranked[key]}, en MLflow = ${shown ?? "—"}.`
        );
      }
    }
  }
  return { candidate: candidateRun, problems };
}

type Exclusion = SelectionState["excluded"][number];
export type ExclusionMark = Pick<Exclusion, "reason" | "detail">;

const EXCLUSION_LABEL: Record<Exclusion["reason"], string> = {
  duplicate_campaign_row: "Reintento de una fila ya aceptada",
  outside_campaign_matrix: "Fuera de la matriz OFAT",
  not_finished: "No terminado",
  invalid_contract: "Contrato inválido",
  manifest_mismatch: "Manifest distinto",
  release_mismatch: "Release distinto",
  provenance_inconsistent: "Procedencia inconsistente",
};

/**
 * run_id → motivo por el que la selección (D05-02) NO lo cuenta, tal cual viene en
 * `excluded` de `GET /api/selection`. Experiments solo lo rotula; no lo decide.
 */
export function exclusionMarks(selection: Pick<SelectionState, "excluded">) {
  const marks = new Map<string, ExclusionMark>();
  for (const { run_id, reason, detail } of selection.excluded) {
    if (run_id !== null) marks.set(run_id, { reason, detail });
  }
  return marks;
}

/** Rótulo de un run de training que no cuenta para la campaña aceptada. */
export function exclusionLabel(mark: ExclusionMark | undefined) {
  return mark === undefined ? "No aceptado por D05-02" : EXCLUSION_LABEL[mark.reason];
}
