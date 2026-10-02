/**
 * D05-03 — Cotejo entre la selección (D04-04/D05-02, `GET /api/selection`) y los runs que
 * Experiments recibe de MLflow (D04-01). No ordena ni elige nada: el ranking y el candidato
 * son los de la API. Solo comprueba que lo que dice la selección existe en Experiments con
 * los mismos valores.
 */
import { z } from "zod";
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

/**
 * Lo que Experiments lee del expediente de campaña de D05-02 (`GET /api/selection/campaign`):
 * el rol de cada intento. No es otra fuente de verdad: solo rotula los runs que la selección
 * no acepta (reintentos, pendientes, excluidos). El resto de campos se ignora.
 */
const dossierAttemptSchema = z.object({
  run_id: z.string().nullable(),
  role: z.enum(["representative", "retry", "pending", "excluded"]),
  reason: z.string().nullable(),
});
export const campaignDossierAttemptsSchema = z.object({
  rows: z.array(z.object({ row: z.number().int(), attempts: z.array(dossierAttemptSchema) })),
  unattributed: z.array(dossierAttemptSchema),
});
export type CampaignDossierAttempts = z.infer<typeof campaignDossierAttemptsSchema>;

export type AttemptMark = { role: "retry" | "pending" | "excluded"; row: number | null };

/** run_id → rol de los intentos que NO son representantes, tal como los clasificó D05-02. */
export function attemptMarks(dossier: CampaignDossierAttempts | null) {
  const marks = new Map<string, AttemptMark>();
  if (dossier === null) return marks;
  const add = (attempt: z.infer<typeof dossierAttemptSchema>, row: number | null) => {
    if (attempt.run_id !== null && attempt.role !== "representative") {
      marks.set(attempt.run_id, { role: attempt.role, row });
    }
  };
  for (const row of dossier.rows) for (const attempt of row.attempts) add(attempt, row.row);
  for (const attempt of dossier.unattributed) add(attempt, null);
  return marks;
}

const ROLE_LABEL: Record<AttemptMark["role"], string> = {
  retry: "Reintento",
  pending: "Pendiente",
  excluded: "Excluido",
};

/** Rótulo de un run que no cuenta para la campaña aceptada. */
export function attemptLabel(mark: AttemptMark | undefined) {
  if (mark === undefined) return "No aceptado por D05-02";
  return mark.row === null ? ROLE_LABEL[mark.role] : `${ROLE_LABEL[mark.role]} · fila ${mark.row}`;
}
