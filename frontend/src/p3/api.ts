/**
 * D01-05 — Hooks del portal para los endpoints P3 (`contracts/p3/README.md`).
 * Cada respuesta pasa por su esquema Zod antes de llegar a un componente.
 */
import { useValidatedFetch } from "@/hooks/useValidatedFetch";
import {
  evaluationResponseSchema,
  experimentRunDetailSchema,
  experimentRunsResponseSchema,
  manifestSummarySchema,
  modelsResponseSchema,
  releasesResponseSchema,
  selectionStateSchema,
  trainingJobListSchema,
} from "./contracts";

// D03-03: el 503 de las fuentes trae el motivo que publicó trainer-worker; se muestra.
const WITH_REASON = { showServerReason: true } as const;
export const useReleases = () =>
  useValidatedFetch("/releases", releasesResponseSchema, WITH_REASON);
export const useManifest = () => useValidatedFetch("/manifest", manifestSummarySchema, WITH_REASON);
export const useTrainingJobs = () => useValidatedFetch("/training/jobs", trainingJobListSchema);
// D04-02: el 503 de MLflow caído (D04-01) trae el motivo; se muestra.
export const useExperimentRuns = () =>
  useValidatedFetch("/experiments/runs", experimentRunsResponseSchema, WITH_REASON);
export const useExperimentRun = (runId: string) =>
  useValidatedFetch(`/experiments/runs/${runId}`, experimentRunDetailSchema, WITH_REASON);
export const useEvaluation = () => useValidatedFetch("/evaluation", evaluationResponseSchema);
export const useModels = () => useValidatedFetch("/models", modelsResponseSchema);
// D05-03: estado de la selección (D04-04/D05-02); el 503 trae el motivo.
export const useSelection = () =>
  useValidatedFetch("/selection", selectionStateSchema, WITH_REASON);
