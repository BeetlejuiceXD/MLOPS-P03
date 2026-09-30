/**
 * D01-05 — Hooks del portal para los endpoints P3 (`contracts/p3/README.md`).
 * Cada respuesta pasa por su esquema Zod antes de llegar a un componente.
 */
import { useValidatedFetch } from "@/hooks/useValidatedFetch";
import {
  evaluationResponseSchema,
  experimentRunsResponseSchema,
  manifestSummarySchema,
  modelsResponseSchema,
  releasesResponseSchema,
  trainingJobListSchema,
} from "./contracts";

// D03-03: el 503 de las fuentes trae el motivo que publicó trainer-worker; se muestra.
const WITH_REASON = { showServerReason: true } as const;
export const useReleases = () =>
  useValidatedFetch("/releases", releasesResponseSchema, WITH_REASON);
export const useManifest = () => useValidatedFetch("/manifest", manifestSummarySchema, WITH_REASON);
export const useTrainingJobs = () => useValidatedFetch("/training/jobs", trainingJobListSchema);
export const useExperimentRuns = () =>
  useValidatedFetch("/experiments/runs", experimentRunsResponseSchema);
export const useEvaluation = () => useValidatedFetch("/evaluation", evaluationResponseSchema);
export const useModels = () => useValidatedFetch("/models", modelsResponseSchema);
