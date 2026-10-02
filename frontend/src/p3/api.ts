/**
 * D01-05 — Hooks del portal para los endpoints P3 (`contracts/p3/README.md`).
 * Cada respuesta pasa por su esquema Zod antes de llegar a un componente.
 */
import { useValidatedFetch } from "@/hooks/useValidatedFetch";
import {
  evaluationDetailsSchema,
  evaluationPredictionsSchema,
  evaluationResponseSchema,
  experimentRunDetailSchema,
  experimentRunsResponseSchema,
  localTestModelsResponseSchema,
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
// D05-05: el 503 de una evaluación incoherente o MariaDB caída trae el motivo; se muestra.
export const useEvaluation = () =>
  useValidatedFetch("/evaluation", evaluationResponseSchema, WITH_REASON);
// D06-05: detalle (procedencia, umbral, ejemplos) y exportación por muestra de la evaluación
// guardada. Solo se piden con la evaluación `ready`: antes, la API responde 409/404.
export const useEvaluationDetails = () =>
  useValidatedFetch("/evaluation/details", evaluationDetailsSchema, WITH_REASON);
export const useEvaluationPredictions = () =>
  useValidatedFetch("/evaluation/predictions", evaluationPredictionsSchema, WITH_REASON);
export const useModels = () => useValidatedFetch("/models", modelsResponseSchema);
// D05-03: estado de la selección (D04-04/D05-02); el 503 trae el motivo.
export const useSelection = () =>
  useValidatedFetch("/selection", selectionStateSchema, WITH_REASON);
// D05-06: registro local_test (MinIO) de D04-06, aparte de la lista official.
export const useLocalTestModels = () =>
  useValidatedFetch("/models/local-test", localTestModelsResponseSchema, WITH_REASON);
