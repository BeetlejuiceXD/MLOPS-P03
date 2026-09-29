/**
 * D02-05 — Acciones sobre jobs de entrenamiento (POST) y lectura de logs.
 * Igual que los GET de `api.ts`: toda respuesta se valida con su contrato.
 */
import { buildApiUrl } from "@/api/client";
import {
  apiErrorSchema,
  type CreateTrainingJobRequest,
  jobLogsSchema,
  type TrainingJob,
  trainingJobSchema,
} from "./contracts";

async function request(path: string, init?: RequestInit): Promise<unknown> {
  let response: Response;
  try {
    response = await fetch(buildApiUrl(path), {
      ...init,
      headers: { Accept: "application/json", "Content-Type": "application/json" },
    });
  } catch {
    throw new Error("No se pudo conectar con el servidor.");
  }
  const body: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const apiError = apiErrorSchema.safeParse(body);
    throw new Error(
      apiError.success
        ? apiError.data.error
        : `El servidor respondió con estado ${response.status}.`
    );
  }
  return body;
}

function parseJob(body: unknown): TrainingJob {
  const parsed = trainingJobSchema.safeParse(body);
  if (!parsed.success) throw new Error("La respuesta del servidor no tiene el formato esperado.");
  return parsed.data;
}

export async function createTrainingJob(body: CreateTrainingJobRequest): Promise<TrainingJob> {
  return parseJob(await request("/training/jobs", { method: "POST", body: JSON.stringify(body) }));
}

export async function cancelTrainingJob(id: number): Promise<TrainingJob> {
  return parseJob(await request(`/training/jobs/${id}/cancel`, { method: "POST", body: "{}" }));
}

export async function fetchJobLogs(id: number) {
  const parsed = jobLogsSchema.safeParse(await request(`/training/jobs/${id}/logs`));
  if (!parsed.success) throw new Error("La respuesta del servidor no tiene el formato esperado.");
  return parsed.data;
}
