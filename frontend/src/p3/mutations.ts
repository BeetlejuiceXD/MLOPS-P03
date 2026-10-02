/**
 * D02-05 — Acciones sobre jobs de entrenamiento (POST) y lectura de logs.
 * D05-07 — Inference: predecir (archivo o crop) y enviar a la cola de anotación.
 * Igual que los GET de `api.ts`: toda respuesta se valida con su contrato.
 */
import { buildApiUrl } from "@/api/client";
import {
  type AnnotationQueueItem,
  annotationQueueItemSchema,
  apiErrorSchema,
  type CreateTrainingJobRequest,
  type InferenceResult,
  inferenceResultSchema,
  jobLogsSchema,
  type TrainingJob,
  trainingJobSchema,
} from "./contracts";

async function request(path: string, init?: RequestInit): Promise<unknown> {
  let response: Response;
  try {
    response = await fetch(buildApiUrl(path), {
      ...init,
      // Un FormData (multipart) lleva su propio Content-Type con el boundary.
      headers:
        init?.body instanceof FormData
          ? { Accept: "application/json" }
          : { Accept: "application/json", "Content-Type": "application/json" },
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

function parseWith<T>(schema: { safeParse: (value: unknown) => { success: boolean; data?: T } }) {
  return (body: unknown): T => {
    const parsed = schema.safeParse(body);
    if (!parsed.success) throw new Error("La respuesta del servidor no tiene el formato esperado.");
    return parsed.data as T;
  };
}

/** Archivo nuevo: el portal manda los bytes; la clase la calcula el motor (D05-04). */
export async function runInferenceUpload(file: File): Promise<InferenceResult> {
  const form = new FormData();
  form.append("image", file);
  return parseWith<InferenceResult>(inferenceResultSchema)(
    await request("/inference", { method: "POST", body: form })
  );
}

/** Crop del portal: solo el id de la anotación; la API recorta la imagen original. */
export async function runInferenceCrop(annotationId: number): Promise<InferenceResult> {
  return parseWith<InferenceResult>(inferenceResultSchema)(
    await request("/inference", {
      method: "POST",
      body: JSON.stringify({ annotation_id: annotationId }),
    })
  );
}

export async function sendToAnnotationQueue(inferenceId: number): Promise<AnnotationQueueItem> {
  return parseWith<AnnotationQueueItem>(annotationQueueItemSchema)(
    await request(`/inference/${inferenceId}/annotation-queue`, { method: "POST", body: "{}" })
  );
}
