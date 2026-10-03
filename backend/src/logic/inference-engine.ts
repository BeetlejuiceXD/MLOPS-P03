/**
 * D05-07 — Puerto del motor de inferencia y su adaptador HTTP.
 *
 * El portal no calcula clases: pasa los bytes al motor de D05-04, que carga el paquete
 * smoke de D05-01 (y, en D06-06, el modelo official recargado de AWS). Contrato del motor
 * (`INFERENCE_ENGINE_URL`):
 *
 *   GET  {url}/identity → inference_engine
 *   POST {url}/predict  → cuerpo = bytes de la imagen, `Content-Type: image/*`;
 *                         200 inference_engine_prediction · 4xx imagen rechazada {error}
 *                         · 5xx modelo/servicio no disponible {error}
 *
 * Este módulo es el punto de sustitución de la fuente: D06-06 apunta la URL a un motor con
 * el modelo official y el resto del portal no cambia.
 */
import type { InferenceModelIdentity } from './p3.contracts.js';

export interface EngineIdentity {
  model: InferenceModelIdentity;
  classes: readonly string[];
  image_size: number;
}

export interface EnginePrediction {
  predicted_class: string;
  probabilities: Record<string, number>;
  model: InferenceModelIdentity;
}

export interface InferenceEngine {
  identity(): Promise<EngineIdentity>;
  predict(image: Buffer, mimeType: string): Promise<EnginePrediction>;
}

/** El motor no responde, no tiene modelo cargado o no está configurado (→ 503). */
export class EngineUnavailableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'EngineUnavailableError';
  }
}

/** El motor respondió que la imagen no se puede clasificar (→ 400). */
export class EngineRejectedInputError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'EngineRejectedInputError';
  }
}

const DEFAULT_TIMEOUT_MS = 30_000;

async function readError(response: Response): Promise<string> {
  const body = (await response.json().catch(() => null)) as { error?: unknown } | null;
  return typeof body?.error === 'string' ? body.error : `HTTP ${response.status}`;
}

export function createHttpInferenceEngine(options: {
  baseUrl: string | undefined;
  fetch?: typeof fetch;
  timeoutMs?: number;
}): InferenceEngine {
  const { baseUrl } = options;
  const doFetch = options.fetch ?? fetch;
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;

  async function call(path: string, init?: RequestInit): Promise<unknown> {
    if (!baseUrl) {
      throw new EngineUnavailableError(
        'INFERENCE_ENGINE_URL no está configurado: el motor de D05-04 aún no está integrado',
      );
    }
    let response: Response;
    try {
      response = await doFetch(`${baseUrl.replace(/\/+$/, '')}${path}`, {
        ...init,
        signal: AbortSignal.timeout(timeoutMs),
      });
    } catch (error) {
      throw new EngineUnavailableError(error instanceof Error ? error.message : String(error));
    }
    if (response.status >= 400 && response.status < 500) {
      throw new EngineRejectedInputError(await readError(response));
    }
    if (!response.ok) throw new EngineUnavailableError(await readError(response));
    try {
      return await response.json();
    } catch {
      throw new EngineUnavailableError('respuesta del motor que no es JSON');
    }
  }

  return {
    // Validadas con su contrato en el servicio: aquí solo el transporte.
    identity: async () => {
      try {
        return (await call('/identity')) as EngineIdentity;
      } catch (error) {
        // Un 4xx al pedir la identidad no es culpa de ninguna imagen: el motor no sirve.
        if (error instanceof EngineRejectedInputError) {
          throw new EngineUnavailableError(error.message);
        }
        throw error;
      }
    },
    predict: async (image, mimeType) =>
      (await call('/predict', {
        method: 'POST',
        headers: { 'Content-Type': mimeType },
        body: new Uint8Array(image),
      })) as EnginePrediction,
  };
}
