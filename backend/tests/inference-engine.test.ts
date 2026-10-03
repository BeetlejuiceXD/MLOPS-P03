/**
 * D05-07 (preparación) — Adaptador HTTP hacia el motor de inferencia (D05-04).
 *
 * Contrato propuesto para el motor (`INFERENCE_ENGINE_URL`):
 *   GET  {url}/identity → inference_engine (identidad del paquete cargado)
 *   POST {url}/predict  → cuerpo = bytes de la imagen (Content-Type image/*);
 *                         responde inference_engine_prediction; 4xx = imagen rechazada
 *
 * Sin URL configurada, el motor está "pendiente" (503 con el motivo), igual que los
 * adaptadores pendientes de D04-04. D06-06 cambia solo la fuente (modelo official).
 */
import { describe, expect, it } from 'vitest';
import {
  createHttpInferenceEngine,
  EngineRejectedInputError,
  EngineUnavailableError,
} from '../src/logic/inference-engine.js';

const SMOKE = {
  source: 'smoke',
  package_id: 'p3-cnn-classifier-smoke-c46e4c3ab2bb',
  format_version: '1.0.0',
  model_version: null,
  mlflow_run_id: 'c46e4c3ab2bb4ee18c37571adbb65d92',
  checkpoint_sha256: 'e93de23e2cf9e72e8efa97bde1e9fb2083402d15a422705dab80ffd11bbd5b97',
};

function fakeFetch(handler: (url: string, init?: RequestInit) => Response | Promise<Response>) {
  const calls: { url: string; init?: RequestInit }[] = [];
  const impl = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    calls.push({ url, init });
    return handler(url, init);
  }) as typeof fetch;
  return { calls, impl };
}
const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });

describe('motor de inferencia por HTTP', () => {
  it('sin INFERENCE_ENGINE_URL: pendiente, con el motivo', async () => {
    const engine = createHttpInferenceEngine({ baseUrl: undefined });
    await expect(engine.identity()).rejects.toThrow(EngineUnavailableError);
    await expect(engine.predict(Buffer.from('x'), 'image/png')).rejects.toThrow(
      /INFERENCE_ENGINE_URL no está configurado.*D05-04/,
    );
  });

  it('identity: GET {url}/identity', async () => {
    const { calls, impl } = fakeFetch(() =>
      json({ model: SMOKE, classes: ['cat', 'dog'], image_size: 224 }),
    );
    const engine = createHttpInferenceEngine({ baseUrl: 'http://motor:8100/', fetch: impl });
    expect((await engine.identity()).model).toEqual(SMOKE);
    expect(calls[0]?.url).toBe('http://motor:8100/identity');
  });

  it('predict: POST {url}/predict con los bytes y su Content-Type', async () => {
    const { calls, impl } = fakeFetch(() =>
      json({ predicted_class: 'dog', probabilities: { cat: 0.1, dog: 0.9 }, model: SMOKE }),
    );
    const engine = createHttpInferenceEngine({ baseUrl: 'http://motor:8100', fetch: impl });
    const bytes = Buffer.from([1, 2, 3]);
    const prediction = await engine.predict(bytes, 'image/jpeg');
    expect(prediction.predicted_class).toBe('dog');
    expect(calls[0]?.url).toBe('http://motor:8100/predict');
    expect(calls[0]?.init?.method).toBe('POST');
    expect(new Headers(calls[0]?.init?.headers).get('content-type')).toBe('image/jpeg');
    expect(Buffer.from(calls[0]?.init?.body as Uint8Array).equals(bytes)).toBe(true);
  });

  it('4xx del motor = imagen rechazada, con su motivo', async () => {
    const { impl } = fakeFetch(() => json({ error: 'imagen ilegible' }, 422));
    const engine = createHttpInferenceEngine({ baseUrl: 'http://motor:8100', fetch: impl });
    await expect(engine.predict(Buffer.from('x'), 'image/png')).rejects.toThrow(
      EngineRejectedInputError,
    );
    await expect(engine.predict(Buffer.from('x'), 'image/png')).rejects.toThrow(/imagen ilegible/);
  });

  it('5xx, red caída o JSON ilegible = motor no disponible', async () => {
    for (const handler of [
      () => json({ error: 'modelo no cargado' }, 503),
      () => {
        throw new TypeError('fetch failed');
      },
      () => new Response('<html>', { status: 200 }),
    ]) {
      const { impl } = fakeFetch(handler);
      const engine = createHttpInferenceEngine({ baseUrl: 'http://motor:8100', fetch: impl });
      await expect(engine.predict(Buffer.from('x'), 'image/png')).rejects.toThrow(
        EngineUnavailableError,
      );
      await expect(engine.identity()).rejects.toThrow(EngineUnavailableError);
    }
  });
});
