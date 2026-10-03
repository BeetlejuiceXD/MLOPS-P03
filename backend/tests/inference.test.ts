/**
 * D05-07 (preparación) — Inference: API y cola de anotación sobre un motor de inferencia.
 *
 *   GET  /inference/engine                  → identidad del paquete que sirve el motor
 *   POST /inference                         → multipart `image` (archivo nuevo) o JSON
 *                                             `{ annotation_id }` (crop de una anotación)
 *   GET  /inference, GET /inference/:id     → inferencias persistidas (tras recargar)
 *   POST /inference/:id/annotation-queue    → elemento de la cola (201 nuevo, 200 reintento)
 *   GET  /inference/annotation-queue        → cola persistida
 *
 * El motor es un FIXTURE (`FakeEngine`): estas pruebas cubren la API, la validación y la
 * persistencia por contrato, no el motor real de D05-04, que se integra y evidencia aparte.
 * La clase nunca se calcula aquí: sale del motor y se guarda con su identidad.
 */
import { createHash } from 'node:crypto';
import type { Server } from 'node:http';
import type { AddressInfo } from 'node:net';
import express from 'express';
import sharp from 'sharp';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import {
  createInferenceService,
  type InferenceRepository,
  type NewInference,
  type PortalImages,
  type QueueRecord,
  type StoredInference,
} from '../src/logic/inference.service.js';
import {
  type EngineIdentity,
  type EnginePrediction,
  EngineRejectedInputError,
  EngineUnavailableError,
  type InferenceEngine,
} from '../src/logic/inference-engine.js';
import {
  annotationQueueItemSchema,
  annotationQueueResponseSchema,
  inferenceEngineSchema,
  inferenceListSchema,
  inferenceResultSchema,
} from '../src/logic/p3.contracts.js';
import { createInferenceRouter } from '../src/ui/inference.routes.js';

const sha = (body: Buffer) => createHash('sha256').update(body).digest('hex');
const SMOKE = {
  source: 'smoke' as const,
  package_id: 'p3-cnn-classifier-smoke-c46e4c3ab2bb',
  format_version: '1.0.0',
  model_version: null,
  mlflow_run_id: 'c46e4c3ab2bb4ee18c37571adbb65d92',
  checkpoint_sha256: 'e93de23e2cf9e72e8efa97bde1e9fb2083402d15a422705dab80ffd11bbd5b97',
  s3_object: null,
};
/**
 * D06-06 (preparación): identidad SINTÉTICA de un modelo official recargado de AWS. Solo
 * prueba que la forma viaja y se conserva; no es el modelo seleccionado ni un objeto real.
 */
const OFFICIAL = {
  source: 'official' as const,
  package_id: 'p3-cnn-classifier-1.0.0',
  format_version: '1.0.0',
  model_version: '1.0.0',
  mlflow_run_id: 'c46e4c3ab2bb4ee18c37571adbb65d92',
  checkpoint_sha256: 'e93de23e2cf9e72e8efa97bde1e9fb2083402d15a422705dab80ffd11bbd5b97',
  s3_object: {
    s3_bucket: 'mlops-p3-models-sintetico',
    s3_key: 'models/p3-cnn-classifier/1.0.0/model.pt',
    version_id: 'sintetico-3HL4kqtJlcpXroDTDmJ.rmSpXd3dIbrHY',
    sha256: 'd'.repeat(64),
  },
};
const NOW = new Date('2026-10-02T15:00:00.000Z');
const MAX_BYTES = 200_000;

const png = (width: number, height: number, rgb: [number, number, number] = [200, 120, 40]) =>
  sharp({ create: { width, height, channels: 3, background: { r: rgb[0], g: rgb[1], b: rgb[2] } } })
    .png()
    .toBuffer();

class FakeEngine implements InferenceEngine {
  calls: { bytes: Buffer; mimeType: string }[] = [];
  mode: 'ok' | 'down' | 'reject' | 'bad-response' = 'ok';
  /** Identidad que reporta el motor (smoke hoy; official con D06-06). */
  model: EnginePrediction['model'] = SMOKE;
  async identity(): Promise<EngineIdentity> {
    if (this.mode === 'down') throw new EngineUnavailableError('connect ECONNREFUSED motor:8100');
    return { model: this.model, classes: ['cat', 'dog'], image_size: 224 };
  }
  async predict(bytes: Buffer, mimeType: string): Promise<EnginePrediction> {
    this.calls.push({ bytes, mimeType });
    if (this.mode === 'down') throw new EngineUnavailableError('connect ECONNREFUSED motor:8100');
    if (this.mode === 'reject')
      throw new EngineRejectedInputError('imagen ilegible para el modelo');
    if (this.mode === 'bad-response') {
      return { predicted_class: 'cat', probabilities: { cat: 0.1, dog: 0.9 }, model: this.model };
    }
    return {
      predicted_class: 'dog',
      probabilities: { cat: 0.0587, dog: 0.9413 },
      model: this.model,
    };
  }
}

class MemoryRepo implements InferenceRepository {
  inferences: StoredInference[] = [];
  queue: QueueRecord[] = [];
  failInsert = false;
  failQueueInsert = false;
  /** MariaDB caída para lecturas: cualquier select falla con el mensaje de drizzle. */
  failReads = false;
  private readFails() {
    if (this.failReads) throw new Error('Failed query: select * from `p3_inference`\nparams: 1');
  }
  async insert(record: NewInference) {
    if (this.failInsert) {
      throw new Error(
        'Failed query: insert into `p3_inference` (`input`) values (?)\nparams: gato.jpg',
      );
    }
    const id = this.inferences.length + 1;
    this.inferences.push({ ...record, id });
    return id;
  }
  async find(id: number) {
    this.readFails();
    return this.inferences.find((row) => row.id === id) ?? null;
  }
  async list(limit: number) {
    this.readFails();
    return [...this.inferences].reverse().slice(0, limit);
  }
  async findQueueItem(inferenceId: number) {
    this.readFails();
    return this.queue.find((row) => row.inference_id === inferenceId) ?? null;
  }
  async insertQueueItem(item: Omit<QueueRecord, 'id'>) {
    if (this.failQueueInsert) throw new Error('connect ECONNREFUSED mariadb:3306');
    const record = { ...item, id: this.queue.length + 1 };
    this.queue.push(record);
    return record;
  }
  /** Transacción: la imagen pending y el elemento se crean juntos o ninguno. */
  portal?: MemoryImages;
  async insertQueueItemWithNewImage(
    image: { storageKey: string; mimeType: string },
    item: Omit<QueueRecord, 'id' | 'image_id'>,
  ) {
    if (this.failQueueInsert) throw new Error('connect ECONNREFUSED mariadb:3306');
    const imageId = (this.portal as MemoryImages).commitPendingImage(image);
    const record = { ...item, image_id: imageId, id: this.queue.length + 1 };
    this.queue.push(record);
    return record;
  }
  async listQueue() {
    this.readFails();
    return [...this.queue].reverse();
  }
}

class MemoryImages implements PortalImages {
  objects = new Map<string, Buffer>();
  images = new Map<number, { storageKey: string; mimeType: string; status: string }>();
  annotations = new Map<number, { image_id: number; bbox: [number, number, number, number] }>();
  createdImages: number[] = [];
  createdAnnotations = 0;
  /** Si hubiera limpieza compensatoria, también fallaría (MariaDB caída a la mitad). */
  failDelete = false;
  /** MariaDB/MinIO caídos para las lecturas del portal. */
  failReads = false;
  private seq = 100;
  async storeInput(bytes: Buffer) {
    const key = `inference/${this.objects.size + 1}`;
    this.objects.set(key, Buffer.from(bytes));
    return key;
  }
  async deleteInput(key: string) {
    this.objects.delete(key);
  }
  async findAnnotation(id: number) {
    if (this.failReads) throw new Error('Failed query: select * from `annotations`');
    const found = this.annotations.get(id);
    return found ? { annotation_id: id, ...found } : null;
  }
  async readImage(imageId: number) {
    const image = this.images.get(imageId);
    if (!image) return null;
    const bytes = this.objects.get(image.storageKey) as Buffer;
    return { bytes, mimeType: image.mimeType };
  }
  async imageExists(imageId: number) {
    if (this.failReads) throw new Error('Failed query: select * from `images`');
    return this.images.has(imageId);
  }
  /** Lo que confirma la transacción del repositorio (fila `images` en pending). */
  commitPendingImage(image: { storageKey: string; mimeType: string }) {
    const id = ++this.seq;
    this.images.set(id, { ...image, status: 'pending' });
    this.createdImages.push(id);
    return id;
  }
}

let server: Server;
let base: string;
let engine: FakeEngine;
let repo: MemoryRepo;
let images: MemoryImages;

beforeEach(async () => {
  engine = new FakeEngine();
  repo = new MemoryRepo();
  images = new MemoryImages();
  repo.portal = images;
  const service = createInferenceService({
    engine,
    repo,
    images,
    maxUploadBytes: MAX_BYTES,
    now: () => NOW,
  });
  const app = express();
  app.use(express.json());
  app.use('/inference', createInferenceRouter(service, { maxUploadBytes: MAX_BYTES }));
  server = app.listen(0);
  base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});

afterEach(() => new Promise<void>((resolve) => server.close(() => resolve())));

function postFile(body: Buffer, type: string, name = 'foto.png') {
  const form = new FormData();
  form.append('image', new Blob([new Uint8Array(body)], { type }), name);
  return fetch(`${base}/inference`, { method: 'POST', body: form });
}
const postCrop = (annotationId: unknown) =>
  fetch(`${base}/inference`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ annotation_id: annotationId }),
  });
const enqueue = (id: number) =>
  fetch(`${base}/inference/${id}/annotation-queue`, { method: 'POST' });

/** Una imagen del portal (en el "bucket") con una anotación sobre ella. */
async function portalImageWithBox() {
  const source = await png(200, 100);
  images.objects.set('images/portal-1', source);
  images.images.set(42, {
    storageKey: 'images/portal-1',
    mimeType: 'image/png',
    status: 'completed',
  });
  images.annotations.set(7, { image_id: 42, bbox: [10, 20, 120, 60] });
}

describe('GET /inference/engine', () => {
  it('devuelve la identidad del paquete que carga el motor (smoke, sin semver)', async () => {
    const res = await fetch(`${base}/inference/engine`);
    expect(res.status).toBe(200);
    const body = inferenceEngineSchema.parse(await res.json());
    expect(body.model).toEqual(SMOKE);
  });

  it('motor ausente → 503 con el motivo', async () => {
    engine.mode = 'down';
    const res = await fetch(`${base}/inference/engine`);
    expect(res.status).toBe(503);
    expect((await res.json()).error).toMatch(/motor de inferencia.*no disponible.*ECONNREFUSED/);
  });
});

describe('POST /inference con un archivo nuevo', () => {
  it('pasa los bytes al motor, guarda el resultado con la identidad y lo devuelve', async () => {
    const body = await png(64, 48);
    const res = await postFile(body, 'image/png', 'perro.png');
    expect(res.status).toBe(201);
    const result = inferenceResultSchema.parse(await res.json());
    expect(result).toEqual({
      inference_id: 1,
      created_at: NOW.toISOString(),
      input: {
        kind: 'upload',
        filename: 'perro.png',
        mime_type: 'image/png',
        size_bytes: body.length,
        width: 64,
        height: 48,
        sha256: sha(body),
      },
      predicted_class: 'dog',
      probabilities: { cat: 0.0587, dog: 0.9413 },
      model: SMOKE,
      annotation_queue_item_id: null,
    });
    // El motor recibe exactamente los bytes subidos; la clase no se calcula en la API.
    expect(engine.calls).toHaveLength(1);
    expect(sha(engine.calls[0]?.bytes as Buffer)).toBe(sha(body));
    expect(engine.calls[0]?.mimeType).toBe('image/png');
    // La imagen queda guardada para poder enviarla después a la cola.
    expect(repo.inferences[0]?.storage_key).toBe('inference/1');
    expect(images.objects.get('inference/1')?.equals(body)).toBe(true);
  });

  it('queda consultable después (GET /inference/:id y GET /inference)', async () => {
    await postFile(await png(32, 32), 'image/png');
    await postFile(await png(40, 30), 'image/png');
    const one = inferenceResultSchema.parse(await (await fetch(`${base}/inference/1`)).json());
    expect(one.inference_id).toBe(1);
    const list = inferenceListSchema.parse(await (await fetch(`${base}/inference`)).json());
    expect(list.inferences.map((item) => item.inference_id)).toEqual([2, 1]);
    expect((await fetch(`${base}/inference/99`)).status).toBe(404);
    expect((await fetch(`${base}/inference/abc`)).status).toBe(400);
  });

  it.each([
    ['tipo no permitido', Buffer.from('GIF89a....'), 'image/gif', 400, /JPEG, PNG o WebP/],
    [
      'contenido que no es imagen',
      Buffer.from('no soy un png'),
      'image/png',
      400,
      /no es una imagen/,
    ],
  ])('%s → %i, sin llamar al motor ni guardar nada', async (_n, body, type, status, message) => {
    const res = await postFile(body as Buffer, type as string);
    expect(res.status).toBe(status);
    expect((await res.json()).error).toMatch(message as RegExp);
    expect(engine.calls).toHaveLength(0);
    expect(repo.inferences).toHaveLength(0);
    expect(images.objects.size).toBe(0);
  });

  it('archivo demasiado grande → 413, sin llamar al motor', async () => {
    const res = await postFile(Buffer.alloc(MAX_BYTES + 1, 1), 'image/png');
    expect(res.status).toBe(413);
    expect(engine.calls).toHaveLength(0);
    expect(repo.inferences).toHaveLength(0);
  });

  it('sin archivo ni annotation_id, o con los dos → 400', async () => {
    const none = await fetch(`${base}/inference`, { method: 'POST' });
    expect(none.status).toBe(400);
    const form = new FormData();
    form.append(
      'image',
      new Blob([new Uint8Array(await png(8, 8))], { type: 'image/png' }),
      'a.png',
    );
    form.append('annotation_id', '7');
    const both = await fetch(`${base}/inference`, { method: 'POST', body: form });
    expect(both.status).toBe(400);
    expect(engine.calls).toHaveLength(0);
  });

  it('motor ausente (modelo no cargado) → 503 con el motivo y nada guardado', async () => {
    engine.mode = 'down';
    const res = await postFile(await png(16, 16), 'image/png');
    expect(res.status).toBe(503);
    expect((await res.json()).error).toMatch(/motor de inferencia.*no disponible/);
    expect(repo.inferences).toHaveLength(0);
    expect(images.objects.size).toBe(0);
  });

  it('el motor rechaza la imagen → 400 con su motivo', async () => {
    engine.mode = 'reject';
    const res = await postFile(await png(16, 16), 'image/png');
    expect(res.status).toBe(400);
    expect((await res.json()).error).toMatch(/imagen ilegible para el modelo/);
    expect(repo.inferences).toHaveLength(0);
  });

  it('respuesta del motor que no cumple el contrato → 503, no se guarda una clase inventada', async () => {
    engine.mode = 'bad-response';
    const res = await postFile(await png(16, 16), 'image/png');
    expect(res.status).toBe(503);
    expect((await res.json()).error).toMatch(/no cumple el contrato/);
    expect(repo.inferences).toHaveLength(0);
    expect(images.objects.size).toBe(0);
  });

  it('error de persistencia → 503 y la imagen guardada se borra (nada a medias)', async () => {
    repo.failInsert = true;
    const res = await postFile(await png(16, 16), 'image/png');
    expect(res.status).toBe(503);
    const { error } = await res.json();
    expect(error).toMatch(/no se pudo guardar la inferencia/i);
    // El motivo interno (SQL y parámetros) no sale de la API.
    expect(error).not.toMatch(/Failed query|insert into|params|gato\.jpg/);
    expect(images.objects.size).toBe(0);
  });
});

describe('POST /inference con un crop del portal', () => {
  it('recorta la anotación de la imagen original y pasa ese recorte al motor', async () => {
    await portalImageWithBox();
    const res = await postCrop(7);
    expect(res.status).toBe(201);
    const result = inferenceResultSchema.parse(await res.json());
    expect(result.input).toMatchObject({
      kind: 'crop',
      image_id: 42,
      annotation_id: 7,
      bbox: [10, 20, 120, 60],
      width: 120,
      height: 60,
    });
    const sent = engine.calls[0];
    expect(sent?.mimeType).toBe('image/png');
    const meta = await sharp(sent?.bytes).metadata();
    expect([meta.width, meta.height]).toEqual([120, 60]);
    expect(result.input.sha256).toBe(sha(sent?.bytes as Buffer));
    // Un crop no copia la imagen: referencia la del portal.
    expect(repo.inferences[0]?.storage_key).toBeNull();
  });

  it.each([
    ['anotación inexistente', 999, 404],
    ['id inválido', 'x', 400],
  ])('%s → %i sin llamar al motor', async (_name, id, status) => {
    await portalImageWithBox();
    const res = await postCrop(id);
    expect(res.status).toBe(status);
    expect(engine.calls).toHaveLength(0);
  });

  it.each([
    ['se sale por la derecha y por abajo', [150, 50, 100, 80]],
    ['se sale solo por la derecha', [150, 10, 100, 20]],
    ['se sale solo por abajo', [10, 60, 20, 80]],
    ['sin área', [10, 10, 0, 20]],
  ])('caja que %s → 400 sin llamar al motor', async (_name, bbox) => {
    await portalImageWithBox();
    images.annotations.set(8, { image_id: 42, bbox: bbox as [number, number, number, number] });
    const res = await postCrop(8);
    expect(res.status).toBe(400);
    expect((await res.json()).error).toMatch(/fuera de la imagen/);
    expect(engine.calls).toHaveLength(0);
  });
});

describe('POST /inference/:id/annotation-queue', () => {
  it('archivo nuevo → crea la imagen pendiente en el portal y el elemento de la cola', async () => {
    await postFile(await png(30, 20), 'image/png', 'nueva.png');
    const res = await enqueue(1);
    expect(res.status).toBe(201);
    const item = annotationQueueItemSchema.parse(await res.json());
    expect(item).toMatchObject({
      queue_item_id: 1,
      inference_id: 1,
      annotation_id: null,
      status: 'pending',
      human_label: null,
      suggestion: { source: 'model', predicted_class: 'dog' },
      model: SMOKE,
    });
    expect(images.createdImages).toEqual([item.image_id]);
    expect(images.images.get(item.image_id)).toMatchObject({
      storageKey: 'inference/1',
      status: 'pending',
    });
    // La predicción no se convierte en anotación ni en etiqueta humana.
    expect(images.createdAnnotations).toBe(0);
    const again = inferenceResultSchema.parse(await (await fetch(`${base}/inference/1`)).json());
    expect(again.annotation_queue_item_id).toBe(1);
  });

  it('reintento → 200 con el mismo elemento; no duplica la imagen ni el elemento', async () => {
    await postFile(await png(30, 20), 'image/png');
    const first = annotationQueueItemSchema.parse(await (await enqueue(1)).json());
    const retry = await enqueue(1);
    expect(retry.status).toBe(200);
    expect(annotationQueueItemSchema.parse(await retry.json())).toEqual(first);
    expect(images.createdImages).toHaveLength(1);
    expect(repo.queue).toHaveLength(1);
  });

  it('crop → referencia la imagen y la anotación existentes, sin crear otra imagen ni tocar su estado', async () => {
    await portalImageWithBox();
    await postCrop(7);
    const res = await enqueue(1);
    expect(res.status).toBe(201);
    const item = annotationQueueItemSchema.parse(await res.json());
    expect(item).toMatchObject({ image_id: 42, annotation_id: 7, status: 'pending' });
    expect(images.createdImages).toEqual([]);
    expect(images.images.get(42)?.status).toBe('completed');
  });

  it('crop cuya imagen ya no existe → 409 y sin elemento', async () => {
    await portalImageWithBox();
    await postCrop(7);
    images.images.delete(42);
    const res = await enqueue(1);
    expect(res.status).toBe(409);
    expect(repo.queue).toHaveLength(0);
  });

  it('inferencia inexistente → 404; id inválido → 400', async () => {
    expect((await enqueue(5)).status).toBe(404);
    expect((await fetch(`${base}/inference/0/annotation-queue`, { method: 'POST' })).status).toBe(
      400,
    );
  });

  it('error de persistencia → 503 sin imagen huérfana; el reintento después crea uno solo', async () => {
    await postFile(await png(30, 20), 'image/png');
    repo.failQueueInsert = true;
    const failed = await enqueue(1);
    expect(failed.status).toBe(503);
    const { error } = await failed.json();
    expect(error).toMatch(/no se pudo guardar el elemento de la cola/i);
    expect(error).not.toMatch(/ECONNREFUSED|mariadb:3306/);
    expect(images.createdImages).toEqual([]); // la transacción no confirmó la imagen
    expect(repo.queue).toHaveLength(0);
    repo.failQueueInsert = false;
    expect((await enqueue(1)).status).toBe(201);
    expect(repo.queue).toHaveLength(1);
  });

  it('MariaDB cae entre crear la imagen y el elemento: no queda imagen pending ni elemento', async () => {
    await postFile(await png(30, 20), 'image/png');
    const before = images.images.size;
    repo.failQueueInsert = true;
    images.failDelete = true; // tampoco funcionaría una limpieza compensatoria
    const failed = await enqueue(1);
    expect(failed.status).toBe(503);
    expect((await failed.json()).error).not.toMatch(/ECONNREFUSED|Failed query/);
    expect(images.images.size).toBe(before);
    expect([...images.images.values()].filter((image) => image.status === 'pending')).toEqual([]);
    expect(repo.queue).toHaveLength(0);
  });

  it('lecturas con MariaDB caída → 503 sin detalle interno, nunca 500', async () => {
    await postFile(await png(30, 20), 'image/png');
    repo.failReads = true;
    for (const [method, path] of [
      ['GET', '/inference'],
      ['GET', '/inference/1'],
      ['GET', '/inference/annotation-queue'],
      ['POST', '/inference/1/annotation-queue'],
    ] as const) {
      const res = await fetch(`${base}${path}`, { method });
      expect(res.status, `${method} ${path}`).toBe(503);
      expect((await res.json()).error).not.toMatch(/Failed query|params|select/);
    }
    expect(repo.queue).toHaveLength(0);
  });

  it('crop con la base del portal caída → 503 sin detalle; no se guarda inferencia', async () => {
    await portalImageWithBox();
    images.failReads = true;
    const res = await postCrop(7);
    expect(res.status).toBe(503);
    expect((await res.json()).error).not.toMatch(/Failed query|select/);
    expect(repo.inferences).toHaveLength(0);
    expect(engine.calls).toHaveLength(0);
  });

  it('GET /inference/annotation-queue lista los elementos persistidos', async () => {
    await postFile(await png(30, 20), 'image/png');
    await enqueue(1);
    const body = annotationQueueResponseSchema.parse(
      await (await fetch(`${base}/inference/annotation-queue`)).json(),
    );
    expect(body.items.map((item) => item.inference_id)).toEqual([1]);
  });
});

describe('D06-06 (preparación): identidad del modelo official recargado de AWS', () => {
  it('GET /inference/engine devuelve run, checkpoint, semver y el objeto S3 (bucket/key/VersionId/SHA)', async () => {
    engine.model = OFFICIAL;
    const body = inferenceEngineSchema.parse(
      await (await fetch(`${base}/inference/engine`)).json(),
    );
    expect(body.model).toEqual(OFFICIAL);
  });

  it('archivo y crop guardan la identidad completa y se releen igual (GET tras recargar)', async () => {
    engine.model = OFFICIAL;
    await portalImageWithBox();
    const upload = inferenceResultSchema.parse(
      await (await postFile(await png(32, 32), 'image/png')).json(),
    );
    const crop = inferenceResultSchema.parse(await (await postCrop(7)).json());
    for (const created of [upload, crop]) {
      expect(created.model).toEqual(OFFICIAL);
      const stored = repo.inferences.find((row) => row.id === created.inference_id);
      expect(stored?.model).toEqual(OFFICIAL);
      const again = inferenceResultSchema.parse(
        await (await fetch(`${base}/inference/${created.inference_id}`)).json(),
      );
      expect(again.model).toEqual(OFFICIAL);
    }
    const list = inferenceListSchema.parse(await (await fetch(`${base}/inference`)).json());
    expect(list.inferences.map((item) => item.model.s3_object)).toEqual([
      OFFICIAL.s3_object,
      OFFICIAL.s3_object,
    ]);
  });

  it('el elemento de la cola conserva la misma identidad AWS y sigue siendo sugerencia, no etiqueta', async () => {
    engine.model = OFFICIAL;
    await postFile(await png(30, 20), 'image/png');
    const item = annotationQueueItemSchema.parse(await (await enqueue(1)).json());
    expect(item.model).toEqual(OFFICIAL);
    expect(item.human_label).toBeNull();
    expect(item.suggestion.source).toBe('model');
    const queue = annotationQueueResponseSchema.parse(
      await (await fetch(`${base}/inference/annotation-queue`)).json(),
    );
    expect(queue.items[0]?.model.s3_object).toEqual(OFFICIAL.s3_object);
  });

  it.each([
    ['sin objeto S3', { ...OFFICIAL, s3_object: null }],
    ['sin VersionId', { ...OFFICIAL, s3_object: { ...OFFICIAL.s3_object, version_id: '' } }],
    [
      'key de otro semver',
      {
        ...OFFICIAL,
        s3_object: { ...OFFICIAL.s3_object, s3_key: 'models/p3-cnn-classifier/0.9.0/model.pt' },
      },
    ],
    ['smoke con objeto S3', { ...SMOKE, s3_object: OFFICIAL.s3_object }],
  ])(
    'motor con identidad inválida (%s) → 503, sin guardar nada ni caer a otro modelo',
    async (_n, model) => {
      engine.model = model as EnginePrediction['model'];
      expect((await fetch(`${base}/inference/engine`)).status).toBe(503);
      const res = await postFile(await png(16, 16), 'image/png');
      expect(res.status).toBe(503);
      expect((await res.json()).error).toMatch(/no cumple el contrato/);
      expect(repo.inferences).toHaveLength(0);
      expect(images.objects.size).toBe(0);
    },
  );
});
