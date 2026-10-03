/**
 * D05-07 — Inference: entrada (archivo nuevo o crop de una anotación del portal) → motor de
 * D05-04 → resultado persistido con la identidad del modelo → cola de anotación.
 *
 * Reglas:
 *  - La clase la calcula el motor; la respuesta se valida con su contrato antes de guardarla.
 *  - Entrada inválida, motor ausente o respuesta inválida no guardan nada.
 *  - Enviar a la cola NO crea anotaciones ni etiqueta humana: deja la imagen pendiente de
 *    revisión (archivo nuevo → imagen `pending` en el portal; crop → la imagen y la
 *    anotación existentes, sin tocar su estado). Un reintento devuelve el mismo elemento.
 */
import { createHash } from 'node:crypto';
import sharp from 'sharp';
import {
  ConflictError,
  NotFoundError,
  ServiceUnavailableError,
  ValidationError,
} from './errors.js';
import { validateImageUpload } from './image-upload.validation.js';
import {
  EngineRejectedInputError,
  EngineUnavailableError,
  type InferenceEngine,
} from './inference-engine.js';
import {
  type AnnotationQueueItem,
  type InferenceInput,
  type InferenceModelIdentity,
  type InferenceResult,
  inferenceEnginePredictionSchema,
  inferenceEngineSchema,
} from './p3.contracts.js';

export interface NewInference {
  created_at: Date;
  input: InferenceInput;
  /** Clave en MinIO de la imagen subida (solo archivos nuevos; un crop referencia el portal). */
  storage_key: string | null;
  predicted_class: string;
  probabilities: Record<string, number>;
  model: InferenceModelIdentity;
}
export type StoredInference = NewInference & { id: number };

export interface QueueRecord {
  id: number;
  inference_id: number;
  image_id: number;
  annotation_id: number | null;
  created_at: Date;
}

/** El mismo `inference_id` ya tiene elemento en la cola (índice único). */
export class DuplicateQueueItemError extends Error {
  constructor(inferenceId: number) {
    super(`La inferencia ${inferenceId} ya está en la cola de anotación`);
    this.name = 'DuplicateQueueItemError';
  }
}

export interface InferenceRepository {
  insert(record: NewInference): Promise<number>;
  find(id: number): Promise<StoredInference | null>;
  list(limit: number): Promise<StoredInference[]>;
  findQueueItem(inferenceId: number): Promise<QueueRecord | null>;
  /** Lanza `DuplicateQueueItemError` si la inferencia ya tiene elemento. */
  insertQueueItem(item: Omit<QueueRecord, 'id'>): Promise<QueueRecord>;
  /**
   * Archivo nuevo: crea la imagen `pending` del portal y el elemento de la cola en UNA
   * transacción (o ninguno de los dos). Lanza `DuplicateQueueItemError` igual que arriba.
   */
  insertQueueItemWithNewImage(
    image: NewPortalImage,
    item: Omit<QueueRecord, 'id' | 'image_id'>,
  ): Promise<QueueRecord>;
  listQueue(): Promise<QueueRecord[]>;
}

/** Lo que Inference usa del portal de anotación (imágenes en MinIO + MariaDB). */
export interface PortalImages {
  storeInput(bytes: Buffer, mimeType: string): Promise<string>;
  deleteInput(storageKey: string): Promise<void>;
  findAnnotation(id: number): Promise<{
    annotation_id: number;
    image_id: number;
    bbox: [number, number, number, number];
  } | null>;
  readImage(imageId: number): Promise<{ bytes: Buffer; mimeType: string } | null>;
  imageExists(imageId: number): Promise<boolean>;
}

/** Fila nueva de `images` (portal) para un archivo enviado a la cola. */
export interface NewPortalImage {
  filename: string;
  storageKey: string;
  mimeType: string;
  width: number;
  height: number;
  sizeBytes: number;
}

export interface UploadedFile {
  filename: string;
  mimeType: string;
  sizeBytes: number;
  buffer: Buffer;
}

const HISTORY_LIMIT = 50;
const sha256 = (bytes: Buffer) => createHash('sha256').update(bytes).digest('hex');

/**
 * MariaDB o MinIO caídos: 503 con un mensaje fijo. La causa (consulta, parámetros, host)
 * va solo al log del servidor. Los errores de dominio (404, 400, 409, duplicado) pasan.
 */
async function infra<T>(what: string, operation: () => Promise<T>): Promise<T> {
  try {
    return await operation();
  } catch (error) {
    if (
      error instanceof NotFoundError ||
      error instanceof ValidationError ||
      error instanceof ConflictError ||
      error instanceof ServiceUnavailableError ||
      error instanceof DuplicateQueueItemError
    ) {
      throw error;
    }
    console.error(`[inference] ${what} falló:`, error);
    throw new ServiceUnavailableError(
      `No se pudo ${what}: la base de datos o el almacenamiento no están disponibles.`,
    );
  }
}

function engineError(error: unknown): Error {
  if (error instanceof EngineRejectedInputError) {
    return new ValidationError(`El motor de inferencia rechazó la imagen: ${error.message}`);
  }
  if (error instanceof EngineUnavailableError) {
    return new ServiceUnavailableError(
      `motor de inferencia (D05-04) no disponible: ${error.message}`,
    );
  }
  return error instanceof Error ? error : new Error(String(error));
}

export function createInferenceService(deps: {
  engine: InferenceEngine;
  repo: InferenceRepository;
  images: PortalImages;
  maxUploadBytes: number;
  now?: () => Date;
}) {
  const { engine, repo, images } = deps;
  const now = deps.now ?? (() => new Date());

  async function predict(bytes: Buffer, mimeType: string) {
    let raw: unknown;
    try {
      raw = await engine.predict(bytes, mimeType);
    } catch (error) {
      throw engineError(error);
    }
    const parsed = inferenceEnginePredictionSchema.safeParse(raw);
    if (!parsed.success) {
      throw new ServiceUnavailableError(
        `La respuesta del motor de inferencia no cumple el contrato: ${parsed.error.issues
          .map((issue) => issue.message)
          .join('; ')}`,
      );
    }
    return parsed.data;
  }

  async function save(record: NewInference): Promise<InferenceResult> {
    let id: number;
    try {
      id = await repo.insert(record);
    } catch (error) {
      if (record.storage_key) await images.deleteInput(record.storage_key).catch(() => undefined);
      // El detalle (consulta y parámetros) va al log del servidor, no a la respuesta.
      console.error('[inference] insert de p3_inference falló:', error);
      throw new ServiceUnavailableError(
        'No se pudo guardar la inferencia: la base de datos no está disponible. No se guardó nada.',
      );
    }
    return toResult({ ...record, id }, null);
  }

  function toResult(stored: StoredInference, queueItemId: number | null): InferenceResult {
    return {
      inference_id: stored.id,
      created_at: stored.created_at.toISOString(),
      input: stored.input,
      predicted_class: stored.predicted_class as InferenceResult['predicted_class'],
      probabilities: stored.probabilities as InferenceResult['probabilities'],
      model: stored.model,
      annotation_queue_item_id: queueItemId,
    };
  }

  function toQueueItem(record: QueueRecord, stored: StoredInference): AnnotationQueueItem {
    return {
      queue_item_id: record.id,
      inference_id: record.inference_id,
      image_id: record.image_id,
      annotation_id: record.annotation_id,
      status: 'pending',
      human_label: null,
      suggestion: {
        source: 'model',
        predicted_class:
          stored.predicted_class as AnnotationQueueItem['suggestion']['predicted_class'],
        probabilities: stored.probabilities as AnnotationQueueItem['suggestion']['probabilities'],
      },
      model: stored.model,
      created_at: record.created_at.toISOString(),
    };
  }

  async function requireInference(id: number): Promise<StoredInference> {
    const stored = await infra('leer la inferencia', () => repo.find(id));
    if (!stored) throw new NotFoundError(`La inferencia ${id} no existe`);
    return stored;
  }

  return {
    async engine() {
      let raw: unknown;
      try {
        raw = await engine.identity();
      } catch (error) {
        throw engineError(error);
      }
      const parsed = inferenceEngineSchema.safeParse(raw);
      if (!parsed.success) {
        throw new ServiceUnavailableError(
          'La identidad del motor de inferencia no cumple el contrato inference_engine',
        );
      }
      return parsed.data;
    },

    /** Archivo nuevo: misma validación que el upload del portal (tipo, tamaño, contenido). */
    async predictUpload(file: UploadedFile): Promise<InferenceResult> {
      const check = validateImageUpload(
        { mimeType: file.mimeType, sizeBytes: file.sizeBytes },
        deps.maxUploadBytes,
      );
      if (!check.success) {
        throw new ValidationError(
          'La imagen no cumple con los requisitos de carga: JPEG, PNG o WebP de hasta ' +
            `${Math.floor(deps.maxUploadBytes / 1024)} KiB.`,
        );
      }
      let width: number | undefined;
      let height: number | undefined;
      try {
        ({ width, height } = await sharp(file.buffer).metadata());
      } catch {
        throw new ValidationError('El archivo no es una imagen válida.');
      }
      if (!width || !height) {
        throw new ValidationError('No se pudieron obtener las dimensiones de la imagen.');
      }
      const prediction = await predict(file.buffer, file.mimeType);
      const storageKey = await infra('guardar la imagen de entrada', () =>
        images.storeInput(file.buffer, file.mimeType),
      );
      return save({
        created_at: now(),
        input: {
          kind: 'upload',
          filename: file.filename,
          mime_type: file.mimeType as 'image/jpeg' | 'image/png' | 'image/webp',
          size_bytes: file.sizeBytes,
          width,
          height,
          sha256: sha256(file.buffer),
        },
        storage_key: storageKey,
        predicted_class: prediction.predicted_class,
        probabilities: prediction.probabilities,
        model: prediction.model,
      });
    },

    /** Crop: la caja de una anotación existente, recortada de la imagen original. */
    async predictCrop(annotationId: number): Promise<InferenceResult> {
      const annotation = await infra('leer la anotación del portal', () =>
        images.findAnnotation(annotationId),
      );
      if (!annotation) throw new NotFoundError(`La anotación ${annotationId} no existe`);
      const image = await infra('leer la imagen del portal', () =>
        images.readImage(annotation.image_id),
      );
      if (!image) throw new NotFoundError(`La imagen ${annotation.image_id} no existe`);
      const meta = await sharp(image.bytes).metadata();
      const [x, y, w, h] = annotation.bbox;
      const left = Math.round(x);
      const top = Math.round(y);
      const width = Math.round(w);
      const height = Math.round(h);
      if (
        width < 1 ||
        height < 1 ||
        left < 0 ||
        top < 0 ||
        left + width > (meta.width ?? 0) ||
        top + height > (meta.height ?? 0)
      ) {
        throw new ValidationError(
          `La caja de la anotación ${annotationId} queda fuera de la imagen ` +
            `(${meta.width}x${meta.height}) o no tiene área`,
        );
      }
      const crop = await sharp(image.bytes).extract({ left, top, width, height }).png().toBuffer();
      const prediction = await predict(crop, 'image/png');
      return save({
        created_at: now(),
        input: {
          kind: 'crop',
          image_id: annotation.image_id,
          annotation_id: annotationId,
          bbox: annotation.bbox,
          width,
          height,
          sha256: sha256(crop),
        },
        storage_key: null,
        predicted_class: prediction.predicted_class,
        probabilities: prediction.probabilities,
        model: prediction.model,
      });
    },

    async get(id: number): Promise<InferenceResult> {
      const stored = await requireInference(id);
      const item = await infra('leer la cola de anotación', () => repo.findQueueItem(id));
      return toResult(stored, item?.id ?? null);
    },

    async list() {
      const inferences = await infra('leer las inferencias', async () => {
        const rows = await repo.list(HISTORY_LIMIT);
        return Promise.all(
          rows.map(async (row) => toResult(row, (await repo.findQueueItem(row.id))?.id ?? null)),
        );
      });
      return { inferences };
    },

    /** Idempotente: el segundo envío devuelve el mismo elemento (`created: false`). */
    async sendToQueue(id: number): Promise<{ item: AnnotationQueueItem; created: boolean }> {
      const stored = await requireInference(id);
      const existing = await infra('leer la cola de anotación', () => repo.findQueueItem(id));
      if (existing) return { item: toQueueItem(existing, stored), created: false };

      try {
        let record: QueueRecord;
        if (stored.input.kind === 'crop') {
          const { image_id: imageId, annotation_id: annotationId } = stored.input;
          if (!(await infra('leer la imagen del portal', () => images.imageExists(imageId)))) {
            throw new ConflictError(`La imagen ${imageId} del crop ya no existe en el portal`);
          }
          record = await repo.insertQueueItem({
            inference_id: id,
            image_id: imageId,
            annotation_id: annotationId,
            created_at: now(),
          });
        } else {
          // Imagen pending y elemento en una sola transacción: no hay limpieza que pueda fallar.
          record = await repo.insertQueueItemWithNewImage(
            {
              filename: stored.input.filename,
              storageKey: stored.storage_key as string,
              mimeType: stored.input.mime_type,
              width: stored.input.width,
              height: stored.input.height,
              sizeBytes: stored.input.size_bytes,
            },
            { inference_id: id, annotation_id: null, created_at: now() },
          );
        }
        return { item: toQueueItem(record, stored), created: true };
      } catch (error) {
        if (error instanceof DuplicateQueueItemError) {
          // Otro envío simultáneo ganó: se devuelve el suyo, sin duplicar.
          const winner = await infra('leer la cola de anotación', () => repo.findQueueItem(id));
          if (winner) return { item: toQueueItem(winner, stored), created: false };
        }
        if (error instanceof ConflictError || error instanceof ServiceUnavailableError) throw error;
        console.error('[inference] insert de p3_annotation_queue falló:', error);
        throw new ServiceUnavailableError(
          'No se pudo guardar el elemento de la cola de anotación: la base de datos no está disponible. No se guardó nada.',
        );
      }
    },

    async queue() {
      const records = await infra('leer la cola de anotación', () => repo.listQueue());
      const items = await Promise.all(
        records.map(async (record) =>
          toQueueItem(record, await requireInference(record.inference_id)),
        ),
      );
      return { items };
    },
  };
}

export type InferenceService = ReturnType<typeof createInferenceService>;
