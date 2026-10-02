import { relations } from 'drizzle-orm';
import {
  bigint,
  boolean,
  char,
  double,
  index,
  int,
  json,
  mysqlEnum,
  mysqlTable,
  primaryKey,
  text,
  timestamp,
  uniqueIndex,
  varchar,
} from 'drizzle-orm/mysql-core';

/**
 * Guarda los metadatos de las imágenes.
 * El archivo real se almacena en MinIO mediante storageKey.
 */

export const images = mysqlTable(
  'images',
  {
    // ID único de la imagen.
    id: bigint('id', {
      mode: 'number',
      unsigned: true,
    })
      .autoincrement()
      .primaryKey(),

    // Nombre original del archivo.
    filename: varchar('filename', {
      length: 255,
    }).notNull(),

    // Ruta o key del archivo dentro de MinIO.
    storageKey: varchar('storage_key', {
      length: 512,
    }).notNull(),

    // Tipo de archivo, por ejemplo image/jpeg.
    mimeType: varchar('mime_type', {
      length: 100,
    }).notNull(),

    // Dimensiones de la imagen en píxeles.
    width: int('width', {
      unsigned: true,
    }).notNull(),

    height: int('height', {
      unsigned: true,
    }).notNull(),

    // Tamaño del archivo en bytes.
    sizeBytes: bigint('size_bytes', {
      mode: 'number',
      unsigned: true,
    }).notNull(),

    // Estado actual del proceso de anotación.
    status: mysqlEnum('status', ['pending', 'in_progress', 'completed'])
      .notNull()
      .default('pending'),

    // Fecha de creación del registro.
    createdAt: timestamp('created_at').notNull().defaultNow(),

    // Fecha de la última actualización.
    updatedAt: timestamp('updated_at').notNull().defaultNow().onUpdateNow(),
  },

  // Índices para mejorar búsquedas y filtros.
  (table) => [
    uniqueIndex('images_storage_key_unique').on(table.storageKey),
    index('images_status_idx').on(table.status),
    index('images_created_at_idx').on(table.createdAt),
    index('images_status_created_at_idx').on(table.status, table.createdAt),
  ],
);
/**
 * categories
 *
 * Categorías/clases usadas para anotar imágenes (equivalente a
 * `categories` en el formato COCO).
 */

export const categories = mysqlTable(
  'categories',
  {
    // ID único de la categoría.
    id: bigint('id', {
      mode: 'number',
      unsigned: true,
    })
      .autoincrement()
      .primaryKey(),

    // Nombre de la categoría, por ejemplo person o car.
    name: varchar('name', {
      length: 150,
    }).notNull(),

    // Color usado para mostrar la bounding box en la interfaz (hexadecimal).
    color: varchar('color', {
      length: 7,
    }).notNull(),

    // Fecha de creación de la categoría.
    createdAt: timestamp('created_at').notNull().defaultNow(),
  },

  // Evita que existan dos categorías con el mismo nombre.
  (table) => [uniqueIndex('categories_name_unique').on(table.name)],
);

/**
 * annotations
 *
 * Bounding box asociado a una imagen y una categoría. Los campos de bbox
 * siguen la convención de COCO (x, y = esquina superior izquierda;
 * width/height = dimensiones de la caja) para que la futura fase de
 * exportación a COCO JSON no requiera cambios de schema.
 */
export const annotations = mysqlTable(
  'annotations',
  {
    // ID único de la anotación.
    id: bigint('id', {
      mode: 'number',
      unsigned: true,
    })
      .autoincrement()
      .primaryKey(),

    // Imagen a la que pertenece la bounding box.
    imageId: bigint('image_id', {
      mode: 'number',
      unsigned: true,
    })
      .notNull()
      .references(() => images.id, {
        onDelete: 'cascade',
      }),

    // Categoría asignada a la bounding box.
    categoryId: bigint('category_id', {
      mode: 'number',
      unsigned: true,
    })
      .notNull()
      .references(() => categories.id, {
        onDelete: 'restrict',
      }),

    // Posición y tamaño de la caja en píxeles.
    bboxX: double('bbox_x').notNull(),
    bboxY: double('bbox_y').notNull(),
    bboxWidth: double('bbox_width').notNull(),
    bboxHeight: double('bbox_height').notNull(),

    // Área de la bounding box para la exportación COCO.
    area: double('area').notNull(),

    // Campo requerido por el formato COCO.
    isCrowd: boolean('iscrowd').notNull().default(false),

    // Fecha de creación de la anotación.
    createdAt: timestamp('created_at').notNull().defaultNow(),

    // Fecha de la última modificación.
    updatedAt: timestamp('updated_at').notNull().defaultNow().onUpdateNow(),
  },

  // Índices para buscar anotaciones por imagen y categoría.
  (table) => [
    index('annotations_image_id_idx').on(table.imageId),
    index('annotations_category_id_idx').on(table.categoryId),
    index('annotations_image_category_idx').on(table.imageId, table.categoryId),
  ],
);

/**
 * Relación: una imagen puede tener muchas anotaciones.
 */
export const imagesRelations = relations(images, ({ many }) => ({
  annotations: many(annotations),
}));

/**
 * Relación: una categoría puede pertenecer a muchas anotaciones.
 */
export const categoriesRelations = relations(categories, ({ many }) => ({
  annotations: many(annotations),
}));

/**
 * Cada anotación pertenece a una imagen y a una categoría.
 */
export const annotationsRelations = relations(annotations, ({ one }) => ({
  image: one(images, {
    fields: [annotations.imageId],
    references: [images.id],
  }),

  category: one(categories, {
    fields: [annotations.categoryId],
    references: [categories.id],
  }),
}));

/**
 * Tipos TypeScript generados automáticamente desde el esquema.
 */
export type Image = typeof images.$inferSelect;
export type NewImage = typeof images.$inferInsert;

export type Category = typeof categories.$inferSelect;
export type NewCategory = typeof categories.$inferInsert;

export type Annotation = typeof annotations.$inferSelect;
export type NewAnnotation = typeof annotations.$inferInsert;

/**
 * D02-05 — Jobs de entrenamiento P3. La API los crea en `queued`; `trainer-worker`
 * los toma por polling (sin otra tecnología de colas) y persiste progreso, logs,
 * error y run de MLflow. El worker usa `worker_id`/`heartbeat_at` para no duplicar
 * un job interrumpido al reiniciarse.
 */
export const trainingJobs = mysqlTable(
  'training_jobs',
  {
    id: bigint('id', { mode: 'number', unsigned: true }).autoincrement().primaryKey(),
    task: mysqlEnum('task', ['controlled', 'training']).notNull(),
    status: mysqlEnum('status', ['queued', 'running', 'succeeded', 'failed', 'cancelled'])
      .notNull()
      .default('queued'),
    datasetVersion: varchar('dataset_version', { length: 32 }).notNull(),
    manifestHash: char('manifest_hash', { length: 64 }).notNull(),
    // TrainingConfig efectivo, ya validado por el contrato (contracts/p3).
    config: json('config').notNull(),
    // Solo tarea controlada: época en la que falla a propósito (prueba de fallos).
    controlledFailAtEpoch: int('controlled_fail_at_epoch', { unsigned: true }),
    progressEpoch: int('progress_epoch', { unsigned: true }),
    totalEpochs: int('total_epochs', { unsigned: true }),
    mlflowRunId: char('mlflow_run_id', { length: 32 }),
    // Estado que el run de MLflow aún debe recibir (FINISHED/FAILED/KILLED). Se escribe
    // en el mismo UPDATE que cierra el job y el worker lo reintenta hasta que MLflow
    // responde; así un run nunca se queda RUNNING si MLflow estaba caído (#53, B2).
    mlflowCloseStatus: mysqlEnum('mlflow_close_status', ['FINISHED', 'FAILED', 'KILLED']),
    error: text('error'),
    cancelRequested: boolean('cancel_requested').notNull().default(false),
    workerId: varchar('worker_id', { length: 128 }),
    heartbeatAt: timestamp('heartbeat_at'),
    createdAt: timestamp('created_at').notNull().defaultNow(),
    startedAt: timestamp('started_at'),
    finishedAt: timestamp('finished_at'),
    updatedAt: timestamp('updated_at').notNull().defaultNow().onUpdateNow(),
  },
  (table) => [index('training_jobs_status_created_idx').on(table.status, table.createdAt)],
);

/**
 * D03-03 — Snapshot de las fuentes oficiales de Training que publica `trainer-worker`
 * (Python: resolver de releases + manifest congelado de D03-01 verificado contra los
 * datos). El backend lo lee para `GET /api/releases`, `GET /api/manifest` y la
 * compuerta de training real; el worker vuelve a verificar contra los archivos antes
 * de cada entrenamiento, así que este snapshot solo no basta para entrenar.
 */
export const p3TrainingSources = mysqlTable('p3_training_sources', {
  name: mysqlEnum('name', ['releases', 'manifest']).primaryKey(),
  status: mysqlEnum('status', ['ok', 'unavailable']).notNull(),
  payload: text('payload'),
  detail: text('detail'),
  updatedAt: timestamp('updated_at').notNull().defaultNow().onUpdateNow(),
});

export const trainingJobLogs = mysqlTable(
  'training_job_logs',
  {
    id: bigint('id', { mode: 'number', unsigned: true }).autoincrement().primaryKey(),
    jobId: bigint('job_id', { mode: 'number', unsigned: true })
      .notNull()
      .references(() => trainingJobs.id, { onDelete: 'cascade' }),
    ts: timestamp('ts', { fsp: 3 }).notNull().defaultNow(),
    level: mysqlEnum('level', ['info', 'warning', 'error']).notNull(),
    message: text('message').notNull(),
  },
  (table) => [index('training_job_logs_job_idx').on(table.jobId, table.id)],
);

export type TrainingJobRow = typeof trainingJobs.$inferSelect;
export type TrainingJobLogRow = typeof trainingJobLogs.$inferSelect;

/**
 * D04-04 — Estado de la selección del candidato (un único registro, `id = 1`, que crea la
 * migración en `open`). `outcome` guarda el ranking SOLO de validation, los runs
 * excluidos y la referencia del manifest congelado; `outcome_hash` es lo que el cierre
 * exige que no haya cambiado. `closed` es definitivo (MODEL SELECTION CLOSED).
 */
export const p3ModelSelection = mysqlTable('p3_model_selection', {
  id: int('id').primaryKey(),
  status: mysqlEnum('status', ['open', 'candidate', 'closed']).notNull(),
  outcome: json('outcome'),
  outcomeHash: char('outcome_hash', { length: 64 }),
  proposedAt: timestamp('proposed_at', { fsp: 3 }),
  closedAt: timestamp('closed_at', { fsp: 3 }),
});

export type P3ModelSelectionRow = typeof p3ModelSelection.$inferSelect;

/**
 * D04-05 — Evaluación del frozen test y su exportación por muestra, una fila por
 * namespace. La escribe el productor Python (`app/evaluation`), nunca la API:
 * `official` una sola vez (D06-01, después de MODEL SELECTION CLOSED) y `synthetic` para
 * recorridos de prueba con predicciones conocidas, que la API no sirve como oficiales.
 * `evaluation` cumple `evaluation_response` (ready) y `predictions`, `evaluation_predictions`.
 */
export const p3Evaluation = mysqlTable('p3_evaluation', {
  namespace: mysqlEnum('namespace', ['official', 'synthetic']).primaryKey(),
  candidateRunId: char('candidate_run_id', { length: 32 }).notNull(),
  evaluatedAt: timestamp('evaluated_at', { fsp: 3 }).notNull(),
  evaluation: json('evaluation').notNull(),
  predictions: json('predictions').notNull(),
  createdAt: timestamp('created_at').notNull().defaultNow(),
});

export type P3EvaluationRow = typeof p3Evaluation.$inferSelect;

/**
 * D04-06 — Registro de versiones del modelo (contrato `model_version`) y de su objeto en
 * el bucket de modelos. `local_test` = roundtrips de prueba contra MinIO; `official` =
 * publicación real (D06-03), la única que servirá `GET /api/models`. El semver es
 * inmutable dentro de su namespace; `published` exige VersionId y objeto verificado
 * (tamaño + SHA-256) y `failed` guarda el motivo. Las transiciones son condicionales.
 */
export const p3ModelRegistry = mysqlTable(
  'p3_model_registry',
  {
    namespace: mysqlEnum('namespace', ['official', 'local_test']).notNull(),
    semver: varchar('semver', { length: 32 }).notNull(),
    mlflowRunId: char('mlflow_run_id', { length: 32 }).notNull(),
    manifestHash: char('manifest_hash', { length: 64 }).notNull(),
    dvcRelease: varchar('dvc_release', { length: 32 }).notNull(),
    dvcReleaseHash: char('dvc_release_hash', { length: 64 }).notNull(),
    s3Bucket: varchar('s3_bucket', { length: 63 }).notNull(),
    s3Key: varchar('s3_key', { length: 512 }).notNull(),
    versionId: varchar('version_id', { length: 1024 }),
    sha256: char('sha256', { length: 64 }).notNull(),
    sizeBytes: bigint('size_bytes', { mode: 'number', unsigned: true }).notNull(),
    status: mysqlEnum('status', ['draft', 'published', 'failed']).notNull(),
    failureReason: varchar('failure_reason', { length: 32 }),
    failureDetail: text('failure_detail'),
    publishedAt: timestamp('published_at', { fsp: 3 }),
    createdAt: timestamp('created_at', { fsp: 3 }).notNull().defaultNow(),
  },
  (table) => [primaryKey({ columns: [table.namespace, table.semver] })],
);

export type P3ModelRegistryRow = typeof p3ModelRegistry.$inferSelect;

/**
 * D06-03 — Tarjeta del modelo de cada versión (`models/p3-cnn-classifier/<semver>/
 * model_card.json`). Misma forma y transiciones que `p3_model_registry`: una versión
 * official solo se publica si su tarjeta quedó verificada por VersionId.
 */
export const p3ModelCard = mysqlTable(
  'p3_model_card',
  {
    namespace: mysqlEnum('namespace', ['official', 'local_test']).notNull(),
    semver: varchar('semver', { length: 32 }).notNull(),
    mlflowRunId: char('mlflow_run_id', { length: 32 }).notNull(),
    manifestHash: char('manifest_hash', { length: 64 }).notNull(),
    dvcRelease: varchar('dvc_release', { length: 32 }).notNull(),
    dvcReleaseHash: char('dvc_release_hash', { length: 64 }).notNull(),
    s3Bucket: varchar('s3_bucket', { length: 63 }).notNull(),
    s3Key: varchar('s3_key', { length: 512 }).notNull(),
    versionId: varchar('version_id', { length: 1024 }),
    sha256: char('sha256', { length: 64 }).notNull(),
    sizeBytes: bigint('size_bytes', { mode: 'number', unsigned: true }).notNull(),
    status: mysqlEnum('status', ['draft', 'published', 'failed']).notNull(),
    failureReason: varchar('failure_reason', { length: 32 }),
    failureDetail: text('failure_detail'),
    publishedAt: timestamp('published_at', { fsp: 3 }),
    createdAt: timestamp('created_at', { fsp: 3 }).notNull().defaultNow(),
  },
  (table) => [primaryKey({ columns: [table.namespace, table.semver] })],
);

export type P3ModelCardRow = typeof p3ModelCard.$inferSelect;
