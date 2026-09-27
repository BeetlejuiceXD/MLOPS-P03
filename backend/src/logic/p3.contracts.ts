/**
 * D01-05 — Contratos P3 entre portal, API y worker.
 *
 * Reglas tomadas del protocolo congelado de D01-03 (#33). Espejo en
 * `frontend/src/p3/contracts.ts`; ambos se prueban contra los mismos fixtures de
 * `contracts/p3/fixtures` (ver `contracts/p3/README.md`). Si cambias una regla
 * aquí, cámbiala también en el espejo y agrega/ajusta el fixture.
 */
import { z } from 'zod';

/** Clases declaradas antes de ver el test (#33). */
export const P3_CLASSES = ['cat', 'dog'] as const;
export const P3_EXPERIMENT = 'p3-cnn-classifier';
export const P3_MANIFEST_SEED = 42;
export const P3_SPLIT_TARGETS = { train: 0.7, val: 0.2, test: 0.1 } as const;
/** Tolerancia de la rúbrica 1.3: ±5 puntos porcentuales por partición. */
export const P3_SPLIT_TOLERANCE = 0.05;
export const PROBABILITY_SUM_TOLERANCE = 1e-3;
/** Métricas reportadas redondeadas (F1, precision, recall): tolerancia al comparar. */
const REPORTED_METRIC_TOLERANCE = 1e-4;
/** accuracy se compara sin redondear (rúbrica 4.3). */
const EXACT_TOLERANCE = 1e-9;

const classSchema = z.enum(P3_CLASSES);
const classListSchema = z
  .array(classSchema)
  .refine(
    (classes) =>
      classes.length === P3_CLASSES.length && P3_CLASSES.every((name) => classes.includes(name)),
    'Debe declarar exactamente las clases congeladas: cat, dog',
  );
const sha256Schema = z.string().regex(/^[0-9a-f]{64}$/, 'sha256 hex en minúsculas (64)');
const dvcMd5Schema = z.string().regex(/^[0-9a-f]{32}(\.dir)?$/, 'md5 de DVC');
const datasetVersionSchema = z.string().regex(/^v\d+\.\d+\.\d+$/, 'vMAJOR.MINOR.PATCH');
const modelSemverSchema = z.string().regex(/^\d+\.\d+\.\d+$/, 'MAJOR.MINOR.PATCH');
const mlflowRunIdSchema = z.string().regex(/^[0-9a-f]{32}$/, 'run_id de MLflow (32 hex)');
const gitCommitSchema = z.string().regex(/^[0-9a-f]{40}$/, 'commit de Git completo (40 hex)');
const timestampSchema = z.iso.datetime({ offset: true });
const unitInterval = z.number().finite().min(0).max(1);
const nonNegativeInt = z.number().int().nonnegative();
const positiveInt = z.number().int().positive();

const close = (a: number, b: number, tolerance: number) => Math.abs(a - b) <= tolerance;

// ---------------------------------------------------------------------------
// TrainingConfig (#33). Sin defaults en el esquema: lo validado es exactamente lo
// que se registra en MLflow. Los defaults para el formulario viven aparte.
// ---------------------------------------------------------------------------
export const trainingConfigSchema = z
  .strictObject({
    architecture: z.literal('resnet18'),
    pretrained: z.boolean(),
    trainable_layers: z.enum(['head_only', 'last_block', 'full']),
    image_size: z.number().int().min(128).max(256),
    batch_size: z.number().int().min(8).max(64),
    learning_rate: z.number().finite().gt(1e-5).lte(1e-2),
    weight_decay: z.number().finite().min(0).max(1e-2),
    optimizer: z.enum(['adam', 'sgd']),
    max_epochs: z.number().int().min(10).max(100),
    patience: z.number().int().min(3).max(15),
    augmentation: z.boolean(),
    seed: z.number().int().min(0).max(Number.MAX_SAFE_INTEGER),
    hidden_layers: z.union([z.literal(0), z.literal(1)]),
    hidden_dim: z.literal(128),
    dropout: z.number().finite().min(0).max(0.5),
  })
  .refine((config) => config.patience <= config.max_epochs, {
    message: 'patience no puede superar max_epochs',
    path: ['patience'],
  });
export type TrainingConfig = z.infer<typeof trainingConfigSchema>;

/** Defaults congelados en #33 para prellenar el formulario. `seed` no tiene default. */
export const TRAINING_CONFIG_DEFAULTS: Omit<TrainingConfig, 'seed'> = {
  architecture: 'resnet18',
  pretrained: true,
  trainable_layers: 'last_block',
  image_size: 224,
  batch_size: 16,
  learning_rate: 1e-3,
  weight_decay: 1e-4,
  optimizer: 'adam',
  max_epochs: 30,
  patience: 5,
  augmentation: true,
  hidden_layers: 0,
  hidden_dim: 128,
  dropout: 0,
};

export const createTrainingJobRequestSchema = z.strictObject({
  dataset_version: datasetVersionSchema,
  manifest_hash: sha256Schema,
  config: trainingConfigSchema,
});

// ---------------------------------------------------------------------------
// Releases: forma de `python -m presentation.release_resolver --all` (#40).
// ---------------------------------------------------------------------------
export const releaseRejectionReasons = [
  'invalid_version',
  'not_in_catalog',
  'not_allowed',
  'quality_failed',
  'quality_mismatch',
  'identity_mismatch',
  'data_missing',
  'insufficient_classes',
] as const;

export const resolvedReleaseSchema = z.strictObject({
  dataset_version: datasetVersionSchema,
  status: z.enum(['passed', 'warning']),
  dataset_dir: z.string().min(1),
  annotations_dir: z.string().min(1),
  images_dir: z.string().min(1),
  annotations_md5: dvcMd5Schema,
  images_md5: dvcMd5Schema,
  annotations_nfiles: nonNegativeInt,
  images_nfiles: nonNegativeInt,
  quality_file: z.string().min(1),
  quality_sha256: sha256Schema,
  policy_sha256: sha256Schema,
  min_images_per_class: z.number().finite().nonnegative(),
  min_classes: positiveInt,
  originals_per_class: z.record(z.string().min(1), nonNegativeInt),
});

export const releasesResponseSchema = z.strictObject({
  approved: z.array(resolvedReleaseSchema),
  rejected: z.array(
    z.strictObject({
      dataset_version: z.string().min(1),
      reason: z.enum(releaseRejectionReasons),
      detail: z.string(),
    }),
  ),
});

// ---------------------------------------------------------------------------
// Manifest P3 70/20/10 (#33: seed 42, grupos indivisibles, clases en val y test).
// ---------------------------------------------------------------------------
const splitCountsSchema = z
  .strictObject({
    crops: nonNegativeInt,
    originals: nonNegativeInt,
    crops_per_class: z.record(classSchema, nonNegativeInt),
  })
  .refine(
    (split) => Object.values(split.crops_per_class).reduce((sum, n) => sum + n, 0) === split.crops,
    'crops_per_class debe sumar crops',
  );

export const manifestSummarySchema = z
  .strictObject({
    manifest_version: z.string().min(1),
    manifest_hash: sha256Schema,
    dataset_version: datasetVersionSchema,
    dvc_release_hash: sha256Schema,
    seed: z.literal(P3_MANIFEST_SEED),
    target_ratios: z.strictObject({
      train: z.literal(P3_SPLIT_TARGETS.train),
      val: z.literal(P3_SPLIT_TARGETS.val),
      test: z.literal(P3_SPLIT_TARGETS.test),
    }),
    frozen: z.boolean(),
    classes: classListSchema,
    splits: z.strictObject({
      train: splitCountsSchema,
      val: splitCountsSchema,
      test: splitCountsSchema,
    }),
  })
  .superRefine((manifest, ctx) => {
    const { train, val, test } = manifest.splits;
    const total = train.crops + val.crops + test.crops;
    for (const [name, split] of Object.entries(manifest.splits)) {
      const target = P3_SPLIT_TARGETS[name as keyof typeof P3_SPLIT_TARGETS];
      if (total === 0 || Math.abs(split.crops / total - target) > P3_SPLIT_TOLERANCE + 1e-12) {
        ctx.addIssue({
          code: 'custom',
          path: ['splits', name, 'crops'],
          message: `${name} fuera de ±5 pp de ${target * 100}% (medido en crops)`,
        });
      }
    }
    for (const splitName of ['val', 'test'] as const) {
      for (const name of P3_CLASSES) {
        if ((manifest.splits[splitName].crops_per_class[name] ?? 0) === 0) {
          ctx.addIssue({
            code: 'custom',
            path: ['splits', splitName, 'crops_per_class', name],
            message: `La clase ${name} debe estar presente en ${splitName}`,
          });
        }
      }
    }
  });

// ---------------------------------------------------------------------------
// Jobs de entrenamiento persistentes (worker fuera del request HTTP).
// ---------------------------------------------------------------------------
export const trainingJobStatuses = [
  'queued',
  'running',
  'succeeded',
  'failed',
  'cancelled',
] as const;

export const trainingJobSchema = z
  .strictObject({
    id: positiveInt,
    status: z.enum(trainingJobStatuses),
    dataset_version: datasetVersionSchema,
    manifest_hash: sha256Schema,
    config: trainingConfigSchema,
    progress: z.strictObject({ epoch: nonNegativeInt, total_epochs: positiveInt }).nullable(),
    mlflow_run_id: mlflowRunIdSchema.nullable(),
    error: z.string().min(1).nullable(),
    created_at: timestampSchema,
    started_at: timestampSchema.nullable(),
    finished_at: timestampSchema.nullable(),
  })
  .superRefine((job, ctx) => {
    const issue = (path: string, message: string) =>
      ctx.addIssue({ code: 'custom', path: [path], message });
    const finished =
      job.status === 'succeeded' || job.status === 'failed' || job.status === 'cancelled';

    if (job.status === 'queued') {
      if (job.mlflow_run_id !== null)
        issue('mlflow_run_id', 'Un job en cola no tiene run de MLflow');
      if (job.started_at !== null) issue('started_at', 'Un job en cola no ha iniciado');
    }
    if (job.status === 'running' && job.started_at === null)
      issue('started_at', 'Falta started_at');
    if (job.status === 'succeeded' && job.mlflow_run_id === null) {
      issue('mlflow_run_id', 'succeeded exige mlflow_run_id');
    }
    if (job.status === 'failed' && job.error === null)
      issue('error', 'failed exige mensaje de error');
    if (job.status !== 'failed' && job.error !== null) issue('error', 'Solo failed lleva error');
    if (finished && job.finished_at === null) issue('finished_at', 'Falta finished_at');
    if (!finished && job.finished_at !== null)
      issue('finished_at', 'Un job activo no tiene finished_at');
    if (job.progress) {
      if (job.progress.total_epochs !== job.config.max_epochs) {
        issue('progress', 'total_epochs debe ser config.max_epochs');
      }
      if (job.progress.epoch > job.progress.total_epochs) {
        issue('progress', 'epoch no puede superar total_epochs');
      }
    }
  });
export type TrainingJob = z.infer<typeof trainingJobSchema>;

export const trainingJobListSchema = z.strictObject({ jobs: z.array(trainingJobSchema) });

export const jobLogsSchema = z.strictObject({
  job_id: positiveInt,
  lines: z.array(
    z.strictObject({
      ts: timestampSchema,
      level: z.enum(['info', 'warning', 'error']),
      message: z.string(),
    }),
  ),
});

// ---------------------------------------------------------------------------
// Runs de MLflow (#33: params, métricas por época, resumen y tags de trazabilidad).
// ---------------------------------------------------------------------------
const epochMetricsSchema = z.strictObject({
  epoch: positiveInt,
  train_loss: z.number().finite().nonnegative(),
  train_accuracy: unitInterval,
  val_loss: z.number().finite().nonnegative(),
  val_accuracy: unitInterval,
  val_macro_f1: unitInterval,
  learning_rate: z.number().finite().positive(),
});

export const experimentRunSchema = z
  .strictObject({
    run_id: mlflowRunIdSchema,
    experiment_name: z.literal(P3_EXPERIMENT),
    status: z.enum(['RUNNING', 'FINISHED', 'FAILED', 'KILLED']),
    start_time: timestampSchema,
    end_time: timestampSchema.nullable(),
    params: trainingConfigSchema,
    tags: z.strictObject({
      git_commit: gitCommitSchema,
      dvc_release: datasetVersionSchema,
      dvc_images_md5: dvcMd5Schema,
      dvc_annotations_md5: dvcMd5Schema,
      dvc_release_hash: sha256Schema,
      manifest_version: z.string().min(1),
      manifest_hash: sha256Schema,
      classes: classListSchema,
      seed: z.number().int().nonnegative(),
      job_id: positiveInt,
    }),
    // Solo validation: un run de la campaña nunca trae métricas de test.
    summary: z
      .strictObject({
        best_epoch: positiveInt,
        best_val_accuracy: unitInterval,
        best_val_macro_f1: unitInterval,
        best_val_loss: z.number().finite().nonnegative(),
      })
      .nullable(),
    history: z.array(epochMetricsSchema),
  })
  .superRefine((run, ctx) => {
    const issue = (path: string, message: string) =>
      ctx.addIssue({ code: 'custom', path: [path], message });
    if (run.tags.seed !== run.params.seed)
      issue('tags', 'tags.seed debe coincidir con params.seed');
    run.history.forEach((row, index) => {
      if (row.epoch !== index + 1) issue('history', 'Las épocas deben ir 1, 2, 3… sin huecos');
    });
    if (run.status === 'FINISHED' && run.summary === null) {
      issue('summary', 'FINISHED exige el resumen de mejores métricas');
    }
    if (run.status === 'RUNNING' && run.end_time !== null)
      issue('end_time', 'RUNNING sin end_time');
    if (run.summary) {
      const best = run.history.find((row) => row.epoch === run.summary?.best_epoch);
      const maxAccuracy = Math.max(...run.history.map((row) => row.val_accuracy));
      if (!best) {
        issue('summary', 'best_epoch no existe en history');
      } else if (
        best.val_accuracy !== maxAccuracy ||
        best.val_accuracy !== run.summary.best_val_accuracy
      ) {
        issue(
          'summary',
          'best_epoch debe ser la de mayor val_accuracy (mejor checkpoint, no el último)',
        );
      }
    }
  });
export type ExperimentRun = z.infer<typeof experimentRunSchema>;

export const experimentRunsResponseSchema = z.strictObject({
  experiment_name: z.literal(P3_EXPERIMENT),
  runs: z.array(experimentRunSchema),
});

// ---------------------------------------------------------------------------
// Evaluation: bloqueada hasta MODEL SELECTION CLOSED (#33, custodia: Ale).
// ---------------------------------------------------------------------------
const evaluationBlockedSchema = z.strictObject({
  state: z.literal('blocked'),
  reason: z.literal('model_selection_open'),
  detail: z.string(),
});

const evaluationReadySchema = z
  .strictObject({
    state: z.literal('ready'),
    selection: z.strictObject({
      candidate_run_id: mlflowRunIdSchema,
      metric: z.literal('val_accuracy'),
      closed_at: timestampSchema,
    }),
    manifest_hash: sha256Schema,
    evaluated_at: timestampSchema,
    n_test: positiveInt,
    classes: classListSchema,
    // Filas = clase real, columnas = clase predicha (rúbrica 4.2).
    confusion_matrix: z.strictObject({
      labels: classListSchema,
      rows: z.array(z.array(nonNegativeInt)),
    }),
    metrics: z.strictObject({
      accuracy: unitInterval,
      macro_f1: unitInterval,
      per_class: z.array(
        z.strictObject({
          class_name: classSchema,
          precision: unitInterval,
          recall: unitInterval,
          f1: unitInterval,
          support: nonNegativeInt,
        }),
      ),
    }),
    majority_baseline_accuracy: unitInterval,
  })
  .superRefine((evaluation, ctx) => {
    const issue = (path: string, message: string) =>
      ctx.addIssue({ code: 'custom', path: [path], message });
    if (Date.parse(evaluation.evaluated_at) <= Date.parse(evaluation.selection.closed_at)) {
      issue('evaluated_at', 'El test se evalúa solo después de MODEL SELECTION CLOSED');
    }
    const { labels, rows } = evaluation.confusion_matrix;
    if (rows.length !== labels.length || rows.some((row) => row.length !== labels.length)) {
      issue('confusion_matrix', 'La matriz debe ser cuadrada con una fila/columna por clase');
      return;
    }
    const total = rows.flat().reduce((sum, n) => sum + n, 0);
    if (total !== evaluation.n_test) issue('confusion_matrix', 'La matriz debe sumar n_test');
    const trace = labels.reduce((sum, _label, i) => sum + (rows[i]?.[i] ?? 0), 0);
    if (!close(evaluation.metrics.accuracy, trace / evaluation.n_test, EXACT_TOLERANCE)) {
      issue('metrics', 'accuracy debe ser traza / n_test, sin redondear');
    }
    const supports = rows.map((row) => row.reduce((sum, n) => sum + n, 0));
    const f1s: number[] = [];
    labels.forEach((label, i) => {
      const stats = evaluation.metrics.per_class.find((entry) => entry.class_name === label);
      if (!stats) {
        issue('metrics', `Faltan métricas de la clase ${label}`);
        return;
      }
      const tp = rows[i]?.[i] ?? 0;
      const predicted = rows.reduce((sum, row) => sum + (row[i] ?? 0), 0);
      const support = supports[i] ?? 0;
      const precision = predicted === 0 ? 0 : tp / predicted;
      const recall = support === 0 ? 0 : tp / support;
      const f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
      f1s.push(stats.f1);
      if (stats.support !== support) issue('metrics', `support de ${label} ≠ suma de su fila`);
      if (
        !close(stats.precision, precision, REPORTED_METRIC_TOLERANCE) ||
        !close(stats.recall, recall, REPORTED_METRIC_TOLERANCE) ||
        !close(stats.f1, f1, REPORTED_METRIC_TOLERANCE)
      ) {
        issue('metrics', `precision/recall/F1 de ${label} no coinciden con la matriz`);
      }
    });
    const macro = f1s.reduce((sum, n) => sum + n, 0) / Math.max(f1s.length, 1);
    if (!close(evaluation.metrics.macro_f1, macro, REPORTED_METRIC_TOLERANCE)) {
      issue('metrics', 'macro_f1 debe ser el promedio de los F1 por clase');
    }
    const majority = Math.max(...supports) / evaluation.n_test;
    if (!close(evaluation.majority_baseline_accuracy, majority, EXACT_TOLERANCE)) {
      issue('majority_baseline_accuracy', 'Baseline = soporte de la clase mayoritaria / n_test');
    }
  });

export const evaluationResponseSchema = z.discriminatedUnion('state', [
  evaluationBlockedSchema,
  evaluationReadySchema,
]);
export type EvaluationResponse = z.infer<typeof evaluationResponseSchema>;

// ---------------------------------------------------------------------------
// Versiones del modelo publicadas en AWS S3 (#33: bucket propio de modelos).
// ---------------------------------------------------------------------------
export const modelVersionSchema = z
  .strictObject({
    semver: modelSemverSchema,
    mlflow_run_id: mlflowRunIdSchema,
    manifest_hash: sha256Schema,
    dvc_release: datasetVersionSchema,
    dvc_release_hash: sha256Schema,
    s3_bucket: z.string().min(3),
    s3_key: z.string().min(1),
    version_id: z.string().min(1).nullable(),
    sha256: sha256Schema,
    status: z.enum(['draft', 'published', 'failed']),
    published_at: timestampSchema.nullable(),
  })
  .superRefine((model, ctx) => {
    const issue = (path: string, message: string) =>
      ctx.addIssue({ code: 'custom', path: [path], message });
    if (!model.s3_key.startsWith(`models/${P3_EXPERIMENT}/${model.semver}/`)) {
      issue('s3_key', `s3_key debe estar bajo models/${P3_EXPERIMENT}/<semver>/`);
    }
    if (
      model.status === 'published' &&
      (model.version_id === null || model.published_at === null)
    ) {
      issue('status', 'published exige VersionId y published_at (objeto verificado en S3)');
    }
    if (model.status !== 'published' && model.published_at !== null) {
      issue('published_at', 'Solo una versión publicada tiene published_at');
    }
  });
export type ModelVersion = z.infer<typeof modelVersionSchema>;

export const modelsResponseSchema = z.strictObject({ models: z.array(modelVersionSchema) });

// ---------------------------------------------------------------------------
// Inferencia y cola de anotación.
// ---------------------------------------------------------------------------
export const inferenceResultSchema = z
  .strictObject({
    inference_id: positiveInt,
    model_version: modelSemverSchema,
    predicted_class: classSchema,
    probabilities: z.record(classSchema, unitInterval),
    created_at: timestampSchema,
  })
  .superRefine((result, ctx) => {
    const values = Object.values(result.probabilities);
    const sum = values.reduce((total, p) => total + p, 0);
    if (!close(sum, 1, PROBABILITY_SUM_TOLERANCE)) {
      ctx.addIssue({ code: 'custom', path: ['probabilities'], message: 'Deben sumar ~1' });
    }
    const top = Math.max(...values);
    if (result.probabilities[result.predicted_class] !== top) {
      ctx.addIssue({
        code: 'custom',
        path: ['predicted_class'],
        message: 'predicted_class debe ser el argmax de probabilities',
      });
    }
  });

export const annotationQueueItemSchema = z.strictObject({
  inference_id: positiveInt,
  image_id: positiveInt,
  status: z.literal('pending'),
});

export const apiErrorSchema = z.strictObject({ error: z.string().min(1) });

/** Nombre del contrato (carpeta en `contracts/p3/fixtures`) → esquema. */
export const P3_CONTRACTS = {
  training_config: trainingConfigSchema,
  create_training_job_request: createTrainingJobRequestSchema,
  releases_response: releasesResponseSchema,
  manifest_summary: manifestSummarySchema,
  training_job: trainingJobSchema,
  job_logs: jobLogsSchema,
  experiment_runs_response: experimentRunsResponseSchema,
  evaluation_response: evaluationResponseSchema,
  models_response: modelsResponseSchema,
  inference_result: inferenceResultSchema,
  annotation_queue_item: annotationQueueItemSchema,
  api_error: apiErrorSchema,
} as const;

/** Catálogo de endpoints P3 (detalle y dueños en `contracts/p3/README.md`). */
export const P3_ENDPOINTS = {
  listReleases: 'GET /api/releases',
  getManifest: 'GET /api/manifest',
  createTrainingJob: 'POST /api/training/jobs',
  listTrainingJobs: 'GET /api/training/jobs',
  getTrainingJob: 'GET /api/training/jobs/:id',
  getTrainingJobLogs: 'GET /api/training/jobs/:id/logs',
  listRuns: 'GET /api/experiments/runs',
  getEvaluation: 'GET /api/evaluation',
  listModels: 'GET /api/models',
  publishModel: 'POST /api/models/:semver/publish',
  runInference: 'POST /api/inference',
  sendToAnnotationQueue: 'POST /api/inference/:id/annotation-queue',
} as const;
