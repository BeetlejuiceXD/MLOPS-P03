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
export const trainingConfigSchema = z.strictObject({
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
  // #33: seed es un entero cualquiera (obligatoria, sin rango adicional).
  seed: z.number().int(),
  hidden_layers: z.union([z.literal(0), z.literal(1)]),
  hidden_dim: z.literal(128),
  dropout: z.number().finite().min(0).max(0.5),
});
// #33 fija rangos independientes para patience y max_epochs: no se cruzan aquí.
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

/**
 * D02-05: `controlled` es una tarea sintética que recorre la máquina de estados, registra
 * un run de MLflow etiquetado y no entrena; `training` (D03-03) exige además release
 * elegible y manifest oficial congelado, comprobados por la API antes de encolar.
 */
export const trainingJobTasks = ['controlled', 'training'] as const;

export const createTrainingJobRequestSchema = z
  .strictObject({
    task: z.enum(trainingJobTasks),
    dataset_version: datasetVersionSchema,
    manifest_hash: sha256Schema,
    config: trainingConfigSchema,
    controlled: z.strictObject({ fail_at_epoch: positiveInt.nullable() }).optional(),
  })
  .superRefine((request, ctx) => {
    if (request.controlled === undefined) return;
    if (request.task !== 'controlled') {
      ctx.addIssue({
        code: 'custom',
        path: ['controlled'],
        message: 'Las opciones de la tarea controlada solo aplican a task=controlled',
      });
    }
    const failAt = request.controlled.fail_at_epoch;
    if (failAt !== null && failAt > request.config.max_epochs) {
      ctx.addIssue({
        code: 'custom',
        path: ['controlled', 'fail_at_epoch'],
        message: 'fail_at_epoch no puede superar config.max_epochs',
      });
    }
  });
export type CreateTrainingJobRequest = z.infer<typeof createTrainingJobRequestSchema>;

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
    task: z.enum(trainingJobTasks),
    status: z.enum(trainingJobStatuses),
    dataset_version: datasetVersionSchema,
    manifest_hash: sha256Schema,
    config: trainingConfigSchema,
    progress: z.strictObject({ epoch: nonNegativeInt, total_epochs: positiveInt }).nullable(),
    mlflow_run_id: mlflowRunIdSchema.nullable(),
    error: z.string().min(1).nullable(),
    cancel_requested: z.boolean(),
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
    status: z.enum(['RUNNING', 'SCHEDULED', 'FINISHED', 'FAILED', 'KILLED']),
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
      seed: z.number().int(),
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
    // D04-01: sha256 del checkpoint que registró el worker (tag `checkpoint_sha256`); null si
    // el run no lo tiene. Elegible para campaña = run de training terminado, con resumen y
    // checkpoint: FINISHED solo no basta. Si no es elegible, dice por qué.
    checkpoint_sha256: sha256Schema.nullable(),
    campaign_eligible: z.boolean(),
    ineligible_reasons: z.array(z.string().min(1)),
  })
  .superRefine((run, ctx) => {
    const issue = (path: string, message: string) =>
      ctx.addIssue({ code: 'custom', path: [path], message });
    if (run.campaign_eligible !== (run.ineligible_reasons.length === 0)) {
      issue('campaign_eligible', 'Elegible ⇔ sin motivos; no elegible ⇔ al menos un motivo');
    }
    if (run.campaign_eligible && (run.status !== 'FINISHED' || run.summary === null)) {
      issue('campaign_eligible', 'Solo un run FINISHED con resumen puede ser elegible');
    }
    if (run.campaign_eligible && run.checkpoint_sha256 === null) {
      issue('campaign_eligible', 'Sin checkpoint_sha256 no es elegible');
    }
    if (run.tags.seed !== run.params.seed)
      issue('tags', 'tags.seed debe coincidir con params.seed');
    run.history.forEach((row, index) => {
      if (row.epoch !== index + 1) issue('history', 'Las épocas deben ir 1, 2, 3… sin huecos');
    });
    if (run.status === 'FINISHED' && run.summary === null) {
      issue('summary', 'FINISHED exige el resumen de mejores métricas');
    }
    if ((run.status === 'RUNNING' || run.status === 'SCHEDULED') && run.end_time !== null)
      issue('end_time', 'RUNNING sin end_time');
    if (run.summary) {
      const summary = run.summary;
      const best = run.history.find((row) => row.epoch === summary.best_epoch);
      const maxAccuracy = Math.max(...run.history.map((row) => row.val_accuracy));
      if (!best) {
        issue('summary', 'best_epoch no existe en history');
      } else {
        if (best.val_accuracy !== maxAccuracy) {
          issue(
            'summary',
            'best_epoch debe ser la de mayor val_accuracy (mejor checkpoint, no el último)',
          );
        }
        // El resumen describe UNA época: las tres métricas salen de history[best_epoch].
        const reported = [
          ['best_val_accuracy', summary.best_val_accuracy, best.val_accuracy],
          ['best_val_macro_f1', summary.best_val_macro_f1, best.val_macro_f1],
          ['best_val_loss', summary.best_val_loss, best.val_loss],
        ] as const;
        for (const [name, value, epochValue] of reported) {
          if (!close(value, epochValue, REPORTED_METRIC_TOLERANCE)) {
            ctx.addIssue({
              code: 'custom',
              path: ['summary', name],
              message: `${name} debe ser el de history[best_epoch] (tolerancia 1e-4)`,
            });
          }
        }
      }
    }
  });
export type ExperimentRun = z.infer<typeof experimentRunSchema>;

// D04-01: runs del experimento que no se pueden presentar como corridas de la campaña
// (auxiliares: controlled_task, short_run_instrumentation, persistence_check; o de training
// sin provenance, params o curvas completas). Se listan con su motivo, no se esconden.
export const excludedRunSchema = z.strictObject({
  run_id: mlflowRunIdSchema,
  run_kind: z.string().min(1).nullable(),
  status: z.enum(['RUNNING', 'SCHEDULED', 'FINISHED', 'FAILED', 'KILLED']),
  start_time: timestampSchema,
  reasons: z.array(z.string().min(1)).min(1),
});
export type ExcludedRun = z.infer<typeof excludedRunSchema>;

export const experimentRunsResponseSchema = z
  .strictObject({
    experiment_name: z.literal(P3_EXPERIMENT),
    runs: z.array(experimentRunSchema),
    excluded: z.array(excludedRunSchema),
  })
  .superRefine((response, ctx) => {
    const ids = [...response.runs, ...response.excluded].map((run) => run.run_id);
    if (new Set(ids).size !== ids.length) {
      ctx.addIssue({ code: 'custom', path: ['excluded'], message: 'run_id repetido' });
    }
  });

// D04-01: detalle de un run. Rutas de artefacto relativas a la raíz de artefactos del run.
const artifactPathSchema = z
  .string()
  .min(1)
  .refine(
    (path) =>
      !path.startsWith('/') && !path.split('/').some((part) => part === '..' || part === ''),
    'Ruta relativa al run, sin .. ni segmentos vacíos',
  );
export const artifactEntrySchema = z
  .strictObject({
    path: artifactPathSchema,
    is_dir: z.boolean(),
    size_bytes: nonNegativeInt.nullable(),
  })
  .refine((entry) => !entry.is_dir || entry.size_bytes === null, {
    message: 'Un directorio no tiene tamaño',
    path: ['size_bytes'],
  });
export type ArtifactEntry = z.infer<typeof artifactEntrySchema>;

export const experimentRunDetailSchema = z.strictObject({
  run: experimentRunSchema,
  artifacts: z.array(artifactEntrySchema),
});
export type ExperimentRunDetail = z.infer<typeof experimentRunDetailSchema>;

// ---------------------------------------------------------------------------
// Evaluation: bloqueada hasta MODEL SELECTION CLOSED (#33, custodia: Ale).
// ---------------------------------------------------------------------------
const evaluationBlockedSchema = z.strictObject({
  state: z.literal('blocked'),
  reason: z.literal('model_selection_open'),
  detail: z.string(),
});

// D05-05: `synthetic` = recorridos de prueba con predicciones conocidas; nunca se presenta
// como evaluación oficial. `local_test` es del registro de modelos (D04-06), no de aquí.
export const evaluationNamespaces = ['official', 'synthetic'] as const;
const evaluationNamespaceSchema = z.enum(evaluationNamespaces);

const evaluationSelectionSchema = z.strictObject({
  candidate_run_id: mlflowRunIdSchema,
  metric: z.literal('val_accuracy'),
  closed_at: timestampSchema,
});

// D05-05: selección cerrada, pero todavía no hay evaluación guardada en ese namespace
// (resultado ausente; ni bloqueo ni fallo). Solo identidad de la selección, sin resultados.
const evaluationPendingSchema = z.strictObject({
  state: z.literal('pending'),
  namespace: evaluationNamespaceSchema,
  reason: z.literal('evaluation_missing'),
  selection: evaluationSelectionSchema,
  manifest_hash: sha256Schema,
  detail: z.string(),
});

const evaluationReadySchema = z
  .strictObject({
    state: z.literal('ready'),
    namespace: evaluationNamespaceSchema,
    selection: evaluationSelectionSchema,
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
  evaluationPendingSchema,
  evaluationReadySchema,
]);
export type EvaluationResponse = z.infer<typeof evaluationResponseSchema>;

// ---------------------------------------------------------------------------
// D04-05: exportación por muestra de una evaluación (`GET /api/evaluation/predictions`).
// Filas en orden de crop_id; probabilidades de cada clase declarada; `synthetic` marca
// recorridos de prueba, que nunca se sirven como evaluación oficial. El backend verifica
// además `test_split_hash` contra los crop_id y la matriz contra `evaluation_response`.
// ---------------------------------------------------------------------------

const evaluationSampleSchema = z.strictObject({
  crop_id: positiveInt,
  true_class: classSchema,
  predicted_class: classSchema,
  // z.record con clave enum es exhaustivo: exige cat y dog, y rechaza cualquier otra.
  probabilities: z.record(classSchema, unitInterval),
});

export const evaluationPredictionsSchema = z
  .strictObject({
    namespace: z.enum(evaluationNamespaces),
    candidate_run_id: mlflowRunIdSchema,
    manifest_hash: sha256Schema,
    test_split_hash: sha256Schema,
    evaluated_at: timestampSchema,
    n_test: positiveInt,
    classes: classListSchema,
    predictions: z.array(evaluationSampleSchema),
  })
  .superRefine((exported, ctx) => {
    const issue = (path: (string | number)[], message: string) =>
      ctx.addIssue({ code: 'custom', path, message });
    if (exported.predictions.length !== exported.n_test) {
      issue(['predictions'], 'Debe haber exactamente n_test predicciones');
    }
    exported.predictions.forEach((sample, i) => {
      const previous = exported.predictions[i - 1];
      if (previous && sample.crop_id <= previous.crop_id) {
        issue(
          ['predictions', i, 'crop_id'],
          'crop_id en orden estrictamente creciente (sin repetidos)',
        );
      }
      const values = Object.values(sample.probabilities);
      const sum = values.reduce((total, p) => total + p, 0);
      if (!close(sum, 1, PROBABILITY_SUM_TOLERANCE)) {
        issue(['predictions', i, 'probabilities'], 'Deben sumar ~1');
      }
      if (sample.probabilities[sample.predicted_class] !== Math.max(...values)) {
        issue(
          ['predictions', i, 'predicted_class'],
          'predicted_class debe ser el argmax de probabilities',
        );
      }
    });
  });
export type EvaluationPredictions = z.infer<typeof evaluationPredictionsSchema>;

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
    // D06-03: tarjeta del modelo junto al modelo (`<semver>/model_card.json`), verificada
    // por su propio VersionId. La publicación official la exige antes de subir el modelo.
    model_card: z
      .strictObject({
        s3_key: z.string().min(1),
        version_id: z.string().min(1).nullable(),
        sha256: sha256Schema,
        size_bytes: positiveInt,
        status: z.enum(['draft', 'published', 'failed']),
      })
      .optional(),
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
    const card = model.model_card;
    if (card !== undefined) {
      if (card.s3_key !== `models/${P3_EXPERIMENT}/${model.semver}/model_card.json`) {
        issue('model_card', `La tarjeta va en models/${P3_EXPERIMENT}/<semver>/model_card.json`);
      }
      if (card.status === 'published' && card.version_id === null) {
        issue('model_card', 'Una tarjeta publicada tiene VersionId');
      }
      if (model.status === 'published' && card.status !== 'published') {
        issue('model_card', 'Un modelo publicado tiene su tarjeta verificada');
      }
    }
  });
export type ModelVersion = z.infer<typeof modelVersionSchema>;

export const modelsResponseSchema = z.strictObject({ models: z.array(modelVersionSchema) });

// D05-06: registro `local_test` (MinIO) que muestra Models aparte de `official`. Mismos campos
// que model_version más namespace, tamaño y motivo de fallo; nunca una publicación AWS.
const registryFailureReasons = [
  'sha256_mismatch',
  'size_mismatch',
  'object_missing',
  'version_id_missing',
  'version_mismatch',
  'model_card_failed',
] as const;
export const localTestModelSchema = z
  .strictObject({
    namespace: z.literal('local_test'),
    semver: modelSemverSchema,
    mlflow_run_id: mlflowRunIdSchema,
    manifest_hash: sha256Schema,
    dvc_release: datasetVersionSchema,
    dvc_release_hash: sha256Schema,
    s3_bucket: z.string().min(3),
    s3_key: z.string().min(1),
    version_id: z.string().min(1).nullable(),
    sha256: sha256Schema,
    size_bytes: positiveInt,
    status: z.enum(['draft', 'published', 'failed']),
    published_at: timestampSchema.nullable(),
    failure_reason: z.enum(registryFailureReasons).nullable(),
    failure_detail: z.string().min(1).nullable(),
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
      issue('status', 'published exige VersionId y published_at (objeto verificado)');
    }
    if (model.status !== 'published' && model.published_at !== null) {
      issue('published_at', 'Solo una versión publicada tiene published_at');
    }
    if ((model.status === 'failed') !== (model.failure_reason !== null)) {
      issue('failure_reason', 'failed ⇔ motivo de fallo');
    }
    if ((model.failure_reason === null) !== (model.failure_detail === null)) {
      issue('failure_detail', 'El motivo de fallo va con su detalle');
    }
  });
export type LocalTestModel = z.infer<typeof localTestModelSchema>;

export const localTestModelsResponseSchema = z.strictObject({
  namespace: z.literal('local_test'),
  models: z.array(localTestModelSchema),
});

/** Integridad comprobada AHORA (head + get por VersionId); no cambia el registro. */
export const localTestModelDetailSchema = z
  .strictObject({
    model: localTestModelSchema,
    integrity: z
      .strictObject({
        checked_at: timestampSchema,
        ok: z.boolean(),
        reason: z.enum(registryFailureReasons).nullable(),
        detail: z.string().min(1).nullable(),
      })
      .refine((i) => i.ok === (i.reason === null) && (i.reason === null) === (i.detail === null), {
        message: 'ok ⇔ sin motivo; un motivo va con su detalle',
      })
      .nullable(),
  })
  .refine((d) => (d.integrity !== null) === (d.model.status === 'published'), {
    message: 'Solo una versión published se audita',
    path: ['integrity'],
  });
export type LocalTestModelDetail = z.infer<typeof localTestModelDetailSchema>;

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
  experiment_run_detail: experimentRunDetailSchema,
  evaluation_response: evaluationResponseSchema,
  evaluation_predictions: evaluationPredictionsSchema,
  models_response: modelsResponseSchema,
  local_test_models_response: localTestModelsResponseSchema,
  local_test_model_detail: localTestModelDetailSchema,
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
  cancelTrainingJob: 'POST /api/training/jobs/:id/cancel',
  listRuns: 'GET /api/experiments/runs',
  getRun: 'GET /api/experiments/runs/:runId',
  getRunArtifact: 'GET /api/experiments/runs/:runId/artifacts/*path',
  getEvaluation: 'GET /api/evaluation',
  exportEvaluationPredictions: 'GET /api/evaluation/predictions',
  listModels: 'GET /api/models',
  listLocalTestModels: 'GET /api/models/local-test',
  getLocalTestModel: 'GET /api/models/local-test/:semver',
  getLocalTestModelObject: 'GET /api/models/local-test/:semver/object',
  publishModel: 'POST /api/models/:semver/publish',
  runInference: 'POST /api/inference',
  sendToAnnotationQueue: 'POST /api/inference/:id/annotation-queue',
} as const;
