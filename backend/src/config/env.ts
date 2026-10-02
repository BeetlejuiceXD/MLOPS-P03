import 'dotenv/config';
import { fileURLToPath } from 'node:url';
import { z } from 'zod';

/**
 * Valida las variables de entorno usadas por la aplicación.
 */
const envSchema = z.object({
  PIPELINE_CONFIG_ROOT: z
    .string()
    .min(1)
    .default(fileURLToPath(new URL('../../../app/', import.meta.url))),

  NODE_ENV: z.enum(['development', 'test', 'production']).default('development'),

  PORT: z.coerce.number().int().positive().default(3000),

  DATABASE_URL: z.string().min(1),

  MINIO_ENDPOINT: z.string().min(1),

  MINIO_PORT: z.coerce.number().int().positive().default(9000),

  MINIO_USE_SSL: z
    .enum(['true', 'false'])
    .default('false')
    .transform((value) => value === 'true'),

  MINIO_ACCESS_KEY: z.string().min(1),

  MINIO_SECRET_KEY: z.string().min(1),

  MINIO_BUCKET: z.string().min(3),

  // D04-06: bucket de modelos (versioning obligatorio). Local/CI = MinIO; en AWS lo crea
  // Terraform con su prefijo models/p3-cnn-classifier/<semver>/ (D06-03).
  MODEL_S3_BUCKET: z
    .string()
    .regex(/^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$/, 'nombre de bucket S3')
    .default('p3-models-local'),

  // D04-01: servidor MLflow que lee el portal (Compose: http://mlflow:5000).
  MLFLOW_TRACKING_URI: z.string().url().default('http://localhost:5000'),

  MAX_UPLOAD_SIZE_BYTES: z.coerce
    .number()
    .int()
    .positive()
    .default(5 * 1024 * 1024),
});

/**
 * Variables ya validadas y tipadas.
 * Si alguna configuración requerida falta, la aplicación falla al iniciar.
 */
export const env = envSchema.parse(process.env);
