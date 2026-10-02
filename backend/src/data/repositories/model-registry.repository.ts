import { and, eq, isNull } from 'drizzle-orm';
import { db } from '../db/client.js';
import { p3ModelRegistry } from '../db/schema.js';

/**
 * D04-06 — `p3_model_registry` (MariaDB). Cada transición es un UPDATE condicional en una
 * sola sentencia (mysql2 cuenta filas coincidentes en `affectedRows`): una versión que
 * otro proceso ya publicó o marcó `failed` no se vuelve a escribir.
 */
export type ModelRegistryNamespace = 'official' | 'local_test';
export type ModelRegistryInsert = typeof p3ModelRegistry.$inferInsert;

type UpdateResult = [{ affectedRows: number }, unknown];

const row = (namespace: ModelRegistryNamespace, semver: string) =>
  and(eq(p3ModelRegistry.namespace, namespace), eq(p3ModelRegistry.semver, semver));

/** `false` si el semver ya existe en el namespace (clave primaria). */
export async function insertModelRegistryDraft(entry: ModelRegistryInsert): Promise<boolean> {
  try {
    await db.insert(p3ModelRegistry).values(entry);
    return true;
  } catch (error) {
    const code = (error as { cause?: { code?: string }; code?: string }).cause?.code;
    if (code === 'ER_DUP_ENTRY' || (error as { code?: string }).code === 'ER_DUP_ENTRY') {
      return false;
    }
    throw error;
  }
}

export async function findModelRegistryRow(namespace: ModelRegistryNamespace, semver: string) {
  const [found] = await db.select().from(p3ModelRegistry).where(row(namespace, semver)).limit(1);
  return found ?? null;
}

export async function listModelRegistryRows(namespace: ModelRegistryNamespace) {
  return db.select().from(p3ModelRegistry).where(eq(p3ModelRegistry.namespace, namespace));
}

/** `draft` sin VersionId → guarda el VersionId del objeto subido. */
export async function markModelRegistryUploaded(
  namespace: ModelRegistryNamespace,
  semver: string,
  versionId: string,
): Promise<boolean> {
  const [result] = (await db
    .update(p3ModelRegistry)
    .set({ versionId })
    .where(
      and(
        row(namespace, semver),
        eq(p3ModelRegistry.status, 'draft'),
        isNull(p3ModelRegistry.versionId),
      ),
    )) as UpdateResult;
  return result.affectedRows === 1;
}

/** `draft` con ese VersionId → `published`. */
export async function markModelRegistryPublished(
  namespace: ModelRegistryNamespace,
  semver: string,
  versionId: string,
  at: Date,
): Promise<boolean> {
  const [result] = (await db
    .update(p3ModelRegistry)
    .set({ status: 'published', publishedAt: at })
    .where(
      and(
        row(namespace, semver),
        eq(p3ModelRegistry.status, 'draft'),
        eq(p3ModelRegistry.versionId, versionId),
      ),
    )) as UpdateResult;
  return result.affectedRows === 1;
}

/** `draft` → `failed` con su motivo. */
export async function markModelRegistryFailed(
  namespace: ModelRegistryNamespace,
  semver: string,
  reason: string,
  detail: string,
): Promise<boolean> {
  const [result] = (await db
    .update(p3ModelRegistry)
    .set({ status: 'failed', failureReason: reason, failureDetail: detail })
    .where(and(row(namespace, semver), eq(p3ModelRegistry.status, 'draft')))) as UpdateResult;
  return result.affectedRows === 1;
}
