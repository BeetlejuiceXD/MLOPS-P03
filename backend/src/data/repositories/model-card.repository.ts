import { and, eq, isNull } from 'drizzle-orm';
import { db } from '../db/client.js';
import { p3ModelCard } from '../db/schema.js';

/**
 * D06-03 — `p3_model_card` (MariaDB), con las mismas transiciones condicionales que
 * `p3_model_registry` (D04-06): una sola sentencia por transición, así una tarjeta que otro
 * proceso ya publicó o marcó `failed` no se vuelve a escribir.
 */
export type ModelCardNamespace = 'official' | 'local_test';
export type ModelCardInsert = typeof p3ModelCard.$inferInsert;

type UpdateResult = [{ affectedRows: number }, unknown];

const row = (namespace: ModelCardNamespace, semver: string) =>
  and(eq(p3ModelCard.namespace, namespace), eq(p3ModelCard.semver, semver));

/** `false` si el semver ya existe en el namespace (clave primaria). */
export async function insertModelCardDraft(entry: ModelCardInsert): Promise<boolean> {
  try {
    await db.insert(p3ModelCard).values(entry);
    return true;
  } catch (error) {
    const code = (error as { cause?: { code?: string }; code?: string }).cause?.code;
    if (code === 'ER_DUP_ENTRY' || (error as { code?: string }).code === 'ER_DUP_ENTRY') {
      return false;
    }
    throw error;
  }
}

export async function findModelCardRow(namespace: ModelCardNamespace, semver: string) {
  const [found] = await db.select().from(p3ModelCard).where(row(namespace, semver)).limit(1);
  return found ?? null;
}

export async function listModelCardRows(namespace: ModelCardNamespace) {
  return db.select().from(p3ModelCard).where(eq(p3ModelCard.namespace, namespace));
}

/** `draft` sin VersionId → guarda el VersionId del objeto subido. */
export async function markModelCardUploaded(
  namespace: ModelCardNamespace,
  semver: string,
  versionId: string,
): Promise<boolean> {
  const [result] = (await db
    .update(p3ModelCard)
    .set({ versionId })
    .where(
      and(row(namespace, semver), eq(p3ModelCard.status, 'draft'), isNull(p3ModelCard.versionId)),
    )) as UpdateResult;
  return result.affectedRows === 1;
}

/** `draft` con ese VersionId → `published`. */
export async function markModelCardPublished(
  namespace: ModelCardNamespace,
  semver: string,
  versionId: string,
  at: Date,
): Promise<boolean> {
  const [result] = (await db
    .update(p3ModelCard)
    .set({ status: 'published', publishedAt: at })
    .where(
      and(
        row(namespace, semver),
        eq(p3ModelCard.status, 'draft'),
        eq(p3ModelCard.versionId, versionId),
      ),
    )) as UpdateResult;
  return result.affectedRows === 1;
}

/** `draft` → `failed` con su motivo. */
export async function markModelCardFailed(
  namespace: ModelCardNamespace,
  semver: string,
  reason: string,
  detail: string,
): Promise<boolean> {
  const [result] = (await db
    .update(p3ModelCard)
    .set({ status: 'failed', failureReason: reason, failureDetail: detail })
    .where(and(row(namespace, semver), eq(p3ModelCard.status, 'draft')))) as UpdateResult;
  return result.affectedRows === 1;
}
