import { desc, eq } from 'drizzle-orm';
import { db } from '../db/client.js';
import { images, type NewImage, p3AnnotationQueue, p3Inference } from '../db/schema.js';

/**
 * D05-07 — `p3_inference` y `p3_annotation_queue` (MariaDB, migración 0009). El índice
 * único de `p3_annotation_queue.inference_id` impide dos elementos para la misma
 * inferencia aunque lleguen dos envíos a la vez.
 */
export type InferenceInsert = typeof p3Inference.$inferInsert;
export type AnnotationQueueInsert = typeof p3AnnotationQueue.$inferInsert;

export async function insertInferenceRow(row: InferenceInsert): Promise<number> {
  const [inserted] = await db.insert(p3Inference).values(row).$returningId();
  if (!inserted) throw new Error('MariaDB no devolvió el id de la inferencia');
  return inserted.id;
}

export async function findInferenceRow(id: number) {
  const [found] = await db.select().from(p3Inference).where(eq(p3Inference.id, id)).limit(1);
  return found ?? null;
}

export async function listInferenceRows(limit: number) {
  return db.select().from(p3Inference).orderBy(desc(p3Inference.id)).limit(limit);
}

export async function findAnnotationQueueRow(inferenceId: number) {
  const [found] = await db
    .select()
    .from(p3AnnotationQueue)
    .where(eq(p3AnnotationQueue.inferenceId, inferenceId))
    .limit(1);
  return found ?? null;
}

function isDuplicate(error: unknown): boolean {
  const code = (error as { cause?: { code?: string }; code?: string }).cause?.code;
  return code === 'ER_DUP_ENTRY' || (error as { code?: string }).code === 'ER_DUP_ENTRY';
}

/** `null` si la inferencia ya tiene elemento en la cola (índice único). */
export async function insertAnnotationQueueRow(row: AnnotationQueueInsert): Promise<number | null> {
  try {
    const [inserted] = await db.insert(p3AnnotationQueue).values(row).$returningId();
    if (!inserted) throw new Error('MariaDB no devolvió el id del elemento de la cola');
    return inserted.id;
  } catch (error) {
    if (isDuplicate(error)) return null;
    throw error;
  }
}

/**
 * Archivo nuevo a la cola: la imagen `pending` del portal y el elemento de la cola en UNA
 * transacción. Si el segundo insert falla (índice único, MariaDB caída), la imagen se
 * revierte con él: nunca queda una imagen pending sin elemento. `null` = ya tenía elemento.
 */
export async function insertAnnotationQueueWithNewImage(
  image: NewImage,
  row: Omit<AnnotationQueueInsert, 'imageId'>,
): Promise<{ id: number; imageId: number } | null> {
  try {
    return await db.transaction(async (tx) => {
      const [createdImage] = await tx.insert(images).values(image).$returningId();
      if (!createdImage) throw new Error('MariaDB no devolvió el id de la imagen');
      const [inserted] = await tx
        .insert(p3AnnotationQueue)
        .values({ ...row, imageId: createdImage.id })
        .$returningId();
      if (!inserted) throw new Error('MariaDB no devolvió el id del elemento de la cola');
      return { id: inserted.id, imageId: createdImage.id };
    });
  } catch (error) {
    if (isDuplicate(error)) return null;
    throw error;
  }
}

export async function listAnnotationQueueRows() {
  return db.select().from(p3AnnotationQueue).orderBy(desc(p3AnnotationQueue.id));
}
