import { eq } from 'drizzle-orm';
import { db } from '../db/client.js';
import { p3TrainingSources } from '../db/schema.js';

/**
 * D03-03 — Lectura del snapshot de fuentes que publica `trainer-worker` (Python). El
 * backend nunca lo escribe: solo el worker verifica las fuentes contra los datos reales.
 */
export async function readP3TrainingSource(name: 'releases' | 'manifest') {
  const [row] = await db
    .select()
    .from(p3TrainingSources)
    .where(eq(p3TrainingSources.name, name))
    .limit(1);
  return row ?? null;
}
