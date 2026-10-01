import { eq } from 'drizzle-orm';
import { db } from '../db/client.js';
import { p3Evaluation } from '../db/schema.js';

/**
 * D04-05 — `p3_evaluation` (MariaDB), solo lectura. La escribe el productor Python
 * (`app/evaluation/store.py`); la API nunca crea ni modifica evaluaciones.
 */
export async function readEvaluation(namespace: 'official' | 'synthetic') {
  const [row] = await db
    .select()
    .from(p3Evaluation)
    .where(eq(p3Evaluation.namespace, namespace))
    .limit(1);
  return row ?? null;
}
