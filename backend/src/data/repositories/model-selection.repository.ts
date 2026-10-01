import { and, eq, ne } from 'drizzle-orm';
import { db } from '../db/client.js';
import { p3ModelSelection } from '../db/schema.js';

/**
 * D04-04 — `p3_model_selection` (MariaDB). Un solo registro (`id = 1`), creado en `open`
 * por la migración 0006. Las transiciones son UPDATE condicionales en una sola sentencia
 * (mysql2 cuenta filas coincidentes en `affectedRows`): un cierre concurrente o una
 * propuesta sobre una selección ya cerrada no escriben nada.
 */
const SELECTION_ID = 1;

type UpdateResult = [{ affectedRows: number }, unknown];

export async function readModelSelection() {
  const [row] = await db
    .select()
    .from(p3ModelSelection)
    .where(eq(p3ModelSelection.id, SELECTION_ID))
    .limit(1);
  return row ?? null;
}

/** → `candidate` salvo que ya esté `closed`. */
export async function saveModelSelectionCandidate(
  outcome: unknown,
  outcomeHash: string,
  at: Date,
): Promise<boolean> {
  const [result] = (await db
    .update(p3ModelSelection)
    .set({ status: 'candidate', outcome, outcomeHash, proposedAt: at })
    .where(
      and(eq(p3ModelSelection.id, SELECTION_ID), ne(p3ModelSelection.status, 'closed')),
    )) as UpdateResult;
  return result.affectedRows === 1;
}

/** `candidate` → `closed`, solo si el resultado guardado sigue siendo `outcomeHash`. */
export async function closeModelSelection(outcomeHash: string, at: Date): Promise<boolean> {
  const [result] = (await db
    .update(p3ModelSelection)
    .set({ status: 'closed', closedAt: at })
    .where(
      and(
        eq(p3ModelSelection.id, SELECTION_ID),
        eq(p3ModelSelection.status, 'candidate'),
        eq(p3ModelSelection.outcomeHash, outcomeHash),
      ),
    )) as UpdateResult;
  return result.affectedRows === 1;
}
