/**
 * D04-04 — Persistencia REAL de la selección en MariaDB (`p3_model_selection`, migración
 * 0006): escrituras condicionales, reinicio del proceso (otra conexión lee lo mismo) y
 * cierre definitivo. Solo corre con `P3_SELECTION_MARIADB_TEST=1` y `DATABASE_URL`
 * apuntando a una base desechable con las migraciones aplicadas (job de CI "Jobs
 * persistentes"): reinicia el registro de selección. Datos sintéticos; ningún run real.
 */
import { describe, expect, it } from 'vitest';
import type { SelectionOutcome } from '../src/logic/model-selection.js';

const enabled = process.env.P3_SELECTION_MARIADB_TEST === '1';
// En CI, saltarse en silencio por falta de URL pasaría el paso sin probar nada.
if (enabled && !process.env.DATABASE_URL) {
  throw new Error('P3_SELECTION_MARIADB_TEST=1 exige DATABASE_URL de una MariaDB desechable.');
}

const outcome = (hash: string): SelectionOutcome => ({
  reference: {
    dataset_version: 'v0.1.1',
    manifest_hash: 'd'.repeat(64),
    dvc_release_hash: 'e'.repeat(64),
  },
  ranking: [
    {
      run_id: 'a'.repeat(32),
      campaign_row: 1,
      start_time: '2026-10-01T10:00:00Z',
      best_epoch: 2,
      val_accuracy: 0.86,
      val_macro_f1: 0.85,
      val_loss: 0.38,
    },
  ],
  excluded: [{ run_id: 'b'.repeat(32), reason: 'not_finished', detail: 'status=FAILED' }],
  candidate: {
    run_id: 'a'.repeat(32),
    campaign_row: 1,
    start_time: '2026-10-01T10:00:00Z',
    best_epoch: 2,
    val_accuracy: 0.86,
    val_macro_f1: 0.85,
    val_loss: 0.38,
  },
  campaign_rows: [1],
  ready_to_close: false,
  outcome_hash: hash,
});

describe.skipIf(!enabled)('p3_model_selection en MariaDB', () => {
  it('open → candidate → closed con escrituras condicionales y lectura tras reconexión', async () => {
    const { db, pool } = await import('../src/data/db/client.js');
    const { p3ModelSelection } = await import('../src/data/db/schema.js');
    const { mariaDbModelSelectionRepository: repo } = await import(
      '../src/logic/model-selection.repository.js'
    );
    const { eq } = await import('drizzle-orm');
    const mysql = (await import('mysql2/promise')).default;

    const reset = () =>
      db
        .update(p3ModelSelection)
        .set({ status: 'open', outcome: null, outcomeHash: null, proposedAt: null, closedAt: null })
        .where(eq(p3ModelSelection.id, 1));

    try {
      // La migración 0006 crea el único registro en `open`.
      const [rows] = await pool.query('SELECT COUNT(*) AS n FROM p3_model_selection');
      expect((rows as { n: number }[])[0]?.n).toBe(1);
      await reset();
      expect(await repo.read()).toEqual({
        status: 'open',
        outcome: null,
        proposedAt: null,
        closedAt: null,
      });

      const first = outcome('1'.repeat(64));
      const at = new Date('2026-10-01T12:00:00.123Z');
      expect(await repo.saveCandidate(first, at)).toBe(true);
      // Mismos valores otra vez: cuenta como fila coincidente (FOUND_ROWS), no como fallo.
      expect(await repo.saveCandidate(first, at)).toBe(true);
      expect(await repo.read()).toEqual({
        status: 'candidate',
        outcome: first,
        proposedAt: at,
        closedAt: null,
      });

      // El cierre exige el mismo outcome_hash guardado.
      expect(await repo.close('2'.repeat(64), new Date())).toBe(false);
      expect((await repo.read()).status).toBe('candidate');
      const closedAt = new Date('2026-10-01T13:00:00.456Z');
      expect(await repo.close(first.outcome_hash, closedAt)).toBe(true);

      // Cerrada es definitiva: ni re-propuesta ni segundo cierre escriben.
      expect(await repo.saveCandidate(outcome('3'.repeat(64)), new Date())).toBe(false);
      expect(await repo.close(first.outcome_hash, new Date())).toBe(false);

      // Otra conexión (equivalente a reiniciar el backend) lee el mismo estado.
      const other = await mysql.createConnection(process.env.DATABASE_URL as string);
      try {
        const [persisted] = await other.query(
          'SELECT status, outcome_hash, closed_at FROM p3_model_selection WHERE id = 1',
        );
        expect(persisted).toEqual([
          { status: 'closed', outcome_hash: first.outcome_hash, closed_at: closedAt },
        ]);
      } finally {
        await other.end();
      }
      expect(await repo.read()).toEqual({
        status: 'closed',
        outcome: first,
        proposedAt: at,
        closedAt,
      });
    } finally {
      await reset();
      await pool.end();
    }
  });
});
