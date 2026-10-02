/**
 * D04-05 — Lectura REAL de `p3_evaluation` (MariaDB, migración 0007) de lo que escribió el
 * productor Python. Corre en el job de CI "Jobs persistentes" DESPUÉS de
 * `.github/scripts/evaluation_e2e.py`, que deja la selección cerrada (sintética) y una
 * corrida `synthetic` con las mismas entradas y fechas que `tests/fixtures/evaluation-synthetic.json`.
 * Solo con `P3_EVALUATION_MARIADB_TEST=1` y `DATABASE_URL` de una base desechable; al
 * final borra la fila y reabre la selección. Datos sintéticos; ningún resultado del test.
 */
import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { NotFoundError } from '../src/logic/errors.js';
import {
  createEvaluationService,
  predictionsToCsv,
  type StoredEvaluation,
} from '../src/logic/evaluation.service.js';
import {
  createModelSelectionService,
  runsAdapterPendingSource,
} from '../src/logic/model-selection.service.js';

const enabled = process.env.P3_EVALUATION_MARIADB_TEST === '1';
// En CI, saltarse en silencio por falta de URL pasaría el paso sin probar nada.
if (enabled && !process.env.DATABASE_URL) {
  throw new Error('P3_EVALUATION_MARIADB_TEST=1 exige DATABASE_URL de una MariaDB desechable.');
}

const FIXTURE: StoredEvaluation = JSON.parse(
  fs.readFileSync(path.resolve('tests/fixtures/evaluation-synthetic.json'), 'utf8'),
);

describe.skipIf(!enabled)('p3_evaluation en MariaDB', () => {
  it('la corrida sintética del productor Python se lee idéntica y nunca como oficial', async () => {
    const { pool } = await import('../src/data/db/client.js');
    const { mariaDbEvaluationRepository: repo } = await import(
      '../src/logic/evaluation.repository.js'
    );
    const { mariaDbModelSelectionRepository } = await import(
      '../src/logic/model-selection.repository.js'
    );
    const guard = createModelSelectionService(
      mariaDbModelSelectionRepository,
      runsAdapterPendingSource,
      {
        manifest: async () => {
          throw new Error('la evaluación no lee el manifest');
        },
      },
    );

    try {
      // Lo que escribió Python es, byte a byte de JSON, lo que el backend usa en sus tests.
      expect(await repo.read('synthetic')).toEqual(FIXTURE);
      expect(await repo.read('official')).toBeNull();

      const synthetic = createEvaluationService(repo, guard, 'synthetic');
      expect(await synthetic.evaluation()).toEqual(FIXTURE.evaluation);
      const exported = await synthetic.predictions();
      expect(exported.namespace).toBe('synthetic');
      expect(predictionsToCsv(exported).trimEnd().split('\n')).toHaveLength(1 + exported.n_test);

      // D05-05: la API oficial responde `pending` (sin resultados) y nunca la sintética.
      const official = createEvaluationService(repo, guard, 'official');
      const pending = await official.evaluation();
      expect(pending).toMatchObject({ state: 'pending', namespace: 'official' });
      expect(JSON.stringify(pending)).not.toMatch(/confusion_matrix|metrics|n_test/);
      await expect(official.predictions()).rejects.toBeInstanceOf(NotFoundError);
    } finally {
      await pool.query("DELETE FROM p3_evaluation WHERE namespace = 'synthetic'");
      await pool.query(
        "UPDATE p3_model_selection SET status = 'open', outcome = NULL, outcome_hash = NULL, " +
          'proposed_at = NULL, closed_at = NULL WHERE id = 1',
      );
      await pool.end();
    }
  });
});
