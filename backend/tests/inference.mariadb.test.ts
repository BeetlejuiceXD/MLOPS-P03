/**
 * D05-07 — Persistencia REAL de inferencias y cola de anotación en MariaDB (migración 0009:
 * `p3_inference`, `p3_annotation_queue`). Escribe con el repositorio del backend y relee
 * desde OTRA conexión (equivalente a reiniciar el backend). Un mismo `inference_id` no puede
 * tener dos elementos en la cola. Solo corre con `P3_INFERENCE_MARIADB_TEST=1` y
 * `DATABASE_URL` hacia una base desechable con las migraciones aplicadas (job de CI
 * "Jobs persistentes"). Datos sintéticos; ningún modelo real.
 */
import { afterAll, describe, expect, it } from 'vitest';

const enabled = process.env.P3_INFERENCE_MARIADB_TEST === '1';
if (enabled && !process.env.DATABASE_URL) {
  throw new Error('P3_INFERENCE_MARIADB_TEST=1 exige DATABASE_URL de una MariaDB desechable.');
}

const SMOKE = {
  source: 'smoke' as const,
  package_id: 'p3-cnn-classifier-smoke-c46e4c3ab2bb',
  format_version: '1.0.0',
  model_version: null,
  mlflow_run_id: 'c46e4c3ab2bb4ee18c37571adbb65d92',
  checkpoint_sha256: 'e93de23e2cf9e72e8efa97bde1e9fb2083402d15a422705dab80ffd11bbd5b97',
};

describe.skipIf(!enabled)('p3_inference y p3_annotation_queue en MariaDB', () => {
  afterAll(async () => {
    const { pool } = await import('../src/data/db/client.js');
    await pool.end();
  });

  it('guarda la inferencia con su identidad, encola una sola vez y se relee tras reconectar', async () => {
    const { db, pool } = await import('../src/data/db/client.js');
    const { images } = await import('../src/data/db/schema.js');
    const { mariaDbInferenceRepository: repo } = await import(
      '../src/logic/inference.repository.js'
    );
    const { DuplicateQueueItemError } = await import('../src/logic/inference.service.js');
    const mysql = (await import('mysql2/promise')).default;

    const storageKey = `inference/mariadb-test-${Date.now()}`;
    const createdAt = new Date('2026-10-02T15:00:00.123Z');
    let imageId: number | undefined;
    let inferenceId: number | undefined;
    try {
      inferenceId = await repo.insert({
        created_at: createdAt,
        input: {
          kind: 'upload',
          filename: 'sintetica.png',
          mime_type: 'image/png',
          size_bytes: 1234,
          width: 32,
          height: 32,
          sha256: 'c'.repeat(64),
        },
        storage_key: storageKey,
        predicted_class: 'dog',
        probabilities: { cat: 0.0587, dog: 0.9413 },
        model: SMOKE,
      });
      const [inserted] = await db
        .insert(images)
        .values({
          filename: 'sintetica.png',
          storageKey,
          mimeType: 'image/png',
          width: 32,
          height: 32,
          sizeBytes: 1234,
        })
        .$returningId();
      imageId = inserted?.id;
      const item = await repo.insertQueueItem({
        inference_id: inferenceId,
        image_id: imageId as number,
        annotation_id: null,
        created_at: createdAt,
      });
      await expect(
        repo.insertQueueItem({
          inference_id: inferenceId,
          image_id: imageId as number,
          annotation_id: null,
          created_at: createdAt,
        }),
      ).rejects.toThrow(DuplicateQueueItemError);

      // Otra conexión lee lo mismo: identidad, probabilidades y elemento de la cola.
      const other = await mysql.createConnection(process.env.DATABASE_URL as string);
      try {
        const [rows] = await other.query(
          'SELECT i.predicted_class, i.checkpoint_sha256, i.model_source, q.id AS queue_id, q.status ' +
            'FROM p3_inference i JOIN p3_annotation_queue q ON q.inference_id = i.id WHERE i.id = ?',
          [inferenceId],
        );
        expect(rows).toEqual([
          {
            predicted_class: 'dog',
            checkpoint_sha256: SMOKE.checkpoint_sha256,
            model_source: 'smoke',
            queue_id: item.id,
            status: 'pending',
          },
        ]);
      } finally {
        await other.end();
      }
      expect(await repo.find(inferenceId)).toEqual({
        id: inferenceId,
        created_at: createdAt,
        input: expect.objectContaining({ kind: 'upload', sha256: 'c'.repeat(64) }),
        storage_key: storageKey,
        predicted_class: 'dog',
        probabilities: { cat: 0.0587, dog: 0.9413 },
        model: SMOKE,
      });
      expect(await repo.findQueueItem(inferenceId)).toEqual(item);
      // Enviar a la cola no crea anotaciones: la predicción no es una etiqueta humana.
      const [annotations] = await pool.query(
        'SELECT COUNT(*) AS n FROM annotations WHERE image_id = ?',
        [imageId],
      );
      expect((annotations as { n: number }[])[0]?.n).toBe(0);
    } finally {
      if (inferenceId !== undefined) {
        await pool.query('DELETE FROM p3_annotation_queue WHERE inference_id = ?', [inferenceId]);
        await pool.query('DELETE FROM p3_inference WHERE id = ?', [inferenceId]);
      }
      if (imageId !== undefined) await pool.query('DELETE FROM images WHERE id = ?', [imageId]);
    }
  });

  it('archivo nuevo a la cola: imagen y elemento en UNA transacción; si falla el elemento no queda la imagen', async () => {
    const { pool } = await import('../src/data/db/client.js');
    const { mariaDbInferenceRepository: repo } = await import(
      '../src/logic/inference.repository.js'
    );
    const { DuplicateQueueItemError } = await import('../src/logic/inference.service.js');
    const storageKey = `inference/mariadb-tx-${Date.now()}`;
    const createdAt = new Date('2026-10-02T16:00:00.000Z');
    const image = {
      filename: 'sintetica-tx.png',
      storageKey,
      mimeType: 'image/png',
      width: 32,
      height: 32,
      sizeBytes: 1234,
    };
    const countImages = async () => {
      const [rows] = await pool.query('SELECT COUNT(*) AS n FROM images WHERE storage_key LIKE ?', [
        `${storageKey}%`,
      ]);
      return Number((rows as { n: number }[])[0]?.n);
    };
    let inferenceId: number | undefined;
    try {
      inferenceId = await repo.insert({
        created_at: createdAt,
        input: {
          kind: 'upload',
          filename: 'sintetica-tx.png',
          mime_type: 'image/png',
          size_bytes: 1234,
          width: 32,
          height: 32,
          sha256: 'd'.repeat(64),
        },
        storage_key: storageKey,
        predicted_class: 'cat',
        probabilities: { cat: 0.9, dog: 0.1 },
        model: SMOKE,
      });
      const item = await repo.insertQueueItemWithNewImage(image, {
        inference_id: inferenceId,
        annotation_id: null,
        created_at: createdAt,
      });
      expect(await countImages()).toBe(1);
      const [status] = await pool.query('SELECT status FROM images WHERE id = ?', [item.image_id]);
      expect((status as { status: string }[])[0]?.status).toBe('pending');

      // Otra imagen válida (otra key), pero el elemento viola el índice único de la cola:
      // la imagen que ya se había insertado dentro de la misma transacción se revierte.
      await expect(
        repo.insertQueueItemWithNewImage(
          { ...image, storageKey: `${storageKey}-retry` },
          {
            inference_id: inferenceId,
            annotation_id: null,
            created_at: createdAt,
          },
        ),
      ).rejects.toThrow(DuplicateQueueItemError);
      expect(await countImages()).toBe(1);
    } finally {
      if (inferenceId !== undefined) {
        await pool.query('DELETE FROM p3_annotation_queue WHERE inference_id = ?', [inferenceId]);
        await pool.query('DELETE FROM p3_inference WHERE id = ?', [inferenceId]);
      }
      await pool.query('DELETE FROM images WHERE storage_key LIKE ?', [`${storageKey}%`]);
    }
  });
});
