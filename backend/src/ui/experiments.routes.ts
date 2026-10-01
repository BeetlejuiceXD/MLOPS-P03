import { Readable } from 'node:stream';
import type { ReadableStream as WebReadableStream } from 'node:stream/web';
import express from 'express';
import type { ExperimentsService } from '../logic/experiments.service.js';
import { sendError } from './http-errors.js';

/**
 * D04-01 — Runs reales de MLflow para el portal (nginx quita el prefijo /api):
 * - `GET /experiments/runs` → `experiment_runs_response`
 * - `GET /experiments/runs/:runId` → `experiment_run_detail` (409 si es auxiliar/incompleto)
 * - `GET /experiments/runs/:runId/artifacts/<ruta>` → bytes del artefacto (404 artifact_missing)
 * MLflow caído → 503 con el motivo.
 */
export function createExperimentsRouter(service: ExperimentsService) {
  const router = express.Router();

  router.get('/runs', async (_req, res) => {
    try {
      res.json(await service.listRuns());
    } catch (error) {
      sendError(res, error, 'No se pudieron leer los runs de MLflow.');
    }
  });

  router.get('/runs/:runId', async (req, res) => {
    try {
      res.json(await service.getRun(req.params.runId));
    } catch (error) {
      sendError(res, error, 'No se pudo leer el run de MLflow.');
    }
  });

  router.get('/runs/:runId/artifacts/*path', async (req, res) => {
    const segments = (req.params as { path?: string | string[] }).path ?? [];
    const path = Array.isArray(segments) ? segments.join('/') : segments;
    try {
      const artifact = await service.downloadArtifact(req.params.runId, path);
      res.status(200);
      res.type('application/octet-stream');
      res.attachment(artifact.filename);
      if (artifact.size !== null) res.setHeader('Content-Length', String(artifact.size));
      if (!artifact.body) {
        res.end();
        return;
      }
      Readable.fromWeb(artifact.body as WebReadableStream<Uint8Array>)
        .on('error', () => res.destroy())
        .pipe(res);
    } catch (error) {
      sendError(res, error, 'No se pudo descargar el artefacto de MLflow.');
    }
  });

  return router;
}
