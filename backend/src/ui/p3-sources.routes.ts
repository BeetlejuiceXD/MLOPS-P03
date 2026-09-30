import express from 'express';
import type { P3SourcesService } from '../logic/p3-sources.service.js';
import { sendError } from './http-errors.js';

/**
 * D03-03 — `GET /api/releases` y `GET /api/manifest` (nginx quita el prefijo /api).
 * 200 con el contrato publicado por `trainer-worker`; 503 con el motivo si no está.
 */
export function createP3SourcesRouter(service: P3SourcesService) {
  const router = express.Router();

  router.get('/releases', async (_req, res) => {
    try {
      res.json(await service.releases());
    } catch (error) {
      sendError(res, error, 'No se pudieron leer los releases.');
    }
  });

  router.get('/manifest', async (_req, res) => {
    try {
      res.json(await service.manifest());
    } catch (error) {
      sendError(res, error, 'No se pudo leer el manifest.');
    }
  });

  return router;
}
