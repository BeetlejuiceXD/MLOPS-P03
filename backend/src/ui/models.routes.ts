import express from 'express';
import type { ModelsPortalService } from '../logic/models-portal.service.js';
import { sendError } from './http-errors.js';

/**
 * D05-06 — Models (nginx quita el prefijo /api):
 * - `GET /models` → `models_response`, solo el namespace `official`.
 * - `GET /models/local-test` → `local_test_models_response` (pruebas locales en MinIO).
 * - `GET /models/local-test/:semver` → `local_test_model_detail` con integridad comprobada.
 * - `GET /models/local-test/:semver/object` → bytes de la versión exacta (VersionId).
 * Storage caído → 503 con el motivo.
 */
export function createModelsRouter(service: ModelsPortalService) {
  const router = express.Router();

  router.get('/', async (_req, res) => {
    try {
      res.json(await service.official());
    } catch (error) {
      sendError(res, error, 'No se pudo leer el registro de modelos.');
    }
  });

  router.get('/local-test', async (_req, res) => {
    try {
      res.json(await service.localTest());
    } catch (error) {
      sendError(res, error, 'No se pudo leer el registro local_test.');
    }
  });

  router.get('/local-test/:semver', async (req, res) => {
    try {
      res.json(await service.localTestDetail(req.params.semver));
    } catch (error) {
      sendError(res, error, 'No se pudo comprobar la versión local_test.');
    }
  });

  router.get('/local-test/:semver/object', async (req, res) => {
    try {
      const object = await service.localTestObject(req.params.semver);
      res.status(200);
      res.type('application/octet-stream');
      res.attachment(object.filename);
      res.setHeader('X-Model-Version-Id', object.versionId);
      res.setHeader('X-Model-Sha256', object.sha256);
      res.send(object.body);
    } catch (error) {
      sendError(res, error, 'No se pudo leer el objeto de la versión local_test.');
    }
  });

  return router;
}
