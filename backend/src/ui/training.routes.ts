import express from 'express';
// Módulos de Logic importados directo (no el índice): el índice abre la conexión a MariaDB
// al importarse y estas rutas se prueban sin base de datos.
import { idParamSchema } from '../logic/annotation.validation.js';
import type { TrainingJobsService } from '../logic/training-jobs.service.js';
import { sendError } from './http-errors.js';

/**
 * D02-05 — Rutas de jobs de entrenamiento (`/api/training/...` detrás de nginx).
 * Solo validan y delegan a Logic: el entrenamiento corre en `trainer-worker`.
 */
export function createTrainingRouter(service: TrainingJobsService): express.Router {
  const router = express.Router();

  const withId =
    (
      handler: (id: number, res: express.Response) => Promise<void>,
      fallback: string,
    ): express.RequestHandler =>
    async (req, res) => {
      const parsed = idParamSchema.safeParse(req.params.id);
      if (!parsed.success) {
        res.status(400).json({ error: 'El id del job debe ser un entero positivo.' });
        return;
      }
      try {
        await handler(parsed.data, res);
      } catch (error) {
        sendError(res, error, fallback);
      }
    };

  router.post('/jobs', async (req, res) => {
    try {
      res.status(201).json(await service.create(req.body));
    } catch (error) {
      sendError(res, error, 'No se pudo crear el job de entrenamiento.');
    }
  });

  router.get('/jobs', async (_req, res) => {
    try {
      res.json(await service.list());
    } catch (error) {
      sendError(res, error, 'No se pudieron listar los jobs de entrenamiento.');
    }
  });

  router.get(
    '/jobs/:id',
    withId(async (id, res) => {
      res.json(await service.get(id));
    }, 'No se pudo leer el job.'),
  );

  router.get(
    '/jobs/:id/logs',
    withId(async (id, res) => {
      res.json(await service.logs(id));
    }, 'No se pudieron leer los logs del job.'),
  );

  router.post(
    '/jobs/:id/cancel',
    withId(async (id, res) => {
      res.json(await service.cancel(id));
    }, 'No se pudo cancelar el job.'),
  );

  return router;
}
