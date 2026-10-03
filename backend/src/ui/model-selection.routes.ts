import express from 'express';
import type { ModelSelectionService } from '../logic/model-selection.service.js';
import { sendError } from './http-errors.js';

/**
 * D04-04 — Selección por validation (`/api/...` detrás de nginx).
 *
 * - `GET /selection`: estado persistido (open | candidate | closed) con ranking y excluidos.
 * - `POST /selection/candidate`: recalcula y guarda el candidato preparatorio.
 * - `POST /selection/close` `{ candidate_run_id }`: MODEL SELECTION CLOSED (definitivo).
 * - `GET /selection/campaign` (D05-02): expediente de campaña por fila (intentos, roles,
 *   conciliación, ranking e identidad del candidato). Solo lectura: nunca escribe.
 *
 * `GET /evaluation` y la exportación por muestra viven en `evaluation.routes.ts` (D04-05),
 * detrás de `requireClosed` / `blockedEvaluation` de este mismo servicio.
 */
export function createModelSelectionRouter(service: ModelSelectionService): express.Router {
  const router = express.Router();

  const handle =
    (run: (req: express.Request) => Promise<unknown>, fallback: string): express.RequestHandler =>
    async (req, res) => {
      try {
        res.json(await run(req));
      } catch (error) {
        sendError(res, error, fallback);
      }
    };

  router.get(
    '/selection',
    handle(() => service.state(), 'No se pudo leer el estado de la selección.'),
  );

  router.get(
    '/selection/campaign',
    handle(() => service.campaign(), 'No se pudo armar el expediente de campaña.'),
  );

  router.post(
    '/selection/candidate',
    handle(() => service.propose(), 'No se pudo proponer el candidato.'),
  );

  router.post(
    '/selection/close',
    handle(
      (req) =>
        service.close((req.body as { candidate_run_id?: unknown } | undefined)?.candidate_run_id),
      'No se pudo cerrar la selección.',
    ),
  );

  return router;
}
