import express from 'express';
import { NotFoundError } from '../logic/errors.js';
import type { ModelSelectionService } from '../logic/model-selection.service.js';
import { sendError } from './http-errors.js';

/**
 * D04-04 — Selección por validation y bloqueo de Evaluation (`/api/...` detrás de nginx).
 *
 * - `GET /selection`: estado persistido (open | candidate | closed) con ranking y excluidos.
 * - `POST /selection/candidate`: recalcula y guarda el candidato preparatorio.
 * - `POST /selection/close` `{ candidate_run_id }`: MODEL SELECTION CLOSED (definitivo).
 * - `GET /evaluation`: `blocked` (contrato `evaluation_response`) hasta el cierre. Cerrada,
 *   responde 404 mientras la evaluación oficial no exista (la produce D04-05/D06-01).
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

  router.get(
    '/evaluation',
    handle(async () => {
      const blocked = await service.blockedEvaluation();
      if (blocked) return blocked;
      throw new NotFoundError(
        'La selección está cerrada, pero la evaluación oficial del frozen test aún no existe.',
      );
    }, 'No se pudo leer la evaluación.'),
  );

  return router;
}
