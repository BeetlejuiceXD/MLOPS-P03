import express from 'express';
import { ValidationError } from '../logic/errors.js';
import { type EvaluationService, predictionsToCsv } from '../logic/evaluation.service.js';
import { sendError } from './http-errors.js';

/**
 * D04-05 — Evaluation (`/api/...` detrás de nginx).
 *
 * - `GET /evaluation`: `evaluation_response`. `blocked` hasta MODEL SELECTION CLOSED;
 *   cerrada, `ready` con la evaluación oficial, o 404 mientras no exista (D06-01).
 * - `GET /evaluation/predictions[?format=json|csv]`: exportación por muestra
 *   (`evaluation_predictions`); 409 antes del cierre, 404 sin evaluación. Con `csv` se
 *   descarga el mismo contenido como archivo.
 */
export function createEvaluationRouter(service: EvaluationService): express.Router {
  const router = express.Router();

  router.get('/evaluation', async (_req, res) => {
    try {
      res.json(await service.evaluation());
    } catch (error) {
      sendError(res, error, 'No se pudo leer la evaluación.');
    }
  });

  router.get('/evaluation/predictions', async (req, res) => {
    try {
      const format = req.query.format ?? 'json';
      if (format !== 'json' && format !== 'csv') {
        throw new ValidationError('format debe ser json o csv.');
      }
      const predictions = await service.predictions();
      if (format === 'json') {
        res.json(predictions);
        return;
      }
      res.type('text/csv; charset=utf-8');
      res.attachment(`p3-evaluation-predictions-${predictions.namespace}.csv`);
      res.send(predictionsToCsv(predictions));
    } catch (error) {
      sendError(res, error, 'No se pudo exportar las predicciones.');
    }
  });

  return router;
}
