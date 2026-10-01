import express from 'express';
import type { EvaluationService } from '../logic/evaluation.service.js';

/** D04-05 — `GET /evaluation` y `GET /evaluation/predictions` (stub Red). */
export function createEvaluationRouter(_service: EvaluationService): express.Router {
  return express.Router();
}
