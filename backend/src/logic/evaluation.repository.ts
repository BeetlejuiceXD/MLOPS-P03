// Módulo directo, no `../data/index.js`: el índice también carga MinIO (y exige su
// configuración), que la evaluación no usa.
import { readEvaluation } from '../data/repositories/evaluation.repository.js';
import type { EvaluationRepository } from './evaluation.service.js';

/** MariaDB guarda JSON como LONGTEXT: según el driver puede llegar como texto. */
const parse = (value: unknown): unknown => (typeof value === 'string' ? JSON.parse(value) : value);

/** Repositorio real de la evaluación (D04-05): MariaDB `p3_evaluation`. */
export const mariaDbEvaluationRepository: EvaluationRepository = {
  async read(namespace) {
    const row = await readEvaluation(namespace);
    if (!row) return null;
    return { evaluation: parse(row.evaluation), predictions: parse(row.predictions) };
  },
};
