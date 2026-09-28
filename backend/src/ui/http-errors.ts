import type express from 'express';
import { ConflictError, NotFoundError, ValidationError } from '../logic/errors.js';

/**
 * SPEC-VALID-001 — Traduce un error de la capa Logic al código HTTP correcto.
 *
 * El mapeo se hace por clase de error, no comparando el texto del mensaje,
 * para que un cambio de redacción no altere la semántica de la respuesta.
 */
export function sendError(res: express.Response, error: unknown, fallback: string): void {
  if (error instanceof NotFoundError) {
    res.status(404).json({ error: error.message });
    return;
  }

  if (error instanceof ValidationError) {
    res.status(400).json({ error: error.message });
    return;
  }

  if (error instanceof ConflictError) {
    res.status(409).json({ error: error.message });
    return;
  }

  console.error(fallback, error);
  res.status(500).json({ error: fallback });
}
