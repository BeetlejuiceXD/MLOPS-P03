import express from 'express';
import type { createP3SourcesService } from '../logic/p3-sources.service.js';

/** D03-03 — esqueleto; las rutas se implementan en el commit Green. */
export function createP3SourcesRouter(_service: ReturnType<typeof createP3SourcesService>) {
  return express.Router();
}
