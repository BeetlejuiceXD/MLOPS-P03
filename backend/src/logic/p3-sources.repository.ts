import { readP3TrainingSource } from '../data/index.js';
import type { P3SourcesRepository } from './p3-sources.service.js';

/** Repositorio real de fuentes P3 (D03-03): MariaDB `p3_training_sources`. */
export const mariaDbP3SourcesRepository: P3SourcesRepository = {
  async read(name) {
    const row = await readP3TrainingSource(name);
    return row ? { status: row.status, payload: row.payload, detail: row.detail } : null;
  },
};
