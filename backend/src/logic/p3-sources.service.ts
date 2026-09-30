/**
 * D03-03 — Fuentes oficiales de Training (esqueleto; implementación en el commit Green).
 */
export type P3SourceName = 'releases' | 'manifest';

export interface P3SourceRecord {
  status: 'ok' | 'unavailable';
  payload: string | null;
  detail: string | null;
}

export interface P3SourcesRepository {
  read(name: P3SourceName): Promise<P3SourceRecord | null>;
}

export function createP3SourcesService(_repo: P3SourcesRepository) {
  return {
    async releases(): Promise<unknown> {
      throw new Error('no implementado');
    },
    async manifest(): Promise<unknown> {
      throw new Error('no implementado');
    },
  };
}
