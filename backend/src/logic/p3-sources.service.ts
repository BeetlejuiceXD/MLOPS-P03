import { ServiceUnavailableError } from './errors.js';
import { manifestSummarySchema, releasesResponseSchema } from './p3.contracts.js';

/**
 * D03-03 — Fuentes oficiales de Training: releases (resolver D01-02) y manifest P3
 * congelado (D03-01), tal como las verificó y publicó `trainer-worker`.
 *
 * Se validan de nuevo contra los contratos compartidos antes de salir de la API: un
 * payload fuera de contrato, una fuente `unavailable` o un snapshot que el worker aún no
 * publicó son ServiceUnavailableError con el motivo — nunca datos vacíos o inventados.
 * La compuerta de training real (`createEligibilityGate`) las consume igual, y el worker
 * vuelve a verificar contra los archivos antes de entrenar.
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

const SCHEMAS = { releases: releasesResponseSchema, manifest: manifestSummarySchema } as const;

export function createP3SourcesService(repo: P3SourcesRepository) {
  async function load(name: P3SourceName): Promise<unknown> {
    const record = await repo.read(name);
    if (!record) {
      throw new ServiceUnavailableError(
        `trainer-worker todavía no publicó la fuente ${name} (¿está corriendo?).`,
      );
    }
    if (record.status !== 'ok' || record.payload === null) {
      throw new ServiceUnavailableError(
        `Fuente ${name} no disponible: ${record.detail ?? 'sin motivo'}`,
      );
    }
    let payload: unknown;
    try {
      payload = JSON.parse(record.payload);
    } catch {
      throw new ServiceUnavailableError(`La fuente ${name} publicada no es JSON válido.`);
    }
    const parsed = SCHEMAS[name].safeParse(payload);
    if (!parsed.success) {
      throw new ServiceUnavailableError(`La fuente ${name} publicada no cumple el contrato.`);
    }
    return parsed.data;
  }

  return {
    releases: () => load('releases'),
    manifest: () => load('manifest'),
  };
}

export type P3SourcesService = ReturnType<typeof createP3SourcesService>;
