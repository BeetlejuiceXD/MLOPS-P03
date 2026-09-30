/**
 * D03-03 — Fuentes oficiales de Training en la API: `GET /api/releases`, `GET /api/manifest`
 * y la compuerta de training real, leídas del snapshot que publica `trainer-worker`
 * (resolver + manifest congelado verificado contra los datos, en Python).
 *
 * Repositorio en memoria solo en estos tests de componente; los payloads son los fixtures
 * compartidos de `contracts/p3`. El recorrido real con v0.1.1 se evidencia aparte.
 */
import fs from 'node:fs';
import type { AddressInfo } from 'node:net';
import path from 'node:path';
import express from 'express';
import { describe, expect, it } from 'vitest';
import { ConflictError, ServiceUnavailableError } from '../src/logic/errors.js';
import {
  createP3SourcesService,
  type P3SourceName,
  type P3SourceRecord,
  type P3SourcesRepository,
} from '../src/logic/p3-sources.service.js';
import {
  createEligibilityGate,
  createTrainingJobsService,
  type NewTrainingJob,
  type TrainingJobRecord,
  type TrainingJobRepository,
} from '../src/logic/training-jobs.service.js';
import { createP3SourcesRouter } from '../src/ui/p3-sources.routes.js';

const FIXTURES = path.resolve('../contracts/p3/fixtures');
const fixture = (contract: string, name: string): unknown =>
  JSON.parse(fs.readFileSync(path.join(FIXTURES, contract, `${name}.json`), 'utf8')).payload;

const RELEASES = fixture('releases_response', 'valid-approved-and-rejected');
const FROZEN = fixture('manifest_summary', 'valid-frozen');

class InMemorySources implements P3SourcesRepository {
  rows = new Map<P3SourceName, P3SourceRecord>();

  async read(name: P3SourceName): Promise<P3SourceRecord | null> {
    return this.rows.get(name) ?? null;
  }

  ok(name: P3SourceName, payload: unknown) {
    this.rows.set(name, { status: 'ok', payload: JSON.stringify(payload), detail: null });
    return this;
  }

  unavailable(name: P3SourceName, detail: string) {
    this.rows.set(name, { status: 'unavailable', payload: null, detail });
    return this;
  }
}

describe('servicio de fuentes P3', () => {
  it('devuelve releases y manifest publicados cuando cumplen el contrato', async () => {
    const service = createP3SourcesService(
      new InMemorySources().ok('releases', RELEASES).ok('manifest', FROZEN),
    );
    await expect(service.releases()).resolves.toEqual(RELEASES);
    await expect(service.manifest()).resolves.toEqual(FROZEN);
  });

  it('una fuente no disponible explica el motivo que publicó el worker', async () => {
    const service = createP3SourcesService(
      new InMemorySources().unavailable('manifest', 'manifest_missing: no existe data/p3'),
    );
    const error = await service.manifest().catch((e) => e);
    expect(error).toBeInstanceOf(ServiceUnavailableError);
    expect(error.message).toMatch(/manifest_missing/);
  });

  it('sin snapshot publicado todavía no se inventa nada', async () => {
    const error = await createP3SourcesService(new InMemorySources())
      .releases()
      .catch((e) => e);
    expect(error).toBeInstanceOf(ServiceUnavailableError);
    expect(error.message).toMatch(/trainer-worker/);
  });

  it('un payload fuera de contrato no sale de la API', async () => {
    const service = createP3SourcesService(
      new InMemorySources().ok('manifest', { frozen: true }).ok('releases', { approved: 'x' }),
    );
    await expect(service.manifest()).rejects.toThrow(/contrato/);
    await expect(service.releases()).rejects.toThrow(/contrato/);
  });
});

describe('compuerta de training real con el snapshot publicado', () => {
  class Jobs implements TrainingJobRepository {
    inserted: NewTrainingJob[] = [];
    async insert(job: NewTrainingJob): Promise<TrainingJobRecord> {
      this.inserted.push(job);
      return {
        ...job,
        id: 1,
        status: 'queued',
        progressEpoch: null,
        totalEpochs: null,
        mlflowRunId: null,
        error: null,
        cancelRequested: false,
        createdAt: new Date('2026-09-29T20:00:00Z'),
        startedAt: null,
        finishedAt: null,
      };
    }
    async list() {
      return [];
    }
    async findById() {
      return null;
    }
    async listLogs() {
      return [];
    }
    async cancelQueued() {
      return false;
    }
    async requestCancel() {
      return false;
    }
  }

  const request = () => fixture('create_training_job_request', 'valid-training');

  it('release aprobado + manifest congelado publicados: encola el training', async () => {
    const jobs = new Jobs();
    const sources = createP3SourcesService(
      new InMemorySources().ok('releases', RELEASES).ok('manifest', FROZEN),
    );
    const service = createTrainingJobsService(jobs, createEligibilityGate(sources));

    const job = await service.create(request());

    expect(job.task).toBe('training');
    expect(jobs.inserted).toHaveLength(1);
  });

  it('manifest no disponible: 409 con el motivo y sin encolar', async () => {
    const jobs = new Jobs();
    const sources = createP3SourcesService(
      new InMemorySources()
        .ok('releases', RELEASES)
        .unavailable('manifest', 'manifest_not_frozen: frozen debe ser true'),
    );
    const service = createTrainingJobsService(jobs, createEligibilityGate(sources));

    const error = await service.create(request()).catch((e) => e);

    expect(error).toBeInstanceOf(ConflictError);
    expect(error.message).toMatch(/manifest_not_frozen/);
    expect(jobs.inserted).toHaveLength(0);
  });
});

describe('API HTTP de fuentes', () => {
  async function withServer(repo: P3SourcesRepository, run: (base: string) => Promise<void>) {
    const app = express();
    app.use(createP3SourcesRouter(createP3SourcesService(repo)));
    const server = app.listen(0);
    const { port } = server.address() as AddressInfo;
    try {
      await run(`http://127.0.0.1:${port}`);
    } finally {
      server.close();
    }
  }

  it('GET /releases y GET /manifest devuelven el contrato publicado', async () => {
    const repo = new InMemorySources().ok('releases', RELEASES).ok('manifest', FROZEN);
    await withServer(repo, async (base) => {
      const releases = await fetch(`${base}/releases`);
      expect(releases.status).toBe(200);
      expect(await releases.json()).toEqual(RELEASES);
      const manifest = await fetch(`${base}/manifest`);
      expect(manifest.status).toBe(200);
      expect(await manifest.json()).toEqual(FROZEN);
    });
  });

  it('una fuente no disponible responde 503 con el motivo, nunca datos vacíos', async () => {
    const repo = new InMemorySources()
      .ok('releases', RELEASES)
      .unavailable('manifest', 'manifest_missing: sin manifest congelado');
    await withServer(repo, async (base) => {
      const response = await fetch(`${base}/manifest`);
      expect(response.status).toBe(503);
      expect(await response.json()).toEqual({ error: expect.stringMatching(/manifest_missing/) });
    });
  });
});
