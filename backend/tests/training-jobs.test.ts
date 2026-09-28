/**
 * D02-05 — Jobs de entrenamiento persistentes: servicio, compuerta de training real y API.
 *
 * El repositorio en memoria sustituye a MariaDB solo en estos tests de componente; la
 * persistencia real (MariaDB, worker, recarga y reinicio) la prueba el job de CI
 * "Jobs persistentes". Los fixtures de `contracts/p3` son evidencia de componente.
 */
import fs from 'node:fs';
import type { AddressInfo } from 'node:net';
import path from 'node:path';
import express from 'express';
import { beforeEach, describe, expect, it } from 'vitest';
import { ConflictError, NotFoundError, ValidationError } from '../src/logic/errors.js';
import { trainingJobSchema } from '../src/logic/p3.contracts.js';
import {
  createEligibilityGate,
  createTrainingJobsService,
  type NewTrainingJob,
  officialSourcesUnavailableGate,
  type TrainingJobLogRecord,
  type TrainingJobRecord,
  type TrainingJobRepository,
} from '../src/logic/training-jobs.service.js';
import { createTrainingRouter } from '../src/ui/training.routes.js';

const FIXTURES = path.resolve('../contracts/p3/fixtures');
const fixture = (contract: string, name: string): unknown =>
  JSON.parse(fs.readFileSync(path.join(FIXTURES, contract, `${name}.json`), 'utf8')).payload;

const validRequest = () =>
  structuredClone(fixture('create_training_job_request', 'valid-ok')) as Record<string, unknown>;

class InMemoryRepository implements TrainingJobRepository {
  jobs: TrainingJobRecord[] = [];
  logs = new Map<number, TrainingJobLogRecord[]>();
  inserts = 0;

  async insert(job: NewTrainingJob): Promise<TrainingJobRecord> {
    this.inserts += 1;
    const record: TrainingJobRecord = {
      ...job,
      id: this.jobs.length + 1,
      status: 'queued',
      progressEpoch: null,
      totalEpochs: null,
      mlflowRunId: null,
      error: null,
      cancelRequested: false,
      createdAt: new Date('2026-09-27T20:00:00Z'),
      startedAt: null,
      finishedAt: null,
    };
    this.jobs.push(record);
    return record;
  }

  async list(): Promise<TrainingJobRecord[]> {
    return [...this.jobs].sort((a, b) => b.id - a.id);
  }

  async findById(id: number): Promise<TrainingJobRecord | null> {
    return this.jobs.find((job) => job.id === id) ?? null;
  }

  async listLogs(id: number): Promise<TrainingJobLogRecord[]> {
    return this.logs.get(id) ?? [];
  }

  async cancelQueued(id: number): Promise<boolean> {
    const job = this.jobs.find((candidate) => candidate.id === id);
    if (job?.status !== 'queued') return false;
    job.status = 'cancelled';
    job.finishedAt = new Date('2026-09-27T20:00:10Z');
    return true;
  }

  async requestCancel(id: number): Promise<boolean> {
    const job = this.jobs.find((candidate) => candidate.id === id);
    if (job?.status !== 'running') return false;
    job.cancelRequested = true;
    return true;
  }

  /** Simula lo que hace el worker al tomar el job. */
  markRunning(id: number, runId: string | null) {
    const job = this.jobs.find((candidate) => candidate.id === id);
    if (!job) throw new Error('no existe');
    job.status = 'running';
    job.startedAt = new Date('2026-09-27T20:00:05Z');
    job.progressEpoch = 1;
    job.totalEpochs = job.config.max_epochs;
    job.mlflowRunId = runId;
  }
}

let repo: InMemoryRepository;
let service: ReturnType<typeof createTrainingJobsService>;

beforeEach(() => {
  repo = new InMemoryRepository();
  service = createTrainingJobsService(repo, officialSourcesUnavailableGate);
});

describe('crear jobs', () => {
  it('encola una tarea controlada y responde con el contrato training_job', async () => {
    const job = await service.create(validRequest());
    expect(trainingJobSchema.safeParse(job).success).toBe(true);
    expect(job).toMatchObject({
      id: 1,
      task: 'controlled',
      status: 'queued',
      mlflow_run_id: null,
      progress: null,
      error: null,
      cancel_requested: false,
    });
    expect(job.config).toEqual(validRequest().config);
  });

  it('guarda fail_at_epoch de la tarea controlada', async () => {
    const request = fixture('create_training_job_request', 'valid-controlled-fail-at');
    await service.create(request);
    expect(repo.jobs[0]?.controlledFailAtEpoch).toBe(2);
  });

  it.each([
    'invalid-invalid-config',
    'invalid-missing-task',
    'invalid-unknown-task',
    'invalid-controlled-options-on-training',
    'invalid-fail-after-last-epoch',
    'invalid-bad-manifest-hash',
  ])('rechaza %s antes de encolar', async (name) => {
    await expect(service.create(fixture('create_training_job_request', name))).rejects.toThrow(
      ValidationError,
    );
    expect(repo.inserts).toBe(0);
  });

  it('el mensaje de validación nombra el campo', async () => {
    const request = validRequest();
    (request.config as Record<string, unknown>).batch_size = 4;
    await expect(service.create(request)).rejects.toThrow(/config\.batch_size/);
  });

  it('training real sin release/manifest oficial disponibles responde conflicto y no encola', async () => {
    await expect(
      service.create(fixture('create_training_job_request', 'valid-training')),
    ).rejects.toThrow(ConflictError);
    expect(repo.inserts).toBe(0);
  });
});

describe('compuerta de training real (fixtures, no habilita el entrenamiento oficial)', () => {
  const releases = fixture('releases_response', 'valid-approved-and-rejected');
  const noneApproved = fixture('releases_response', 'valid-none-approved');
  const frozen = fixture('manifest_summary', 'valid-frozen') as { manifest_hash: string };
  const notFrozen = fixture('manifest_summary', 'valid-not-frozen');
  const request = fixture('create_training_job_request', 'valid-training') as {
    dataset_version: string;
    manifest_hash: string;
  };

  it('acepta release aprobado + manifest congelado con el mismo hash', async () => {
    const gate = createEligibilityGate({
      releases: async () => releases,
      manifest: async () => frozen,
    });
    await expect(gate.check(request.dataset_version, request.manifest_hash)).resolves.toEqual({
      eligible: true,
    });
  });

  it.each([
    ['release no aprobado', noneApproved, frozen, request.manifest_hash, /release/],
    ['manifest sin congelar', releases, notFrozen, request.manifest_hash, /congelado/],
    ['hash distinto al manifest oficial', releases, frozen, 'e'.repeat(64), /hash/],
  ])('rechaza: %s', async (_case, releaseData, manifestData, hash, reason) => {
    const gate = createEligibilityGate({
      releases: async () => releaseData,
      manifest: async () => manifestData,
    });
    const result = await gate.check(request.dataset_version, hash);
    expect(result.eligible).toBe(false);
    expect(result.eligible === false && result.reason).toMatch(reason);
  });

  it('una fuente fuera de contrato no se toma como elegible', async () => {
    const gate = createEligibilityGate({
      releases: async () => ({ approved: 'todo' }),
      manifest: async () => frozen,
    });
    expect((await gate.check(request.dataset_version, request.manifest_hash)).eligible).toBe(false);
  });

  it('con la compuerta elegible se encola un training', async () => {
    const eligible = createTrainingJobsService(
      repo,
      createEligibilityGate({ releases: async () => releases, manifest: async () => frozen }),
    );
    const job = await eligible.create(fixture('create_training_job_request', 'valid-training'));
    expect(job.task).toBe('training');
    expect(job.status).toBe('queued');
  });
});

describe('consultar y cancelar', () => {
  it('lista del más reciente al más antiguo y cada job cumple el contrato', async () => {
    await service.create(validRequest());
    await service.create(validRequest());
    const { jobs } = await service.list();
    expect(jobs.map((job) => job.id)).toEqual([2, 1]);
    for (const job of jobs) expect(trainingJobSchema.safeParse(job).success).toBe(true);
  });

  it('un job inexistente es 404', async () => {
    await expect(service.get(99)).rejects.toThrow(NotFoundError);
    await expect(service.logs(99)).rejects.toThrow(NotFoundError);
  });

  it('cancelar un job en cola lo termina como cancelled, no como éxito', async () => {
    await service.create(validRequest());
    const job = await service.cancel(1);
    expect(job.status).toBe('cancelled');
    expect(job.finished_at).not.toBeNull();
    expect(trainingJobSchema.safeParse(job).success).toBe(true);
  });

  it('cancelar un job en ejecución solo marca la petición; el worker la aplica', async () => {
    await service.create(validRequest());
    repo.markRunning(1, 'a'.repeat(32));
    const job = await service.cancel(1);
    expect(job.status).toBe('running');
    expect(job.cancel_requested).toBe(true);
  });

  it('cancelar un job terminado es conflicto', async () => {
    await service.create(validRequest());
    await service.cancel(1);
    await expect(service.cancel(1)).rejects.toThrow(ConflictError);
  });

  it('expone progreso y run de MLflow que persistió el worker', async () => {
    await service.create(validRequest());
    repo.markRunning(1, 'b'.repeat(32));
    const job = await service.get(1);
    expect(job.progress).toEqual({ epoch: 1, total_epochs: 30 });
    expect(job.mlflow_run_id).toBe('b'.repeat(32));
  });

  it('devuelve los logs persistidos', async () => {
    await service.create(validRequest());
    repo.logs.set(1, [
      { ts: new Date('2026-09-27T20:00:06Z'), level: 'info', message: 'Época 1/30' },
    ]);
    expect(await service.logs(1)).toEqual({
      job_id: 1,
      lines: [{ ts: '2026-09-27T20:00:06.000Z', level: 'info', message: 'Época 1/30' }],
    });
  });

  it('un registro corrupto en la base no sale fuera de contrato', async () => {
    await service.create(validRequest());
    const job = repo.jobs[0];
    if (!job) throw new Error('sin job');
    job.status = 'succeeded'; // sin run ni fechas: incoherente
    await expect(service.get(1)).rejects.toThrow(/contrato/);
  });
});

describe('API HTTP', () => {
  async function withServer(run: (base: string) => Promise<void>) {
    const app = express();
    app.use(express.json());
    app.use('/training', createTrainingRouter(service));
    const server = app.listen(0);
    const { port } = server.address() as AddressInfo;
    try {
      await run(`http://127.0.0.1:${port}`);
    } finally {
      server.close();
    }
  }

  const post = (url: string, body: unknown) =>
    fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

  it('POST válido responde 201 con el job en cola, sin esperar al entrenamiento', async () => {
    await withServer(async (base) => {
      const started = Date.now();
      const res = await post(`${base}/training/jobs`, validRequest());
      expect(res.status).toBe(201);
      expect(Date.now() - started).toBeLessThan(1000);
      const body = await res.json();
      expect(body.status).toBe('queued');
      expect(trainingJobSchema.safeParse(body).success).toBe(true);
    });
  });

  it('POST inválido responde 400 con mensaje útil y no encola', async () => {
    await withServer(async (base) => {
      const request = validRequest();
      (request.config as Record<string, unknown>).learning_rate = 0.05;
      const res = await post(`${base}/training/jobs`, request);
      expect(res.status).toBe(400);
      expect((await res.json()).error).toMatch(/config\.learning_rate/);
      expect(repo.inserts).toBe(0);
    });
  });

  it('training real sin fuentes oficiales responde 409', async () => {
    await withServer(async (base) => {
      const res = await post(
        `${base}/training/jobs`,
        fixture('create_training_job_request', 'valid-training'),
      );
      expect(res.status).toBe(409);
    });
  });

  it('GET lista, detalle, logs y cancelación', async () => {
    await withServer(async (base) => {
      await post(`${base}/training/jobs`, validRequest());
      expect((await (await fetch(`${base}/training/jobs`)).json()).jobs).toHaveLength(1);
      expect((await fetch(`${base}/training/jobs/1`)).status).toBe(200);
      expect((await (await fetch(`${base}/training/jobs/1/logs`)).json()).lines).toEqual([]);
      expect((await fetch(`${base}/training/jobs/99`)).status).toBe(404);
      expect((await fetch(`${base}/training/jobs/abc`)).status).toBe(400);
      expect((await post(`${base}/training/jobs/1/cancel`, {})).status).toBe(200);
      expect((await post(`${base}/training/jobs/1/cancel`, {})).status).toBe(409);
    });
  });
});
