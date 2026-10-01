/**
 * D04-01 — Adaptador MLflow → portal: `GET /api/experiments/runs`, el detalle de un run y
 * la descarga de sus artefactos.
 *
 * El adaptador habla con MLflow por su API REST. Aquí MLflow es un servidor HTTP falso que
 * responde con las mismas formas que el real (`runs/search`, `runs/get`,
 * `metrics/get-history`, `mlflow-artifacts`), sembrado con runs como los que escriben
 * `trainer_worker.runner` (training, controlled_task), `tracking.short_run` y
 * `tracking.verify`. Son fixtures de componente: el contraste con el run integrado real de
 * D03-04 se evidencia aparte, contra el MLflow del stack.
 */
import { createHash } from 'node:crypto';
import type { Server } from 'node:http';
import type { AddressInfo } from 'node:net';
import express from 'express';
import { afterEach, describe, expect, it } from 'vitest';
import { createMlflowRestClient } from '../src/data/mlflow/mlflow-rest.client.js';
import { createExperimentsService } from '../src/logic/experiments.service.js';
import {
  experimentRunDetailSchema,
  experimentRunsResponseSchema,
} from '../src/logic/p3.contracts.js';
import { createExperimentsRouter } from '../src/ui/experiments.routes.js';

const EXPERIMENT = 'p3-cnn-classifier';
const EXPERIMENT_ID = '1';
const T0 = Date.parse('2026-09-30T23:40:13Z');

// ---------------------------------------------------------------------------
// MLflow falso (forma de la API REST 2.0 de MLflow 3.x con artefactos proxied).
// ---------------------------------------------------------------------------
interface FakeRun {
  run_id: string;
  experiment_id: string;
  status: 'RUNNING' | 'FINISHED' | 'FAILED' | 'KILLED';
  start_time: number;
  end_time: number | null;
  tags: Record<string, string>;
  params: Record<string, string>;
  metrics: { key: string; value: number; step: number }[];
  artifacts: Record<string, Buffer>;
}

function listen(app: express.Express): Promise<Server> {
  return new Promise((resolve) => {
    const server = app.listen(0, '127.0.0.1', () => resolve(server));
  });
}

class FakeMlflow {
  experiments = new Map<string, string>([[EXPERIMENT, EXPERIMENT_ID]]);
  runs = new Map<string, FakeRun>();
  pageSize = 2;
  server?: Server;
  url = '';
  requests: string[] = [];

  add(run: FakeRun) {
    this.runs.set(run.run_id, run);
    return run;
  }

  private wire(run: FakeRun) {
    const latest = new Map<string, { key: string; value: number; step: number }>();
    for (const metric of run.metrics) {
      const seen = latest.get(metric.key);
      if (!seen || metric.step >= seen.step) latest.set(metric.key, metric);
    }
    return {
      info: {
        run_id: run.run_id,
        run_uuid: run.run_id,
        experiment_id: run.experiment_id,
        status: run.status,
        start_time: run.start_time,
        ...(run.end_time === null ? {} : { end_time: run.end_time }),
        artifact_uri: `mlflow-artifacts:/${run.experiment_id}/${run.run_id}/artifacts`,
        lifecycle_stage: 'active',
      },
      data: {
        metrics: [...latest.values()].map((m) => ({ ...m, timestamp: T0 + m.step })),
        params: Object.entries(run.params).map(([key, value]) => ({ key, value })),
        tags: Object.entries(run.tags).map(([key, value]) => ({ key, value })),
      },
    };
  }

  async start() {
    const app = express();
    app.use(express.json());
    app.use((req, _res, next) => {
      this.requests.push(`${req.method} ${req.path}`);
      next();
    });
    const missing = (res: express.Response, message: string) =>
      res.status(404).json({ error_code: 'RESOURCE_DOES_NOT_EXIST', message });

    app.get('/api/2.0/mlflow/experiments/get-by-name', (req, res) => {
      const id = this.experiments.get(String(req.query.experiment_name));
      if (!id) return missing(res, 'Could not find experiment');
      res.json({ experiment: { experiment_id: id, name: req.query.experiment_name } });
    });
    app.post('/api/2.0/mlflow/runs/search', (req, res) => {
      const ids: string[] = req.body.experiment_ids ?? [];
      const all = [...this.runs.values()]
        .filter((run) => ids.includes(run.experiment_id))
        .sort((a, b) => b.start_time - a.start_time);
      const offset = Number(req.body.page_token ?? 0);
      const size = Math.min(Number(req.body.max_results ?? 1000), this.pageSize);
      const page = all.slice(offset, offset + size);
      const next = offset + size < all.length ? String(offset + size) : undefined;
      res.json({
        runs: page.map((run) => this.wire(run)),
        ...(next ? { next_page_token: next } : {}),
      });
    });
    app.get('/api/2.0/mlflow/runs/get', (req, res) => {
      const run = this.runs.get(String(req.query.run_id));
      if (!run) return missing(res, `Run '${req.query.run_id}' not found`);
      res.json({ run: this.wire(run) });
    });
    app.get('/api/2.0/mlflow/metrics/get-history', (req, res) => {
      const run = this.runs.get(String(req.query.run_id));
      if (!run) return missing(res, 'Run not found');
      res.json({
        metrics: run.metrics
          .filter((m) => m.key === req.query.metric_key)
          .map((m) => ({ ...m, timestamp: T0 + m.step })),
      });
    });
    // mlflow-artifacts: listado (`?path=<exp>/<run>/artifacts/<dir>`, nombres base) y descarga.
    app.get('/api/2.0/mlflow-artifacts/artifacts', (req, res) => {
      const [experimentId, runId, root, ...rest] = String(req.query.path ?? '').split('/');
      const run = this.runs.get(runId);
      if (!run || run.experiment_id !== experimentId || root !== 'artifacts') {
        return res.json({});
      }
      const dir = rest.filter(Boolean).join('/');
      const prefix = dir ? `${dir}/` : '';
      const children = new Map<string, { is_dir: boolean; file_size?: number }>();
      for (const [file, body] of Object.entries(run.artifacts)) {
        if (!file.startsWith(prefix)) continue;
        const [head, ...tail] = file.slice(prefix.length).split('/');
        children.set(
          head,
          tail.length ? { is_dir: true } : { is_dir: false, file_size: body.length },
        );
      }
      if (children.size === 0) return res.json({});
      res.json({ files: [...children].map(([path, info]) => ({ path, ...info })) });
    });
    app.get(/^\/api\/2\.0\/mlflow-artifacts\/artifacts\/(.+)$/, (req, res) => {
      const [experimentId, runId, root, ...rest] = String(req.params[0]).split('/');
      const body = this.runs.get(runId)?.artifacts[rest.join('/')];
      if (!body || root !== 'artifacts' || experimentId !== EXPERIMENT_ID) {
        return res.status(500).json({ error_code: 'INTERNAL_ERROR', message: 'NoSuchKey' });
      }
      res.type('application/octet-stream').send(body);
    });

    const server = await listen(app);
    this.server = server;
    this.url = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
    return this;
  }

  async stop() {
    await new Promise<void>((resolve) => this.server?.close(() => resolve()));
  }
}

// ---------------------------------------------------------------------------
// Runs con la forma que escriben los productores reales.
// ---------------------------------------------------------------------------
const CONFIG = {
  architecture: 'resnet18',
  pretrained: 'True',
  trainable_layers: 'last_block',
  image_size: '224',
  batch_size: '16',
  learning_rate: '0.001',
  weight_decay: '0.0001',
  optimizer: 'adam',
  max_epochs: '10',
  patience: '3',
  augmentation: 'True',
  seed: '7',
  hidden_layers: '0',
  hidden_dim: '128',
  dropout: '0.0',
};

const PROVENANCE = {
  git_commit: 'e1020c5f3a8f0d1b2c3d4e5f60718293a4b5c6d7',
  dvc_release: 'v0.1.1',
  dvc_images_md5: '951150dd4fb053f4665089fcb37a1c87.dir',
  dvc_annotations_md5: 'c7cb86ae7ece94ef7b853620e464a4d7.dir',
  dvc_release_hash: '2e7029bd794da00c9962263bd42f4e4fa94bba25e2350e4cf934505e98f937e8',
  manifest_version: 'p3-v0.1.1-s42',
  manifest_hash: '0c03c3951554b5dbf096c37468f0fb5f04e9c62c033604acabf366fb11375c43',
  classes: 'cat,dog',
  seed: '7',
  job_id: '1',
};

// Curvas del run de D03-04: la época 7 empata en val_accuracy con la 6 y gana por macro-F1.
const CURVES = [
  [0.3441, 0.85, 0.6011, 0.8284, 0.8262],
  [0.1139, 0.96, 0.2724, 0.903, 0.903],
  [0.1139, 0.96, 0.3001, 0.8955, 0.8936],
  [0.0449, 0.98, 0.2521, 0.9104, 0.9101],
  [0.0476, 0.98, 0.4003, 0.903, 0.9029],
  [0.066, 0.98, 0.359, 0.9254, 0.925],
  [0.0634, 0.98, 0.1598, 0.9254, 0.9253],
  [0.05, 0.99, 0.3, 0.91, 0.9099],
  [0.04, 0.99, 0.31, 0.9179, 0.9177],
];
const NAMES = ['train_loss', 'train_accuracy', 'val_loss', 'val_accuracy', 'val_macro_f1'];

function epochMetrics(epochs: number, skip?: { epoch: number; key: string }) {
  return CURVES.slice(0, epochs).flatMap((row, index) => {
    const step = index + 1;
    return [
      ...NAMES.map((key, i) => ({ key, value: row[i], step })),
      {
        key: 'learning_rate',
        value: 0.001,
        step,
      },
    ].filter((m) => !(skip && skip.epoch === step && skip.key === m.key));
  });
}

const MODEL = Buffer.from('pesos reales del checkpoint (bytes de prueba)');
const SHA = createHash('sha256').update(MODEL).digest('hex');
let counter = 0;
const runId = () => (++counter).toString(16).padStart(32, 'a');

function trainingRun(overrides: Partial<FakeRun> = {}): FakeRun {
  return {
    run_id: runId(),
    experiment_id: EXPERIMENT_ID,
    status: 'FINISHED',
    start_time: T0 + counter * 1000,
    end_time: T0 + counter * 1000 + 160_000,
    tags: {
      'p3.run_kind': 'training',
      'p3.job_id': '1',
      'p3.task': 'training',
      ...PROVENANCE,
      stopped_early: 'true',
      device: 'cpu',
      checkpoint_sha256: SHA,
    },
    params: { ...CONFIG, classes: 'cat,dog' },
    metrics: [
      ...epochMetrics(9),
      { key: 'best_epoch', value: 7, step: 0 },
      { key: 'best_val_accuracy', value: 0.9253731343283582, step: 0 },
      { key: 'best_val_macro_f1', value: 0.9253065774804905, step: 0 },
      { key: 'best_val_loss', value: 0.1598, step: 0 },
      { key: 'duration_seconds', value: 150.2, step: 0 },
    ],
    artifacts: {
      'checkpoint/model.pt': MODEL,
      'checkpoint/training_config.json': Buffer.from('{"seed": 7}'),
      'checkpoint/sources.json': Buffer.from(`{"checkpoint_sha256": "${SHA}"}`),
    },
    ...overrides,
  };
}

function auxiliaryRun(kind: string | null, extraTags: Record<string, string> = {}): FakeRun {
  const tags: Record<string, string> = { ...extraTags };
  if (kind) tags['p3.run_kind'] = kind;
  return {
    ...trainingRun(),
    tags,
    params: kind === 'persistence_check' ? {} : { ...CONFIG },
    metrics:
      kind === 'persistence_check' ? [{ key: 'verify_value', value: 1, step: 0 }] : epochMetrics(2),
    artifacts: {},
  };
}

// ---------------------------------------------------------------------------
// Portal (router real + servicio real + cliente REST real) contra el MLflow falso.
// ---------------------------------------------------------------------------
let mlflow: FakeMlflow;
let portal: Server | undefined;

async function startPortal(mlflowUrl: string) {
  const service = createExperimentsService(
    createMlflowRestClient({ baseUrl: mlflowUrl, timeoutMs: 2000 }),
  );
  const app = express();
  app.use('/experiments', createExperimentsRouter(service));
  const server = await listen(app);
  portal = server;
  return `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
}

async function setup(...runs: FakeRun[]) {
  mlflow = await new FakeMlflow().start();
  for (const run of runs) mlflow.add(run);
  return startPortal(mlflow.url);
}

async function getJson(url: string) {
  const res = await fetch(url);
  return { status: res.status, body: await res.json() };
}

afterEach(async () => {
  await new Promise<void>((resolve) => (portal ? portal.close(() => resolve()) : resolve()));
  portal = undefined;
  await mlflow?.stop();
});

describe('GET /experiments/runs — listado desde MLflow', () => {
  it('un run de training terminado se traduce al contrato con sus valores reales', async () => {
    const run = trainingRun();
    const api = await setup(run);
    const { status, body } = await getJson(`${api}/experiments/runs`);

    expect(status).toBe(200);
    expect(experimentRunsResponseSchema.safeParse(body).success).toBe(true);
    expect(body.excluded).toEqual([]);
    const [listed] = body.runs;
    expect(listed.run_id).toBe(run.run_id);
    expect(listed.status).toBe('FINISHED');
    expect(listed.start_time).toBe(new Date(run.start_time).toISOString());
    expect(listed.end_time).toBe(new Date(run.end_time as number).toISOString());
    // params de MLflow son texto: se tipan con TrainingConfig, no se reinterpretan.
    expect(listed.params).toEqual({
      architecture: 'resnet18',
      pretrained: true,
      trainable_layers: 'last_block',
      image_size: 224,
      batch_size: 16,
      learning_rate: 0.001,
      weight_decay: 0.0001,
      optimizer: 'adam',
      max_epochs: 10,
      patience: 3,
      augmentation: true,
      seed: 7,
      hidden_layers: 0,
      hidden_dim: 128,
      dropout: 0,
    });
    expect(listed.tags).toEqual({
      ...PROVENANCE,
      classes: ['cat', 'dog'],
      seed: 7,
      job_id: 1,
    });
    expect(listed.history).toHaveLength(9);
    expect(listed.history[6]).toEqual({
      epoch: 7,
      train_loss: 0.0634,
      train_accuracy: 0.98,
      val_loss: 0.1598,
      val_accuracy: 0.9254,
      val_macro_f1: 0.9253,
      learning_rate: 0.001,
    });
    expect(listed.summary).toEqual({
      best_epoch: 7,
      best_val_accuracy: 0.9253731343283582,
      best_val_macro_f1: 0.9253065774804905,
      best_val_loss: 0.1598,
    });
    expect(listed.checkpoint_sha256).toBe(SHA);
    expect(listed.campaign_eligible).toBe(true);
    expect(listed.ineligible_reasons).toEqual([]);
  });

  it.each([['controlled_task'], ['short_run_instrumentation'], ['persistence_check']])(
    'un run auxiliar (%s) va a excluded con su motivo, nunca a runs',
    async (kind) => {
      const run = auxiliaryRun(kind, kind === 'controlled_task' ? { job_id: '3' } : {});
      const api = await setup(run);
      const { body } = await getJson(`${api}/experiments/runs`);

      expect(experimentRunsResponseSchema.safeParse(body).success).toBe(true);
      expect(body.runs).toEqual([]);
      expect(body.excluded).toEqual([
        {
          run_id: run.run_id,
          run_kind: kind,
          status: 'FINISHED',
          start_time: new Date(run.start_time).toISOString(),
          reasons: [`run auxiliar (${kind}): no es de campaña`],
        },
      ]);
    },
  );

  it('un run sin p3.run_kind no se adivina: excluded', async () => {
    const run = auxiliaryRun(null, { ...PROVENANCE });
    const api = await setup(run);
    const { body } = await getJson(`${api}/experiments/runs`);
    expect(body.runs).toEqual([]);
    expect(body.excluded[0].run_kind).toBeNull();
    expect(body.excluded[0].reasons[0]).toMatch(/sin tag p3\.run_kind/);
  });

  it('provenance ausente no se rellena con valores oficiales: el run queda excluded', async () => {
    const run = trainingRun();
    delete run.tags.dvc_release_hash;
    const api = await setup(run);
    const { body } = await getJson(`${api}/experiments/runs`);
    expect(body.runs).toEqual([]);
    expect(body.excluded).toHaveLength(1);
    expect(body.excluded[0].run_kind).toBe('training');
    expect(body.excluded[0].reasons.join(' ')).toMatch(/dvc_release_hash/);
  });

  it('un hueco en la curva no se inventa: el run queda excluded', async () => {
    const run = trainingRun({ metrics: epochMetrics(9, { epoch: 3, key: 'val_macro_f1' }) });
    const api = await setup(run);
    const { body } = await getJson(`${api}/experiments/runs`);
    expect(body.runs).toEqual([]);
    expect(body.excluded[0].reasons.join(' ')).toMatch(/época 3.*val_macro_f1/);
  });

  it('un run RUNNING aparece con sus épocas reales pero no es elegible', async () => {
    const run = trainingRun({
      status: 'RUNNING',
      end_time: null,
      metrics: epochMetrics(2),
    });
    delete run.tags.checkpoint_sha256;
    const api = await setup(run);
    const { body } = await getJson(`${api}/experiments/runs`);

    expect(experimentRunsResponseSchema.safeParse(body).success).toBe(true);
    const [listed] = body.runs;
    expect(listed.history.map((row: { epoch: number }) => row.epoch)).toEqual([1, 2]);
    expect(listed.summary).toBeNull();
    expect(listed.end_time).toBeNull();
    expect(listed.campaign_eligible).toBe(false);
    expect(listed.ineligible_reasons.join(' ')).toMatch(/RUNNING/);
  });

  it('FINISHED no basta: sin checkpoint_sha256 no es elegible', async () => {
    const run = trainingRun();
    delete run.tags.checkpoint_sha256;
    const api = await setup(run);
    const { body } = await getJson(`${api}/experiments/runs`);
    const [listed] = body.runs;
    expect(listed.checkpoint_sha256).toBeNull();
    expect(listed.campaign_eligible).toBe(false);
    expect(listed.ineligible_reasons.join(' ')).toMatch(/checkpoint_sha256/);
  });

  it('un run FAILED con épocas parciales se muestra como no elegible', async () => {
    const run = trainingRun({
      status: 'FAILED',
      metrics: epochMetrics(3),
    });
    delete run.tags.checkpoint_sha256;
    const api = await setup(run);
    const { body } = await getJson(`${api}/experiments/runs`);
    const [listed] = body.runs;
    expect(listed.status).toBe('FAILED');
    expect(listed.summary).toBeNull();
    expect(listed.campaign_eligible).toBe(false);
    expect(listed.ineligible_reasons.join(' ')).toMatch(/FAILED/);
  });

  it('recorre todas las páginas de MLflow y ordena del más reciente al más antiguo', async () => {
    const runs = [trainingRun(), auxiliaryRun('controlled_task'), trainingRun(), trainingRun()];
    const api = await setup(...runs);
    const { body } = await getJson(`${api}/experiments/runs`);
    expect(body.runs.length + body.excluded.length).toBe(4);
    const starts = body.runs.map((run: { start_time: string }) => run.start_time);
    expect(starts).toEqual([...starts].sort().reverse());
  });

  it('sin experimento p3-cnn-classifier todavía: listado vacío, no error', async () => {
    const api = await setup();
    mlflow.experiments.clear();
    const { status, body } = await getJson(`${api}/experiments/runs`);
    expect(status).toBe(200);
    expect(body).toEqual({ experiment_name: EXPERIMENT, runs: [], excluded: [] });
  });

  it('MLflow caído → 503 con el motivo (nunca un listado vacío)', async () => {
    const api = await setup(trainingRun());
    await mlflow.stop();
    const { status, body } = await getJson(`${api}/experiments/runs`);
    expect(status).toBe(503);
    expect(body.error).toMatch(/^mlflow_unavailable: /);
  });
});

describe('GET /experiments/runs/:runId — detalle', () => {
  it('es el mismo run del listado más sus artefactos reales', async () => {
    const run = trainingRun();
    const api = await setup(run);
    const list = await getJson(`${api}/experiments/runs`);
    const { status, body } = await getJson(`${api}/experiments/runs/${run.run_id}`);

    expect(status).toBe(200);
    expect(experimentRunDetailSchema.safeParse(body).success).toBe(true);
    expect(body.run).toEqual(list.body.runs[0]);
    expect(body.artifacts).toEqual([
      { path: 'checkpoint', is_dir: true, size_bytes: null },
      { path: 'checkpoint/model.pt', is_dir: false, size_bytes: MODEL.length },
      { path: 'checkpoint/sources.json', is_dir: false, size_bytes: expect.any(Number) },
      { path: 'checkpoint/training_config.json', is_dir: false, size_bytes: expect.any(Number) },
    ]);
  });

  it('un run sin artefactos devuelve la lista vacía', async () => {
    const run = trainingRun({ artifacts: {} });
    delete run.tags.checkpoint_sha256;
    const api = await setup(run);
    const { body } = await getJson(`${api}/experiments/runs/${run.run_id}`);
    expect(body.artifacts).toEqual([]);
  });

  it('un run auxiliar → 409 con el motivo', async () => {
    const run = auxiliaryRun('short_run_instrumentation');
    const api = await setup(run);
    const { status, body } = await getJson(`${api}/experiments/runs/${run.run_id}`);
    expect(status).toBe(409);
    expect(body.error).toMatch(/short_run_instrumentation/);
  });

  it('run inexistente → 404; run de otro experimento → 404; id mal formado → 400', async () => {
    const other = trainingRun({ experiment_id: '9' });
    const api = await setup(other);
    expect((await getJson(`${api}/experiments/runs/${'b'.repeat(32)}`)).status).toBe(404);
    expect((await getJson(`${api}/experiments/runs/${other.run_id}`)).status).toBe(404);
    expect((await getJson(`${api}/experiments/runs/no-es-un-run`)).status).toBe(400);
  });

  it('MLflow caído → 503', async () => {
    const run = trainingRun();
    const api = await setup(run);
    await mlflow.stop();
    expect((await getJson(`${api}/experiments/runs/${run.run_id}`)).status).toBe(503);
  });
});

describe('GET /experiments/runs/:runId/artifacts/<ruta> — recuperación', () => {
  it('descarga los bytes reales; el sha256 coincide con el tag del run', async () => {
    const run = trainingRun();
    const api = await setup(run);
    const res = await fetch(`${api}/experiments/runs/${run.run_id}/artifacts/checkpoint/model.pt`);
    expect(res.status).toBe(200);
    expect(res.headers.get('content-type')).toMatch(/application\/octet-stream/);
    const bytes = Buffer.from(await res.arrayBuffer());
    expect(createHash('sha256').update(bytes).digest('hex')).toBe(run.tags.checkpoint_sha256);
  });

  it('artefacto inexistente → 404 artifact_missing (no un 500 ni bytes vacíos)', async () => {
    const run = trainingRun();
    const api = await setup(run);
    const { status, body } = await getJson(
      `${api}/experiments/runs/${run.run_id}/artifacts/checkpoint/no-existe.pt`,
    );
    expect(status).toBe(404);
    expect(body.error).toMatch(/^artifact_missing: /);
  });

  it('un directorio no se descarga → 404 artifact_missing', async () => {
    const run = trainingRun();
    const api = await setup(run);
    const { status } = await getJson(`${api}/experiments/runs/${run.run_id}/artifacts/checkpoint`);
    expect(status).toBe(404);
  });

  it('rutas que salen del run → 400', async () => {
    const run = trainingRun();
    const api = await setup(run);
    const res = await fetch(
      `${api}/experiments/runs/${run.run_id}/artifacts/checkpoint%2F..%2F..%2Fotro%2Fmodel.pt`,
    );
    expect(res.status).toBe(400);
  });

  it('artefactos de un run auxiliar → 409', async () => {
    const run = { ...auxiliaryRun('controlled_task'), artifacts: { 'x.txt': Buffer.from('x') } };
    const api = await setup(run);
    expect((await getJson(`${api}/experiments/runs/${run.run_id}/artifacts/x.txt`)).status).toBe(
      409,
    );
  });

  it('MLflow caído → 503', async () => {
    const run = trainingRun();
    const api = await setup(run);
    await mlflow.stop();
    const res = await fetch(`${api}/experiments/runs/${run.run_id}/artifacts/checkpoint/model.pt`);
    expect(res.status).toBe(503);
  });
});
