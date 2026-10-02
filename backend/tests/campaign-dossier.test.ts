/**
 * D05-02 — Expediente de campaña: conciliación request → job → run por fila OFAT,
 * representante cronológico y ranking validation-only.
 *
 * Todo con jobs y runs SINTÉTICOS (`selection-runs.ts`), con datos conocidos: no son runs
 * reales de la campaña ni traen datos del frozen test. El ranking lo decide
 * `selectCandidate` (D04-04); el expediente lo explica fila por fila y dice qué impide
 * cerrar el conteo.
 */
import { describe, expect, it } from 'vitest';
import { buildCampaignDossier, type CampaignDossier } from '../src/logic/campaign-dossier.js';
import { MIN_COMPARABLE_RUNS, selectCandidate } from '../src/logic/model-selection.js';
import {
  campaign,
  jobFor,
  jobsFor,
  MANIFEST_HASH,
  makeRun,
  matrixConfig,
  queuedJob,
  REFERENCE,
  RELEASE_HASH,
  runId,
} from './selection-runs.js';

type Run = Record<string, unknown>;

const dossier = (runs: Run[], jobs: Run[] = jobsFor(runs), excluded: unknown[] = []) =>
  buildCampaignDossier(REFERENCE, { runs, excluded }, jobs);

const rowOf = (result: CampaignDossier, row: number) => {
  const found = result.rows.find((entry) => entry.row === row);
  if (!found) throw new Error(`sin fila ${row}`);
  return found;
};

const attemptOf = (result: CampaignDossier, row: number, id: string) =>
  rowOf(result, row).attempts.find((attempt) => attempt.run_id === runId(id));

/** Campaña completa con la fila 1 sustituida por `row1`. */
const withRow1 = (...row1: Run[]) => [...campaign().slice(1), ...row1];

describe('campaña completa y conciliada', () => {
  it('12 filas aceptadas, un intento cada una, lista para cerrar', () => {
    const result = dossier(campaign());
    expect(result.rows.map((row) => [row.row, row.status])).toEqual(
      Array.from({ length: 12 }, (_, i) => [i + 1, 'accepted']),
    );
    for (const row of result.rows) {
      expect(row.attempts).toHaveLength(1);
      expect(row.attempts[0]).toMatchObject({ role: 'representative', problems: [] });
      expect(row.representative).toBe(row.attempts[0]?.run_id);
      expect(row.config).toEqual(matrixConfig(row.row));
    }
    expect(result.accepted_rows).toHaveLength(12);
    expect(result.min_comparable_runs).toBe(MIN_COMPARABLE_RUNS);
    expect(result.close_blockers).toEqual([]);
    expect(result.ready_to_close).toBe(true);
    expect(result.unattributed).toEqual([]);
  });

  it('decide lo mismo que la selección (D04-04): mismo ranking y mismo outcome_hash', () => {
    const runs = campaign();
    const outcome = selectCandidate(runs, REFERENCE);
    const result = dossier(runs);
    expect(result.ranking).toEqual(outcome.ranking);
    expect(result.outcome_hash).toBe(outcome.outcome_hash);
    expect(result.candidate?.run_id).toBe(outcome.candidate?.run_id);
  });

  it('identidad completa del candidato: run, job, commit, config, checkpoint y procedencia', () => {
    const result = dossier(campaign());
    expect(result.candidate).toEqual({
      run_id: runId('c'),
      campaign_row: 12,
      job_id: 12,
      git_commit: 'c'.repeat(40),
      start_time: '2026-10-01T10:00:00Z',
      config: matrixConfig(12),
      checkpoint_sha256: 'e'.repeat(64),
      dataset_version: 'v0.1.1',
      dvc_release_hash: RELEASE_HASH,
      dvc_images_md5: '22222222222222222222222222222222.dir',
      dvc_annotations_md5: '11111111111111111111111111111111.dir',
      manifest_version: 'p3-v0.1.1-s42',
      manifest_hash: MANIFEST_HASH,
      classes: ['cat', 'dog'],
      best_epoch: 2,
      val_accuracy: 0.811,
      val_macro_f1: 0.85,
      val_loss: 0.38,
    });
  });

  it('el resultado no depende del orden de runs ni de jobs', () => {
    const runs = withRow1(
      makeRun({ id: '1', row: 1, job: 1, start: '2026-10-01T09:00:00Z' }),
      makeRun({ id: '0', row: 1, job: 20, start: '2026-10-01T12:00:00Z' }),
    );
    const a = dossier(runs);
    const b = dossier([...runs].reverse(), jobsFor(runs).reverse());
    expect(b).toEqual(a);
  });
});

describe('representante por fila: cronológico, nunca por métricas', () => {
  it('un retry con mejor val_accuracy NO reemplaza al intento más temprano', () => {
    const result = dossier(
      withRow1(
        makeRun({ id: '1', row: 1, job: 1, acc: 0.7, start: '2026-10-01T09:00:00Z' }),
        makeRun({ id: '0', row: 1, job: 20, acc: 0.99, start: '2026-10-01T12:00:00Z' }),
      ),
    );
    expect(rowOf(result, 1).representative).toBe(runId('1'));
    expect(attemptOf(result, 1, '0')).toMatchObject({
      role: 'retry',
      reason: 'duplicate_campaign_row',
      job_id: 20,
    });
    expect(result.candidate?.run_id).not.toBe(runId('0'));
    // Intentos en orden cronológico, no por run_id.
    expect(rowOf(result, 1).attempts.map((attempt) => attempt.run_id)).toEqual([
      runId('1'),
      runId('0'),
    ]);
    // El retry se documenta, pero no agrega otra configuración.
    expect(result.accepted_rows).toHaveLength(12);
    expect(result.ready_to_close).toBe(true);
  });

  it('empate real de start_time: representa el menor run_id', () => {
    const same = '2026-10-01T09:00:00Z';
    const result = dossier(
      withRow1(
        makeRun({ id: `${'b'.repeat(31)}1`, row: 1, job: 21, acc: 0.99, start: same }),
        makeRun({ id: `${'b'.repeat(31)}0`, row: 1, job: 20, acc: 0.5, start: same }),
      ),
    );
    expect(rowOf(result, 1).representative).toBe(`${'b'.repeat(31)}0`);
  });

  it('un intento fallido anterior se documenta y no ocupa la fila', () => {
    const result = dossier(
      withRow1(
        makeRun({ id: '0', row: 1, job: 20, status: 'FAILED', start: '2026-10-01T08:00:00Z' }),
        makeRun({ id: '1', row: 1, job: 1, start: '2026-10-01T09:00:00Z' }),
      ),
    );
    expect(rowOf(result, 1)).toMatchObject({ status: 'accepted', representative: runId('1') });
    expect(attemptOf(result, 1, '0')).toMatchObject({
      role: 'excluded',
      reason: 'not_finished',
      job_status: 'failed',
      problems: [],
    });
    // Orden cronológico dentro de la fila.
    expect(rowOf(result, 1).attempts.map((attempt) => attempt.run_id)).toEqual([
      runId('0'),
      runId('1'),
    ]);
    expect(result.ready_to_close).toBe(true);
  });

  it('un job fallido antes de crear su run es un intento documentado, sin run', () => {
    const runs = campaign();
    const failed = {
      ...queuedJob(30, 2),
      status: 'failed',
      error: 'OOM al cargar el dataset',
      started_at: '2026-10-01T12:00:30Z',
      finished_at: '2026-10-01T12:01:00Z',
    };
    const result = dossier(runs, [...jobsFor(runs), failed]);
    expect(rowOf(result, 2).attempts).toContainEqual(
      expect.objectContaining({
        job_id: 30,
        run_id: null,
        role: 'excluded',
        reason: 'job_failed',
        detail: 'OOM al cargar el dataset',
      }),
    );
    expect(rowOf(result, 2).status).toBe('accepted');
    expect(result.ready_to_close).toBe(true);
  });
});

describe('exclusiones documentadas en su fila', () => {
  it.each([
    [
      'métrica de test en el resumen',
      () => {
        const run = makeRun({ id: '0', row: 3, job: 3 });
        (run.summary as Record<string, unknown>).test_accuracy = 0.99;
        return run;
      },
      'invalid_contract',
    ],
    [
      'otro manifest',
      () => makeRun({ id: '0', row: 3, job: 3, tags: { manifest_hash: 'e'.repeat(64) } }),
      'manifest_mismatch',
    ],
    ['cancelado', () => makeRun({ id: '0', row: 3, job: 3, status: 'KILLED' }), 'not_finished'],
    [
      'sin checkpoint verificado',
      () => makeRun({ id: '0', row: 3, job: 3, checkpoint: null }),
      'not_campaign_eligible',
    ],
  ])(
    '%s: queda en la fila con su motivo y la fila queda sin representante',
    (_n, build, reason) => {
      const runs = [...campaign().filter((_, i) => i !== 2), build()];
      const jobs = runs.map((run) =>
        // El job conserva el request de la fila aunque el run no se pueda leer.
        run.run_id === runId('0') ? jobFor(run, { config: matrixConfig(3) }) : jobFor(run),
      );
      const result = dossier(runs, jobs);
      expect(attemptOf(result, 3, '0')).toMatchObject({ role: 'excluded', reason });
      expect(rowOf(result, 3)).toMatchObject({ status: 'missing', representative: null });
      expect(result.accepted_rows).toHaveLength(11);
      expect(result.ready_to_close).toBe(true); // 11 ≥ 10
    },
  );

  it('un smoke (fuera de la matriz) no entra a ninguna fila: queda sin atribuir', () => {
    const smoke = makeRun({ id: '0', row: 1, job: 40, config: { max_epochs: 10 } });
    const runs = [...campaign(), smoke];
    const result = dossier(runs);
    expect(result.rows.flatMap((row) => row.attempts.map((a) => a.run_id))).not.toContain(
      runId('0'),
    );
    expect(result.unattributed).toContainEqual(
      expect.objectContaining({ run_id: runId('0'), reason: 'outside_campaign_matrix' }),
    );
    expect(result.ready_to_close).toBe(true);
  });

  it('los jobs controlled (D02-05) no son intentos de la campaña', () => {
    const runs = campaign();
    const controlled = { ...queuedJob(41, 1), task: 'controlled' };
    const result = dossier(runs, [...jobsFor(runs), controlled]);
    expect(JSON.stringify(result)).not.toContain('"job_id":41');
    expect(result.ready_to_close).toBe(true);
  });

  it('un run que D04-01 excluyó (p. ej. sin procedencia) se lista con sus motivos', () => {
    const runs = campaign().filter((_, i) => i !== 2);
    const adapterExcluded = {
      run_id: runId('0'),
      run_kind: 'training',
      status: 'FINISHED',
      start_time: '2026-10-01T10:00:00Z',
      reasons: ['falta el tag dvc_release_hash'],
    };
    const job = jobFor(makeRun({ id: '0', row: 3, job: 3 }));
    const auxiliary = { ...adapterExcluded, run_id: runId('f'), run_kind: 'persistence_check' };
    const result = dossier(runs, [...jobsFor(runs), job], [adapterExcluded, auxiliary]);
    expect(attemptOf(result, 3, '0')).toMatchObject({
      role: 'excluded',
      reason: 'adapter_excluded',
      detail: 'falta el tag dvc_release_hash',
      problems: [],
    });
    expect(result.unattributed).toContainEqual(
      expect.objectContaining({ run_id: runId('f'), reason: 'adapter_excluded' }),
    );
  });
});

describe('qué impide cerrar el conteo', () => {
  it(`con menos de ${MIN_COMPARABLE_RUNS} filas aceptadas hay candidato pero no se cierra`, () => {
    const result = dossier(campaign(MIN_COMPARABLE_RUNS - 1));
    expect(result.candidate).not.toBeNull();
    expect(result.rows.filter((row) => row.status === 'missing').map((row) => row.row)).toEqual([
      10, 11, 12,
    ]);
    expect(result.ready_to_close).toBe(false);
    expect(result.close_blockers.join(' ')).toMatch(/9 filas aceptadas/);
  });

  it('un job de la matriz en cola deja su fila pendiente y bloquea, aunque haya 12 filas', () => {
    const runs = campaign();
    const result = dossier(runs, [...jobsFor(runs), queuedJob(31, 4)]);
    expect(rowOf(result, 4).status).toBe('pending');
    expect(rowOf(result, 4).attempts).toContainEqual(
      expect.objectContaining({ job_id: 31, role: 'pending', run_id: null }),
    );
    expect(result.ready_to_close).toBe(false);
    expect(result.close_blockers.join(' ')).toMatch(/pendiente/);
  });

  it('un run de la matriz en curso deja su fila pendiente y bloquea', () => {
    const runs = [...campaign(), makeRun({ id: '0', row: 5, job: 32, status: 'RUNNING' })];
    const result = dossier(runs);
    expect(rowOf(result, 5).status).toBe('pending');
    expect(attemptOf(result, 5, '0')).toMatchObject({ role: 'pending', job_status: 'running' });
    expect(result.ready_to_close).toBe(false);
  });

  it.each([
    [
      'request del job ≠ config del run',
      (runs: Run[]) =>
        runs.map((run) =>
          run.run_id === runId('5') ? jobFor(run, { config: matrixConfig(6) }) : jobFor(run),
        ),
      /config/,
    ],
    [
      'manifest del job ≠ manifest del run',
      (runs: Run[]) =>
        runs.map((run) =>
          run.run_id === runId('5') ? jobFor(run, { manifest_hash: 'f'.repeat(64) }) : jobFor(run),
        ),
      /manifest/,
    ],
    [
      'release del job ≠ release del run',
      (runs: Run[]) =>
        runs.map((run) =>
          run.run_id === runId('5') ? jobFor(run, { dataset_version: 'v0.1.0' }) : jobFor(run),
        ),
      /release/,
    ],
    [
      'job_id repetido en la fuente',
      (runs: Run[]) => [
        ...jobsFor(runs),
        { ...queuedJob(5, 7), status: 'cancelled', finished_at: '2026-10-01T12:00:00Z' },
      ],
      /repetido/,
    ],
    [
      'el run declara otro job_id',
      (runs: Run[]) =>
        runs.map((run) => (run.run_id === runId('5') ? jobFor(run, { id: 77 }) : jobFor(run))),
      /job_id/,
    ],
    [
      'job failed con un run FINISHED',
      (runs: Run[]) =>
        runs.map((run) =>
          run.run_id === runId('5') ? jobFor(run, { status: 'failed', error: 'x' }) : jobFor(run),
        ),
      /estado/,
    ],
    [
      'run sin job que lo registre',
      (runs: Run[]) => jobsFor(runs).filter((job) => job.mlflow_run_id !== runId('5')),
      /ningún job/,
    ],
  ])('no concilia (%s): problema en el intento y bloqueo del cierre', (_n, jobsOf, message) => {
    const runs = campaign();
    const result = dossier(runs, jobsOf(runs));
    // El intento queda en la fila de lo que se entrenó (la config del run), no del request.
    const attempt = attemptOf(result, 5, '5');
    expect(attempt).toBeDefined();
    expect(attempt?.problems.join(' ') ?? '').toMatch(message);
    expect(result.ready_to_close).toBe(false);
    expect(result.close_blockers.join(' ')).toMatch(/concilia/);
  });

  it('un job succeeded cuyo run no está en MLflow bloquea el cierre', () => {
    const runs = campaign();
    const lost = jobFor(makeRun({ id: '0', row: 6, job: 33 }));
    const result = dossier(runs, [...jobsFor(runs), lost]);
    expect(rowOf(result, 6).attempts).toContainEqual(
      expect.objectContaining({ job_id: 33, run_id: runId('0') }),
    );
    expect(
      rowOf(result, 6)
        .attempts.flatMap((attempt) => attempt.problems)
        .join(' '),
    ).toMatch(/no está en MLflow/);
    expect(result.ready_to_close).toBe(false);
  });

  it('un job fuera de contrato no se ignora: queda sin atribuir y bloquea', () => {
    const runs = campaign();
    const result = dossier(runs, [...jobsFor(runs), { id: 34, task: 'training' }]);
    expect(result.unattributed).toContainEqual(
      expect.objectContaining({ job_id: 34, reason: 'invalid_contract' }),
    );
    expect(result.ready_to_close).toBe(false);
  });

  it('sin runs elegibles no hay candidato', () => {
    const runs = [makeRun({ id: '0', row: 1, status: 'FAILED' })];
    const result = dossier(runs);
    expect(result.candidate).toBeNull();
    expect(result.ranking).toEqual([]);
    expect(result.ready_to_close).toBe(false);
  });
});
