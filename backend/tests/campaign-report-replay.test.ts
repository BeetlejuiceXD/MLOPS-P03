/**
 * D05-02 sobre la evidencia REAL de D04-03 (`reports/campaign_p3.json`, PR #82) y el
 * manifest congelado (`reports/manifest_p3.json`), contrastada con la respuesta de la API
 * sobre el MLflow real (`reports/selection_p3/`). Solo métricas de validation: ninguno de
 * estos archivos trae ni puede traer el frozen test.
 */
import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { type CampaignReport, replayCampaignReport } from '../src/cli/replay-campaign-report.js';

const REPORTS = path.resolve('../reports');
const read = (name: string) => JSON.parse(fs.readFileSync(path.join(REPORTS, name), 'utf8'));
const evidence = (): CampaignReport => read('campaign_p3.json');
const manifest = () => read('manifest_p3.json');
const api = (name: string) => read(`selection_p3/${name}`);

// Inventario de D04-07 (#76), filas 2–12: fila → job → run del representante. La fila 1
// tiene varios intentos con la misma config; su representante es el de menor start_time
// entre los que trae el reporte (ver el test de la fila 1).
const INVENTORY: [number, number, string][] = [
  [2, 6, 'a09c7b6e5102491f9dfc2391b88bb9b8'],
  [3, 7, '2d56233c886142b7824e1551b90e8327'],
  [4, 8, '379b553913254241a2efece3ed451d2d'],
  [5, 9, 'c2adc19a9fb94b6999937289bf2a1f2c'],
  [6, 10, '6646544b73734ac1abdf413d1b63b3d9'],
  [7, 11, 'a8b7545b465b465990fe7e7c594c1798'],
  [8, 12, '5fe9d9d9ca46440b9e591daff45acf35'],
  [9, 13, '02397ddadec047809ac0dd2421aeff7b'],
  [10, 14, '438cc06bdf32461fa2982963e8be3727'],
  [11, 15, '33715e5e663e4ace82265b3d05b48158'],
  [12, 16, 'bd4ee53765bc4a9c96ff3cb3b1f5330d'],
];

describe('D05-02 · expediente sobre la campaña real de D04-03', () => {
  const result = replayCampaignReport(evidence(), manifest());

  it('concilia las 12 filas con el inventario de D04-07 y deja la campaña lista', () => {
    expect(result.attempts).toBe(evidence().results.length);
    expect(result.accepted_rows).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    expect(result.close_blockers).toEqual([]);
    expect(result.unattributed).toEqual([]);
    expect(result.ready_to_close).toBe(true);
    expect(result.matches_selection).toBe(true);
    for (const [row, job, run] of INVENTORY) {
      const representative = result.rows
        .find((entry) => entry.row === row)
        ?.attempts.find((attempt) => attempt.role === 'representative');
      expect({ row, job: representative?.job_id, run: representative?.run_id }).toEqual({
        row,
        job,
        run,
      });
    }
  });

  it('fila 1: representa el intento más temprano; los demás son retry y no agregan filas', () => {
    const reported = evidence()
      .results.filter((attempt) => attempt.row === 1)
      .sort((a, b) => a.details.start_time - b.details.start_time);
    expect(reported.length).toBeGreaterThan(1);
    const attempts = result.rows.find((entry) => entry.row === 1)?.attempts ?? [];
    expect(attempts.map((a) => [a.run_id, a.role])).toEqual(
      reported.map((attempt, i) => [attempt.report.run_id, i === 0 ? 'representative' : 'retry']),
    );
    expect(result.ranking.filter((entry) => entry.campaign_row === 1)).toHaveLength(1);
  });

  it('coincide con GET /selection/campaign y la propuesta sobre el MLflow real', () => {
    const dossier = api('2-expediente.json');
    const proposal = api('3-propuesta.json');
    expect(dossier).toMatchObject({
      ready_to_close: true,
      close_blockers: [],
      unattributed: [],
      matches_proposal: true,
      accepted_rows: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    });
    expect(dossier.reference).toEqual(result.reference);
    expect(proposal.status).toBe('candidate');
    expect(proposal.outcome_hash).toBe(dossier.outcome_hash);
    // Misma decisión: candidato con identidad completa y mismo orden y métricas por fila.
    expect(dossier.candidate).toEqual(result.candidate);
    expect(proposal.candidate.run_id).toBe(result.candidate?.run_id);
    const byRow = (ranking: { campaign_row: number; val_accuracy: number }[]) =>
      ranking.map(({ campaign_row, val_accuracy }) => [campaign_row, val_accuracy]);
    expect(byRow(dossier.ranking)).toEqual(byRow(result.ranking));
    // La API ve los 5 intentos de la fila 1 y elige el más temprano (job 1).
    const row1 = dossier.rows.find((entry: { row: number }) => entry.row === 1);
    expect(row1.attempts.map((a: { job_id: number; role: string }) => [a.job_id, a.role])).toEqual([
      [1, 'representative'],
      [2, 'retry'],
      [3, 'retry'],
      [4, 'retry'],
      [5, 'retry'],
    ]);
  });

  it('con la selección en candidate el test sigue bloqueado', () => {
    expect(api('4-estado-final.json')).toMatchObject({ status: 'candidate', closed_at: null });
    expect(api('5-evaluation.json')).toMatchObject({
      state: 'blocked',
      reason: 'model_selection_open',
    });
  });

  it('ranking validation-only: empate a 4 decimales en accuracy y F1 lo decide el val_loss', () => {
    const top = result.ranking.slice(0, 3).map((entry) => ({
      row: entry.campaign_row,
      acc: entry.val_accuracy.toFixed(4),
      f1: entry.val_macro_f1.toFixed(4),
      loss: entry.val_loss.toFixed(4),
    }));
    expect(top).toEqual([
      { row: 3, acc: '0.9552', f1: '0.9552', loss: '0.1317' },
      { row: 2, acc: '0.9552', f1: '0.9552', loss: '0.1498' },
      { row: 5, acc: '0.9552', f1: '0.9551', loss: '0.1351' },
    ]);
    expect(result.ranking).toHaveLength(12);
  });

  it('candidato con identidad completa (fila 3, learning_rate 3e-4)', () => {
    expect(result.candidate).toMatchObject({
      run_id: '2d56233c886142b7824e1551b90e8327',
      campaign_row: 3,
      job_id: 7,
      git_commit: 'bb7deb5da3c5d7f5b44edfdf36659c8945e34168',
      checkpoint_sha256: '0c6b589bdd8ba639ed6890386db5bdd20555adc3452df7e9657c2b4a4b9d563b',
      dataset_version: 'v0.1.1',
      dvc_release_hash: '2e7029bd794da00c9962263bd42f4e4fa94bba25e2350e4cf934505e98f937e8',
      manifest_hash: '0c03c3951554b5dbf096c37468f0fb5f04e9c62c033604acabf366fb11375c43',
      classes: ['cat', 'dog'],
      best_epoch: 5,
      config: { learning_rate: 0.0003, optimizer: 'adam', seed: 7, max_epochs: 30 },
    });
    expect(result.outcome_hash).toMatch(/^[0-9a-f]{64}$/);
  });

  it('el resultado no depende del orden del reporte', () => {
    const reversed = { results: [...evidence().results].reverse() };
    const again = replayCampaignReport(reversed, manifest());
    expect(again.outcome_hash).toBe(result.outcome_hash);
    expect(again.candidate?.run_id).toBe(result.candidate?.run_id);
  });

  it('contra otro manifest no acepta ninguna fila ni cierra', () => {
    const other = { ...manifest(), manifest_hash: 'f'.repeat(64) };
    const replay = replayCampaignReport(evidence(), other);
    expect(replay.accepted_rows).toEqual([]);
    expect(replay.ready_to_close).toBe(false);
    expect(replay.candidate).toBeNull();
  });

  it('con solo 9 filas de la evidencia no se puede cerrar', () => {
    const nine = { results: evidence().results.filter((attempt) => attempt.row <= 9) };
    const replay = replayCampaignReport(nine, manifest());
    expect(replay.accepted_rows).toHaveLength(9);
    expect(replay.ready_to_close).toBe(false);
    expect(replay.close_blockers.join(' ')).toMatch(/10/);
  });

  it('rechaza un manifest que no está congelado', () => {
    expect(() => replayCampaignReport(evidence(), { ...manifest(), frozen: false })).toThrow(
      /congelado/,
    );
  });
});
