/**
 * D01-05 — Contratos P3 (backend). Valida cada fixture compartido de
 * `contracts/p3/fixtures` con el esquema Zod del backend. El frontend corre el
 * mismo recorrido con su espejo (`frontend/tests/p3-contracts.test.ts`), así que
 * si los dos esquemas divergen, uno de los dos tests falla.
 */
import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { P3_CONTRACTS, P3_ENDPOINTS, trainingConfigSchema } from '../src/logic/p3.contracts.js';

const FIXTURES_DIR = path.resolve('../contracts/p3/fixtures');

interface Fixture {
  file: string;
  contract: string;
  valid: boolean;
  why: string;
  payload: unknown;
}

function loadFixtures(): Fixture[] {
  return fs
    .readdirSync(FIXTURES_DIR)
    .sort()
    .flatMap((contract) =>
      fs
        .readdirSync(path.join(FIXTURES_DIR, contract))
        .filter((name) => name.endsWith('.json'))
        .sort()
        .map((name) => {
          const raw = JSON.parse(fs.readFileSync(path.join(FIXTURES_DIR, contract, name), 'utf8'));
          return { file: `${contract}/${name}`, ...raw } as Fixture;
        }),
    );
}

const fixtures = loadFixtures();

describe('fixtures compartidos de contratos P3', () => {
  it('cada contrato del esquema tiene fixtures válidos e inválidos, y viceversa', () => {
    const contractsWithFixtures = new Set(fixtures.map((fixture) => fixture.contract));
    expect([...contractsWithFixtures].sort()).toEqual(Object.keys(P3_CONTRACTS).sort());
    for (const contract of Object.keys(P3_CONTRACTS)) {
      const own = fixtures.filter((fixture) => fixture.contract === contract);
      expect(
        own.some((fixture) => fixture.valid),
        `${contract} sin fixture válido`,
      ).toBe(true);
      expect(
        own.some((fixture) => !fixture.valid),
        `${contract} sin fixture inválido`,
      ).toBe(true);
    }
  });

  it.each(fixtures.map((fixture) => [fixture.file, fixture] as const))('%s', (_file, fixture) => {
    const schema = P3_CONTRACTS[fixture.contract as keyof typeof P3_CONTRACTS];
    expect(schema, `contrato desconocido: ${fixture.contract}`).toBeDefined();
    const result = schema.safeParse(fixture.payload);
    expect(result.success, fixture.why).toBe(fixture.valid);
  });
});

describe('TrainingConfig congelado en #33', () => {
  it('no aplica defaults silenciosos: lo validado es lo que se registra', () => {
    expect(trainingConfigSchema.safeParse({}).success).toBe(false);
  });
});

describe('catálogo de endpoints', () => {
  it('cubre las cinco páginas bajo /api y sin duplicados', () => {
    const routes = Object.values(P3_ENDPOINTS);
    expect(new Set(routes).size).toBe(routes.length);
    for (const route of routes) {
      expect(route).toMatch(/^(GET|POST) \/api\//);
    }
    for (const needed of [
      'GET /api/releases',
      'POST /api/training/jobs',
      'GET /api/experiments/runs',
      'GET /api/evaluation',
      'GET /api/models',
      'POST /api/inference',
    ]) {
      expect(routes).toContain(needed);
    }
  });
});
