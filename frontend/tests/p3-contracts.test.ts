/**
 * D01-05 — Contratos P3 (frontend). Mismos fixtures que el backend
 * (`backend/tests/p3-contracts.test.ts`): si el espejo del portal diverge del
 * esquema de la API, este test o el del backend falla.
 */
import { describe, expect, it } from "vitest";
import { P3_CONTRACTS } from "../src/p3/contracts";
import { loadAllFixtures } from "./p3-fixtures";

const fixtures = loadAllFixtures();

describe("fixtures compartidos de contratos P3 (portal)", () => {
  it("el portal tiene esquema para cada contrato con fixtures", () => {
    const contracts = new Set(fixtures.map((fixture) => fixture.contract));
    expect([...contracts].sort()).toEqual(Object.keys(P3_CONTRACTS).sort());
  });

  it.each(fixtures.map((fixture) => [fixture.file, fixture] as const))("%s", (_file, fixture) => {
    const schema = P3_CONTRACTS[fixture.contract as keyof typeof P3_CONTRACTS];
    expect(schema, `contrato desconocido: ${fixture.contract}`).toBeDefined();
    expect(schema.safeParse(fixture.payload).success, fixture.why).toBe(fixture.valid);
  });
});
