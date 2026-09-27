/**
 * D01-05 — Lectura de los fixtures compartidos `contracts/p3/fixtures`.
 * No termina en `.test.ts`, así que Vitest no lo recolecta como suite.
 */
import fs from "node:fs";
import path from "node:path";

export const FIXTURES_DIR = path.resolve(__dirname, "../../contracts/p3/fixtures");

export interface ContractFixture {
  file: string;
  contract: string;
  valid: boolean;
  why: string;
  payload: unknown;
}

export function loadAllFixtures(): ContractFixture[] {
  return fs
    .readdirSync(FIXTURES_DIR)
    .sort()
    .flatMap((contract) =>
      fs
        .readdirSync(path.join(FIXTURES_DIR, contract))
        .filter((name) => name.endsWith(".json"))
        .sort()
        .map((name) => {
          const raw = JSON.parse(
            fs.readFileSync(path.join(FIXTURES_DIR, contract, name), "utf8")
          ) as Omit<ContractFixture, "file">;
          return { file: `${contract}/${name}`, ...raw };
        })
    );
}

/** Payload de `contracts/p3/fixtures/<contract>/<name>.json`. */
export function fixturePayload(contract: string, name: string): unknown {
  const raw = JSON.parse(
    fs.readFileSync(path.join(FIXTURES_DIR, contract, `${name}.json`), "utf8")
  ) as ContractFixture;
  return raw.payload;
}
