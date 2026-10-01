// D03-06: el arranque limpio encontró una imagen del backend sin el binario de
// `tsx/node_modules/esbuild` (npm omite en silencio un opcional que no pudo descargar).
// `drizzle-kit migrate` y `db:seed` usan tsx, así que la imagen se construía "bien" y el
// backend moría al arrancar. `docker/check-esbuild.mjs` corre en el build y lo detecta.
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
// @ts-expect-error módulo .mjs sin tipos; se ejecuta tal cual en el Dockerfile
import { findMissingEsbuildBinaries } from '../docker/check-esbuild.mjs';

const PLATFORM = '@esbuild/linux-x64';
let root: string;

function pkg(dir: string, manifest: Record<string, unknown>) {
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, 'package.json'), JSON.stringify(manifest));
}

function esbuild(at: string, version: string) {
  pkg(join(root, at), {
    name: 'esbuild',
    version,
    optionalDependencies: { [PLATFORM]: version, '@esbuild/darwin-arm64': version },
  });
}

function binary(at: string, version: string) {
  pkg(join(root, at), { name: PLATFORM, version });
}

function check() {
  return findMissingEsbuildBinaries(join(root, 'node_modules'), PLATFORM);
}

afterEach(() => rmSync(root, { recursive: true, force: true }));

describe('check-esbuild: cada esbuild instalado tiene su binario de plataforma', () => {
  it('árbol completo (raíz + anidados) → nada falta', () => {
    root = mkdtempSync(join(tmpdir(), 'esb-'));
    esbuild('node_modules/esbuild', '0.25.12');
    binary('node_modules/@esbuild/linux-x64', '0.25.12');
    esbuild('node_modules/tsx/node_modules/esbuild', '0.28.2');
    binary('node_modules/tsx/node_modules/@esbuild/linux-x64', '0.28.2');
    expect(check()).toEqual([]);
  });

  it('falta el binario anidado de tsx (el caso real del arranque limpio) → se reporta', () => {
    root = mkdtempSync(join(tmpdir(), 'esb-'));
    esbuild('node_modules/esbuild', '0.25.12');
    binary('node_modules/@esbuild/linux-x64', '0.25.12');
    esbuild('node_modules/tsx/node_modules/esbuild', '0.28.2');
    expect(check()).toEqual([
      expect.objectContaining({
        esbuild: join('node_modules', 'tsx', 'node_modules', 'esbuild'),
        version: '0.28.2',
      }),
    ]);
  });

  it('el binario de la raíz con otra versión no cubre al esbuild anidado', () => {
    root = mkdtempSync(join(tmpdir(), 'esb-'));
    esbuild('node_modules/esbuild', '0.25.12');
    binary('node_modules/@esbuild/linux-x64', '0.25.12');
    esbuild('node_modules/tsx/node_modules/esbuild', '0.28.2');
    // Node resolvería subiendo hasta node_modules/@esbuild/linux-x64 (0.25.12): esbuild
    // 0.28.2 lo rechaza en tiempo de ejecución por versión distinta.
    expect(check()).toEqual([expect.objectContaining({ version: '0.28.2', found: '0.25.12' })]);
  });

  it('un esbuild anidado sin copia propia usa el binario de un node_modules superior con la misma versión', () => {
    root = mkdtempSync(join(tmpdir(), 'esb-'));
    esbuild('node_modules/esbuild', '0.25.12');
    esbuild('node_modules/drizzle-kit/node_modules/esbuild', '0.25.12');
    binary('node_modules/@esbuild/linux-x64', '0.25.12');
    expect(check()).toEqual([]);
  });

  it('falta el binario de la raíz → se reporta', () => {
    root = mkdtempSync(join(tmpdir(), 'esb-'));
    esbuild('node_modules/esbuild', '0.25.12');
    expect(check()).toEqual([
      expect.objectContaining({ esbuild: join('node_modules', 'esbuild'), found: null }),
    ]);
  });

  it('sin ningún esbuild instalado → error (no un falso "todo bien")', () => {
    root = mkdtempSync(join(tmpdir(), 'esb-'));
    mkdirSync(join(root, 'node_modules'), { recursive: true });
    expect(() => check()).toThrow(/ningún esbuild/);
  });
});
