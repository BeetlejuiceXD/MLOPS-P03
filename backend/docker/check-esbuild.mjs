// D03-06: comprueba que cada esbuild instalado tenga el binario de su plataforma.
//
// Con `npm ci --ignore-scripts`, el binario llega como dependencia opcional
// (`@esbuild/<plataforma>`). Si su descarga falla, npm la omite EN SILENCIO y la imagen
// queda rota hasta que algo usa esbuild: `drizzle-kit migrate` y `db:seed` usan tsx,
// que trae su propio esbuild anidado. Este script corre en el build (ver Dockerfile) y
// termina con 1 si falta alguno, en vez de dejar que el backend muera al arrancar.
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { dirname, join, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

function readJson(path) {
  return JSON.parse(readFileSync(path, 'utf8'));
}

/** Directorios `.../node_modules/esbuild` (raíz y anidados) bajo `nodeModules`. */
function findEsbuildDirs(nodeModules) {
  const found = [];
  const walk = (dir) => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      if (!entry.isDirectory() || entry.name.startsWith('.')) continue;
      const path = join(dir, entry.name);
      if (entry.name.startsWith('@')) {
        walk(path);
        continue;
      }
      if (entry.name === 'esbuild' && existsSync(join(path, 'package.json'))) {
        if (readJson(join(path, 'package.json')).name === 'esbuild') found.push(path);
      }
      const nested = join(path, 'node_modules');
      if (existsSync(nested)) walk(nested);
    }
  };
  walk(nodeModules);
  return found;
}

/**
 * Resuelve `pkg` desde un esbuild instalado como lo haría Node: busca en cada
 * node_modules ancestro, del más cercano a `stopAt` (el node_modules raíz).
 */
function resolveFrom(esbuildDir, pkg, stopAt) {
  for (let dir = dirname(esbuildDir); ; dir = dirname(dir)) {
    if (dir.endsWith(`${sep}node_modules`)) {
      const candidate = join(dir, pkg, 'package.json');
      if (existsSync(candidate)) return readJson(candidate);
    }
    if (dir === stopAt || dirname(dir) === dir) return null;
  }
}

/**
 * Lista los esbuild sin binario utilizable para `platformPkg` (p. ej. `@esbuild/linux-x64`).
 * `found` es la versión que Node resolvería (null si ninguna). esbuild exige la misma
 * versión exacta, así que un binario de otra versión en un node_modules superior no cuenta.
 */
export function findMissingEsbuildBinaries(nodeModules, platformPkg) {
  const dirs = findEsbuildDirs(nodeModules);
  if (dirs.length === 0) {
    throw new Error(`No hay ningún esbuild instalado bajo ${nodeModules}`);
  }
  const base = dirname(nodeModules);
  const missing = [];
  for (const dir of dirs) {
    const { version, optionalDependencies = {} } = readJson(join(dir, 'package.json'));
    if (!(platformPkg in optionalDependencies)) continue; // esbuild no publica esta plataforma
    const resolved = resolveFrom(dir, platformPkg, nodeModules);
    if (resolved?.version === version) continue;
    missing.push({ esbuild: relative(base, dir), version, found: resolved?.version ?? null });
  }
  return missing;
}

/** Nombre del paquete de binario que usa esbuild en esta máquina (Linux glibc/musl x64/arm64). */
export function currentPlatformPackage() {
  return `@esbuild/${process.platform}-${process.arch}`;
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  const nodeModules = join(process.cwd(), 'node_modules');
  const platformPkg = currentPlatformPackage();
  const missing = findMissingEsbuildBinaries(nodeModules, platformPkg);
  if (missing.length > 0) {
    for (const m of missing) {
      console.error(
        `[check-esbuild] ${m.esbuild}@${m.version}: falta ${platformPkg}@${m.version} (Node resolvería: ${m.found ?? 'nada'})`,
      );
    }
    process.exit(1);
  }
  const count = findEsbuildDirs(nodeModules).length;
  console.log(`[check-esbuild] OK: ${count} esbuild con ${platformPkg}`);
}
