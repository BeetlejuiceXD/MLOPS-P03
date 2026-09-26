# Cómo contribuir

Guía corta para trabajar en equipo sin pisarnos. Lo que aquí se pide lo revisa el CI
(`.github/workflows/`), así que si algo se pone en rojo, casi siempre el mensaje dice qué
corregir.

## Flujo en cinco pasos

1. **Un ticket = una rama = un PR.** No mezcles tickets en una misma rama salvo que el
   propio ticket los agrupe (por ejemplo `D02-03 D02-04`).
2. **Parte de `main` actualizado:**
   ```bash
   git fetch origin
   git checkout -b dNN-NN-descripcion origin/main
   ```
3. **Haz commits pequeños** con el formato de abajo.
4. **Corre en local lo mismo que el CI** (tabla más abajo) antes de subir.
5. **Abre el PR** con el título en formato y la plantilla completa. Si el CI está en rojo,
   no se mergea.

## Ramas

`dNN-NN-descripcion` (día y ticket del issue diario, p. ej. `D01-06` → `d01-06-...`),
todo en minúsculas y con guiones.

| Caso | Ejemplo |
|---|---|
| Un ticket | `d01-06-ci-exclusions` |
| Varios tickets | `d02-03-d02-04-trainer-mlflow` |
| Sin ticket | `fix/annotation-id-collisions`, `chore/gitignore-node-modules` |

Los prefijos válidos para ramas sin ticket son `feat`, `fix`, `test`, `chore`, `docs`,
`refactor`, `ci`, `build`, `perf` y `style`.

## Commits

```
D01-06: qué cambia, en una línea
fix: qué se arregla, en una línea
```

- Formato `DNN-NN: descripción` cuando hay ticket, o `tipo: descripción` cuando no
  (mismos tipos que en las ramas).
- Primera línea de unos 72 caracteres, sin punto final. Español o inglés, como prefieras.
- Si hace falta, deja una línea en blanco y explica **por qué** en el cuerpo: el qué ya
  lo dice el diff.
- Un commit = un cambio con sentido. Evita `wip`, `arreglos` o `cambios varios`.
- **Nunca** subas `.env`, claves, credenciales de AWS ni datos: `.env` está ignorado y los
  datos van por DVC.

Esta guía no se verifica commit por commit; sí se verifica el nombre de la rama y el título
del PR (siguiente sección).

## Pull requests

- **Título** con el mismo formato que un commit: `D01-06: CI efectiva y exclusiones`,
  `D02-03 D02-04: trainer y MLflow` o `fix: main tiene un test roto`. Ojo con
  `D01-06 - texto`, `D01 06 texto` o `D01-06 texto`: no cumplen.
- **Descripción:** completa la plantilla. Di qué probaste **y qué no pudiste probar**.
- **Cierra el issue** con `Cierra #NN` en la descripción.
- Si tu PR **depende de otro sin mergear**, apúntalo a la rama de ese ticket en vez de
  `main`, y cuando el padre se mergee, reapunta el tuyo a `main`.
- Antes de mergear, actualiza tu rama con `main` si otro PR entró mientras trabajabas, y
  espera a que el CI vuelva a pasar. Dos PR que pasan por separado pueden romper `main`
  juntos: pasó con #120 y #121 (ver #131).

Si el título o la rama no cumplen, el job **Nombre de rama y título del PR** falla. Para el
título basta editarlo en GitHub: el job se vuelve a correr solo. Para comprobarlo antes de
abrir el PR:

```bash
bash .github/scripts/check-naming.sh "$(git branch --show-current)" "DNN-NN: mi título"
```

## Qué corre el CI

Se ejecuta en cada PR (contra cualquier rama base) y en cada push a `main`. Estos son los
mismos comandos que puedes correr en tu máquina:

| Paquete | Comandos (desde su carpeta) |
|---|---|
| `app/` (Python) | `uv sync --locked --no-build` · `uv run --locked --no-build ruff check .` · `uv run --locked --no-build ruff format --check .` · `uv run --locked --no-build pytest -q` · `uv run --locked --no-build python ../.github/scripts/run_gate_mutation.py` |
| `backend/` | `npm ci --ignore-scripts` · `npm run lint` · `npm run typecheck` · `npm test` · `npm run build` |
| `frontend/` | `npm ci --ignore-scripts` · `npm run lint` · `npm run typecheck` · `npm test` · `npm run build` |
| `docker-compose.yml` | `docker compose config --quiet` (con `MINIO_ROOT_USER` y `MINIO_ROOT_PASSWORD` definidas) |

`--no-build` (uv) y `--ignore-scripts` (npm) evitan ejecutar scripts de instalación o de
compilación de dependencias de terceros: lo pide SonarCloud y `app/Dockerfile` ya lo hacía. Si
una dependencia nueva solo se publica como código fuente, o necesita un script de instalación,
el CI fallará a propósito para que el equipo lo revise y decida.

Para arreglar de golpe lo que Ruff o Biome pueden corregir solos: `uv run ruff check --fix .`
y `uv run ruff format .` en `app/`; `npm run lint:fix` en `backend/` y `frontend/`.

`CI OK` es el check que resume a todos los demás. Si un job falla, `CI OK` también falla.

### Higiene del repositorio

`app/tests/test_repo_hygiene.py` corre dentro de `pytest` y falla si `.gitignore` deja de
excluir secretos (`.env`, `.aws/`, `.dvc/config.local`), stores de MLflow (`mlruns/`,
`mlartifacts/`), pesos (`*.pt`, `*.pth`, `*.ckpt`, `*.onnx`, `*.h5`, `*.keras`,
`*.safetensors`) o `data/crops/`; si se versiona un peso, un archivo de más de 1 MiB o algo
con forma de clave (AWS, llave privada, Anthropic, GitHub); o si un workflow usa
`continue-on-error`, `|| true` o `CI OK` deja de esperar a todos los jobs. Los pesos van a
MLflow/S3 y los datos por DVC.

`.gitignore` no impide `git add -f`, así que `app/tests/test_tracked_files.py` revisa el
índice real (`git ls-files`) con las reglas de `app/tests/_repo_hygiene_rules.py`: falla si
se versiona `.env`, `.aws/`, `.dvc/config.local`, contenido de `mlruns/`, `mlartifacts/`,
`data/crops/` o de `data/raw/images|annotations/`, entornos o estado de Terraform. Siguen
permitidos `.env.example`, lockfiles y punteros `.dvc`.

### Mutation test del quality gate

`.github/scripts/run_gate_mutation.py` sale con `0` solo si, con el mutante, pytest termina
con tests **fallidos** (exit 1, sin errores). Sale con `1` si la mutación sobrevive y con `2`
ante cualquier error: objetivo no encontrado, tests en rojo sin mutar, o un error de pytest
(colección, import, fixture). Un error de pytest nunca cuenta como mutación detectada.

### Ruff

La configuración vive en `app/pyproject.toml`. Reglas activas: `E`, `F` (errores y
pyflakes), `I` (orden de imports), `B` (bugbear: bugs probables), `SIM` (simplificaciones),
`C4` (comprensiones y literales) y `RUF` (reglas propias de Ruff). Si de verdad necesitas
silenciar una regla en una línea, usa `# noqa: CODIGO` con el motivo al lado; Ruff (`RUF100`)
avisa cuando un `noqa` ya no hace falta.

### Los reportes commiteados también se validan

`app/tests/test_committed_reports.py` comprueba que los JSON reales de `reports/`
(`quality.json`, `projections.json`, `versions.json` y cada release) cumplan los contratos y
que la versión de cada release coincida con el catálogo. Si regeneras un reporte y lo
commiteas roto, este test lo detecta antes del merge.

## Finales de línea en Windows

El repo guarda LF y `.gitattributes` pide LF también en tu carpeta de trabajo. Si tu copia
todavía tiene archivos con CRLF de antes (Biome lo notará con `npm run lint` como un error de
formato), guarda o commitea tu trabajo y ejecuta:

```bash
git add --renormalize .
git status   # no debería mostrar cambios de contenido
```

Y para que la carpeta de trabajo quede en LF, con todo commiteado o guardado:

```bash
git rm --cached -r . && git reset --hard
```

Este último comando **descarta cambios sin commitear**: no lo uses si no has guardado tu
trabajo.

## Para quien administra el repositorio

Estos ajustes se hacen en *Settings → Branches → Branch protection rules* para `main` y
requieren permisos de administración; el código por sí solo no puede activarlos:

- **Require a pull request before merging.**
- **Require status checks to pass**, marcando `CI OK`. Opcionalmente también
  `Nombre de rama y título del PR`.
- **Require branches to be up to date before merging.** Es la que evita el caso de #131:
  obliga a que cada PR se pruebe contra el `main` actual antes de entrar.

Mientras eso no esté activado, los checks en rojo avisan pero no bloquean el botón de merge.
