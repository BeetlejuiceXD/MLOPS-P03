# Portal de anotación de imágenes

Monolito para subir, anotar y exportar un dataset de detección de objetos.
Las imágenes se almacenan en MinIO; los metadatos y las anotaciones en MariaDB.

## Estructura

```text
backend/    API HTTP con Express (UI → Logic → Data)
frontend/   Interfaz React + Vite (upload, anotación, dashboard, búsqueda)
```

Cada carpeta es un paquete npm independiente con su propio `package.json`.

## Arquitectura

```text
frontend  →  backend
                ├── src/ui      Endpoints HTTP
                ├── src/logic   Reglas de negocio y validación con Zod
                └── src/data    Drizzle (MariaDB) y MinIO
                                    ├── MariaDB  metadatos y anotaciones
                                    └── MinIO    archivos binarios
```

La capa UI nunca accede a MariaDB ni a MinIO: solo invoca a `logic`. La capa
`logic` es la única que puede importar de `data`.

Todo dato que entra por HTTP se valida con Zod antes de llegar a la capa de
datos, y los tipos se infieren del esquema con `z.infer`. La capa Logic lanza
errores tipados que la UI mapea a códigos HTTP:

| Error             | HTTP | Cuándo                                       |
|-------------------|------|----------------------------------------------|
| `ValidationError` | 400  | Dato mal formado o regla de negocio violada  |
| `NotFoundError`   | 404  | El recurso no existe en la base de datos     |

## Contribuir

Convención de ramas, commits y PR, qué valida el CI y cómo correrlo en tu máquina:
[CONTRIBUTING.md](CONTRIBUTING.md).

## Onboarding de desarrollo

Sigue esta sección de arriba hacia abajo en un clon nuevo. El proyecto tiene
dos entornos Python separados: `app/.venv` para el pipeline y `.venv-dvc`
para DVC. No los mezcles.

### 1. Herramientas necesarias

Obligatorias para el trabajo habitual:

- Git.
- Python 3.12.
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) para el
  entorno Python de `app/`.
- Docker Desktop y Docker Compose si vas a levantar los servicios locales.
- Node.js y npm si vas a desarrollar o probar `backend/` y `frontend/`.
- AWS CLI v2 si necesitas leer el dataset de producción desde S3.

Opcionales:

- Terraform CLI, únicamente para validar o trabajar en infraestructura con
  autorización explícita.
- MinIO local, únicamente para el remote DVC `dev`. No es necesario para
  recuperar el dataset de producción.

No necesitas access keys permanentes para el onboarding. El acceso de AWS de
este proyecto usa IAM Identity Center / SSO con tu propia identidad.

### 2. Clonar el repositorio

```bash
git clone https://github.com/BeetlejuiceXD/MLOPS-P03.git
cd MLOPS-P03
git status
```

La comprobación inicial debe mostrar la rama y el estado del clon. No asumas
una ruta local concreta: trabaja desde la carpeta que acabas de clonar.

### 3. Preparar Python del pipeline

Desde `app/`, instala exactamente las dependencias fijadas por `uv.lock`:

```bash
cd app
uv sync --locked --no-build
uv run python --version
uv run pytest -q
cd ..
```

`uv` crea o utiliza `app/.venv`. Este entorno corresponde al pipeline,
quality gate, analyzers, tests y Copilot Python. No lo sustituyas por
`.venv-dvc`, que es exclusivo de DVC.

### 4. Preparar el entorno separado de DVC

Desde la raíz del repositorio:

```bash
python3.12 -m venv .venv-dvc
source .venv-dvc/bin/activate
python -m pip install 'dvc[s3]==3.67.1'
python --version
dvc --version
dvc remote list
```

En PowerShell, activa el mismo entorno con:

```powershell
py -3.12 -m venv .venv-dvc
.venv-dvc\Scripts\Activate.ps1
```

El repositorio ya está inicializado y ya contiene sus remotes. **No ejecutes
`dvc init`.**

### 5. Comprobar AWS CLI v2

Comprueba primero si ya está instalada:

```bash
aws --version
```

Si no aparece el comando, instala AWS CLI v2 siguiendo el instalador oficial
para macOS de [AWS](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).
Por ejemplo, el instalador oficial puede ejecutarse así:

```bash
curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "/tmp/AWSCLIV2.pkg"
sudo installer -pkg "/tmp/AWSCLIV2.pkg" -target /
aws --version
```

En Windows (PowerShell), con el instalador MSI oficial; abre una terminal nueva al
terminar:

```powershell
msiexec.exe /i https://awscli.amazonaws.com/AWSCLIV2.msi
aws --version
```

Un perfil de otro curso (p. ej. AWS Academy) no sirve: el perfil de este proyecto es
`mlops-p2` (paso 6). Al configurarlo, elige el rol con acceso al bucket de DVC
(`MLOPS-S3-DVC`).

### 6. Configurar IAM Identity Center / SSO

Cada integrante tiene su propio usuario de IAM Identity Center. No recibas ni
copies el password, la sesión SSO o las credenciales de otra persona.

Ejecuta:

```bash
aws configure sso --profile mlops-p2
```

Cuando la CLI lo solicite, responde:

| Pregunta | Valor |
|---|---|
| SSO session name | `mlops-p2` |
| SSO start URL | `https://d-90667fbedf.awsapps.com/start` |
| SSO region | `us-east-1` |
| Registration scopes | `sso:account:access` |

Se abrirá el navegador. Inicia sesión con **tu propio usuario** de IAM
Identity Center, selecciona la cuenta que te haya asignado el administrador y
elige el permission set que te haya autorizado. El README no fija nombres de
usuarios, contraseñas, cuentas ni identificadores personales.

Después inicia la sesión y comprueba la identidad efectiva:

```bash
aws sso login --profile mlops-p2
aws sts get-caller-identity --profile mlops-p2
```

La salida debe corresponder a tu sesión autorizada. Cuando expire, normalmente
basta con renovar la sesión:

```bash
aws sso login --profile mlops-p2
```

La configuración del perfil y la caché de la sesión se guardan fuera del
repositorio, en la configuración local de AWS CLI. No copies esos archivos al
proyecto ni los compartas.

### 7. Conectar DVC con el perfil local de AWS

Con `.venv-dvc` activado, configura el remote `prod` solo en tu máquina:

```bash
dvc remote modify --local prod profile mlops-p2
git check-ignore .dvc/config.local
dvc remote list
```

La opción `--local` es obligatoria: escribe el perfil en `.dvc/config.local`,
no en la configuración versionada de `.dvc/config`. Ese archivo local debe
permanecer ignorado y nunca subirse a Git.

### 8. Comprobar lectura del bucket de producción

Estas comprobaciones son de lectura y no modifican datos:

```bash
aws s3api head-bucket \
  --bucket mlops-p2-dvc-cache \
  --profile mlops-p2

aws s3api list-objects-v2 \
  --bucket mlops-p2-dvc-cache \
  --max-keys 1 \
  --query KeyCount \
  --profile mlops-p2
```

Si terminan correctamente, tu sesión puede alcanzar el bucket y tiene los
permisos requeridos para esas operaciones. Después puedes consultar el estado
de los metadatos DVC sin subir datos:

```bash
dvc status -r prod data/raw/images.dvc data/raw/annotations.dvc data/p3/manifest.json.dvc
```

Descarga el dataset únicamente cuando realmente lo necesites. Para P3 (Training) hacen
falta el release v0.1.1 **y** el manifest congelado de D03-01:

```bash
dvc pull -r prod data/raw/images.dvc data/raw/annotations.dvc data/p3/manifest.json.dvc
```

`dvc pull` materializa archivos en tu máquina, pero no sube nada a S3. No uses
`dvc push` como prueba de conectividad; publicar requiere autorización de
escritura y una tarea explícita.

### 9. Permisos y responsabilidades del administrador

Para que el flujo funcione, el administrador debe haber creado o asignado tu
usuario en IAM Identity Center, asignado la cuenta AWS correspondiente y
asignado un permission set con acceso S3. Para lectura del remote DVC se
necesitan conceptualmente permisos equivalentes a `s3:ListBucket` y
`s3:GetObject`. Publicar datasets requiere permisos adicionales de escritura
definidos por el administrador.

### 10. MinIO / `dev` (opcional)

Los remotes no son intercambiables:

- `prod` = AWS S3 compartido, bucket `mlops-p2-dvc-cache`.
- `dev` = MinIO local, bucket `dvc-cache`.

Si solo necesitas recuperar el dataset de producción, no levantes MinIO. El
flujo opcional completo está documentado en [P2-04 — MinIO local y remotes
DVC](#p2-04--minio-local-y-remotes-dvc). Sus credenciales son locales de
MinIO y no tienen relación con AWS SSO.

### 11. Terraform (opcional y autorizado)

Terraform local puede usar el perfil AWS `mlops-p2` mediante la cadena normal de
credenciales. GitHub Actions usa OIDC, que es un mecanismo distinto; OIDC de
GitHub no autentica automáticamente tu Mac. No ejecutes `terraform apply` ni
`terraform destroy` como parte del onboarding. Tampoco inicialices el backend
remoto hasta que el administrador proporcione y confirme el bucket de state.
Consulta [terraform/README.md](terraform/README.md) para la validación estática
y los límites operativos.

### 12. Variables locales y seguridad

El `.env` de la raíz es para configuración local de Compose/MinIO/Copilot. No
coloques credenciales AWS, passwords, sesiones SSO ni access keys en `.env`.
`ANTHROPIC_API_KEY` es opcional y solo se necesita para utilizar el chat
Copilot; nunca pongas una API key real en esta documentación.

Cada persona usa su propia identidad AWS, no comparte passwords, sesiones SSO
ni access keys, y mantiene `.dvc/config.local` fuera de Git.

## Requisitos

La lista completa y secuencial de instalación está en
[Onboarding de desarrollo](#onboarding-de-desarrollo). Para ejecutar la
aplicación local también se necesitan Docker y Docker Compose.

## Despliegue con un solo comando

Antes del primer arranque, crea el `.env` local para Compose y completa los
**tres** valores obligatorios con credenciales locales inventadas por ti:
`MARIADB_ROOT_PASSWORD` (hexadecimal), `MINIO_ROOT_USER` y `MINIO_ROOT_PASSWORD`.
Sin ellos `docker compose` se niega a arrancar (`Define … en .env`).

```bash
cp .env.example .env
chmod 600 .env
# edita .env; por ejemplo, contraseñas con: openssl rand -hex 32
```

En PowerShell, el mismo paso genera los valores sin escribirlos en pantalla:

```powershell
Copy-Item .env.example .env
$hex = { -join ((1..32) | ForEach-Object { '{0:x}' -f (Get-Random -Maximum 16) }) }
(Get-Content .env) `
  -replace '^MARIADB_ROOT_PASSWORD=$', "MARIADB_ROOT_PASSWORD=$(& $hex)" `
  -replace '^MINIO_ROOT_USER=$', 'MINIO_ROOT_USER=minio-local' `
  -replace '^MINIO_ROOT_PASSWORD=$', "MINIO_ROOT_PASSWORD=$(& $hex)" |
  Set-Content .env -Encoding ascii
```

`ANTHROPIC_API_KEY` puede permanecer vacío si no vas a usar el chat Copilot.
No pongas credenciales AWS en este archivo.

```bash
export GIT_COMMIT=$(git rev-parse HEAD)   # PowerShell: $env:GIT_COMMIT = git rev-parse HEAD
docker compose up --build
```

`GIT_COMMIT` queda como tag de procedencia en cada run de Training (si no se define,
el tag dice `unknown`). Este comando levanta los servicios de MariaDB, MinIO,
MLflow, backend, frontend, pipeline `app`, `trainer-worker` y Copilot. Para que
Training pueda entrenar con datos reales, antes hay que bajar el release y el manifest
congelado con DVC (paso 8 del onboarding); sin ellos todo arranca, pero el portal
explica que las fuentes no están disponibles y solo ofrece la tarea controlada. El backend espera a que MariaDB y MinIO estén listos, aplica las
migraciones y siembra únicamente las categorías `dog` y `cat` antes de
arrancar; no crea imágenes demo ni hace falta ejecutar otro paso manual.

| Servicio        | URL                              |
|-----------------|-----------------------------------|
| Frontend        | http://localhost:8080            |
| Backend (API)   | http://localhost:3100            |
| Consola MinIO   | http://localhost:9001 (usuario y contraseña de tu `.env`) |
| MLflow          | http://localhost:5000 (`MLFLOW_HOST_PORT` en `.env` para cambiarlo) |

Para apagar normalmente los servicios, sin borrar los datos persistidos:

```bash
docker compose down
```

`docker compose down -v` elimina también los volúmenes de MariaDB y MinIO.
Úsalo únicamente cuando quieras reiniciar desde cero los datos locales.

Las credenciales de MariaDB/MinIO usadas en `docker-compose.yml` son las de
desarrollo del proyecto; para un despliegue real, cámbialas ahí antes de
publicar los puertos a una red no confiable.

## Desarrollo local sin Docker para las apps

Para iterar con hot reload en backend y frontend, puedes levantar solo la
infraestructura con Docker y correr los paquetes Node directamente en tu
máquina:

### 1. Infraestructura

```bash
docker run --name proyecto1-mariadb \
  -e MARIADB_ROOT_PASSWORD=password \
  -e MARIADB_DATABASE=image_repo \
  -p 3306:3306 -d mariadb:11

docker run --name proyecto1-minio \
  -p 9000:9000 -p 9001:9001 \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin \
  --user 0:0 \
  -d cgr.dev/chainguard/minio@sha256:bd014394a80898e68c149f2311fdf8d5a2c2f3bb2c33b9327ae6d02b4b065ae1 \
  server /data --console-address ":9001"
```

El bucket se crea automáticamente al arrancar el backend.

### 2. Backend

Antes de ejecutar esos comandos, crea `backend/.env` con la configuración de
desarrollo siguiente. Este archivo es distinto del `.env` de la raíz que usa
Docker Compose:

```dotenv
DATABASE_URL=mysql://root:password@localhost:3306/image_repo
MINIO_ENDPOINT=localhost
MINIO_PORT=9000
MINIO_USE_SSL=false
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=image-annotations
MAX_UPLOAD_SIZE_BYTES=5242880
```

No pongas credenciales AWS en `backend/.env` tampoco. Si cambias el puerto
publicado de MariaDB, ajusta `DATABASE_URL` en este archivo.

```bash
cd backend
npm ci
npm run db:migrate
npm run db:seed
npm run dev
```

Queda escuchando en `http://localhost:3000`.

Si el puerto 3306 ya está ocupado en tu máquina, publica MariaDB en otro
puerto (por ejemplo `-p 3307:3306`) y ajusta `DATABASE_URL` en `backend/.env`.
Nada está fijo en el código: puertos, credenciales y bucket salen del `.env`.

### 3. Frontend

```bash
cd frontend
npm ci
cp .env.example .env
npm run dev
```

Queda escuchando en `http://localhost:5173` y consume la API del backend a
través del proxy `/api` configurado en `vite.config.ts`.

### 4. Comprobación

```bash
curl http://localhost:3000/health
```

Respuesta esperada:

```json
{"status":"ok","database":"connected","timestamp":"..."}
```

## Producción

La forma recomendada de desplegar es `docker compose up --build` (ver
[Despliegue con un solo comando](#despliegue-con-un-solo-comando)): construye
las imágenes de backend y frontend y levanta MariaDB y MinIO junto con ellas.

Si necesitas correr el backend fuera de Docker contra tu propia
infraestructura:

```bash
cd backend
npm run build
npm run start:prod
```

El servidor de producción escucha en `http://localhost:3100`. El script usa
`cross-env`, por lo que funciona igual en Windows, macOS y Linux.

La plantilla `.env.production.example` contiene la configuración de
producción, con `PORT=3100`.

## Variables de entorno

Se copian de `.env.example`. Ningún valor real se versiona: `.gitignore`
ignora todo `.env*` salvo las plantillas de ejemplo.

| Variable                | Propósito                                      |
|-------------------------|------------------------------------------------|
| `PORT`                  | Puerto HTTP (3000 desarrollo, 3100 producción) |
| `DATABASE_URL`          | Cadena de conexión a MariaDB                   |
| `MINIO_ENDPOINT`        | Host de MinIO                                  |
| `MINIO_PORT`            | Puerto de la API de MinIO                      |
| `MINIO_USE_SSL`         | `true` o `false`                               |
| `MINIO_ACCESS_KEY`      | Credencial de acceso                           |
| `MINIO_SECRET_KEY`      | Credencial secreta                             |
| `MINIO_BUCKET`          | Bucket donde se guardan las imágenes           |
| `MAX_UPLOAD_SIZE_BYTES` | Tamaño máximo por imagen (5 MiB por defecto)   |

Los `.env` son configuración local de Compose/backend/MinIO/Copilot. Las
credenciales AWS se obtienen mediante el perfil SSO `mlops-p2`; nunca las
copies a un `.env`.

## API

| Método | Ruta                        | Descripción                                 |
|--------|-----------------------------|---------------------------------------------|
| GET    | `/health`                   | Estado del servicio y de la base de datos   |
| POST   | `/images`                   | Sube una imagen (`multipart/form-data`)     |
| GET    | `/images/search`            | Búsqueda con filtros y paginación           |
| DELETE | `/images/:id`               | Elimina imagen, binario y anotaciones       |
| GET    | `/images/:id/file`          | Sirve el binario desde MinIO                |
| PATCH  | `/images/:id/status`        | Transiciona el estado de anotación          |
| GET    | `/images/:id/annotations`   | Cajas de una imagen, con su categoría       |
| POST   | `/images/:id/annotations`   | Crea una bounding box                       |
| PATCH  | `/annotations/:id`          | Mueve, redimensiona o reclasifica una caja  |
| DELETE | `/annotations/:id`          | Elimina una caja                            |
| GET    | `/categories`               | Categorías disponibles con su color         |
| GET    | `/dashboard/summary`        | Métricas calculadas en SQL                  |
| GET    | `/export/coco`              | Descarga el dataset en formato COCO         |

### Búsqueda

`GET /images/search` acepta:

| Query param         | Descripción                                                |
|---------------------|------------------------------------------------------------|
| `q`                 | Clases con operadores, ej. `car AND person`, `car OR dog`   |
| `categories`        | Ids de categoría separados por coma                        |
| `status`            | `pending`, `in_progress`, `completed` (separados por coma)  |
| `dateFrom`/`dateTo` | Rango sobre la fecha de subida                             |
| `page`/`pageSize`   | Paginación                                                 |

Los operadores se resuelven con subconsultas `EXISTS` en SQL, nunca filtrando
en memoria. Con `AND` la imagen debe contener todas las clases; con `OR`, al
menos una. Mezclar `AND` con `OR` devuelve `400`, porque la precedencia
sería ambigua.

```bash
curl "http://localhost:3000/images/search?q=car%20AND%20person&status=pending&page=1&pageSize=24"
```

### Exportación COCO

```bash
curl -O -J http://localhost:3000/export/coco
```

```json
{
  "images":      [{ "id", "file_name", "width", "height" }],
  "annotations": [{ "id", "image_id", "category_id",
                    "bbox": [x, y, width, height],
                    "area", "iscrowd", "segmentation" }],
  "categories":  [{ "id", "name" }]
}
```

El `bbox` va en píxeles absolutos, `area` es coherente con `width × height`,
e `iscrowd` siempre está presente. Los `id` son consistentes entre las tres
secciones.

## Calidad

Desde `backend/`:

```bash
npm run typecheck   # TypeScript en modo strict
npm run lint        # Biome: cero errores y cero advertencias
npm test            # Vitest
npm run build       # Compilación a dist/
```

Desde `frontend/`:

```bash
npm run typecheck
npm run lint        # Biome: cero errores y cero advertencias
npm run build
```

## Especificaciones y pruebas

Cada regla crítica está trazada de la especificación al escenario Gherkin y
de ahí a la prueba automatizada.

| SPEC            | Regla                                   | Implementación               |
|-----------------|-----------------------------------------|------------------------------|
| SPEC-UPLOAD-001 | Tipo y tamaño de la imagen subida       | `image-upload.validation.ts` |
| SPEC-ANNOT-001  | Geometría y categoría de las cajas      | `annotation.validation.ts`   |
| SPEC-COCO-001   | Estructura y consistencia del JSON COCO | `coco-export.builder.ts`     |
| SPEC-SEARCH-001 | Operadores `AND` / `OR` de búsqueda     | `search-query.parser.ts`     |
| SPEC-VALID-001  | Validación de la frontera HTTP con Zod  | `annotation.validation.ts`   |
| SPEC-DASH-001   | Métricas del dashboard desde SQL        | `dashboard.builder.ts`       |

```text
backend/specs/<nombre>.spec.md
        ↓
backend/features/<nombre>.feature    (Given / When / Then)
        ↓
backend/tests/<nombre>.test.ts       (Vitest)
        ↓
backend/src/logic/<nombre>.ts        (implementación)
```

Las pruebas están diseñadas para fallar si la lógica se rompe: invertir
`width` y `height` en la exportación COCO, permitir un `categoryId` no
positivo o dejar de validar `imageId` hace fallar la suite.

## Fuera de alcance

El entrenamiento del modelo y MLOps corresponden a una fase posterior.

## Etapas del proyecto

El proyecto se construyó por etapas, cada una sobre la anterior:

| Etapa | Qué aportó                                                                 |
|-------|----------------------------------------------------------------------------|
| 1     | Esqueleto: TypeScript, Biome, arquitectura UI/Logic/Data, esquema Drizzle. |
| 2     | Persistencia: MariaDB, MinIO, migraciones, upload de imágenes, seeder.     |
| 3     | Frontend React: portal de anotación, canvas, dashboard y búsqueda.         |
| 4     | Integración final: lógica de negocio, COCO, dashboard y validación Zod.    |

### Qué agrega la etapa final (integración)

Esta etapa conecta el frontend con el backend y completa lo que faltaba para
que el portal funcione de punta a punta:

- **Exportación COCO** (`GET /export/coco`): documento JSON descargable con
  `images`, `annotations` y `categories`, con ids consistentes entre
  secciones (SPEC-COCO-001).
- **Métricas del dashboard** (`GET /dashboard/summary`): totales, objetos por
  clase y progreso de anotación, todo calculado en SQL (SPEC-DASH-001).
- **Búsqueda por clases con operadores** en `GET /images/search`: `AND` / `OR`
  resueltos con subconsultas `EXISTS` en SQL, más filtros por categoría,
  estado y rango de fechas (SPEC-SEARCH-001).
- **Validación de la frontera HTTP con Zod**: todo body, query param y route
  param se valida antes de llegar a la base de datos, con errores tipados que
  la UI mapea a códigos HTTP (SPEC-VALID-001).
- **Reglas de anotación**: la caja debe caber dentro de la imagen, el área la
  calcula el backend, y una imagen sin cajas no puede quedar como completada
  (SPEC-ANNOT-001).

### Notas de puesta en marcha

- Usa `npm install` la primera vez en cada paquete (`backend/` y `frontend/`).
  `node_modules` no se versiona: se reconstruye desde `package-lock.json`.
- El backend valida sus variables de entorno al arrancar (fail-fast con Zod).
  Si falta `backend/.env` o alguna variable, el proceso termina indicando
  cuáles faltan; usa el bloque de variables de backend documentado arriba.
- Si publicaste MariaDB en un puerto distinto al 3306 (por ejemplo 3307
  porque el 3306 ya estaba ocupado), ajusta `DATABASE_URL` en `backend/.env`
  para que coincida.
- El frontend habla con el backend a través del proxy `/api` de Vite en
  desarrollo. `VITE_API_BASE_URL` puede dejarse en `/api`; en producción se
  apunta a la URL real del backend.

## P2-04 — MinIO local y remotes DVC

Esta sección contiene los detalles del remote DVC opcional de desarrollo. Para
el onboarding completo, empieza por [Onboarding de desarrollo](#onboarding-de-desarrollo).
No necesitas MinIO para leer el dataset compartido de producción.

- `dev` usa `s3://dvc-cache` con endpoint `http://localhost:9000` (MinIO local).
- `prod` usa `s3://mlops-p2-dvc-cache` en AWS S3.
- `mlops-p2-dataset-releases` se reserva para releases finales del dataset; no es un remote DVC.

### Resumen rápido

- `dev` → MinIO local, bucket `dvc-cache`.
- `prod` → AWS S3, bucket `mlops-p2-dvc-cache`.
- `mlops-p2-dataset-releases` → releases finales del dataset.
- Git versiona la configuración y los archivos `.dvc`; los binarios se guardan en los remotes.
- Las credenciales de MinIO son locales de cada integrante; no son credenciales
  de AWS ni se usan para `prod`.
- Cada integrante necesita su propio acceso SSO a AWS para `prod`.
- No se comparten contraseñas, sesiones SSO, access keys, secret keys ni tokens.

### Flujo opcional: preparar MinIO local

Solo realiza estos pasos si necesitas usar el remote `dev`. Desde la raíz del
proyecto, crea tu archivo `.env` local a partir de la plantilla:

```bash
test -e .env || cp .env.example .env
chmod 600 .env
```

Completa:

```text
MINIO_ROOT_USER=
MINIO_ROOT_PASSWORD=
```

con valores locales propios.

Puedes generar una contraseña con:

```bash
openssl rand -hex 32
```

No uses claves AWS en `.env`. La autenticación de AWS se configura con IAM
Identity Center / SSO y el perfil local `mlops-p2`.

`frontend/.env.example` es independiente y no cambia para este ticket.

Levanta MinIO:

```bash
docker compose up -d --no-deps minio
```

### Instalar DVC

Instala DVC con soporte S3 en un entorno Python separado del entorno de `app/`:

```bash
python3.12 -m venv .venv-dvc
. .venv-dvc/bin/activate
python -m pip install 'dvc[s3]==3.67.1'
python --version
dvc --version
dvc remote list
```

El repositorio ya contiene la inicialización de DVC y la configuración de los
remotes. **No ejecutes `dvc init`.**

### Configurar `dev` con MinIO local

Carga las variables de tu `.env`:

```bash
set -a
. ./.env
set +a
```

Configura las credenciales de MinIO únicamente de forma local y solo para
`dev`:

```bash
dvc remote modify --local dev access_key_id "$MINIO_ROOT_USER"
dvc remote modify --local dev secret_access_key "$MINIO_ROOT_PASSWORD"
chmod 600 .dvc/config.local
```

No omitas `--local`: `.dvc/config.local` es local, está ignorado por Git y no
debe subirse al repositorio. No mezcles estas credenciales con el perfil SSO
de AWS usado por `prod`.

No agregues `.env` ni `.dvc/config.local` a Git.

### Crear el bucket local `dvc-cache`

Si el bucket `dvc-cache` todavía no existe en MinIO, créalo con:

```bash
python - <<'PY'
import os
from botocore.session import get_session
from botocore.exceptions import ClientError

client = get_session().create_client(
    's3',
    endpoint_url='http://localhost:9000',
    region_name='us-east-1',
    aws_access_key_id=os.environ['MINIO_ROOT_USER'],
    aws_secret_access_key=os.environ['MINIO_ROOT_PASSWORD'],
)

try:
    client.head_bucket(Bucket='dvc-cache')
except ClientError as error:
    if error.response['ResponseMetadata']['HTTPStatusCode'] != 404:
        raise
    client.create_bucket(Bucket='dvc-cache')

print('Bucket local dvc-cache disponible')
PY
```

Este paso solo opera contra MinIO local en `localhost:9000`.

No modifica el bucket `image-annotations` usado por el portal.

### Verificar los remotes

Ejecuta:

```bash
dvc remote list -v
```

La salida debe incluir:

```text
dev     s3://dvc-cache
prod    s3://mlops-p2-dvc-cache
```

### Subir y bajar archivos con `dev`

Primero registra el archivo o directorio con DVC:

```bash
dvc add ruta/al/dataset
```

Para subirlo a MinIO:

```bash
dvc push -r dev
```

Para recuperarlo:

```bash
dvc pull -r dev
```

Los archivos `.dvc` generados se comparten mediante Git.

Los binarios se guardan en MinIO, no directamente en GitHub.

### AWS S3 y remote `prod`

La configuración completa de AWS CLI, IAM Identity Center / SSO, el perfil
`mlops-p2`, las comprobaciones de lectura y la conexión local de DVC está en
[Onboarding de desarrollo](#onboarding-de-desarrollo). `prod` usa AWS S3 real,
sin endpoint personalizado, en `us-east-1`:

| Bucket | Uso |
|---|---|
| `mlops-p2-dvc-cache` | Remote DVC `prod`. |
| `mlops-p2-dataset-releases` | Releases finales del dataset. |

Para lectura del remote DVC, el permission set debe tener permisos
conceptualmente equivalentes a `s3:ListBucket` y `s3:GetObject`. Si además
publicas datasets, el administrador debe asignarte permisos de escritura. No
uses `dvc push` como prueba de conexión.

### Flujo recomendado para el equipo

1. Hacer `git pull` para obtener los metadatos `.dvc` más recientes.
2. Activar el entorno de DVC.
3. Iniciar sesión con AWS SSO si se va a usar `prod`.
4. Ejecutar `dvc status -r prod data/raw/images.dvc data/raw/annotations.dvc`
   para consultar el estado sin subir datos.
5. Ejecutar `dvc pull -r prod data/raw/images.dvc data/raw/annotations.dvc`
   solo si necesitas materializar el dataset localmente.
6. Si eres una persona mantenedora autorizada, agregar o actualizar datos,
   ejecutar `dvc add <ruta>` y publicar con `dvc push -r prod` como una
   operación explícita, no como prueba de conexión.
7. Versionar con Git los archivos `.dvc` y los cambios de código
   correspondientes.

No subas los binarios grandes directamente al repositorio de GitHub.

### Seguridad y validación

Los siguientes archivos o datos no deben versionarse:

- `.env`
- `.dvc/config.local`
- credenciales AWS, incluidas access keys y secret keys
- sesiones y tokens SSO
- credenciales reales de MinIO

Cada persona debe usar su propia identidad AWS. No compartas passwords,
sesiones SSO, access keys ni tokens, y no pongas credenciales AWS en ningún
`.env`.

Comprueba que los archivos privados estén ignorados:

```bash
git check-ignore .env .dvc/config.local
```

La salida debe incluir:

```text
.env
.dvc/config.local
```

Comprueba que `.env.example` sí pueda versionarse:

```bash
git check-ignore .env.example
```

Ese comando no debe mostrar salida.

Comprueba que no existan access keys AWS con prefijo `AKIA` en el historial:

```bash
git log --all -p -S 'AKIA'
```

La salida debe estar vacía.

P2-04 externaliza las credenciales MinIO usadas por Docker Compose y evita agregar secretos AWS al repositorio.

No se reescribe el historial de Git ni se modifica la configuración heredada de MariaDB.

### Criterios de aceptación

Antes de cerrar P2-04, verificar:

- `docker compose up -d --no-deps minio` levanta MinIO.
- `.env.example` existe y no contiene credenciales reales.
- `dvc remote list -v` muestra `dev` y `prod`.
- `dvc push -r dev` funciona para una persona autorizada que use MinIO local.
- `aws s3api head-bucket`, `aws s3api list-objects-v2` y `dvc status -r prod`
  funcionan con el perfil SSO autorizado; `dvc pull` se usa solo cuando se
  necesita materializar el dataset.
- `dvc push -r prod` se prueba únicamente con autorización explícita de
  escritura; no es una prueba de conectividad.
- `git log --all -p -S 'AKIA'` no devuelve resultados.
- `.env` y `.dvc/config.local` permanecen fuera de Git.
- Los buckets `mlops-p2-dvc-cache` y `mlops-p2-dataset-releases` existen en AWS.

## P2-42 — Pipeline DVC completo (`dvc.yaml`)

Hasta este ticket, el dataset se manejaba con `dvc add` suelto: reproducible
como almacenamiento de archivos, pero sin un pipeline declarado con
dependencias/salidas. `dvc.yaml` separa el cálculo del reporte (`quality_report`)
de la decisión de la compuerta (`quality_gate`) y agrega `split` como etapa
posterior. Un reporte `failed` hace que DVC termine con código distinto de cero
y evita ejecutar las etapas dependientes.

```bash
dvc repro
```

- **`app/dvc_quality_report_stage.py`** calcula `reports/quality.json` sin
  decidir si el dataset puede avanzar. **`app/dvc_gate_stage.py`** lee ese
  reporte y devuelve `exit 1` cuando `status: failed`; solo en caso aprobado
  escribe `reports/.quality_gate.passed`, que es la dependencia explícita de
  `split`.
- **`reports/quality.json` es un `metrics`, no un `outs`**, con
  `cache: false`: es un reporte chico y legible, pensado para diffs de PR y
  `dvc metrics diff`, no un artefacto binario que amerite el object store
  de DVC.
- El pipeline no usa `always_changed`: una segunda ejecución de `dvc repro`
  puede reutilizar el run-cache y no rehacer etapas cuando sus entradas no
  cambiaron.
- Los remotes `dev`/`prod` de P2-04 ya existían; lo que faltaba en un
  checkout nuevo era el paso local `dvc remote modify --local dev
  access_key_id/secret_access_key` (con `$MINIO_ROOT_USER`/
  `$MINIO_ROOT_PASSWORD`) y crear el bucket `dvc-cache` si no existía —
  ambos ya documentados arriba en P2-04, solo faltaba ejecutarlos en este
  checkout.

### Criterios de aceptación

- `dvc.yaml` define `quality_report`, `quality_gate` y `split` con dependencias
  y salidas reales; un `failed` bloquea el downstream.
- `dvc.lock` y `dvc.yaml` versionados en Git; los datos siguen fuera de Git.
- `dvc repro` regenera `reports/quality.json`; si el reporte queda `failed`,
  termina con código distinto de cero y no ejecuta `split`.
- `dvc push`/`dvc pull` funcionan contra `dev` y `prod` (verificado: 612
  archivos sincronizados en `dev`, `prod` ya en uso durante todo el proyecto).

## P2-45 — Versionado semántico y diff entre releases

Depende de P2-42. Un release congela `dataset_version` (formato
`vMAJOR.MINOR.PATCH`) junto con su `quality.json` y `splits.json` bajo
`reports/releases/<version>/`, y agrega la entrada al catálogo
`reports/versions.json` (`VersionsReport`, contrato v1.0 de P2-12).

```bash
# Desde app/, con las mismas variables placeholder que P2-42 (ver dvc_gate_stage.py):
uv run python -m presentation.release cut v0.1.0
uv run python -m presentation.release diff v0.1.0 v0.2.0
```

- **Content hash DEV/PROD**: el hash del dataset ya es el md5 en
  `data/raw/annotations.dvc`/`data/raw/images.dvc` — el mismo valor sin
  importar el remote, por construcción de DVC. Verificar que ambos
  remotes lo tengan de verdad es `dvc status -r dev` y `dvc status -r
  prod`, ambos reportando "Cache and remote 'X' are in sync." — no hace
  falta recalcular nada; reimplementarlo sería redundante con lo que DVC
  ya garantiza.
- **`splits.json` nunca se había escrito a disco**: P2-32 dejó
  `split_dataset()`/`build_splits_report()` puros a propósito (ver
  `app/splits/README.md`, "antes de persistir artefactos hay que acordar
  su ubicación/ignore o seguimiento DVC"), porque `DatasetRelease` exige
  `quality_file` y `splits_file`, este ticket fue quien tuvo que decidirlo:
  `reports/releases/<version>/splits.json`, escrito por `cut_release()`.
- **El release respeta la compuerta**: el catálogo histórico `v0.1.0` puede
  conservar un reporte `failed`, pero `cut_release()` rechaza cualquier nuevo
  corte cuyo reporte esté en `status: failed`.
- **Un release es inmutable**: `cut_release()` rechaza un `version` que ya
  existe en el catálogo en vez de sobreescribirlo.
- **`diff_releases()` no vuelve a correr el gate**: lee los dos
  `quality.json` ya congelados — un diff no debe poder ver un dataset
  distinto al que el release realmente describió en su momento.

### Criterios de aceptación

- `dvc status -r dev` y `dvc status -r prod` reportan "in sync" (content
  hash idéntico, verificado).
- `reports/versions.json` sigue el contrato `VersionsReport` v1.0 y usa
  versionado semántico (`v0.1.0` histórico y `v0.1.1` válido, ver
  `reports/releases/`).
- `presentation.release diff <a> <b>` genera un diff real entre dos
  releases (conteo por categoría y status de cada check).

## Frente 1 — Arquitectura y entorno del pipeline de calidad

El portal de anotación (arriba) ya no es el entregable de la Fase 2: es la
fuente del COCO crudo. El entregable es un pipeline en Python que mide la
calidad de ese COCO, decide si se libera y versiona el resultado con DVC.

Este frente deja listo el esqueleto; la lógica de cada tier la completan los
frentes 2 a 6.

### Capas (`app/`)

El pipeline vive en `app/`, como paquete Python independiente (hermano de
`backend/` y `frontend/`), con una carpeta por capa:

```text
app/
  ingestion/      Tier 1 — COCO crudo del Proyecto 1
  analyzers/      Tier 2 — 5 analizadores de calidad (objetos pequeños,
                  desbalance, duplicados, cajas inválidas, sesgo espacial)
  policies/       Tier 3 — compuerta de calidad (policies/quality.yaml)
  splits/         Tier 4 — split estratificado train/val/test
  storage/        Tier 5 — MariaDB y MinIO/S3 (DVC)
  presentation/   Expone los resultados a la app web y al Dataset Copilot
```

**Regla de la compuerta de acoplamiento:** solo `storage/` importa `os`
(para leer variables de entorno), crea clientes `boto3`/`Minio(` o abre un
`create_engine`/`pymysql.connect`. Todas las demás capas son funciones puras
que reciben los datos ya cargados como argumento — así se pueden probar con
`pytest` sin levantar MariaDB/MinIO reales. Se verifica con:

```bash
grep -rn "os\.environ\|os\.getenv\|boto3\.client\|Minio(\|create_engine\|pymysql\.connect" app/analyzers/
```

(sin resultados) y con `app/tests/test_architecture.py`, que corre lo mismo
en CI.

### Levantar todo

```bash
docker compose up
```

Además de `mariadb`, `minio`, `backend` y `frontend` (portal P1, se mantiene
porque la cola de re-anotación —cuando la compuerta bloquea el release—
ocurre ahí), se agrega el servicio `app`: el pipeline Python, que reutiliza
el mismo MariaDB y el mismo MinIO del portal (mismas credenciales de
`.env`, sin variables nuevas). Al arrancar, `app` valida que puede
conectarse a ambos, ejecuta el quality gate y puede regenerar
`reports/quality.json`. Un estado `failed` queda registrado en los logs y debe
revisarse antes de promover o publicar el dataset; levantar `app` no sustituye
la compuerta de DVC.

El servicio `copilot` (P2-52) usa la misma imagen que `app` y atiende el chat
de la pantalla Copilot a través de nginx (`/copilot-api/`), sin publicar
puertos. Necesita `ANTHROPIC_API_KEY` en `.env` (opcional: sin ella todo
arranca y el chat explica qué falta). Detalles en `app/copilot/README.md`.

### Python y lockfile

`app/pyproject.toml` fija `requires-python = "==3.12.*"` y `app/Dockerfile`
usa `python:3.12-slim` — misma versión en ambos lados. Las dependencias
quedan resueltas y pineadas en `app/uv.lock` (generado con `uv lock`, no a
mano); el `Dockerfile` instala desde ese lockfile con
`uv sync --locked`, así que build local y build en CI siempre resuelven
exactamente las mismas versiones.

Para trabajar en `app/` localmente con [uv](https://docs.astral.sh/uv/):

```bash
cd app
uv sync            # crea .venv con dependencias + grupo dev (ruff, pytest)
uv run pytest -q
uv run ruff check .
```

### Supuestos de este frente pendientes de confirmar con Karen/Heri

Lo siguiente se infirió a partir del diagrama de tiers y los mockups del
profe, y del trabajo ya mergeado de DVC (P2-04); si Karen decide otra cosa,
son fáciles de mover porque todo el pipeline está aislado en `app/`:

- El portal Node (`backend`/`frontend`) se queda corriendo junto al pipeline
  en el mismo `docker-compose.yml`, en vez de retirarse porque "el portal ya
  no es el entregable".
- El pipeline reutiliza el MariaDB/MinIO del portal (misma base
  `image_repo`, mismo bucket `image-annotations`) en vez de tener su propia
  infraestructura de datos en dev.
- Quién es responsable del frente 4 (compuerta) no estaba claro en el
  reparto compartido — confirmar con Karen.

## P2-36 — Settings persistente

`/pipeline/settings` tiene dos formularios independientes. La API Node
existente ofrece `GET /settings`, `PUT /settings/quality` y
`PUT /settings/splits` (desde el navegador, `/api/settings/...`).

- Quality permite editar threshold/action de los siete checks reales y
  width_px/height_px de objetos pequeños. La similitud pHash es un umbral de
  detección; el cumplimiento sigue exigiendo cero pares.
- Splits permite editar train/val/test (fracciones estrictamente entre 0 y 1,
  suma 1 con tolerancia 1e-6) y seed (entero seguro de JavaScript).
- `cross_split_leakage` se preserva sin exponerlo. No se publican credenciales,
  rutas, variables de infraestructura ni una supuesta versión activa.
- GET devuelve `{quality, splits}`. Cada PUT recibe directamente su sección
  completa y devuelve esa sección validada. Campos desconocidos o valores
  inválidos producen 400; los errores internos producen 500.
- Persisten en `app/policies/quality.yaml` y `app/splits/splits.yaml`. Se
  conservan comentarios y campos no editables; el backend escribe un temporal,
  sincroniza/cierra y renombra en el mismo directorio. Serializa escrituras
  dentro de su proceso. No hay transacción entre ambos archivos ni control de
  edición obsoleta: la última escritura válida gana.

Guardar **no ejecuta el pipeline**, no crea releases y no cambia reportes
existentes. Los valores afectan la siguiente ejecución de quality/release.
El contenedor Python ejecuta el gate al arrancar, no observa archivos para
recalcular automáticamente. Los comandos de pipeline/release existentes
siguen siendo operaciones explícitas.

Compose comparte los directorios de políticas y splits: backend RW bajo
`/pipeline`, Python RO bajo `/app`. No se publican mediante Nginx. Se montan
directorios para que los reemplazos atómicos sean visibles. En desarrollo,
el backend resuelve el directorio `app/` hermano; `PIPELINE_CONFIG_ROOT` es
una opción de despliegue confiable, nunca un parámetro HTTP.

Python carga la política YAML al construir Settings. `QUALITY` del entorno
o de `.env` se ignora, incluso si contiene JSON inválido: no puede sustituir
silenciosamente la política gestionada por UI. El resto de variables de
infraestructura conserva su semántica. Una política inyectada explícitamente
por código sigue siendo válida para tests/operaciones explícitas.
Cada release recibe una política para quality y pHash, y una SplitsConfig
cargada una vez; ya no recarga otra política para agrupar duplicados.

Estos YAML siguen siendo configuración versionada en Git; guardar puede
dejar cambios locales que deben revisarse. DVC observa `policies/` y
`splits/splits.yaml` desde `quality_report`/`split`. Desde P2-53, quality_gate
también registra ratios y seed de splits. Cambiarlos
afecta la próxima evaluación de leakage y el próximo corte de release,
sin reescribir los splits congelados.

Se añadió `yaml` como dependencia directa del backend para leer/escribir
YAML sin un parser artesanal. El backend actual no tiene autenticación ni
autorización: esta edición está destinada al despliegue controlado existente,
no constituye un panel administrativo protegido para exposición pública.

## D01-02 — Resolvedor de release (fuente de datos de P3)

`presentation.release_resolver` convierte una versión de release en una fuente de
datos verificada, o la rechaza con un motivo estable. P3 solo consume releases que
pasen por aquí; no usa una carpeta fija.

```bash
# Con los datos recuperados (dvc pull -r prod data/raw/images.dvc data/raw/annotations.dvc)
cd app
uv run python -m presentation.release_resolver v0.1.1   # una versión
uv run python -m presentation.release_resolver --all      # todas las versiones conocidas
```

Un release es elegible solo si cumple todo lo siguiente (en este orden):

| Motivo (`reason`) | Regla |
|---|---|
| `invalid_version` | Tiene forma `vMAJOR.MINOR.PATCH`. |
| `not_in_catalog` | Está en `reports/versions.json`. |
| `not_allowed` | Está en la allowlist `app/ingestion/release_sources.yaml`. |
| `quality_failed` / `quality_mismatch` | Su `quality.json` describe esa versión, no está `failed` y ningún check con `action: fail` está reprobado. Un `warning` se acepta. |
| `identity_mismatch` | `data/raw/*.dvc` declaran exactamente los md5 fijados en la allowlist. |
| `data_missing` | Los datos existen y su cantidad de archivos coincide con `nfiles` de DVC. |
| `insufficient_classes` | Al menos 2 clases con ≥ `min_images_per_class` (300) originales distintos, contados desde el COCO real. |

- `--all` (exit 0) imprime `{"approved": [...], "rejected": [{"dataset_version", "reason", "detail"}]}`
  con las versiones de la allowlist y del catálogo, ordenadas por semver. Los rechazados se reportan con
  su motivo, no se omiten. Es el listado que consume `GET /api/releases`.
- El descriptor devuelto incluye hashes DVC, `quality_sha256`, `policy_sha256` (hash de la
  política *aplicada*, no del archivo) y `originals_per_class`. Las rutas son relativas a la
  raíz del repo.
- No recalcula el md5 de los directorios: eso lo garantiza `dvc status` (`-c -r prod` tras
  `dvc pull`). El resolvedor solo comprueba que lo declarado en git es lo autorizado.
- Agregar un release nuevo a la allowlist es una decisión de protocolo, no de código. `v0.1.0`
  no está porque su reporte de calidad es `failed`.
- Los tests `real_data` de `app/tests/test_release_resolver.py` se omiten si `data/raw` no está
  recuperado; en un clon sin `dvc pull` solo corren los de componente.

## D01-05 — Contratos P3 y navegación de las cinco páginas

El portal agrega un tercer grupo al menú global, en el mismo `AppLayout` de P1/P2:
**Training** (`/ml/training`), **Experiments** (`/ml/experiments`), **Evaluation**
(`/ml/evaluation`), **Models** (`/ml/models`) e **Inference** (`/ml/inference`).

- Contratos de la API P3 (endpoints, reglas y dueños): [`contracts/p3/README.md`](contracts/p3/README.md).
- Esquemas Zod: `backend/src/logic/p3.contracts.ts` y su espejo `frontend/src/p3/contracts.ts`.
  Un test del backend exige que ambos tengan las mismas reglas.
- Fixtures compartidos en `contracts/p3/fixtures/`: los validan backend y frontend
  (`npm test` en cada paquete). Son **datos de ejemplo**, no resultados reales.
- Cada página valida la respuesta con su contrato: un error HTTP, un endpoint todavía no
  implementado (404) o una respuesta fuera de contrato muestran el estado de error, nunca datos.
  Hasta que existan los endpoints (D02–D06), las páginas muestran ese estado de error.
- Estados bloqueados: Training sin release aprobado o con el manifest sin congelar; Evaluation
  antes de MODEL SELECTION CLOSED; Inference sin ninguna versión publicada.

Para agregar o cambiar una regla: ajusta el esquema en backend **y** frontend, agrega el fixture
válido o inválido correspondiente en `contracts/p3/fixtures/<contrato>/` y corre `npm test`
en ambos paquetes.

## D02-01 — MLflow persistente (MariaDB + MinIO)

Servicio `mlflow` en `docker-compose.yml` (imagen `infra/mlflow`, MLflow 3.16.1 fijado; el
worker usa `mlflow-skinny==3.16.1` desde `app/uv.lock`).

| Qué | Dónde |
|---|---|
| Backend store (experimentos, runs, params, métricas) | base `mlflow` en el MariaDB del portal (volumen `mariadb_data`) |
| Artefactos (curvas, checkpoints) | bucket `mlflow-artifacts` en MinIO (volumen `minio_data`), servidos por MLflow |
| Tracking URI | `http://mlflow:5000` dentro de Compose · `http://localhost:5000` desde el host |
| Experimento | `p3-cnn-classifier` |

### Arrancar

```bash
cp .env.example .env            # si no lo tienes; completa MARIADB_ROOT_PASSWORD y MINIO_ROOT_*
docker compose up -d --build --wait mariadb minio mlflow
curl -s http://localhost:5000/health   # OK
```

Al arrancar, `infra/mlflow/bootstrap.py` crea la base `mlflow` y el bucket
`mlflow-artifacts` si no existen, así que funciona también con volúmenes que ya tenían
datos del portal. No hay credenciales nuevas: usa `MARIADB_ROOT_PASSWORD` y
`MINIO_ROOT_*` de tu `.env` (nunca en Git).

- **macOS:** si el puerto 5000 está ocupado por AirPlay Receiver, define
  `MLFLOW_HOST_PORT=5001` en `.env` y usa `http://localhost:5001` desde el host.
- MLflow 3 rechaza (403) peticiones cuyo `Host` no esté permitido. Compose permite
  `mlflow:5000`, `localhost:*` y `127.0.0.1:*` (`MLFLOW_ALLOWED_HOSTS`).
- Los artefactos siempre pasan por el servidor MLflow: los clientes usan
  `MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=false` y `MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD=false`
  (ya fijados en Compose y en `tracking/settings.py`). Sin eso, MLflow 3.16 entrega URLs
  prefirmadas hacia `http://minio:9000`, que no se resuelven desde el host.

### Verificar persistencia

```bash
cd app
export MLFLOW_TRACKING_URI=http://localhost:5000
uv run python -m tracking.verify write --evidence ../mlflow-evidence.json
cd .. && docker compose down && docker compose up -d --wait mariadb minio mlflow && cd app
uv run python -m tracking.verify check --evidence ../mlflow-evidence.json
```

`write` registra en `p3-cnn-classifier` un run con una métrica por pasos y un artefacto y
guarda IDs y SHA-256; `check` vuelve a leerlos del servidor y los compara. Salida: `0`
verificado, `1` algo cambió, `2` servidor no disponible (nunca reporta éxito sin leer del
servidor). Desde el worker: `docker compose run --rm --no-deps app python -m tracking.verify ...`.
El run lleva el tag `p3.run_kind=persistence_check`: no cuenta como corrida de la campaña.

El job de CI **MLflow persistente** hace exactamente esto con los servicios reales, desde
el host y desde la imagen del worker, antes y después de `down`/`up`, y comprueba que con
MLflow detenido `check` falla con `2`.

### Persistencia y recuperación

- `docker compose restart` y `docker compose down` **conservan** los datos (volúmenes
  nombrados). `docker compose down -v` **los borra**: no lo uses si quieres conservar las
  corridas.
- Si MLflow no arranca, revisa `docker compose logs mlflow` (conexión a MariaDB/MinIO) y
  vuelve a `docker compose up -d --wait mlflow`; el arranque es idempotente.
- Respaldo de metadatos: `docker compose exec mariadb sh -c 'mariadb-dump -uroot
  -p"$MARIADB_ROOT_PASSWORD" --databases mlflow' > mlflow-metadata.sql`. El snapshot
  completo (metadatos + artefactos) para el clean clone se versiona con DVC al cerrar la
  campaña (D05), como quedó en #33.

## D02-05 — Jobs persistentes y formulario Training

Los entrenamientos no corren dentro del request HTTP: la API **encola** un job en MariaDB
(`training_jobs` + `training_job_logs`, migración `0003_training_jobs.sql`) y el servicio
`trainer-worker` lo toma por polling y lo ejecuta. Todo el estado (config, estado,
progreso, logs, error, `mlflow_run_id`) vive en la base de datos, así que sobrevive a
recargas del portal y a reinicios de contenedores.

```bash
docker compose up -d --build            # incluye trainer-worker
docker compose logs -f trainer-worker   # "trainer-worker <id> escuchando training_jobs"
```

Portal: **Training** (`/ml/training`) → formulario → *Encolar job*. La tabla se refresca
sola mientras haya jobs `queued`/`running`; cada job muestra progreso, logs, error y run.

| Endpoint | Uso |
| --- | --- |
| `POST /api/training/jobs` | Crea un job (`create_training_job_request`). `400` si el `TrainingConfig` es inválido (nombra el campo), `409` si se pide entrenamiento real y no hay release/manifest elegibles. |
| `GET /api/training/jobs` · `GET /api/training/jobs/:id` | Lista / detalle (`training_job`). |
| `GET /api/training/jobs/:id/logs` | Últimas 500 líneas. |
| `POST /api/training/jobs/:id/cancel` | `queued` → `cancelled` inmediato; `running` → `cancel_requested` y el worker lo detiene en la siguiente época; terminado → `409`. |

**Estados:** `queued → running → succeeded | failed | cancelled`. Un estado terminal
nunca se sobrescribe. El `TrainingConfig` se valida en la UI, en la API y otra vez en el
worker (un job inválido que llegue a la tabla termina `failed` sin crear run).

**Tareas:**

- `controlled` — tarea controlada para probar el ciclo de vida: no entrena ni lee datos,
  registra por época métricas sintéticas `controlled_*` y crea un run con tag
  `p3.run_kind=controlled_task` (Experiments debe filtrarlo: no es corrida de la campaña).
  `controlled.fail_at_epoch` fuerza un fallo en esa época.
- `training` — entrenamiento real con el trainer de D02-03 (ver **D03-03** abajo). La
  API responde `409` con el motivo mientras no haya release aprobado y manifest
  congelado oficiales verificados.

**Interrupciones (sin duplicar entrenamientos):**

- Primero se escribe el estado terminal en MariaDB y después se cierra el run en MLflow
  con el estado equivalente (`succeeded`→FINISHED, `failed`→FAILED/KILLED,
  `cancelled`→KILLED).
- Éxito vs. cancelación lo decide un único UPDATE (`status='running' AND
  cancel_requested=false`): una cancelación que llegue hasta justo antes del cierre
  termina `cancelled`/KILLED, nunca `succeeded`.
- `docker compose stop trainer-worker` (SIGTERM): el job en curso termina `failed` con
  "Interrumpido ... (SIGTERM)"; su run queda `KILLED`. Al persistir el éxito, la parada se
  revisa dentro de la transacción (UPDATE ya ejecutado, justo antes del COMMIT) y, si la
  señal llega después de ese check, el handler corta la transacción antes del COMMIT
  (la conexión se descarta y MariaDB deshace el cambio): un SIGTERM recibido antes de
  que el estado terminal quede persistido nunca deja `succeeded`/FINISHED. Si llega con
  el COMMIT ya hecho, el éxito ya estaba persistido y solo se detiene el worker.
- Worker matado sin aviso (SIGKILL, OOM, host caído): el job queda `running` con el
  último latido. Al arrancar, cualquier worker marca `failed` los jobs `running` de otro
  worker con latido más viejo que `TRAINER_STALE_AFTER_SECONDS` (60 s, mínimo 30),
  conserva progreso y run, y deja pendiente cerrar el run como `FAILED`.
- Si MLflow no responde al cerrar un run, el estado pendiente queda en
  `training_jobs.mlflow_close_status` (migración `0004`) y el worker lo reintenta en cada
  vuelta hasta que MLflow vuelve. Un run nunca se queda `RUNNING` para siempre.
- Nunca se reencola un job interrumpido: se vuelve a lanzar a mano si corresponde.

Variables: `TRAINER_POLL_SECONDS` (2), `TRAINER_CONTROLLED_EPOCH_SECONDS` (0.5),
`TRAINER_STALE_AFTER_SECONDS` (60), más `DATABASE_URL` y `MLFLOW_TRACKING_URI` del worker.

**Evidencia:** el job de CI **Jobs persistentes** levanta MariaDB, MinIO, MLflow, backend
y trainer-worker reales y ejecuta `.github/scripts/jobs_e2e.py`: config inválido (400),
entrenamiento real (409), éxito, fallo controlado, cancelación, SIGTERM, SIGKILL con
recuperación y `down`/`up` con jobs y logs idénticos; imprime Job IDs, estados y run IDs.
Los fixtures de `contracts/p3/fixtures` son evidencia de componente, no de integración.

**Mutaciones:** `.github/scripts/run_jobs_mutations.py` aplica 7 mutantes de D02-05 (claim atómico,
latido vencido, cierre pendiente del run huérfano, compuerta release/manifest y las dos
protecciones del cierre), corre solo los tests del área, informa qué tests detectaron
cada uno y restaura el archivo. CI lo corre en los jobs "Mutaciones Python" (`--suite app`) y Backend:

```bash
cd app && uv run python ../.github/scripts/run_jobs_mutations.py   # ambas suites
```

## D03-03 — Training real con trainer y datos reales

`task=training` entrena de verdad desde el portal: `trainer-worker` toma el job (mismo
claim, estados, cancelación, SIGTERM y recuperación de D02-05) y corre `trainer.engine.train`
sobre el **release aprobado** + el **manifest P3 congelado** de D03-01, solo con train/val.

**Fuentes (una sola verdad, verificada en Python):**

1. El manifest congelado es `data/p3/manifest.json` + `data/p3/manifest.json.dvc`
   (ruta configurable con `P3_MANIFEST_PATH`). Contrato: `FrozenManifest` en
   `app/presentation/contracts.py` (`frozen: true`, `seed: 42`, identidad DVC,
   `test_split_hash` y `assignments` train/val/test por `crop_id`).
2. `trainer_worker/sources.py::verify_training_sources` rechaza **antes de entrenar**, con
   un motivo estable, si: falta el artefacto (`manifest_missing`), no está congelado
   (`manifest_not_frozen`), no está versionado o no coincide con el md5 de su `.dvc`
   (`manifest_not_versioned`/`manifest_dvc_mismatch`), su `manifest_hash` o
   `test_split_hash` no corresponden al contenido (`manifest_hash_mismatch`/
   `test_hash_mismatch`), el job pide otro release/hash (`request_mismatch`), el release
   no está aprobado por el resolver D01-02 (`release_<motivo>`), la identidad DVC no es la
   del release (`identity_mismatch`) o la partición no es válida contra los crops y grupos
   near-duplicate **recalculados desde los datos** (`invalid_partition`).
3. Cada `TRAINER_SOURCES_REFRESH_SECONDS` (300) el worker publica el resultado en
   `p3_training_sources` (migración `0005`). El backend lo sirve en `GET /api/releases` y
   `GET /api/manifest` (validado otra vez contra los contratos; `503` con el motivo si no
   está disponible, nunca datos vacíos) y la compuerta de `POST /api/training/jobs` lo
   usa para responder `409`. El worker **vuelve a verificar contra los archivos** al tomar
   cada job: el snapshot solo sirve para la UI y la compuerta.

**Qué llega al trainer:** `build_training_dataset` recorta los crops (bbox D01-07) de
train y val; `TrainingDataset.test` va vacío y los píxeles de test nunca se recortan ni se
cargan para entrenar (Ale custodia el frozen test, #33). La verificación sí lee las
imágenes del release para recalcular crops y near-duplicates (auditoría de integridad,
igual que D03-01), sin pasarlas al modelo.

**Trazabilidad en MLflow (run `p3.run_kind=training`):**

- Tags `git_commit`, `dvc_release`, `dvc_images_md5`, `dvc_annotations_md5`,
  `dvc_release_hash`, `manifest_version`, `manifest_hash`, `classes`, `seed`, `job_id`,
  `stopped_early`, `device`, `checkpoint_sha256`.
- Params: el `TrainingConfig` completo + `classes`.
- Métricas por época (`train_loss`, `train_accuracy`, `val_loss`, `val_accuracy`,
  `val_macro_f1`, `learning_rate`) y resumen (`best_epoch`, `best_val_*`,
  `duration_seconds`, `peak_memory_mb`). Nunca métricas de test.
- Artefactos `checkpoint/`: `model.pt` (mejor época), `training_config.json`,
  `class_map.json`, `environment.json`, `sources.json`. Tras subirlo, el worker lo
  **descarga del servidor y compara el sha256**; si no coincide, el job termina `failed`.
- El progreso del job avanza con cada época real (`on_epoch_end` de `train()`), y
  cancelación/SIGTERM se atienden al final de cada época.

**Compose:** `trainer-worker` monta `./data/raw`, `./reports` y `./data/p3` en solo
lectura (`TRAINER_REPO_ROOT=/app`), recibe `GIT_COMMIT` (exporta
`GIT_COMMIT=$(git rev-parse HEAD)` antes de `docker compose up`) y guarda los pesos
preentrenados de torchvision en el volumen `torch_cache`. Para entrenar con v0.1.1:

```bash
# release v0.1.1 + manifest congelado de D03-01 (SSO de AWS, perfil mlops-p2)
dvc pull -r prod data/raw/images.dvc data/raw/annotations.dvc data/p3/manifest.json.dvc
GIT_COMMIT=$(git rev-parse HEAD) docker compose up -d --build
```

Sin datos o sin manifest (p. ej. en CI), el portal muestra el motivo y solo ofrece la
tarea controlada.

**Evidencia:** `tests/test_training_sources.py` (verificación y adaptador),
`tests/test_trainer_worker_training.py` (entrenamiento real con worker, MLflow,
cancelación, SIGTERM, error del trainer y checkpoint no verificable),
`backend/tests/p3-sources.test.ts` y `frontend/tests/p3-training.test.tsx`. El job de CI
**Jobs persistentes** comprueba además que el worker publica las fuentes reales y que,
sin datos, la API responde `503`/`409` con el motivo. Mutantes D03-03 (8 a 14) en
`run_jobs_mutations.py`.

## D03-04 — Smoke real Training → MLflow → checkpoint

Recorrido integrado antes de la campaña: un job `task=training` lanzado **desde el
portal** sobre train/val del manifest congelado, seguido hasta su estado terminal y
contrastado con su run en `p3-cnn-classifier` (MariaDB + MinIO) y con su checkpoint
descargado del servidor. **No es una corrida de la campaña (#33)**: se identifica como
smoke en la evidencia, no se cuenta para selección y nunca consulta el frozen test.

`app/tracking/smoke.py` (`python -m tracking.smoke`) verifica:

- job `training` + `succeeded`, con run en el experimento `p3-cnn-classifier` y `FINISHED`;
- procedencia contrastada con las **fuentes oficiales** que publica la API, no solo con lo
  que declara el run: release aprobado en `GET /releases` (`dvc_release`,
  `dvc_images_md5`, `dvc_annotations_md5`), manifest congelado de `GET /manifest`
  (`manifest_version`, `manifest_hash`, `dvc_release_hash`, `classes`), `dvc_release_hash`
  recalculado como sha256(`images_md5:annotations_md5`), `classes` = class map congelado,
  y `job_id`/`seed`/params = el job y su `TrainingConfig`;
- historial por época con pasos **únicos y exactamente 1..epoch** (= progreso del job) en
  las seis curvas; `best_epoch` entero y registrado; `best_val_accuracy`,
  `best_val_macro_f1` y `best_val_loss` iguales a su curva en `best_epoch` (tolerancia
  1e-4 del contrato `experiment_runs_response`); **ninguna métrica de test**;
- `checkpoint/model.pt` descargado con sha256 = tag, `training_config.json` = config del
  job, `class_map.json` = class map congelado, `sources.json` = procedencia del run, y
  carga estricta en el modelo de su config con una predicción usando el transform de
  evaluación compartido.

Con Compose levantado (D03-03: `dvc pull -r prod` de `data/raw` y `data/p3/manifest.json`):

```bash
export GIT_COMMIT=$(git rev-parse HEAD)          # PowerShell: $env:GIT_COMMIT = git rev-parse HEAD
docker compose up -d --build
# 1) Portal → http://localhost:8080 → Training: entrenamiento real, seed 7,
#    épocas máximas 10, patience 3 → "Encolar job" (anota el Job #).
# 2) Verificar ese job, su run y su checkpoint (dentro de la red de Compose):
mkdir -p _smoke
docker compose run --rm --no-deps -v "$PWD/_smoke:/out" app \
  python -m tracking.smoke verify --job-id <N> \
  --api http://frontend/api --tracking-uri http://mlflow:5000 \
  --evidence /out/smoke-d03-04.json
```

`tracking.smoke run --seed 7` hace lo mismo creando el job por la API del portal.
Sale con `0` solo si todo coincide (`1` = algo no coincide, `2` = servicios/fuentes no
disponibles). `_smoke/` no se versiona.

**Tests:** `app/tests/test_smoke.py` usa el worker real sobre el release sintético y
altera una sola cosa por test (job no `succeeded`, config o manifest distintos, métrica
de test, checkpoint reemplazado en el servidor, state_dict de otro modelo, class map o
`training_config.json` distintos, run de otro experimento) y, tras la auditoría de #69,
negativos independientes de historial/`best_*` (B1: pasos duplicados o incompletos,
`best_epoch` inexistente o no entero, cada `best_*` distinto de su curva, resumen
faltante) y de procedencia (B2: un solo tag falso — `classes`, `dvc_release_hash`, md5,
`manifest_version`, `dvc_release`, `seed`, `job_id` —, hash no recalculable, release no
aprobado o distinto al oficial, `sources.json` incoherente). El clon fiel del run real
pasa, así cada negativo cambia una sola pieza. Mutantes D03-04 en `run_jobs_mutations.py`.

## D03-06 — Arranque limpio verificado

Recorrido completo desde un clon nuevo hasta el smoke de D03-04 (job real → run de
MLflow → checkpoint verificado), sin depender de nada que no esté en este README. Es la
secuencia que se ejecutó para D03-06; sirve como checklist de instalación. **No evalúa
el frozen test ni es una corrida de la campaña.**

**Prerrequisitos fuera del repo** (una vez por máquina, ver
[Onboarding](#onboarding-de-desarrollo)): Git, Python 3.12, Docker Desktop corriendo,
AWS CLI v2 con el perfil SSO `mlops-p2` (rol con acceso al bucket DVC) y unos 10 GB
libres para las imágenes. Puertos libres: 3306, 9000/9001, 5000, 3100 y 8080; si ya
tienes otro clon levantado, apágalo antes (`docker compose down` en esa carpeta).

```bash
# 1) Clon limpio e identificación del commit
git clone https://github.com/BeetlejuiceXD/MLOPS-P03.git MLOPS-P03-clean
cd MLOPS-P03-clean
git rev-parse HEAD

# 2) .env local (ver "Despliegue con un solo comando"; tres valores obligatorios)
cp .env.example .env && chmod 600 .env   # y completa MARIADB_ROOT_PASSWORD, MINIO_ROOT_*

# 3) Datos oficiales con DVC (entorno separado .venv-dvc, perfil SSO mlops-p2)
python3.12 -m venv .venv-dvc && source .venv-dvc/bin/activate
python -m pip install 'dvc[s3]==3.67.1'
aws sso login --profile mlops-p2
dvc remote modify --local prod profile mlops-p2
dvc pull -r prod data/raw/images.dvc data/raw/annotations.dvc data/p3/manifest.json.dvc
dvc status data/raw/images.dvc data/raw/annotations.dvc data/p3/manifest.json.dvc

# 4) Servicios, construidos desde cero
export GIT_COMMIT=$(git rev-parse HEAD)
docker compose build --no-cache --pull
docker compose up -d
docker compose ps

# 5) Salud y fuentes publicadas (esperar ~30 s a que trainer-worker publique)
curl -s http://localhost:3100/health
curl -s http://localhost:5000/health
curl -s http://localhost:8080/api/releases
curl -s http://localhost:8080/api/manifest

# 6) Flujo corto integrado de D03-04: crea el job por la API del portal, lo sigue hasta
#    su estado terminal y verifica job, run, procedencia y checkpoint
mkdir -p _smoke
docker compose run --rm --no-deps -v "$PWD/_smoke:/out" app \
  python -m tracking.smoke run --seed 7 --max-epochs 10 --patience 3 \
  --api http://frontend/api --tracking-uri http://mlflow:5000 \
  --label "D03-06 arranque limpio (no es corrida de campaña)" \
  --evidence /out/smoke-d03-06.json
```

En PowerShell cambian solo estas líneas: `Copy-Item .env.example .env` (y el bloque que
genera los valores de "Despliegue con un solo comando"), `py -3.12 -m venv .venv-dvc` y
`.venv-dvc\Scripts\Activate.ps1`, `$env:GIT_COMMIT = git rev-parse HEAD`,
`curl.exe` en vez de `curl`, `mkdir _smoke` y `"${PWD}/_smoke:/out"`. Si PowerShell
bloquea `Activate.ps1`, corre antes `Set-ExecutionPolicy -Scope Process Bypass`.

**Qué esperar en cada punto:**

| Paso | Resultado esperado |
|---|---|
| 3 | `dvc status` → `Data and pipelines are up to date.`; `data/raw/images` con 600 archivos |
| 4 | `docker compose ps`: `mariadb`, `minio` y `mlflow` *healthy*; los demás *running* |
| 5 | `/health` → `"status":"ok"`; `/api/releases` con `v0.1.1` en `approved`; `/api/manifest` con `p3-v0.1.1-s42` y `"frozen":true` |
| 6 | `SMOKE OK: job #1 → run … → checkpoint … cargado (['cat', 'dog'])` y salida `0` |

Si un paso no da lo esperado, el motivo suele estar en la propia respuesta: por
ejemplo `GET /api/manifest` → `503 manifest_missing` significa que faltó el
`dvc pull` del manifest, y `docker compose` → `Define MARIADB_ROOT_PASSWORD` que faltó
el `.env`. Si el build del backend termina con `[check-esbuild] … falta @esbuild/…`, npm
no pudo descargar un binario opcional ni en el reintento (casi siempre la red): vuelve a
correr `docker compose build --no-cache backend`. Antes de D03-06 ese caso producía una
imagen que se construía "bien" y moría al arrancar (`/api/*` → 502); ahora el build falla.
La evidencia del ensayo está en el PR que cierra #63.

## D04-01 — Runs de MLflow en la API del portal

El backend lee el experimento `p3-cnn-classifier` del MLflow persistente por su API REST
(`MLFLOW_TRACKING_URI`, en Compose `http://mlflow:5000`) y lo sirve al portal con el
contrato de `contracts/p3/README.md`. No guarda ni inventa runs: lo que muestra está en MLflow.

```bash
# Listado: runs de training en `runs`; auxiliares e incompletos en `excluded` con su motivo
curl -s http://localhost:8080/api/experiments/runs
# Detalle de un run (mismos valores del listado + artefactos)
curl -s http://localhost:8080/api/experiments/runs/<run_id>
# Recuperar un artefacto (el sha256 de model.pt coincide con el tag checkpoint_sha256)
curl -s -o model.pt http://localhost:8080/api/experiments/runs/<run_id>/artifacts/checkpoint/model.pt
```

| Situación | Respuesta |
|---|---|
| Run de training completo | En `runs`; `campaign_eligible: true` solo si es `FINISHED`, con resumen y `checkpoint_sha256` |
| Training `RUNNING`, `SCHEDULED`, `FAILED` o `KILLED` | En `runs` con su estado real de MLflow y sus épocas reales, `campaign_eligible: false` y el motivo |
| Training con cualquier métrica `test*` | En `excluded`: el frozen test no se evalúa antes de MODEL SELECTION CLOSED (#33) |
| `controlled_task` (D02-05), `short_run_instrumentation` (D02-06), `persistence_check` (D02-01) o sin `p3.run_kind` | En `excluded` con el motivo; detalle y artefactos → 409 |
| Training sin un tag de provenance o con un hueco en la curva | En `excluded`; el valor no se rellena con el oficial ni se interpola |
| Artefacto que no existe | 404 `artifact_missing: …` |
| MLflow caído | 503 `mlflow_unavailable: …` (nunca un listado vacío) |

`campaign_eligible` es elegibilidad estructural. Que el run pertenezca a la matriz OFAT de
la campaña y la selección por validation son de D04-03/D04-04.

## D04-02 — Experiments en el portal

`http://localhost:8080/ml/experiments` muestra los runs reales del experimento
`p3-cnn-classifier` tal como los sirve el adaptador de D04-01 (`/api/experiments/runs`).

- **Conteo:** runs de training, elegibles para campaña y, aparte, auxiliares y excluidos.
  Los auxiliares (`controlled_task`, `short_run_instrumentation`, `persistence_check`) y los
  runs excluidos aparecen en su propia sección con el motivo: no suman al conteo ni se pueden
  elegir para comparar.
- **Filtros:** estado, elegibilidad, seed, capas entrenables, optimizador y learning rate. Las
  opciones salen de los runs presentes.
- **Comparar:** marca dos o más runs. Se muestran los 15 campos de `TrainingConfig` lado a lado
  (resaltando los que difieren), las mejores métricas de validation y las curvas de
  `val_accuracy` y `val_loss` superpuestas.
- **Ver:** abre el detalle desde `/api/experiments/runs/<run_id>`. Muestra provenance, resumen,
  curvas por época con su tabla de valores y artefactos descargables
  (`checkpoint/model.pt`, etc.).
- **Actualizar:** vuelve a pedir los runs a la API, por ejemplo mientras un job está `RUNNING`.
- Si MLflow no responde, la página muestra el motivo (`503 mlflow_unavailable`), nunca una
  lista vacía.

La aceptación del conjunto de diez runs de campaña en Experiments es de D05-03.

## D05-03 — Campaña aceptada y candidato en Experiments (preparación)

Experiments lee la selección de D04-04/D05-02 (`GET /api/selection`, contrato
`selection_state`) junto a los runs de MLflow. **No ordena ni elige nada.** El ranking, las
filas de campaña y el candidato son los que entrega la API.

- **Panel "Selección por validation":**
  - `open`: "Todavía no hay candidato propuesto".
  - `candidate`: muestra el run_id, la fila OFAT y las métricas de validation del mejor
    checkpoint. Dice **"Propuesta pendiente de cierre (D05-08)"**: no es un cierre formal.
  - `closed`: "MODEL SELECTION CLOSED" con la fecha.
  - En los tres casos muestra "Filas de campaña comparables: N (mínimo 10)".
- **Cotejo selección ↔ MLflow** (`frontend/src/p3/selection.ts`):
  - El candidato y cada fila aceptada deben aparecer en Experiments y ser elegibles.
  - `best_epoch`, `val_accuracy`, `val_macro_f1` y `val_loss` deben ser iguales a los de MLflow.
  - Si algo no coincide, el panel lo dice con el motivo y no presenta la aceptación ni marca
    filas.
- **Tabla:** cada run aceptado lleva su badge "Fila N", y el candidato "Candidato propuesto"
  o "Candidato seleccionado". El filtro **Campaña → Campaña aceptada** deja solo esas filas.
  El training fuera de la matriz, los auxiliares y los excluidos no llevan badge ni cuentan.
- **Fallos:**
  - Si `/api/selection` falla (p. ej. 503 mientras D05-02 no integra el adaptador de runs),
    el motivo sale en el panel y los runs siguen visibles.
  - **Actualizar** recarga runs y selección.
- Sin métricas de test: el contrato rechaza un ranking con `test_*`.

Para **cerrar #86** falta la lista aceptada, el candidato y los hashes de D05-02 (Ale). Los
fixtures solo prueban el render y el cotejo, no acreditan la campaña.

## D05-06 — Models con el registro de modelos (preparación)

Models lee el registro de D04-06 (`p3_model_registry` + bucket `MODEL_S3_BUCKET` con
versioning). No hay otro registry ni otro adaptador.

| Ruta | Qué sirve |
|---|---|
| `GET /api/models` | Solo el namespace `official` (contrato `models_response`). Una prueba local nunca aparece aquí. |
| `GET /api/models/local-test` | Registro `local_test` (MinIO): semver, run, bucket/key, VersionId, SHA-256, tamaño, estado y motivo de fallo. |
| `GET /api/models/local-test/<semver>` | La versión y su integridad comprobada en ese momento (head + get por VersionId). No cambia el registro. |
| `GET /api/models/local-test/<semver>/object` | Los bytes de esa versión exacta, solo si su SHA-256 sigue siendo el registrado (si no, 409). Objeto ausente → 404 `object_missing`; storage caído → 503. |

En el portal, `Models` muestra arriba las versiones official y abajo, rotuladas como
**pruebas locales (no es una publicación en AWS)**, las `local_test`, con **Comprobar**
(integridad) y **Descargar** (versión exacta).

Para registrar un paquete en `local_test`, dentro del contenedor del backend:

```bash
docker compose cp <paquete>/model.pt backend:/tmp/model.pt
docker compose exec backend node dist/cli/register-local-test-model.js \
  --file /tmp/model.pt --semver 0.0.1 --run-id <run_id> \
  --manifest-hash <manifest_hash> --release v0.1.1 --release-hash <dvc_release_hash>
```

Imprime la evidencia (bucket/key/VersionId/SHA-256/tamaño/estado y la relectura por
VersionId) y sale con 0 solo si la versión quedó `published`. El semver es inmutable: repetirlo
falla. El paquete smoke real y su loader son de D05-01; la publicación `official` en AWS es de
D06-03.

## D05-01 — Paquete smoke y loader autocontenido

`app/model_package/` empaqueta el checkpoint real de un run de training para recargarlo
fuera del entrenamiento, sin memoria ni estado del trainer. El detalle está en
`app/model_package/README.md`.

- **Formato `p3-model-package` 1.0.0 (`kind: smoke`):**
  - Contiene los pesos de origen tal cual, la config de D01-04, el `class_map`, el
    preprocessing, las dependencias, la salida de referencia y la tarjeta smoke.
  - Trae un inventario con el SHA-256 y el tamaño de cada archivo.
  - El campo `source` liga el paquete a su origen: `run_id`, `checkpoint/model.pt`,
    `checkpoint_sha256`, `best_epoch` y el manifest/release.
- **Loader:** valida inventario, hashes, dependencias, arquitectura, `class_map` y
  preprocessing **antes** de instanciar la CNN. Cualquier discrepancia es `PackageError`
  y no hay fallback.
- **Interfaz:** `identity()` y `predict()` para D05-04 y D06-02/D06-04.
- **Sin métricas oficiales de test:** el frozen test y la model card oficial son de D06-02.
  `format_version` no es el semver del modelo.

```bash
# dentro del contenedor trainer-worker
python -m model_package build --run-id <run_id> --tracking-uri http://mlflow:5000 --out /tmp/smoke-package
python -m model_package predict --package /tmp/smoke-package   # proceso limpio
```

## D05-07 — Inference con motor y cola de anotación (preparación)

`http://localhost:8080/ml/inference` clasifica una imagen con el modelo que sirve el
**motor de inferencia de D05-04**. La clase la calcula el motor, nunca el portal: la API
le pasa los bytes y guarda lo que responde con la **identidad del modelo**.

- **Entrada:**
  - **Archivo nuevo:** multipart `image`, con la misma validación que el upload del portal
    (JPEG, PNG o WebP, hasta 5 MiB y contenido verificado con sharp).
  - **Crop del portal:** se elige una anotación de una imagen del portal; la API recorta la
    caja de la imagen original y manda ese recorte al motor.
- **Resultado:** clase, probabilidades e identidad: `source` (smoke u official),
  `package_id`, `format_version`, `mlflow_run_id` y `checkpoint_sha256`. Un paquete smoke
  no tiene `model_version`, porque el semver lo asigna D06-02. La página lo rotula como
  "Sugerencia del modelo (smoke), no es una etiqueta validada".
- **Persistencia:** cada inferencia se guarda en `p3_inference` (migración 0009) y queda
  consultable después de recargar o reiniciar (`GET /api/inference` y `GET /api/inference/:id`).
- **Cola de anotación** (`POST /api/inference/:id/annotation-queue`):
  - Archivo nuevo: crea la imagen en el portal con estado `pending`, así entra en la cola
    de anotación existente.
  - Crop: referencia la imagen y la anotación originales, sin tocar su estado.
  - Nunca crea anotaciones ni etiquetas humanas: el elemento nace `pending` con
    `human_label: null`, y la predicción viaja como `suggestion` del modelo.
  - Un reintento devuelve el mismo elemento (índice único por inferencia).
- **Errores:**

  | Caso | Respuesta |
  |---|---|
  | Tipo, tamaño o contenido inválido | 400 o 413 |
  | Caja fuera de la imagen | 400 |
  | Anotación inexistente | 404 |
  | Motor ausente o sin modelo | 503 con el motivo |
  | Respuesta del motor fuera de contrato | 503 |
  | Fallo de MariaDB | 503 |

  En ninguno se guarda nada a medias ni aparece una anotación.

### Motor (D05-04) y punto de sustitución (D06-06)

El backend llama al motor en `INFERENCE_ENGINE_URL` (`backend/src/logic/inference-engine.ts`):

| Ruta del motor | Responde |
|---|---|
| `GET /identity` | `inference_engine`: identidad del paquete cargado y clases `cat`, `dog` |
| `POST /predict` | Recibe los bytes de la imagen (`Content-Type: image/*`) y responde `inference_engine_prediction`. Un 4xx = imagen rechazada; un 5xx = modelo o servicio no disponible |

Sin `INFERENCE_ENGINE_URL`, Inference responde 503 con el motivo. D06-06 solo cambia esa
URL a un motor con el modelo official recargado de AWS (`source: "official"`, con semver):
el resto del portal no cambia.

Para **cerrar #90** falta el motor real de D05-04 (Esteban). Los tests usan un motor de
**fixture**: prueban la API, la validación y la persistencia, no el motor.

## D05-04 — Motor de inferencia con el paquete smoke

`app/inference_engine/` carga un paquete de D05-01 y clasifica imágenes. El detalle está en
`app/inference_engine/README.md`.

- **Carga:** usa `load_package` de D05-01, así que la transformación, el `class_map` y la
  config salen del paquete y no de los defaults del portal. Antes de aceptar imágenes,
  coteja la salida de referencia del paquete y, si se le pasa, el `checkpoint_sha256`
  esperado.
- **Predicción:** devuelve la clase, las probabilidades en el orden del `class_map` y la
  identidad del paquete (`package_id`, `mlflow_run_id`, `checkpoint_sha256`). La
  identidad sale del mismo paquete que predijo. No guarda etiquetas humanas.
- **Errores:** distingue imagen rechazada, paquete rechazado e inferencia incoherente.
  Ninguno de los tres emite predicción.
- **HTTP para D05-07:** `GET /identity` y `POST /predict`, con el contrato de
  `INFERENCE_ENGINE_URL`.

```bash
# proceso limpio: carga, predice y escribe JSON
python -m inference_engine predict --package ./smoke-package --image gato.jpg \
  --expected-sha256 <checkpoint_sha256>
# servicio para el portal (INFERENCE_ENGINE_URL=http://<host>:8090)
python -m inference_engine serve --package ./smoke-package --port 8090
```
