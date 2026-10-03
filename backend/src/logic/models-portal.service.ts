/**
 * D05-06 — Models conectado al registro de D04-06. No hay otro registry ni otro adaptador:
 * lee `p3_model_registry` y el bucket de modelos con el mismo repositorio, el mismo store
 * y el mismo `audit` de `model-registry.service.ts`.
 *
 * - `official()`: lo que sirve `GET /api/models` (contrato `models_response`). Nunca
 *   incluye `local_test`.
 * - `localTest()`/`localTestDetail()`: el registro de pruebas locales (MinIO) con tamaño,
 *   motivo de fallo e integridad comprobada en el momento (head + get por VersionId). La
 *   comprobación no cambia el registro: una versión `published` cuyo objeto desaparece se
 *   muestra con el motivo, pero sigue registrada como estaba.
 * - `localTestObject()`: los bytes de la versión exacta (VersionId), solo si su SHA-256
 *   sigue siendo el registrado. Es lo que descarga el loader de D05-01 para recargarla.
 *
 * Un error del storage es 503, nunca "objeto ausente".
 */
import {
  ConflictError,
  NotFoundError,
  ServiceUnavailableError,
  ValidationError,
} from './errors.js';
import { compareSemver, type RegistryEntry, sha256Hex, toModelVersion } from './model-registry.js';
import {
  createModelRegistryService,
  type ModelObjectStore,
  type ModelRegistryRepository,
} from './model-registry.service.js';
import {
  type LocalTestModel,
  type LocalTestModelDetail,
  localTestModelDetailSchema,
  localTestModelSchema,
  localTestModelsResponseSchema,
  modelsResponseSchema,
} from './p3.contracts.js';

const SEMVER = /^\d+\.\d+\.\d+$/;

export interface LocalTestObject {
  body: Buffer;
  versionId: string;
  sha256: string;
  filename: string;
}

export interface ModelsPortalService {
  official(): Promise<ReturnType<typeof modelsResponseSchema.parse>>;
  localTest(): Promise<ReturnType<typeof localTestModelsResponseSchema.parse>>;
  localTestDetail(semver: string): Promise<LocalTestModelDetail>;
  localTestObject(semver: string): Promise<LocalTestObject>;
}

function toLocalTestModel(entry: RegistryEntry): LocalTestModel {
  return localTestModelSchema.parse({
    ...toModelVersion(entry),
    namespace: 'local_test',
    size_bytes: entry.size_bytes,
    failure_reason: entry.failure_reason,
    failure_detail: entry.failure_detail,
  });
}

export function createModelsPortalService(deps: {
  repo: ModelRegistryRepository;
  store: ModelObjectStore;
  /** D06-03: tarjetas de las versiones (`p3_model_card`); official las muestra con su modelo. */
  cardRepo?: ModelRegistryRepository;
  now?: () => Date;
}): ModelsPortalService {
  const { repo, store, cardRepo } = deps;
  const now = deps.now ?? (() => new Date());
  const official = createModelRegistryService({ repo, store, namespace: 'official', now });
  const localTest = createModelRegistryService({ repo, store, namespace: 'local_test', now });

  async function findLocal(semver: string): Promise<RegistryEntry> {
    if (!SEMVER.test(semver)) throw new ValidationError('semver inválido (MAJOR.MINOR.PATCH)');
    const entry = await repo.find('local_test', semver);
    if (!entry) throw new NotFoundError(`La versión ${semver} no está registrada en local_test`);
    return entry;
  }

  return {
    /** Lo que sirve GET /api/models: solo `official`, cada versión con su tarjeta si existe. */
    async official() {
      if (!cardRepo) return official.list();
      const [rows, cards] = await Promise.all([repo.list('official'), cardRepo.list('official')]);
      const cardBySemver = new Map(cards.map((card) => [card.semver, card]));
      rows.sort((a, b) => compareSemver(a.semver, b.semver));
      return modelsResponseSchema.parse({
        models: rows.map((row) => toModelVersion(row, cardBySemver.get(row.semver) ?? null)),
      });
    },

    async localTest() {
      const rows = await repo.list('local_test');
      rows.sort((a, b) => compareSemver(a.semver, b.semver));
      return localTestModelsResponseSchema.parse({
        namespace: 'local_test',
        models: rows.map(toLocalTestModel),
      });
    },

    async localTestDetail(semver) {
      const entry = await findLocal(semver);
      if (entry.status !== 'published') {
        return localTestModelDetailSchema.parse({
          model: toLocalTestModel(entry),
          integrity: null,
        });
      }
      const audit = await localTest.audit(semver);
      return localTestModelDetailSchema.parse({
        model: toLocalTestModel(entry),
        integrity: {
          checked_at: now().toISOString(),
          ok: audit.ok,
          reason: audit.ok ? null : audit.reason,
          detail: audit.ok ? null : audit.detail,
        },
      });
    },

    async localTestObject(semver) {
      const entry = await findLocal(semver);
      if (entry.status !== 'published' || entry.version_id === null) {
        throw new ConflictError(
          `La versión ${semver} está ${entry.status}: no tiene un objeto verificado que entregar`,
        );
      }
      const versionId = entry.version_id;
      const where = `${entry.s3_bucket}/${entry.s3_key}@${versionId}`;
      let body: Buffer | null;
      try {
        body = await store.get(entry.s3_key, versionId);
      } catch (error) {
        const detail = error instanceof Error ? error.message : String(error);
        throw new ServiceUnavailableError(`Storage de modelos no disponible: ${detail}`);
      }
      if (body === null) throw new NotFoundError(`object_missing: no existe ${where}`);
      const actual = sha256Hex(body);
      if (actual !== entry.sha256) {
        throw new ConflictError(
          `sha256_mismatch: el contenido de ${where} tiene SHA-256 ${actual}; registrado ${entry.sha256}`,
        );
      }
      return { body, versionId, sha256: actual, filename: `model-${semver}.pt` };
    },
  };
}
