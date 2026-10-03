/**
 * D06-03 — Publicación `official` del paquete final (D06-02) y su tarjeta en el bucket de
 * modelos de AWS, con el registro de D04-06 para los dos objetos:
 *
 *   1. Lee (sin cambiarla) la seguridad del bucket: versioning, SSE AES256, bloqueo público,
 *      DenyInsecureTransport y ningún lifecycle que expire versiones. Si algo no cumple o no
 *      se puede leer (permiso insuficiente) → 409 y no se registra ni se sube nada.
 *   2. Contrasta la tarjeta con la identidad del paquete (JSON, no smoke, mismo run y semver).
 *   3. register (modelo y tarjeta, semver inmutable) → upload + verify de la TARJETA → solo
 *      si quedó `published`, upload + verify del MODELO. Verificar = HEAD y GET por el
 *      VersionId exacto y comparar tamaño y SHA-256; nunca el latest.
 *   4. Cualquier fallo deja `failed` con su motivo: nada termina `published` por error.
 */
import { ConflictError, ValidationError } from './errors.js';
import { sha256Hex } from './model-registry.js';
import type {
  ModelObjectStore,
  ModelRegistryService,
  RegistryOutcome,
} from './model-registry.service.js';

export interface BucketSecurityCheck {
  name: string;
  ok: boolean;
  detail: string;
}

export interface BucketSecurityReport {
  bucket: string;
  ok: boolean;
  checks: BucketSecurityCheck[];
}

export interface PackageIdentity {
  semver: string;
  mlflow_run_id: string;
  manifest_hash: string;
  dvc_release: string;
  dvc_release_hash: string;
}

export interface OfficialPublicationResult {
  bucket_security: BucketSecurityReport;
  card: RegistryOutcome;
  model: RegistryOutcome;
  /** SHA-256 de lo que se vuelve a leer por VersionId, si los dos quedaron published. */
  read_back: { model_sha256: string; card_sha256: string } | null;
  published: boolean;
}

/** La tarjeta debe ser la del paquete final de ESTE run y semver (no la smoke de D05-01). */
function checkCard(card: Buffer, identity: PackageIdentity): void {
  let parsed: unknown;
  try {
    parsed = JSON.parse(card.toString('utf8'));
  } catch {
    throw new ValidationError('La tarjeta del modelo no es JSON válido');
  }
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
    throw new ValidationError('La tarjeta del modelo no es JSON válido (se espera un objeto)');
  }
  const fields = parsed as Record<string, unknown>;
  if (fields.kind === 'smoke') {
    throw new ValidationError(
      'La tarjeta es de un paquete smoke: official solo publica el paquete final de D06-02',
    );
  }
  if (fields.mlflow_run_id !== undefined && fields.mlflow_run_id !== identity.mlflow_run_id) {
    throw new ValidationError(
      `La tarjeta es de otro run (${String(fields.mlflow_run_id)}), no de ${identity.mlflow_run_id}`,
    );
  }
  if (fields.semver !== undefined && fields.semver !== identity.semver) {
    throw new ValidationError(
      `La tarjeta declara semver ${String(fields.semver)}, no ${identity.semver}`,
    );
  }
}

export function createOfficialPublication(deps: {
  model: ModelRegistryService;
  card: ModelRegistryService;
  store: ModelObjectStore;
  inspectBucket: () => Promise<BucketSecurityReport>;
}) {
  const { model, card, store } = deps;

  return {
    async publish(input: {
      identity: PackageIdentity;
      model: Buffer;
      card: Buffer;
    }): Promise<OfficialPublicationResult> {
      const security = await deps.inspectBucket();
      if (!security.ok) {
        const failed = security.checks
          .filter((check) => !check.ok)
          .map((check) => `${check.name}: ${check.detail}`)
          .join('; ');
        throw new ConflictError(
          `El bucket ${security.bucket} no cumple los controles para publicar official: ${failed}`,
        );
      }
      checkCard(input.card, input.identity);

      const { semver } = input.identity;
      await model.register({
        ...input.identity,
        sha256: sha256Hex(input.model),
        size_bytes: input.model.length,
      });
      await card.register({
        ...input.identity,
        sha256: sha256Hex(input.card),
        size_bytes: input.card.length,
      });

      let cardOutcome = await card.upload(semver, input.card);
      if (cardOutcome.failure === null) cardOutcome = await card.verify(semver);
      if (cardOutcome.model.status !== 'published') {
        const why = cardOutcome.failure
          ? `${cardOutcome.failure.reason}: ${cardOutcome.failure.detail}`
          : cardOutcome.model.status;
        const rejected = await model.reject(
          semver,
          'model_card_failed',
          `La tarjeta ${cardOutcome.model.s3_key} no quedó verificada (${why})`,
        );
        return {
          bucket_security: security,
          card: cardOutcome,
          model: rejected,
          read_back: null,
          published: false,
        };
      }

      let modelOutcome = await model.upload(semver, input.model);
      if (modelOutcome.failure === null) modelOutcome = await model.verify(semver);
      const published = modelOutcome.model.status === 'published';

      let readBack: OfficialPublicationResult['read_back'] = null;
      if (published) {
        const [modelBytes, cardBytes] = await Promise.all([
          store.get(modelOutcome.model.s3_key, modelOutcome.model.version_id as string),
          store.get(cardOutcome.model.s3_key, cardOutcome.model.version_id as string),
        ]);
        if (modelBytes && cardBytes) {
          readBack = { model_sha256: sha256Hex(modelBytes), card_sha256: sha256Hex(cardBytes) };
        }
      }
      return {
        bucket_security: security,
        card: cardOutcome,
        model: modelOutcome,
        read_back: readBack,
        published,
      };
    },
  };
}
