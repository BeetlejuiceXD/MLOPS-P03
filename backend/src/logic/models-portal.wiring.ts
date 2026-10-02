/**
 * D05-06 — Models con el registro real de D04-06: tabla `p3_model_registry` (MariaDB) y el
 * bucket de modelos en el MinIO del portal (`MODEL_S3_BUCKET`, versioning obligatorio).
 */
import { minioClient } from '../data/storage/minio.client.js';
import { createMinioModelStore } from '../data/storage/model-object.storage.js';
import { mariaDbModelRegistryRepository } from './model-registry.repository.js';
import { createModelsPortalService } from './models-portal.service.js';

export function createRegistryModelsPortal(modelBucket: string) {
  return createModelsPortalService({
    repo: mariaDbModelRegistryRepository,
    store: createMinioModelStore(minioClient, modelBucket),
  });
}
