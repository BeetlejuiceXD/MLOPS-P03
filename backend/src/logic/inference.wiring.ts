import { randomUUID } from 'node:crypto';
import { env } from '../config/env.js';
import {
  deleteImageObject,
  findAnnotationById,
  findImageById,
  getImageObjectStream,
  uploadImageObject,
} from '../data/index.js';
import { mariaDbInferenceRepository } from './inference.repository.js';
import { createInferenceService, type PortalImages } from './inference.service.js';
import { createHttpInferenceEngine } from './inference-engine.js';

/**
 * D05-07 — Inference sobre el portal existente: las imágenes van al mismo bucket y a la
 * misma tabla `images` que el upload, así un archivo enviado a la cola aparece en la cola
 * de anotación del portal (estado `pending`) sin otro sistema de anotaciones.
 */
export const portalImages: PortalImages = {
  async storeInput(bytes, mimeType) {
    const key = `inference/${randomUUID()}`;
    await uploadImageObject(key, bytes, mimeType);
    return key;
  },
  deleteInput: (key) => deleteImageObject(key),
  async findAnnotation(id) {
    const annotation = await findAnnotationById(id);
    if (!annotation) return null;
    return {
      annotation_id: annotation.id,
      image_id: annotation.imageId,
      bbox: [annotation.bboxX, annotation.bboxY, annotation.bboxWidth, annotation.bboxHeight],
    };
  },
  async readImage(imageId) {
    const image = await findImageById(imageId);
    if (!image) return null;
    const chunks: Buffer[] = [];
    for await (const chunk of await getImageObjectStream(image.storageKey)) {
      chunks.push(Buffer.from(chunk as Buffer));
    }
    return { bytes: Buffer.concat(chunks), mimeType: image.mimeType };
  },
  imageExists: async (imageId) => (await findImageById(imageId)) !== null,
};

export function createPortalInferenceService() {
  return createInferenceService({
    engine: createHttpInferenceEngine({ baseUrl: env.INFERENCE_ENGINE_URL }),
    repo: mariaDbInferenceRepository,
    images: portalImages,
    maxUploadBytes: env.MAX_UPLOAD_SIZE_BYTES,
  });
}
