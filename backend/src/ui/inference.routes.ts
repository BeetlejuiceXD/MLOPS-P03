import express from 'express';
import multer from 'multer';
import { idParamSchema } from '../logic/annotation.validation.js';
import type { InferenceService } from '../logic/inference.service.js';
import { sendError } from './http-errors.js';

/**
 * D05-07 — Inference (montado en `/inference`; el portal lo ve como `/api/inference`).
 *
 *   GET  /engine                  identidad del paquete que sirve el motor
 *   POST /                        multipart `image` (archivo nuevo) o JSON `{ annotation_id }`
 *   GET  /                        historial persistido (más reciente primero)
 *   GET  /annotation-queue        cola persistida
 *   GET  /:id                     una inferencia
 *   POST /:id/annotation-queue    envía a la cola (201 nuevo · 200 el mismo, si se reintenta)
 */
export function createInferenceRouter(
  service: InferenceService,
  options: { maxUploadBytes: number },
): express.Router {
  const router = express.Router();
  const upload = multer({
    storage: multer.memoryStorage(),
    limits: { fileSize: options.maxUploadBytes, files: 1 },
  }).single('image');

  const parseId = (raw: unknown) => {
    const parsed = idParamSchema.safeParse(raw);
    return parsed.success ? parsed.data : null;
  };

  router.get('/engine', async (_req, res) => {
    try {
      res.json(await service.engine());
    } catch (error) {
      sendError(res, error, 'No se pudo leer la identidad del motor de inferencia.');
    }
  });

  router.post('/', (req, res) => {
    upload(req, res, async (uploadError: unknown) => {
      if (uploadError instanceof multer.MulterError) {
        if (uploadError.code === 'LIMIT_FILE_SIZE') {
          res.status(413).json({ error: 'La imagen excede el tamaño máximo permitido.' });
        } else {
          res.status(400).json({ error: 'No se pudo procesar el archivo.' });
        }
        return;
      }
      if (uploadError) {
        sendError(res, uploadError, 'No se pudo recibir la imagen.');
        return;
      }
      const rawAnnotation = (req.body as { annotation_id?: unknown } | undefined)?.annotation_id;
      const hasAnnotation = rawAnnotation !== undefined && rawAnnotation !== '';
      if (Boolean(req.file) === hasAnnotation) {
        res.status(400).json({
          error:
            'Envía un archivo (campo image) o un annotation_id de un crop del portal, no ambos.',
        });
        return;
      }
      try {
        if (req.file) {
          const result = await service.predictUpload({
            filename: req.file.originalname,
            mimeType: req.file.mimetype,
            sizeBytes: req.file.size,
            buffer: req.file.buffer,
          });
          res.status(201).json(result);
          return;
        }
        const annotationId = parseId(rawAnnotation);
        if (annotationId === null) {
          res.status(400).json({ error: 'annotation_id inválido.' });
          return;
        }
        res.status(201).json(await service.predictCrop(annotationId));
      } catch (error) {
        sendError(res, error, 'No se pudo completar la inferencia.');
      }
    });
  });

  router.get('/', async (_req, res) => {
    try {
      res.json(await service.list());
    } catch (error) {
      sendError(res, error, 'No se pudieron leer las inferencias.');
    }
  });

  router.get('/annotation-queue', async (_req, res) => {
    try {
      res.json(await service.queue());
    } catch (error) {
      sendError(res, error, 'No se pudo leer la cola de anotación.');
    }
  });

  router.get('/:id', async (req, res) => {
    const id = parseId(req.params.id);
    if (id === null) {
      res.status(400).json({ error: 'ID de inferencia inválido.' });
      return;
    }
    try {
      res.json(await service.get(id));
    } catch (error) {
      sendError(res, error, 'No se pudo leer la inferencia.');
    }
  });

  router.post('/:id/annotation-queue', async (req, res) => {
    const id = parseId(req.params.id);
    if (id === null) {
      res.status(400).json({ error: 'ID de inferencia inválido.' });
      return;
    }
    try {
      const { item, created } = await service.sendToQueue(id);
      res.status(created ? 201 : 200).json(item);
    } catch (error) {
      sendError(res, error, 'No se pudo enviar a la cola de anotación.');
    }
  });

  return router;
}
