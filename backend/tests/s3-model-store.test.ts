/**
 * D06-03 (preparación) — Bucket de modelos en AWS S3 con el SDK oficial.
 *
 * - `createS3ModelStore`: la misma interfaz que el store MinIO de D04-06 (put → VersionId,
 *   head/get POR VersionId exacto). Objeto o versión inexistente → null; cualquier otro
 *   error (AccessDenied, red) se propaga y el registro no decide nada con él.
 * - `inspectModelBucket`: lee, sin cambiarla, la configuración de seguridad del bucket
 *   (versioning, SSE AES256, bloqueo público, DenyInsecureTransport, lifecycle). Lo que no
 *   cumple o no se puede leer (permiso insuficiente) es un check fallido.
 *
 * Cliente S3 SIMULADO (`send` con los comandos del SDK): no toca AWS. Credenciales por la
 * cadena estándar del SDK (perfil SSO del principal operacional); nunca en el repo.
 */
import {
  GetBucketEncryptionCommand,
  GetBucketLifecycleConfigurationCommand,
  GetBucketPolicyCommand,
  GetBucketVersioningCommand,
  GetObjectCommand,
  GetPublicAccessBlockCommand,
  HeadObjectCommand,
  PutObjectCommand,
} from '@aws-sdk/client-s3';
import { describe, expect, it } from 'vitest';
import { createS3ModelStore, inspectModelBucket } from '../src/data/storage/s3-model.storage.js';

const BUCKET = 'mlops-p3-models-test';
const KEY = 'models/p3-cnn-classifier/1.0.0/model.pt';

const awsError = (name: string, status: number) =>
  Object.assign(new Error(`${name}: simulado`), { name, $metadata: { httpStatusCode: status } });

type Handler = (command: { input: Record<string, unknown> }) => unknown;
function fakeClient(handlers: Partial<Record<string, Handler>>) {
  const sent: { name: string; input: Record<string, unknown> }[] = [];
  return {
    sent,
    client: {
      async send(command: { constructor: { name: string }; input: Record<string, unknown> }) {
        const name = command.constructor.name;
        sent.push({ name, input: command.input });
        const handler = handlers[name];
        if (!handler) throw new Error(`comando no simulado: ${name}`);
        return handler(command);
      },
    },
  };
}

describe('createS3ModelStore', () => {
  it('put: sube con el SHA-256 como metadato y devuelve el VersionId', async () => {
    const { client, sent } = fakeClient({
      [PutObjectCommand.name]: () => ({ VersionId: '3HL4kqtJlcpXroDTDmJ' }),
    });
    const store = createS3ModelStore(client, BUCKET);
    const body = Buffer.from('bytes');
    expect(await store.put(KEY, body, 'f'.repeat(64))).toEqual({
      versionId: '3HL4kqtJlcpXroDTDmJ',
    });
    expect(sent[0]?.input).toMatchObject({
      Bucket: BUCKET,
      Key: KEY,
      Metadata: { sha256: 'f'.repeat(64) },
    });
    expect(Buffer.from(sent[0]?.input.Body as Uint8Array).equals(body)).toBe(true);
  });

  it('put sin VersionId (bucket sin versioning) → null, para que el registro lo marque failed', async () => {
    const { client } = fakeClient({ [PutObjectCommand.name]: () => ({}) });
    expect(
      await createS3ModelStore(client, BUCKET).put(KEY, Buffer.from('x'), 'f'.repeat(64)),
    ).toEqual({
      versionId: null,
    });
  });

  it('head y get piden la VERSIÓN exacta, no el latest', async () => {
    const { client, sent } = fakeClient({
      [HeadObjectCommand.name]: () => ({
        ContentLength: 5,
        VersionId: 'v-1',
        Metadata: { sha256: 'f'.repeat(64) },
      }),
      [GetObjectCommand.name]: () => ({
        Body: { transformToByteArray: async () => new Uint8Array(Buffer.from('bytes')) },
      }),
    });
    const store = createS3ModelStore(client, BUCKET);
    expect(await store.head(KEY, 'v-1')).toEqual({
      size: 5,
      versionId: 'v-1',
      sha256: 'f'.repeat(64),
    });
    expect((await store.get(KEY, 'v-1'))?.toString()).toBe('bytes');
    expect(sent.map((call) => call.input)).toEqual([
      { Bucket: BUCKET, Key: KEY, VersionId: 'v-1' },
      { Bucket: BUCKET, Key: KEY, VersionId: 'v-1' },
    ]);
  });

  it.each([
    ['NotFound', 404],
    ['NoSuchKey', 404],
    ['NoSuchVersion', 404],
  ])('objeto o versión inexistente (%s) → null', async (name, status) => {
    const { client } = fakeClient({
      [HeadObjectCommand.name]: () => {
        throw awsError(name, status);
      },
      [GetObjectCommand.name]: () => {
        throw awsError(name, status);
      },
    });
    const store = createS3ModelStore(client, BUCKET);
    expect(await store.head(KEY, 'v-x')).toBeNull();
    expect(await store.get(KEY, 'v-x')).toBeNull();
  });

  it('AccessDenied (permiso insuficiente) se propaga: no es evidencia de objeto ausente', async () => {
    const { client } = fakeClient({
      [HeadObjectCommand.name]: () => {
        throw awsError('AccessDenied', 403);
      },
    });
    await expect(createS3ModelStore(client, BUCKET).head(KEY, 'v-1')).rejects.toThrow(
      /AccessDenied/,
    );
  });
});

const SECURE_POLICY = JSON.stringify({
  Version: '2012-10-17',
  Statement: [
    {
      Sid: 'DenyInsecureTransport',
      Effect: 'Deny',
      Principal: '*',
      Action: 's3:*',
      Resource: [`arn:aws:s3:::${BUCKET}`, `arn:aws:s3:::${BUCKET}/*`],
      Condition: { Bool: { 'aws:SecureTransport': 'false' } },
    },
  ],
});

function bucketClient(overrides: Partial<Record<string, Handler>> = {}) {
  return fakeClient({
    [GetBucketVersioningCommand.name]: () => ({ Status: 'Enabled' }),
    [GetBucketEncryptionCommand.name]: () => ({
      ServerSideEncryptionConfiguration: {
        Rules: [{ ApplyServerSideEncryptionByDefault: { SSEAlgorithm: 'AES256' } }],
      },
    }),
    [GetPublicAccessBlockCommand.name]: () => ({
      PublicAccessBlockConfiguration: {
        BlockPublicAcls: true,
        IgnorePublicAcls: true,
        BlockPublicPolicy: true,
        RestrictPublicBuckets: true,
      },
    }),
    [GetBucketPolicyCommand.name]: () => ({ Policy: SECURE_POLICY }),
    [GetBucketLifecycleConfigurationCommand.name]: () => {
      throw awsError('NoSuchLifecycleConfiguration', 404);
    },
    ...overrides,
  });
}

describe('inspectModelBucket', () => {
  it('bucket seguro: los cinco checks pasan', async () => {
    const report = await inspectModelBucket(bucketClient().client, BUCKET);
    expect(report.ok).toBe(true);
    expect(report.bucket).toBe(BUCKET);
    expect(report.checks.map((check) => [check.name, check.ok])).toEqual([
      ['versioning', true],
      ['encryption', true],
      ['public_access_block', true],
      ['deny_insecure_transport', true],
      ['lifecycle', true],
    ]);
  });

  it.each<[string, string, Handler, RegExp]>([
    ['versioning', 'GetBucketVersioningCommand', () => ({ Status: 'Suspended' }), /Suspended/],
    ['versioning', 'GetBucketVersioningCommand', () => ({}), /sin versioning/],
    [
      'encryption',
      'GetBucketEncryptionCommand',
      () => ({
        ServerSideEncryptionConfiguration: {
          Rules: [{ ApplyServerSideEncryptionByDefault: { SSEAlgorithm: 'aws:kms' } }],
        },
      }),
      /aws:kms/,
    ],
    [
      'public_access_block',
      'GetPublicAccessBlockCommand',
      () => ({
        PublicAccessBlockConfiguration: {
          BlockPublicAcls: true,
          IgnorePublicAcls: true,
          BlockPublicPolicy: false,
          RestrictPublicBuckets: true,
        },
      }),
      /BlockPublicPolicy/,
    ],
    [
      'public_access_block',
      'GetPublicAccessBlockCommand',
      () => {
        throw awsError('NoSuchPublicAccessBlockConfiguration', 404);
      },
      /sin bloqueo/,
    ],
    [
      'deny_insecure_transport',
      'GetBucketPolicyCommand',
      () => ({ Policy: JSON.stringify({ Version: '2012-10-17', Statement: [] }) }),
      /DenyInsecureTransport/,
    ],
    [
      'deny_insecure_transport',
      'GetBucketPolicyCommand',
      () => {
        throw awsError('NoSuchBucketPolicy', 404);
      },
      /sin política/,
    ],
    [
      'lifecycle',
      'GetBucketLifecycleConfigurationCommand',
      () => ({
        Rules: [
          {
            ID: 'limpieza',
            Status: 'Enabled',
            NoncurrentVersionExpiration: { NoncurrentDays: 30 },
          },
        ],
      }),
      /limpieza/,
    ],
    [
      'versioning',
      'GetBucketVersioningCommand',
      () => {
        throw awsError('AccessDenied', 403);
      },
      /permiso insuficiente.*AccessDenied/,
    ],
  ])('%s falla (%s) → ok=false con el motivo', async (name, command, handler, detail) => {
    const report = await inspectModelBucket(bucketClient({ [command]: handler }).client, BUCKET);
    expect(report.ok).toBe(false);
    const check = report.checks.find((item) => item.name === name);
    expect(check?.ok).toBe(false);
    expect(check?.detail).toMatch(detail);
  });

  it('un lifecycle deshabilitado no expira versiones: pasa', async () => {
    const report = await inspectModelBucket(
      bucketClient({
        GetBucketLifecycleConfigurationCommand: () => ({
          Rules: [
            { ID: 'vieja', Status: 'Disabled', NoncurrentVersionExpiration: { NoncurrentDays: 1 } },
          ],
        }),
      }).client,
      BUCKET,
    );
    expect(report.ok).toBe(true);
  });
});
