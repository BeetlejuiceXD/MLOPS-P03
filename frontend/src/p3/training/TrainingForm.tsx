import { type FormEvent, type ReactNode, useEffect, useState } from "react";
import {
  type CreateTrainingJobRequest,
  createTrainingJobRequestSchema,
  TRAINING_CONFIG_DEFAULTS,
  type TrainingConfig,
} from "../contracts";
import { createTrainingJob } from "../mutations";

type Task = CreateTrainingJobRequest["task"];

export interface RealTrainingStatus {
  eligible: boolean;
  /** Motivo por el que el entrenamiento real no está disponible (null si es elegible). */
  reason: string | null;
}

type NumericField =
  | "image_size"
  | "batch_size"
  | "learning_rate"
  | "weight_decay"
  | "max_epochs"
  | "patience"
  | "dropout";

const NUMERIC_FIELDS: { key: NumericField; label: string; step: string; hint: string }[] = [
  { key: "batch_size", label: "Batch size", step: "1", hint: "8–64" },
  { key: "learning_rate", label: "Learning rate", step: "any", hint: "(1e-5, 1e-2]" },
  { key: "weight_decay", label: "Weight decay", step: "any", hint: "0–0.01" },
  { key: "max_epochs", label: "Épocas máximas", step: "1", hint: "10–100" },
  { key: "patience", label: "Patience", step: "1", hint: "3–15 (early stopping)" },
  { key: "image_size", label: "Tamaño de imagen", step: "1", hint: "128–256 px" },
  { key: "dropout", label: "Dropout", step: "any", hint: "0–0.5" },
];

const inputClass = "rounded-lg border border-border bg-white px-3 py-2 text-sm";

/** Número desde un input: vacío o texto no numérico quedan como NaN para que Zod lo rechace. */
const toNumber = (value: string) => (value.trim() === "" ? Number.NaN : Number(value));

function Field({
  id,
  label,
  hint,
  children,
}: Readonly<{ id: string; label: string; hint?: string; children: ReactNode }>) {
  return (
    <div className="flex flex-col gap-1">
      <label htmlFor={id} className="text-xs font-medium text-ink">
        {label}
      </label>
      {children}
      {hint && <span className="text-xs text-ink-muted">{hint}</span>}
    </div>
  );
}

/**
 * D02-05 — Formulario de TrainingConfig (#33). Valida con el mismo contrato que la API
 * (contracts/p3) antes de enviar; la API vuelve a validar y el worker también.
 */
export function TrainingForm({
  realTraining,
  defaultDatasetVersion,
  defaultManifestHash,
  onCreated,
}: Readonly<{
  realTraining: RealTrainingStatus | null;
  defaultDatasetVersion: string;
  defaultManifestHash: string;
  onCreated: () => void;
}>) {
  const [task, setTask] = useState<Task>("controlled");
  const [datasetVersion, setDatasetVersion] = useState(defaultDatasetVersion);
  const [manifestHash, setManifestHash] = useState(defaultManifestHash);
  const [architecture, setArchitecture] = useState<string>(TRAINING_CONFIG_DEFAULTS.architecture);
  const [pretrained, setPretrained] = useState(TRAINING_CONFIG_DEFAULTS.pretrained);
  const [augmentation, setAugmentation] = useState(TRAINING_CONFIG_DEFAULTS.augmentation);
  const [trainableLayers, setTrainableLayers] = useState<string>(
    TRAINING_CONFIG_DEFAULTS.trainable_layers
  );
  const [optimizer, setOptimizer] = useState<string>(TRAINING_CONFIG_DEFAULTS.optimizer);
  const [hiddenLayers, setHiddenLayers] = useState(String(TRAINING_CONFIG_DEFAULTS.hidden_layers));
  const [numbers, setNumbers] = useState<Record<NumericField, string>>(() => ({
    image_size: String(TRAINING_CONFIG_DEFAULTS.image_size),
    batch_size: String(TRAINING_CONFIG_DEFAULTS.batch_size),
    learning_rate: String(TRAINING_CONFIG_DEFAULTS.learning_rate),
    weight_decay: String(TRAINING_CONFIG_DEFAULTS.weight_decay),
    max_epochs: String(TRAINING_CONFIG_DEFAULTS.max_epochs),
    patience: String(TRAINING_CONFIG_DEFAULTS.patience),
    dropout: String(TRAINING_CONFIG_DEFAULTS.dropout),
  }));
  const [seed, setSeed] = useState("");
  const [failAtEpoch, setFailAtEpoch] = useState("");
  const [errors, setErrors] = useState<string[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [created, setCreated] = useState<number | null>(null);

  // Cuando llegan release/manifest, se usan como valores por defecto si el campo sigue vacío.
  useEffect(() => {
    setDatasetVersion((current) => current || defaultDatasetVersion);
  }, [defaultDatasetVersion]);
  useEffect(() => {
    setManifestHash((current) => current || defaultManifestHash);
  }, [defaultManifestHash]);

  // Si el entrenamiento real deja de estar disponible, se vuelve a la tarea controlada.
  const realEnabled = realTraining?.eligible === true;
  useEffect(() => {
    if (!realEnabled) setTask("controlled");
  }, [realEnabled]);

  function buildRequest(): unknown {
    const config: Record<keyof TrainingConfig, unknown> = {
      architecture,
      pretrained,
      trainable_layers: trainableLayers,
      image_size: toNumber(numbers.image_size),
      batch_size: toNumber(numbers.batch_size),
      learning_rate: toNumber(numbers.learning_rate),
      weight_decay: toNumber(numbers.weight_decay),
      optimizer,
      max_epochs: toNumber(numbers.max_epochs),
      patience: toNumber(numbers.patience),
      augmentation,
      seed: seed.trim() === "" ? undefined : toNumber(seed),
      hidden_layers: Number(hiddenLayers),
      hidden_dim: TRAINING_CONFIG_DEFAULTS.hidden_dim,
      dropout: toNumber(numbers.dropout),
    };
    return {
      task,
      dataset_version: datasetVersion.trim(),
      manifest_hash: manifestHash.trim(),
      config,
      ...(task === "controlled" && failAtEpoch.trim() !== ""
        ? { controlled: { fail_at_epoch: toNumber(failAtEpoch) } }
        : {}),
    };
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    setCreated(null);
    const parsed = createTrainingJobRequestSchema.safeParse(buildRequest());
    if (!parsed.success) {
      setErrors(
        parsed.error.issues.map(
          (issue) => `${issue.path.map(String).join(".") || "(formulario)"}: ${issue.message}`
        )
      );
      return;
    }
    setErrors([]);
    setSubmitting(true);
    try {
      const job = await createTrainingJob(parsed.data);
      setCreated(job.id);
      onCreated();
    } catch (error) {
      setErrors([error instanceof Error ? error.message : "No se pudo crear el job."]);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <form
      onSubmit={submit}
      noValidate
      className="flex flex-col gap-5 rounded-2xl border border-border bg-surface p-5"
    >
      <fieldset className="flex flex-col gap-2">
        <legend className="text-sm font-semibold text-ink">Tipo de job</legend>
        <label className="flex items-center gap-2 text-sm">
          <input
            type="radio"
            name="task"
            value="controlled"
            checked={task === "controlled"}
            onChange={() => setTask("controlled")}
          />
          Tarea controlada (sin datos: prueba la plataforma y registra un run de MLflow)
        </label>
        <label className="flex items-center gap-2 text-sm">
          <input
            type="radio"
            name="task"
            value="training"
            checked={task === "training"}
            disabled={!realEnabled}
            onChange={() => setTask("training")}
          />
          Entrenamiento real (D03-03: release aprobado + manifest oficial congelado)
        </label>
      </fieldset>

      <div className="grid gap-4 sm:grid-cols-2">
        <Field id="p3-dataset-version" label="Release (dataset)">
          <input
            id="p3-dataset-version"
            className={inputClass}
            value={datasetVersion}
            onChange={(event) => setDatasetVersion(event.target.value)}
          />
        </Field>
        <Field id="p3-manifest-hash" label="Hash del manifest 70/20/10">
          <input
            id="p3-manifest-hash"
            className={`${inputClass} font-mono text-xs`}
            value={manifestHash}
            onChange={(event) => setManifestHash(event.target.value)}
          />
        </Field>
      </div>

      <fieldset className="grid gap-4 sm:grid-cols-3">
        <legend className="mb-2 text-sm font-semibold text-ink">TrainingConfig</legend>
        <Field id="p3-architecture" label="Arquitectura">
          <select
            id="p3-architecture"
            className={inputClass}
            value={architecture}
            onChange={(event) => setArchitecture(event.target.value)}
          >
            <option value="resnet18">resnet18</option>
          </select>
        </Field>
        <Field id="p3-trainable-layers" label="Capas entrenables">
          <select
            id="p3-trainable-layers"
            className={inputClass}
            value={trainableLayers}
            onChange={(event) => setTrainableLayers(event.target.value)}
          >
            <option value="head_only">head_only</option>
            <option value="last_block">last_block</option>
            <option value="full">full</option>
          </select>
        </Field>
        <Field id="p3-optimizer" label="Optimizador">
          <select
            id="p3-optimizer"
            className={inputClass}
            value={optimizer}
            onChange={(event) => setOptimizer(event.target.value)}
          >
            <option value="adam">adam</option>
            <option value="sgd">sgd</option>
          </select>
        </Field>
        {NUMERIC_FIELDS.map(({ key, label, step, hint }) => (
          <Field key={key} id={`p3-${key}`} label={label} hint={hint}>
            <input
              id={`p3-${key}`}
              type="number"
              step={step}
              className={inputClass}
              value={numbers[key]}
              onChange={(event) =>
                setNumbers((current) => ({ ...current, [key]: event.target.value }))
              }
            />
          </Field>
        ))}
        <Field id="p3-hidden-layers" label="Capas ocultas de la cabeza">
          <select
            id="p3-hidden-layers"
            className={inputClass}
            value={hiddenLayers}
            onChange={(event) => setHiddenLayers(event.target.value)}
          >
            <option value="0">0</option>
            <option value="1">1</option>
          </select>
        </Field>
        <Field id="p3-hidden-dim" label="Dimensión oculta" hint="Fija en 128 (#33)">
          <input
            id="p3-hidden-dim"
            className={inputClass}
            value={String(TRAINING_CONFIG_DEFAULTS.hidden_dim)}
            readOnly
          />
        </Field>
        <Field id="p3-seed" label="Seed" hint="Obligatoria; campaña: 7, 21, 77">
          <input
            id="p3-seed"
            type="number"
            step="1"
            className={inputClass}
            value={seed}
            onChange={(event) => setSeed(event.target.value)}
          />
        </Field>
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={pretrained}
            onChange={(event) => setPretrained(event.target.checked)}
          />
          Pesos preentrenados (ImageNet)
        </label>
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={augmentation}
            onChange={(event) => setAugmentation(event.target.checked)}
          />
          Augmentation (solo train)
        </label>
      </fieldset>

      {task === "controlled" && (
        <Field
          id="p3-fail-at"
          label="Fallar en la época (opcional)"
          hint="Solo tarea controlada: provoca un fallo para probar el manejo de errores."
        >
          <input
            id="p3-fail-at"
            type="number"
            step="1"
            className={`${inputClass} max-w-xs`}
            value={failAtEpoch}
            onChange={(event) => setFailAtEpoch(event.target.value)}
          />
        </Field>
      )}

      {errors.length > 0 && (
        <div role="alert" className="rounded-lg bg-status-pending-soft p-3 text-sm text-ink">
          <ul className="list-disc pl-5">
            {errors.map((error) => (
              <li key={error}>{error}</li>
            ))}
          </ul>
        </div>
      )}
      {created !== null && (
        <p className="text-sm text-ink-muted">Job #{created} en cola. El worker lo tomará.</p>
      )}

      <div>
        <button
          type="submit"
          disabled={submitting}
          className="rounded-lg bg-ink px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
        >
          Encolar job
        </button>
      </div>
    </form>
  );
}
