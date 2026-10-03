import { useState } from "react";
import { Link } from "react-router-dom";
import { ErrorState } from "@/components/ui/ErrorState";
import { Skeleton } from "@/components/ui/Skeleton";
import { getAnnotations } from "@/lib/api/annotations";
import { validateImageFile } from "@/lib/api/images";
import { PageHeader } from "@/pipeline/components/PageHeader";
import type { Annotation } from "@/types/schemas";
import { useInferenceEngine, useInferences } from "../api";
import type { AnnotationQueueItem, InferenceModelIdentity, InferenceResult } from "../contracts";
import { dateTime } from "../format";
import { runInferenceCrop, runInferenceUpload, sendToAnnotationQueue } from "../mutations";

const pct = (value: number) => `${(value * 100).toFixed(2)} %`;
const inputClass =
  "rounded-lg border border-border bg-white px-3 py-2 text-sm text-ink focus:outline-none focus:ring-2 focus:ring-accent/40";

const modelLabel = (model: InferenceModelIdentity) =>
  model.source === "smoke" ? "smoke" : `official ${model.model_version}`;

/**
 * D05-07 — Inference: el portal envía una imagen nueva o un crop (anotación) de una imagen
 * del portal a la API; el motor de D05-04 da la clase y la API guarda el resultado con la
 * identidad del paquete. La predicción se puede enviar a la cola de anotación como
 * sugerencia; nunca se convierte en etiqueta humana. D06-06 solo cambia la fuente del
 * motor (modelo official recargado de AWS).
 */
export function InferencePage() {
  const engine = useInferenceEngine();
  const history = useInferences();
  const [result, setResult] = useState<InferenceResult | null>(null);

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Inference"
        subtitle="Clasifica una imagen con el modelo que sirve el motor de inferencia."
      />
      <section
        data-testid="inference-engine"
        aria-label="Motor de inferencia"
        className="rounded-2xl border border-border bg-surface px-5 py-4 text-sm"
      >
        {engine.status === "loading" && <Skeleton className="h-10" />}
        {engine.status === "error" && (
          <ErrorState
            title="El motor de inferencia no está disponible."
            message={engine.message}
            onRetry={engine.reload}
          />
        )}
        {engine.status === "success" && <EngineIdentity model={engine.data.model} />}
      </section>

      <InferenceForm
        enabled={engine.status === "success"}
        onResult={(next) => {
          setResult(next);
          history.reload();
        }}
      />

      {result !== null && (
        <ResultCard key={result.inference_id} result={result} onQueued={history.reload} />
      )}

      <section
        aria-label="Inferencias guardadas"
        className="flex flex-col gap-2 rounded-2xl border border-border p-5"
      >
        <h2 className="text-sm font-semibold text-ink">Inferencias guardadas</h2>
        {history.status === "loading" && <Skeleton className="h-16" />}
        {history.status === "error" && (
          <ErrorState message={history.message} onRetry={history.reload} />
        )}
        {history.status === "success" && <History inferences={history.data.inferences} />}
      </section>
    </div>
  );
}

function Identity({ model }: Readonly<{ model: InferenceModelIdentity }>) {
  return (
    <dl className="grid grid-cols-[max-content_1fr] gap-x-4 gap-y-1 text-xs">
      <dt className="text-ink-muted">Paquete</dt>
      <dd className="font-mono">{model.package_id}</dd>
      <dt className="text-ink-muted">Run MLflow</dt>
      <dd className="font-mono">{model.mlflow_run_id}</dd>
      <dt className="text-ink-muted">Checkpoint SHA-256</dt>
      <dd className="break-all font-mono">{model.checkpoint_sha256}</dd>
      <dt className="text-ink-muted">Formato</dt>
      <dd>{model.format_version}</dd>
    </dl>
  );
}

function EngineIdentity({ model }: Readonly<{ model: InferenceModelIdentity }>) {
  return (
    <div className="flex flex-col gap-2">
      <p className="font-medium text-ink">
        Modelo:{" "}
        {model.source === "smoke" ? "paquete smoke (D05-01)" : `official ${model.model_version}`}
      </p>
      {model.source === "smoke" && (
        <p className="rounded-lg bg-status-pending-soft px-3 py-2 text-xs text-status-pending">
          Paquete smoke para probar el recorrido: no es el modelo seleccionado, evaluado ni
          publicado.
        </p>
      )}
      <Identity model={model} />
    </div>
  );
}

function InferenceForm({
  enabled,
  onResult,
}: Readonly<{ enabled: boolean; onResult: (result: InferenceResult) => void }>) {
  const [mode, setMode] = useState<"upload" | "crop">("upload");
  const [file, setFile] = useState<File | null>(null);
  const [inputError, setInputError] = useState<string | null>(null);
  const [imageId, setImageId] = useState("");
  const [annotations, setAnnotations] = useState<Annotation[] | null>(null);
  const [annotationId, setAnnotationId] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const ready = enabled && !busy && (mode === "upload" ? file !== null : annotationId !== "");

  const loadAnnotations = async () => {
    setError(null);
    setAnnotations(null);
    setAnnotationId("");
    const id = Number(imageId);
    if (!Number.isInteger(id) || id <= 0) {
      setError("Escribe el ID numérico de una imagen del portal.");
      return;
    }
    try {
      setAnnotations(await getAnnotations(id));
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudieron leer las anotaciones.");
    }
  };

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!ready) return;
    setBusy(true);
    setError(null);
    try {
      onResult(
        mode === "upload" && file
          ? await runInferenceUpload(file)
          : await runInferenceCrop(Number(annotationId))
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudo completar la inferencia.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <form
      data-testid="inference-form"
      onSubmit={submit}
      className="flex max-w-xl flex-col gap-4 rounded-2xl border border-border bg-surface p-5"
    >
      <fieldset className="flex gap-4 text-sm">
        <legend className="mb-2 text-sm font-medium text-ink">Entrada</legend>
        {(
          [
            ["upload", "Archivo nuevo"],
            ["crop", "Crop del portal"],
          ] as const
        ).map(([value, label]) => (
          <label key={value} className="flex items-center gap-2">
            <input
              type="radio"
              name="inference-mode"
              value={value}
              checked={mode === value}
              onChange={() => {
                setMode(value);
                setError(null);
              }}
            />
            {label}
          </label>
        ))}
      </fieldset>

      {mode === "upload" ? (
        <div className="flex flex-col gap-1">
          <label htmlFor="inference-image" className="text-sm font-medium text-ink">
            Imagen
          </label>
          <input
            id="inference-image"
            type="file"
            accept="image/jpeg,image/png,image/webp"
            onChange={(event) => {
              const chosen = event.target.files?.[0] ?? null;
              const problem = chosen ? validateImageFile(chosen) : null;
              setInputError(problem);
              setFile(problem ? null : chosen);
            }}
          />
          {inputError && (
            <p data-testid="inference-input-error" className="text-xs text-status-pending">
              {inputError}
            </p>
          )}
          <p className="text-xs text-ink-muted">JPEG, PNG o WebP de hasta 5 MiB, como el upload.</p>
        </div>
      ) : (
        <div className="flex flex-col gap-2">
          <label htmlFor="inference-image-id" className="text-sm font-medium text-ink">
            ID de la imagen del portal
          </label>
          <div className="flex gap-2">
            <input
              id="inference-image-id"
              inputMode="numeric"
              className={inputClass}
              value={imageId}
              onChange={(event) => setImageId(event.target.value)}
            />
            <button
              type="button"
              className="rounded-lg border border-border px-3 py-1.5 text-sm"
              onClick={loadAnnotations}
            >
              Ver anotaciones
            </button>
          </div>
          {annotations !== null &&
            (annotations.length === 0 ? (
              <p className="text-xs text-ink-muted">Esa imagen no tiene anotaciones.</p>
            ) : (
              <>
                <label htmlFor="inference-annotation" className="text-sm font-medium text-ink">
                  Anotación (crop)
                </label>
                <select
                  id="inference-annotation"
                  className={inputClass}
                  value={annotationId}
                  onChange={(event) => setAnnotationId(event.target.value)}
                >
                  <option value="">Elige una anotación</option>
                  {annotations.map((annotation) => (
                    <option key={annotation.id} value={String(annotation.id)}>
                      #{annotation.id} · {annotation.category.name} · ({annotation.bboxX},{" "}
                      {annotation.bboxY}, {annotation.bboxWidth}×{annotation.bboxHeight})
                    </option>
                  ))}
                </select>
              </>
            ))}
        </div>
      )}

      <button
        type="submit"
        disabled={!ready}
        className="self-start rounded-lg bg-ink px-4 py-2 text-sm font-medium text-white disabled:opacity-40"
      >
        Predecir
      </button>
      {error && (
        <p data-testid="inference-error" role="alert" className="text-sm text-status-pending">
          {error}
        </p>
      )}
    </form>
  );
}

function ResultCard({
  result,
  onQueued,
}: Readonly<{ result: InferenceResult; onQueued: () => void }>) {
  const [queued, setQueued] = useState<AnnotationQueueItem | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const queueId = queued?.queue_item_id ?? result.annotation_queue_item_id;
  const imageId = queued?.image_id ?? (result.input.kind === "crop" ? result.input.image_id : null);

  const enqueue = async () => {
    setBusy(true);
    setError(null);
    try {
      setQueued(await sendToAnnotationQueue(result.inference_id));
      onQueued();
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudo enviar a la cola de anotación.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <section
      data-testid="inference-result"
      aria-label="Resultado"
      className="flex max-w-xl flex-col gap-3 rounded-2xl border border-border bg-surface p-5 text-sm"
    >
      <p className="text-xs text-ink-muted">
        Inferencia #{result.inference_id} ·{" "}
        {result.input.kind === "upload"
          ? `Archivo ${result.input.filename} (${result.input.mime_type}, ${result.input.width}×${result.input.height})`
          : `Crop de la anotación #${result.input.annotation_id} (imagen #${result.input.image_id})`}
      </p>
      <p className="text-lg font-semibold text-ink">{result.predicted_class}</p>
      <ul className="flex flex-col gap-1">
        {Object.entries(result.probabilities).map(([name, value]) => (
          <li key={name} className="flex justify-between gap-6">
            <span>{name}</span>
            <span className="font-mono">{pct(value)}</span>
          </li>
        ))}
      </ul>
      <p className="rounded-lg bg-status-pending-soft px-3 py-2 text-xs text-status-pending">
        Sugerencia del modelo ({modelLabel(result.model)}), no es una etiqueta validada.
      </p>
      <Identity model={result.model} />

      {queueId !== null ? (
        <p data-testid="inference-queued" className="text-sm">
          En la cola de anotación: elemento #{queueId}
          {imageId !== null && <> (imagen #{imageId})</>}, pendiente de revisión humana.{" "}
          {imageId !== null && (
            <Link className="text-accent underline" to={`/annotate/${imageId}`}>
              Abrir en Anotar
            </Link>
          )}
        </p>
      ) : (
        <button
          type="button"
          disabled={busy}
          onClick={enqueue}
          className="self-start rounded-lg border border-border px-3 py-1.5 text-sm disabled:opacity-40"
        >
          Enviar a anotación
        </button>
      )}
      {error && (
        <p data-testid="inference-error" role="alert" className="text-sm text-status-pending">
          {error}
        </p>
      )}
    </section>
  );
}

function History({ inferences }: Readonly<{ inferences: readonly InferenceResult[] }>) {
  if (inferences.length === 0) {
    return <p className="text-xs text-ink-muted">Todavía no hay inferencias.</p>;
  }
  return (
    <div data-testid="inference-history" className="overflow-x-auto">
      <table className="w-full text-left text-xs">
        <thead className="text-ink-muted">
          <tr>
            <th className="px-2 py-1">#</th>
            <th className="px-2 py-1">Fecha</th>
            <th className="px-2 py-1">Entrada</th>
            <th className="px-2 py-1">Clase</th>
            <th className="px-2 py-1">Prob.</th>
            <th className="px-2 py-1">Modelo</th>
            <th className="px-2 py-1">Cola</th>
          </tr>
        </thead>
        <tbody>
          {inferences.map((item) => (
            <tr
              key={item.inference_id}
              data-testid={`inference-row-${item.inference_id}`}
              className="border-t border-border"
            >
              <td className="px-2 py-1">{item.inference_id}</td>
              <td className="px-2 py-1">{dateTime(item.created_at)}</td>
              <td className="px-2 py-1">
                {item.input.kind === "upload"
                  ? item.input.filename
                  : `crop #${item.input.annotation_id} · imagen #${item.input.image_id}`}
              </td>
              <td className="px-2 py-1">{item.predicted_class}</td>
              <td className="px-2 py-1 font-mono">
                {pct(item.probabilities[item.predicted_class] ?? 0)}
              </td>
              <td className="px-2 py-1">{modelLabel(item.model)}</td>
              <td className="px-2 py-1">
                {item.annotation_queue_item_id === null
                  ? "sin enviar"
                  : `#${item.annotation_queue_item_id}`}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
