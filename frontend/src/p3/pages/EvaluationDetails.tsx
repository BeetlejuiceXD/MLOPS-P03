import { useState } from "react";
import { API_BASE_URL } from "@/lib/api/client";
import { useEvaluationDetails, useEvaluationPredictions } from "../api";
import { FetchBoundary } from "../components/FetchBoundary";
import type { EvaluationDetails, EvaluationPredictions, EvaluationResponse } from "../contracts";
import { dateTime, percent, shortHash } from "../format";

/**
 * D06-05 — Lo que Evaluation muestra de una evaluación `ready`, además de la matriz: la
 * procedencia completa, el umbral 0.85 con conteos, métricas por clase, ejemplos por
 * `crop_id` y las predicciones guardadas (consulta y exportación). Todas las cifras vienen
 * de la API (`/evaluation`, `/evaluation/details`, `/evaluation/predictions`); la página no
 * calcula ninguna ni dispara evaluaciones: solo hace GET.
 */
type Ready = Extract<EvaluationResponse, { state: "ready" }>;
type Example = EvaluationDetails["examples"]["errors"][number];

function Field({
  label,
  value,
  mono = false,
}: Readonly<{ label: string; value: string; mono?: boolean }>) {
  return (
    <div>
      <dt className="text-xs text-ink-muted">{label}</dt>
      <dd className={`text-sm text-ink ${mono ? "break-all font-mono" : ""}`}>{value}</dd>
    </div>
  );
}

function Provenance({ details }: Readonly<{ details: EvaluationDetails }>) {
  const p = details.provenance;
  return (
    <section
      data-testid="evaluation-full-provenance"
      className="rounded-2xl border border-border bg-surface p-5"
    >
      <h2 className="mb-3 text-sm font-semibold text-ink">Procedencia</h2>
      <dl className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        <Field label="Run del candidato" value={p.candidate_run_id} mono />
        <Field
          label="Fila de la campaña · mejor época"
          value={`${p.campaign_row} · ${p.best_epoch}`}
        />
        <Field label="Checkpoint (sha256)" value={p.checkpoint_sha256} mono />
        <Field label="Commit" value={p.git_commit} mono />
        <Field
          label="Release DVC"
          value={`${p.dataset_version} · ${shortHash(p.dvc_release_hash)}`}
        />
        <Field label="Manifest" value={p.manifest_hash} mono />
        <Field label="test_split_hash" value={p.test_split_hash} mono />
        <Field
          label="Cronología"
          value={`Selección cerrada ${dateTime(p.closed_at)} → evaluada ${dateTime(p.evaluated_at)}`}
        />
        <Field label="Muestras de test" value={String(p.n_test)} />
      </dl>
    </section>
  );
}

function Target({ target }: Readonly<{ target: EvaluationDetails["target"] }>) {
  return (
    <p
      data-testid="evaluation-target"
      className={`rounded-2xl border px-5 py-3 text-sm ${
        target.met
          ? "border-status-done/40 bg-status-done-soft"
          : "border-status-pending/40 bg-status-pending-soft"
      } text-ink`}
    >
      Objetivo de #33, accuracy ≥ {percent(target.accuracy)}:{" "}
      <strong>{`${target.correct} / ${target.n_test}`}</strong>{" "}
      {target.met ? "cumple el objetivo." : "no alcanza el objetivo; el candidato sigue cerrado."}
    </p>
  );
}

function PerClass({ metrics }: Readonly<{ metrics: Ready["metrics"] }>) {
  return (
    <table data-testid="evaluation-per-class" className="w-fit border-collapse text-sm">
      <caption className="mb-2 text-left text-xs text-ink-muted">Métricas por clase</caption>
      <thead>
        <tr>
          {["Clase", "Precision", "Recall", "F1", "Support"].map((title) => (
            <th key={title} className="px-3 py-2 text-left text-xs text-ink-muted">
              {title}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {metrics.per_class.map((row) => (
          <tr key={row.class_name}>
            <th className="px-3 py-2 text-left">{row.class_name}</th>
            <td className="border border-border px-3 py-2">{percent(row.precision)}</td>
            <td className="border border-border px-3 py-2">{percent(row.recall)}</td>
            <td className="border border-border px-3 py-2">{percent(row.f1)}</td>
            <td className="border border-border px-3 py-2">{row.support}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function ExampleList({
  title,
  testId,
  total,
  items,
}: Readonly<{ title: string; testId: string; total: number; items: Example[] }>) {
  return (
    <div data-testid={testId}>
      <h3 className="text-xs text-ink-muted">{`${title} (${items.length} de ${total})`}</h3>
      <ul className="mt-1 flex flex-col gap-1 text-sm">
        {items.map((item) => (
          <li key={item.crop_id} data-crop-id={item.crop_id}>
            <span className="font-mono">crop {item.crop_id}</span>: real {item.true_class} →
            predicha {item.predicted_class} ({percent(item.confidence)})
          </li>
        ))}
      </ul>
    </div>
  );
}

function Predictions({ predictions }: Readonly<{ predictions: EvaluationPredictions }>) {
  const [onlyErrors, setOnlyErrors] = useState(false);
  const rows = predictions.predictions.filter(
    (sample) => !onlyErrors || sample.true_class !== sample.predicted_class
  );
  const exportUrl = (format: "csv" | "json") =>
    `${API_BASE_URL}/evaluation/predictions?format=${format}`;
  return (
    <section data-testid="evaluation-predictions" className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-4 text-sm">
        <h2 className="font-semibold text-ink">Predicciones guardadas</h2>
        <label className="flex items-center gap-1">
          <input
            type="checkbox"
            checked={onlyErrors}
            onChange={(event) => setOnlyErrors(event.target.checked)}
          />
          Solo errores
        </label>
        <a data-testid="export-csv" href={exportUrl("csv")} className="underline">
          Exportar CSV
        </a>
        <a data-testid="export-json" href={exportUrl("json")} className="underline">
          Exportar JSON
        </a>
      </div>
      <table className="w-fit border-collapse text-sm">
        <thead>
          <tr>
            {["crop_id", "Real", "Predicha", ...predictions.classes.map((c) => `p(${c})`)].map(
              (title) => (
                <th key={title} className="px-3 py-1 text-left text-xs text-ink-muted">
                  {title}
                </th>
              )
            )}
          </tr>
        </thead>
        <tbody>
          {rows.map((sample) => (
            <tr key={sample.crop_id} data-testid="prediction-row">
              <td className="px-3 py-1 font-mono">{sample.crop_id}</td>
              <td className="px-3 py-1">{sample.true_class}</td>
              <td className="px-3 py-1">{sample.predicted_class}</td>
              {predictions.classes.map((name) => (
                <td key={name} className="px-3 py-1">
                  {percent(sample.probabilities[name] ?? 0)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

export function EvaluationDetailsView({ evaluation }: Readonly<{ evaluation: Ready }>) {
  const details = useEvaluationDetails();
  const predictions = useEvaluationPredictions();
  return (
    <FetchBoundary results={[details, predictions]}>
      {(detail, exported) => (
        <div className="flex flex-col gap-6">
          {!detail.final && (
            <p data-testid="evaluation-not-final" className="text-sm text-ink-muted">
              Este resultado es del namespace {detail.namespace}: no es el resultado final.
            </p>
          )}
          <Target target={detail.target} />
          <Provenance details={detail} />
          <PerClass metrics={evaluation.metrics} />
          <section className="grid gap-4 sm:grid-cols-2">
            <ExampleList
              title="Errores"
              testId="evaluation-errors"
              total={detail.examples.n_errors}
              items={detail.examples.errors}
            />
            <ExampleList
              title="Aciertos"
              testId="evaluation-correct"
              total={detail.examples.n_correct}
              items={detail.examples.correct}
            />
          </section>
          <Predictions predictions={exported} />
        </div>
      )}
    </FetchBoundary>
  );
}
