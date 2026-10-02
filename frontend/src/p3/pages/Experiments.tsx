import { useState } from "react";
import { PageHeader } from "@/pipeline/components/PageHeader";
import { useCampaignDossier, useExperimentRuns, useSelection } from "../api";
import { FetchBoundary } from "../components/FetchBoundary";
import { StatePanel } from "../components/StatePanel";
import type { ExcludedRun, ExperimentRun, SelectionState } from "../contracts";
import {
  ALL,
  acceptedCounts,
  campaignCounts,
  type EligibilityFilter,
  EMPTY_FILTERS,
  filterOptions,
  filterRuns,
  type RunFilters,
} from "../experiments";
import { CompareRuns } from "../experiments/CompareRuns";
import { RunDetail } from "../experiments/RunDetail";
import { SelectionPanel } from "../experiments/SelectionPanel";
import { dateTime, percent, shortHash } from "../format";
import {
  type AttemptMark,
  attemptLabel,
  attemptMarks,
  campaignRows,
  selectionCrossCheck,
} from "../selection";

const selectClass =
  "rounded-lg border border-border bg-surface px-2 py-1 text-sm text-ink focus:outline-none focus:ring-2 focus:ring-accent/40";

/**
 * D01-05 / D04-02 — Experiments: runs reales del experimento MLflow `p3-cnn-classifier`
 * servidos por el adaptador de D04-01. Filtros, comparación, curvas y detalle; los runs
 * auxiliares y excluidos se muestran aparte y nunca cuentan para la campaña. La selección
 * del modelo usa solo validation y no se hace aquí (D04-04 / D05-02).
 */
export function ExperimentsPage() {
  const runs = useExperimentRuns();
  const selection = useSelection();
  const dossier = useCampaignDossier();
  const [filters, setFilters] = useState<RunFilters>(EMPTY_FILTERS);
  const [compared, setCompared] = useState<string[]>([]);
  const [detailId, setDetailId] = useState<string | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);

  const toggleCompare = (runId: string) =>
    setCompared((current) =>
      current.includes(runId) ? current.filter((id) => id !== runId) : [...current, runId]
    );

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-start justify-between gap-4">
        <PageHeader
          title="Experiments"
          subtitle="Corridas registradas en MLflow; la selección usa solo métricas de validation."
        />
        <button
          type="button"
          className="rounded-lg border border-border px-3 py-1.5 text-sm text-ink hover:bg-surface"
          onClick={() => {
            runs.reload();
            selection.reload();
            dossier.reload();
            setRefreshKey((key) => key + 1);
          }}
        >
          Actualizar
        </button>
      </div>
      <FetchBoundary results={[runs]}>
        {(data) => {
          const marks = selectionMarks(
            data.runs,
            selection.status === "success" ? selection.data : null,
            dossier.status === "success" ? dossier.data : null
          );
          const counts = campaignCounts(data);
          const accepted = acceptedCounts(data.runs, marks.accepted);
          const shown = filterRuns(data.runs, filters, marks.accepted);
          const selected = compared
            .map((id) => data.runs.find((run) => run.run_id === id))
            .filter((run): run is ExperimentRun => run !== undefined);
          return (
            <>
              <section
                data-testid="experiments-counts"
                className="flex flex-wrap gap-x-6 gap-y-1 rounded-2xl border border-border bg-surface px-5 py-3 text-sm"
              >
                <span>Runs de training: {counts.training}</span>
                {accepted === null ? (
                  <span>Elegibles para campaña: {counts.eligible}</span>
                ) : (
                  <>
                    <span>
                      Cuentan para la campaña (filas aceptadas por D05-02): {accepted.accepted}
                    </span>
                    <span className="text-ink-muted">
                      Reintentos y no aceptados: {accepted.notAccepted} (no cuentan)
                    </span>
                  </>
                )}
                <span className="text-ink-muted">
                  Auxiliares y excluidos: {counts.excluded} (no cuentan para la campaña)
                </span>
              </section>

              <SelectionPanel selection={selection} runs={data.runs} />

              {data.runs.length === 0 ? (
                <StatePanel
                  variant="empty"
                  title={
                    data.excluded.length === 0
                      ? "Todavía no hay corridas en MLflow."
                      : "Todavía no hay corridas de training en MLflow."
                  }
                >
                  La campaña de al menos diez corridas válidas aparecerá aquí.
                </StatePanel>
              ) : (
                <>
                  <Filters
                    runs={data.runs}
                    filters={filters}
                    onChange={setFilters}
                    withCampaign={marks.accepted.size > 0}
                  />
                  <p data-testid="experiments-shown" className="text-xs text-ink-muted">
                    Mostrando {shown.length} de {data.runs.length} runs de training
                  </p>
                  {shown.length === 0 ? (
                    <p
                      data-testid="experiments-no-match"
                      className="rounded-2xl border border-dashed border-border-strong px-5 py-6 text-center text-sm text-ink-muted"
                    >
                      Ningún run de training cumple los filtros.
                    </p>
                  ) : (
                    <RunsTable
                      runs={shown}
                      compared={compared}
                      onToggleCompare={toggleCompare}
                      onShowDetail={setDetailId}
                      marks={marks}
                    />
                  )}
                </>
              )}

              {selected.length >= 2 && <CompareRuns runs={selected} />}
              {detailId !== null && (
                <RunDetail
                  key={`${detailId}-${refreshKey}`}
                  runId={detailId}
                  onClose={() => setDetailId(null)}
                />
              )}
              {data.excluded.length > 0 && <ExcludedRuns excluded={data.excluded} />}
            </>
          );
        }}
      </FetchBoundary>
    </div>
  );
}

function Filters({
  runs,
  filters,
  onChange,
  withCampaign,
}: Readonly<{
  runs: readonly ExperimentRun[];
  filters: RunFilters;
  onChange: (filters: RunFilters) => void;
  withCampaign: boolean;
}>) {
  const options = filterOptions(runs);
  const select = (
    id: keyof RunFilters,
    label: string,
    values: readonly string[],
    labels?: Record<string, string>
  ) => (
    <label key={id} className="flex flex-col gap-1 text-xs font-medium text-ink">
      {label}
      <select
        className={selectClass}
        value={filters[id]}
        onChange={(event) => onChange({ ...filters, [id]: event.target.value })}
      >
        <option value={ALL}>Todos</option>
        {values.map((value) => (
          <option key={value} value={value}>
            {labels?.[value] ?? value}
          </option>
        ))}
      </select>
    </label>
  );
  return (
    <section aria-label="Filtros" className="flex flex-wrap items-end gap-4">
      {withCampaign &&
        select("campaign", "Campaña", ["accepted"], { accepted: "Campaña aceptada" })}
      {select("status", "Estado", options.status)}
      {select("eligibility", "Elegibilidad", ["eligible", "not_eligible"] as EligibilityFilter[], {
        eligible: "Elegibles",
        not_eligible: "No elegibles",
      })}
      {select("seed", "Seed", options.seed)}
      {select("trainable_layers", "Capas entrenables", options.trainable_layers)}
      {select("optimizer", "Optimizador", options.optimizer)}
      {select("learning_rate", "Learning rate", options.learning_rate)}
      <button
        type="button"
        className="text-xs text-ink-muted underline"
        onClick={() => onChange(EMPTY_FILTERS)}
      >
        Limpiar filtros
      </button>
    </section>
  );
}

function RunsTable({
  runs,
  compared,
  onToggleCompare,
  onShowDetail,
  marks,
}: Readonly<{
  runs: readonly ExperimentRun[];
  compared: readonly string[];
  onToggleCompare: (runId: string) => void;
  onShowDetail: (runId: string) => void;
  marks: SelectionMarks;
}>) {
  return (
    <div data-testid="p3-content" className="overflow-x-auto rounded-2xl border border-border">
      <table className="w-full text-left text-sm">
        <thead className="bg-surface text-xs text-ink-muted">
          <tr>
            <th className="px-3 py-2">Comparar</th>
            <th className="px-3 py-2">Run</th>
            <th className="px-3 py-2">Estado</th>
            <th className="px-3 py-2">Campaña</th>
            <th className="px-3 py-2">Seed</th>
            <th className="px-3 py-2">Capas</th>
            <th className="px-3 py-2">LR</th>
            <th className="px-3 py-2">Optimizador</th>
            <th className="px-3 py-2">Batch</th>
            <th className="px-3 py-2">Épocas</th>
            <th className="px-3 py-2">Mejor val acc.</th>
            <th className="px-3 py-2">Inicio</th>
            <th className="px-3 py-2" />
          </tr>
        </thead>
        <tbody>
          {runs.map((run) => (
            <tr
              key={run.run_id}
              data-testid={`run-row-${run.run_id}`}
              className="border-t border-border"
            >
              <td className="px-3 py-2">
                <input
                  type="checkbox"
                  aria-label={`Comparar ${shortHash(run.run_id)}`}
                  checked={compared.includes(run.run_id)}
                  onChange={() => onToggleCompare(run.run_id)}
                />
              </td>
              <td className="px-3 py-2 font-mono text-xs">{shortHash(run.run_id)}</td>
              <td className="px-3 py-2">{run.status}</td>
              <td className="px-3 py-2 text-xs">
                {run.campaign_eligible ? (
                  <span className="rounded-full bg-status-done-soft px-2 py-0.5 text-status-done">
                    Elegible
                  </span>
                ) : (
                  <span>
                    <span className="rounded-full bg-status-pending-soft px-2 py-0.5 text-status-pending">
                      No elegible
                    </span>
                    <span className="mt-1 block text-ink-muted">
                      {run.ineligible_reasons.join("; ")}
                    </span>
                  </span>
                )}
                <SelectionBadges runId={run.run_id} marks={marks} />
              </td>
              <td className="px-3 py-2">{run.params.seed}</td>
              <td className="px-3 py-2">{run.params.trainable_layers}</td>
              <td className="px-3 py-2">{run.params.learning_rate}</td>
              <td className="px-3 py-2">{run.params.optimizer}</td>
              <td className="px-3 py-2">{run.params.batch_size}</td>
              <td className="px-3 py-2">
                {run.history.length}/{run.params.max_epochs}
              </td>
              <td className="px-3 py-2">
                {run.summary ? percent(run.summary.best_val_accuracy) : "—"}
              </td>
              <td className="px-3 py-2 text-xs">{dateTime(run.start_time)}</td>
              <td className="px-3 py-2">
                <button
                  type="button"
                  className="text-xs text-accent underline"
                  aria-label={`Ver ${shortHash(run.run_id)}`}
                  onClick={() => onShowDetail(run.run_id)}
                >
                  Ver
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ExcludedRuns({ excluded }: Readonly<{ excluded: readonly ExcludedRun[] }>) {
  return (
    <section
      data-testid="experiments-excluded"
      aria-label="Runs auxiliares y excluidos"
      className="flex flex-col gap-2 rounded-2xl border border-dashed border-border-strong p-5"
    >
      <h2 className="text-sm font-semibold text-ink">
        Runs auxiliares y excluidos ({excluded.length})
      </h2>
      <p className="text-xs text-ink-muted">
        Existen en MLflow pero no son corridas comparables de la campaña: no cuentan en el total ni
        se pueden elegir para comparar o seleccionar.
      </p>
      <table className="w-full text-left text-xs">
        <thead className="text-ink-muted">
          <tr>
            <th className="px-2 py-1">Run</th>
            <th className="px-2 py-1">Tipo</th>
            <th className="px-2 py-1">Estado</th>
            <th className="px-2 py-1">Motivo</th>
          </tr>
        </thead>
        <tbody>
          {excluded.map((run) => (
            <tr
              key={run.run_id}
              data-testid={`excluded-row-${run.run_id}`}
              className="border-t border-border"
            >
              <td className="px-2 py-1 font-mono">{shortHash(run.run_id)}</td>
              <td className="px-2 py-1">
                <span className="rounded-full bg-surface px-2 py-0.5 font-mono">
                  {run.run_kind ?? "sin tipo"}
                </span>
              </td>
              <td className="px-2 py-1">{run.status}</td>
              <td className="px-2 py-1">{run.reasons.join("; ")}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

type SelectionMarks = {
  accepted: ReadonlySet<string>;
  attempts: ReadonlyMap<string, AttemptMark>;
  rows: ReadonlyMap<string, number>;
  candidateId: string | null;
  candidateLabel: string;
};

const NO_MARKS: SelectionMarks = {
  accepted: new Set(),
  attempts: new Map(),
  rows: new Map(),
  candidateId: null,
  candidateLabel: "",
};

/**
 * D05-03 — Marcas que vienen tal cual de la selección (filas aceptadas y candidato). Si el
 * cotejo con MLflow encuentra discrepancias, no se marca nada como aceptado.
 */
function selectionMarks(
  runs: readonly ExperimentRun[],
  selection: SelectionState | null,
  dossier: Parameters<typeof attemptMarks>[0]
): SelectionMarks {
  if (selection === null || selection.candidate === null) return NO_MARKS;
  if (selectionCrossCheck(runs, selection).problems.length > 0) return NO_MARKS;
  const rows = campaignRows(selection);
  return {
    accepted: new Set(rows.keys()),
    attempts: attemptMarks(dossier),
    rows,
    candidateId: selection.candidate.run_id,
    candidateLabel:
      selection.status === "closed" ? "Candidato seleccionado" : "Candidato propuesto",
  };
}

function SelectionBadges({ runId, marks }: Readonly<{ runId: string; marks: SelectionMarks }>) {
  const row = marks.rows.get(runId);
  if (row === undefined) {
    if (marks.accepted.size === 0) return null;
    return (
      <span className="mt-1 flex flex-wrap gap-1">
        <span className="rounded-full bg-status-pending-soft px-2 py-0.5 text-status-pending">
          {attemptLabel(marks.attempts.get(runId))}
        </span>
      </span>
    );
  }
  return (
    <span className="mt-1 flex flex-wrap gap-1">
      <span className="rounded-full bg-surface px-2 py-0.5 font-mono">Fila {row}</span>
      {runId === marks.candidateId && (
        <span className="rounded-full bg-accent/10 px-2 py-0.5 font-medium text-accent">
          {marks.candidateLabel}
        </span>
      )}
    </span>
  );
}
