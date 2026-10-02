/**
 * D04-02 — Experiments conectado al adaptador MLflow de D04-01 (`/api/experiments/runs`,
 * `/api/experiments/runs/:id` y sus artefactos): filtros, comparación, curvas, detalle,
 * runs auxiliares visibles pero fuera del conteo de campaña, y estados vacío/error.
 *
 * Los datos son los fixtures compartidos de `contracts/p3/fixtures` (más variantes armadas
 * aquí a partir de ellos): acreditan el componente, no la integración real, que se evidencia
 * aparte contra el run de D03-04.
 */
import "@testing-library/jest-dom/vitest";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { App } from "../src/App";
import type { ExperimentRun } from "../src/p3/contracts";
import {
  campaignCounts,
  compareParams,
  curveRows,
  EMPTY_FILTERS,
  filterOptions,
  filterRuns,
} from "../src/p3/experiments";
import { fixturePayload } from "./p3-fixtures";

type RunsResponse = {
  experiment_name: string;
  runs: ExperimentRun[];
  excluded: { run_id: string; run_kind: string | null; status: string; reasons: string[] }[];
};

const clone = <T,>(value: T): T => JSON.parse(JSON.stringify(value)) as T;
const TWO_RUNS = fixturePayload("experiment_runs_response", "valid-two-runs") as RunsResponse;
const MIXED = fixturePayload(
  "experiment_runs_response",
  "valid-auxiliary-and-incomplete-excluded"
) as RunsResponse;
const [FINISHED, RUNNING] = TWO_RUNS.runs as [ExperimentRun, ExperimentRun];

/** Otro run terminado y elegible, con params distintos (fila OFAT diferente). */
function otherFinished(): ExperimentRun {
  const run = clone(FINISHED);
  run.run_id = "c".repeat(32);
  run.start_time = "2026-09-27T18:00:00Z";
  run.end_time = "2026-09-27T18:20:00Z";
  run.params = { ...run.params, seed: 77, learning_rate: 0.0003, optimizer: "sgd" };
  run.tags = { ...run.tags, seed: 77, job_id: 2 };
  return run;
}

/** Respuesta mixta: 2 runs de training elegibles + 1 RUNNING + los 5 excluidos del fixture. */
function mixedResponse(): RunsResponse {
  return {
    experiment_name: "p3-cnn-classifier",
    runs: [otherFinished(), clone(FINISHED), clone(RUNNING)],
    excluded: clone(MIXED.excluded),
  };
}

// ---------------------------------------------------------------------------
// Lógica pura: filtros, conteo, comparación y curvas.
// ---------------------------------------------------------------------------
describe("lógica de Experiments", () => {
  it("sin filtros devuelve todos los runs de training en el orden de la API", () => {
    const response = mixedResponse();
    expect(filterRuns(response.runs, EMPTY_FILTERS).map((r) => r.run_id)).toEqual(
      response.runs.map((r) => r.run_id)
    );
  });

  it("filtra por estado, elegibilidad y params (y combinados)", () => {
    const { runs } = mixedResponse();
    const ids = (filters: Partial<typeof EMPTY_FILTERS>) =>
      filterRuns(runs, { ...EMPTY_FILTERS, ...filters }).map((r) => r.run_id);

    expect(ids({ status: "RUNNING" })).toEqual([RUNNING.run_id]);
    expect(ids({ eligibility: "eligible" })).toEqual(["c".repeat(32), FINISHED.run_id]);
    expect(ids({ eligibility: "not_eligible" })).toEqual([RUNNING.run_id]);
    expect(ids({ seed: "77" })).toEqual(["c".repeat(32)]);
    expect(ids({ optimizer: "adam" })).toEqual([FINISHED.run_id, RUNNING.run_id]);
    expect(ids({ learning_rate: "0.0003" })).toEqual(["c".repeat(32)]);
    expect(ids({ trainable_layers: "full" })).toEqual([]);
    expect(ids({ optimizer: "adam", eligibility: "eligible" })).toEqual([FINISHED.run_id]);
  });

  it("las opciones de cada filtro salen de los datos reales (sin duplicados)", () => {
    const options = filterOptions(mixedResponse().runs);
    expect(options.status).toEqual(["FINISHED", "RUNNING"]);
    expect(options.seed).toEqual(["7", "21", "77"]);
    expect(options.optimizer).toEqual(["adam", "sgd"]);
    expect(options.learning_rate).toEqual(["0.0003", "0.001"]);
  });

  it("los auxiliares y excluidos no inflan el conteo de campaña, pero se cuentan aparte", () => {
    const counts = campaignCounts(mixedResponse());
    expect(counts).toEqual({ training: 3, eligible: 2, excluded: 5 });
  });

  it("la comparación marca qué params difieren entre los runs elegidos", () => {
    const rows = compareParams([FINISHED, otherFinished()]);
    const byKey = Object.fromEntries(rows.map((row) => [row.key, row]));
    expect(byKey.seed).toMatchObject({ values: ["7", "77"], differs: true });
    expect(byKey.optimizer).toMatchObject({ values: ["adam", "sgd"], differs: true });
    expect(byKey.batch_size).toMatchObject({ values: ["16", "16"], differs: false });
    expect(rows).toHaveLength(15); // los 15 campos de TrainingConfig
  });

  it("las curvas son exactamente el history de cada run (sin interpolar épocas faltantes)", () => {
    const rows = curveRows([FINISHED, RUNNING], "val_accuracy");
    expect(rows.map((row) => row.epoch)).toEqual(FINISHED.history.map((h) => h.epoch));
    for (const h of FINISHED.history) {
      expect(rows[h.epoch - 1]?.[FINISHED.run_id]).toBe(h.val_accuracy);
    }
    // RUNNING tiene menos épocas: donde no hay dato queda null, no un valor inventado.
    for (const row of rows) {
      const own = RUNNING.history.find((h) => h.epoch === row.epoch);
      expect(row[RUNNING.run_id]).toBe(own ? own.val_accuracy : null);
    }
  });
});

// ---------------------------------------------------------------------------
// Página conectada a la API (fetch simulado con los contratos compartidos).
// ---------------------------------------------------------------------------
type Route = { status?: number; body: unknown };

function mockApi(routes: Record<string, Route | (() => Route)>) {
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const url = typeof input === "string" ? input : input.toString();
    const path = url.replace(/^https?:\/\/[^/]+/, "");
    const entry = routes[path];
    const route = typeof entry === "function" ? entry() : entry;
    if (!route) {
      return new Response(JSON.stringify({ error: "Recurso no encontrado." }), { status: 404 });
    }
    return new Response(JSON.stringify(route.body), {
      status: route.status ?? 200,
      headers: { "Content-Type": "application/json" },
    });
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function renderExperiments() {
  return render(
    <MemoryRouter initialEntries={["/ml/experiments"]}>
      <App />
    </MemoryRouter>
  );
}

const short = (id: string) => id.slice(0, 12);
const detailOf = (run: ExperimentRun) => ({
  run,
  artifacts: [
    { path: "checkpoint", is_dir: true, size_bytes: null },
    { path: "checkpoint/model.pt", is_dir: false, size_bytes: 44781003 },
  ],
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("página Experiments con datos de la API", () => {
  it("muestra el conteo de campaña y los auxiliares en una sección aparte, sin esconderlos", async () => {
    mockApi({ "/api/experiments/runs": { body: mixedResponse() } });
    renderExperiments();

    const counts = await screen.findByTestId("experiments-counts");
    expect(counts).toHaveTextContent("Runs de training: 3");
    expect(counts).toHaveTextContent("Elegibles para campaña: 2");
    expect(counts).toHaveTextContent("Auxiliares y excluidos: 5");

    const table = screen.getByTestId("p3-content");
    expect(within(table).getAllByTestId(/^run-row-/)).toHaveLength(3);

    const excluded = screen.getByTestId("experiments-excluded");
    expect(within(excluded).getAllByTestId(/^excluded-row-/)).toHaveLength(5);
    expect(within(excluded).getByText("controlled_task")).toBeInTheDocument();
    expect(within(excluded).getByText("short_run_instrumentation")).toBeInTheDocument();
    expect(within(excluded).getByText("persistence_check")).toBeInTheDocument();
    expect(within(excluded).getByText(/dvc_release_hash: falta en MLflow/)).toBeInTheDocument();
    // Un excluido nunca se puede elegir para comparar ni cuenta como run de training.
    expect(within(excluded).queryByRole("checkbox")).not.toBeInTheDocument();
  });

  it("los motivos de no elegibilidad de un run de training se ven en su fila", async () => {
    mockApi({ "/api/experiments/runs": { body: mixedResponse() } });
    renderExperiments();
    const row = await screen.findByTestId(`run-row-${RUNNING.run_id}`);
    expect(row).toHaveTextContent("No elegible");
    expect(row).toHaveTextContent(RUNNING.ineligible_reasons[0] as string);
    expect(screen.getByTestId(`run-row-${FINISHED.run_id}`)).toHaveTextContent("Elegible");
  });

  it("los filtros reducen la tabla y el conteo filtrado", async () => {
    mockApi({ "/api/experiments/runs": { body: mixedResponse() } });
    renderExperiments();
    await screen.findByTestId("p3-content");

    fireEvent.change(screen.getByLabelText("Elegibilidad"), { target: { value: "eligible" } });
    expect(screen.getAllByTestId(/^run-row-/)).toHaveLength(2);
    fireEvent.change(screen.getByLabelText("Optimizador"), { target: { value: "sgd" } });
    expect(screen.getAllByTestId(/^run-row-/)).toHaveLength(1);
    expect(screen.getByTestId("experiments-shown")).toHaveTextContent("1 de 3");

    fireEvent.change(screen.getByLabelText("Seed"), { target: { value: "7" } });
    expect(screen.queryAllByTestId(/^run-row-/)).toHaveLength(0);
    expect(screen.getByTestId("experiments-no-match")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Limpiar filtros" }));
    expect(screen.getAllByTestId(/^run-row-/)).toHaveLength(3);
  });

  it("comparar dos runs muestra sus params lado a lado, marca diferencias y superpone curvas", async () => {
    mockApi({ "/api/experiments/runs": { body: mixedResponse() } });
    renderExperiments();
    await screen.findByTestId("p3-content");
    expect(screen.queryByTestId("experiments-compare")).not.toBeInTheDocument();

    fireEvent.click(screen.getByLabelText(`Comparar ${short(FINISHED.run_id)}`));
    fireEvent.click(screen.getByLabelText(`Comparar ${short("c".repeat(32))}`));

    const compare = await screen.findByTestId("experiments-compare");
    const seedRow = within(compare).getByTestId("compare-param-seed");
    expect(seedRow).toHaveAttribute("data-differs", "true");
    expect(seedRow).toHaveTextContent("7");
    expect(seedRow).toHaveTextContent("77");
    expect(within(compare).getByTestId("compare-param-batch_size")).toHaveAttribute(
      "data-differs",
      "false"
    );
    const curves = within(compare).getByTestId("curve-table");
    expect(within(curves).getAllByRole("row")).toHaveLength(1 + FINISHED.history.length);
  });

  it("el detalle carga el run de la API y sus curvas son las de ese mismo run", async () => {
    const fetchMock = mockApi({
      "/api/experiments/runs": { body: mixedResponse() },
      [`/api/experiments/runs/${FINISHED.run_id}`]: { body: detailOf(FINISHED) },
    });
    renderExperiments();
    await screen.findByTestId("p3-content");
    fireEvent.click(screen.getByRole("button", { name: `Ver ${short(FINISHED.run_id)}` }));

    const detail = await screen.findByTestId("experiments-detail");
    expect(fetchMock).toHaveBeenCalledWith(`/api/experiments/runs/${FINISHED.run_id}`);
    expect(detail).toHaveTextContent(FINISHED.run_id);
    expect(detail).toHaveTextContent(FINISHED.tags.manifest_hash);
    expect(detail).toHaveTextContent(FINISHED.tags.git_commit);
    expect(within(detail).getByTestId("detail-job")).toHaveTextContent(
      new RegExp(`^${FINISHED.tags.job_id}$`)
    );
    expect(detail).toHaveTextContent(`Mejor época: ${FINISHED.summary?.best_epoch}`);

    const curves = within(detail).getByTestId("curve-table");
    const rows = within(curves).getAllByRole("row").slice(1);
    expect(rows).toHaveLength(FINISHED.history.length);
    FINISHED.history.forEach((h, i) => {
      expect(rows[i]).toHaveTextContent(String(h.epoch));
      expect(rows[i]).toHaveTextContent(h.val_accuracy.toFixed(4));
      expect(rows[i]).toHaveTextContent(h.val_loss.toFixed(4));
    });

    const link = within(detail).getByRole("link", { name: "checkpoint/model.pt" });
    expect(link).toHaveAttribute(
      "href",
      `/api/experiments/runs/${FINISHED.run_id}/artifacts/checkpoint/model.pt`
    );
  });

  it("el detalle muestra el motivo si la API lo rechaza (409/503), sin datos inventados", async () => {
    mockApi({
      "/api/experiments/runs": { body: mixedResponse() },
      [`/api/experiments/runs/${RUNNING.run_id}`]: {
        status: 503,
        body: { error: "mlflow_unavailable: no se pudo conectar con MLflow" },
      },
    });
    renderExperiments();
    await screen.findByTestId("p3-content");
    fireEvent.click(screen.getByRole("button", { name: `Ver ${short(RUNNING.run_id)}` }));
    expect(
      await screen.findByText(
        "El servidor respondió con estado 503: mlflow_unavailable: no se pudo conectar con MLflow"
      )
    ).toBeInTheDocument();
    expect(screen.queryByTestId("curve-table")).not.toBeInTheDocument();
  });

  it("Actualizar vuelve a pedir los runs y muestra el estado nuevo", async () => {
    let call = 0;
    const fetchMock = mockApi({
      "/api/experiments/runs": () => {
        call += 1;
        const body = mixedResponse();
        if (call > 1) {
          const now = body.runs.find((r) => r.run_id === RUNNING.run_id) as ExperimentRun;
          Object.assign(now, clone(FINISHED), { run_id: RUNNING.run_id });
        }
        return { body };
      },
    });
    renderExperiments();
    expect(await screen.findByTestId(`run-row-${RUNNING.run_id}`)).toHaveTextContent("RUNNING");
    fireEvent.click(screen.getByRole("button", { name: "Actualizar" }));
    await waitFor(() =>
      expect(screen.getByTestId(`run-row-${RUNNING.run_id}`)).toHaveTextContent("FINISHED")
    );
    expect(fetchMock.mock.calls.filter(([url]) => url === "/api/experiments/runs")).toHaveLength(2);
    expect(screen.getByTestId("experiments-counts")).toHaveTextContent("Elegibles para campaña: 3");
  });

  it("sin runs de training pero con auxiliares: vacío de campaña y auxiliares visibles", async () => {
    mockApi({
      "/api/experiments/runs": { body: { ...mixedResponse(), runs: [] } },
    });
    renderExperiments();
    expect(await screen.findByTestId("p3-state-empty")).toHaveTextContent(
      "Todavía no hay corridas de training en MLflow"
    );
    expect(screen.getByTestId("experiments-excluded")).toBeInTheDocument();
  });

  it("MLflow caído: error con el motivo que da la API (no una lista vacía)", async () => {
    mockApi({
      "/api/experiments/runs": {
        status: 503,
        body: { error: "mlflow_unavailable: no se pudo conectar con MLflow en http://mlflow:5000" },
      },
    });
    renderExperiments();
    expect(
      await screen.findByText(
        "El servidor respondió con estado 503: mlflow_unavailable: no se pudo conectar con MLflow en http://mlflow:5000"
      )
    ).toBeInTheDocument();
    expect(screen.queryByTestId("p3-state-empty")).not.toBeInTheDocument();
  });
});
