/**
 * D05-03 (preparación) — Experiments muestra la campaña aceptada y el candidato tal como
 * los entrega la selección (D04-04/D05-02, `GET /api/selection`), sin un segundo ranking:
 * marca las filas aceptadas y el candidato, lo rotula como propuesta hasta el cierre
 * (D05-08) y coteja que el candidato y sus métricas sean los del run de MLflow que
 * muestra Experiments; si no coinciden, lo dice y no presenta la aceptación.
 *
 * Fixtures compartidos (`contracts/p3`): prueban el render y el cotejo, no acreditan la
 * campaña real, que se evidencia aparte con la lista y el candidato de D05-02.
 */
import "@testing-library/jest-dom/vitest";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { App } from "../src/App";
import type { ExperimentRun } from "../src/p3/contracts";
import {
  attemptLabel,
  attemptMarks,
  campaignRows,
  selectionCrossCheck,
} from "../src/p3/selection";
import { fixturePayload } from "./p3-fixtures";

type Selection = {
  status: "open" | "candidate" | "closed";
  candidate: RankedRun | null;
  ranking: RankedRun[];
  excluded: { run_id: string | null; reason: string; detail: string }[];
  campaign_rows: number[];
  min_comparable_runs: number;
  ready_to_close: boolean;
  closed_at: string | null;
  proposed_at: string | null;
};
type RankedRun = {
  run_id: string;
  campaign_row: number;
  start_time: string;
  best_epoch: number;
  val_accuracy: number;
  val_macro_f1: number;
  val_loss: number;
};

const clone = <T,>(value: T): T => JSON.parse(JSON.stringify(value)) as T;
const CANDIDATE = fixturePayload("selection_state", "valid-candidate") as Selection;
const CLOSED = fixturePayload("selection_state", "valid-closed") as Selection;
const OPEN = fixturePayload("selection_state", "valid-open") as Selection;
const BASE = (
  fixturePayload("experiment_runs_response", "valid-two-runs") as { runs: ExperimentRun[] }
).runs[0] as ExperimentRun;
const AUX = (
  fixturePayload("experiment_runs_response", "valid-auxiliary-and-incomplete-excluded") as {
    excluded: unknown[];
  }
).excluded;

/** Run de MLflow (contrato D04-01) cuyo resumen es el de una fila del ranking. */
function runFor(ranked: RankedRun): ExperimentRun {
  const run = clone(BASE);
  run.run_id = ranked.run_id;
  run.start_time = ranked.start_time;
  run.end_time = ranked.start_time;
  const history = Array.from({ length: ranked.best_epoch }, (_, i) => ({
    ...BASE.history[0],
    epoch: i + 1,
    val_accuracy: i + 1 === ranked.best_epoch ? ranked.val_accuracy : ranked.val_accuracy - 0.05,
    val_macro_f1: i + 1 === ranked.best_epoch ? ranked.val_macro_f1 : ranked.val_macro_f1 - 0.05,
    val_loss: i + 1 === ranked.best_epoch ? ranked.val_loss : ranked.val_loss + 0.1,
  })) as ExperimentRun["history"];
  run.history = history;
  run.summary = {
    best_epoch: ranked.best_epoch,
    best_val_accuracy: ranked.val_accuracy,
    best_val_macro_f1: ranked.val_macro_f1,
    best_val_loss: ranked.val_loss,
  };
  return run;
}

/** Los 11 runs aceptados + un training no aceptado (smoke) + auxiliares excluidos. */
function campaignRuns(selection: Selection = CANDIDATE) {
  const smoke = clone(BASE);
  smoke.run_id = "c46e4c3ab2bb4ee18c37571adbb65d92";
  return {
    experiment_name: "p3-cnn-classifier",
    runs: [...selection.ranking.map(runFor), smoke],
    // Los auxiliares del fixture reusan ids 1…5; aquí llevan ids propios para no chocar con el ranking.
    excluded: (clone(AUX) as { run_id: string }[]).map((aux, i) => ({
      ...aux,
      run_id: `e${i}`.padEnd(32, "e"),
    })),
  };
}


const RETRY_ID = "7576d02bb5564dbf94429373e9587a3a";

/** Forma de `GET /api/selection/campaign` (D05-02): solo lo que Experiments lee. */
function dossierFor(selection: Selection) {
  return {
    rows: selection.ranking.map((ranked) => ({
      row: ranked.campaign_row,
      status: "accepted",
      representative: ranked.run_id,
      attempts: [
        { job_id: 1, run_id: ranked.run_id, role: "representative", reason: null, detail: null },
        ...(ranked.campaign_row === selection.ranking[0]?.campaign_row
          ? [
              {
                job_id: 5,
                run_id: RETRY_ID,
                role: "retry",
                reason: "duplicate_campaign_row",
                detail: `La fila ya cuenta con ${ranked.run_id} (corrida más temprana).`,
              },
            ]
          : []),
      ],
    })),
    unattributed: [],
  };
}

describe("cotejo selección ↔ Experiments (sin segundo ranking)", () => {
  it("candidato y filas aceptadas coinciden con los runs de MLflow → sin problemas", () => {
    const check = selectionCrossCheck(campaignRuns().runs, CANDIDATE);
    expect(check.problems).toEqual([]);
    expect(check.candidate?.run_id).toBe(CANDIDATE.candidate?.run_id);
  });

  it("candidato que no aparece en Experiments → problema que lo nombra", () => {
    const runs = campaignRuns().runs.filter((r) => r.run_id !== CANDIDATE.candidate?.run_id);
    const check = selectionCrossCheck(runs, CANDIDATE);
    expect(check.problems.join(" ")).toMatch(
      new RegExp(`candidato ${CANDIDATE.candidate?.run_id}.*no aparece`)
    );
  });

  it("métricas del candidato distintas de las del run (best_epoch o val_accuracy) → problema", () => {
    const runs = campaignRuns().runs;
    const run = runs.find((r) => r.run_id === CANDIDATE.candidate?.run_id) as ExperimentRun;
    (run.summary as NonNullable<ExperimentRun["summary"]>).best_val_accuracy = 0.5;
    const check = selectionCrossCheck(runs, CANDIDATE);
    expect(check.problems.join(" ")).toMatch(/val_accuracy/);
  });

  it("una fila aceptada (no candidata) que no aparece en Experiments → problema que la nombra", () => {
    const missing = CANDIDATE.ranking[5] as RankedRun;
    const runs = campaignRuns().runs.filter((r) => r.run_id !== missing.run_id);
    const check = selectionCrossCheck(runs, CANDIDATE);
    expect(check.problems).toContain(
      `La fila ${missing.campaign_row} (${missing.run_id}) no aparece en Experiments.`
    );
  });

  it("una fila aceptada que en Experiments no es elegible → problema", () => {
    const runs = campaignRuns().runs;
    const row = runs.find((r) => r.run_id === CANDIDATE.ranking[3]?.run_id) as ExperimentRun;
    row.campaign_eligible = false;
    row.ineligible_reasons = ["sin checkpoint_sha256: no hay checkpoint verificado"];
    row.checkpoint_sha256 = null;
    const check = selectionCrossCheck(runs, CANDIDATE);
    expect(check.problems.join(" ")).toMatch(new RegExp(`${CANDIDATE.ranking[3]?.run_id}.*no es elegible`));
  });

  it("la fila de campaña de cada run aceptado sale de la selección, no se recalcula", () => {
    const rows = campaignRows(CANDIDATE);
    expect(rows.get(CANDIDATE.ranking[0]?.run_id as string)).toBe(1);
    expect(rows.size).toBe(CANDIDATE.ranking.length);
  });

  it("el expediente solo rotula intentos no representantes, con su fila", () => {
    const marks = attemptMarks({
      rows: [
        {
          row: 1,
          attempts: [
            { run_id: "a".repeat(32), role: "representative", reason: null },
            { run_id: RETRY_ID, role: "retry", reason: "duplicate_campaign_row" },
            { run_id: null, role: "excluded", reason: "job_failed" },
          ],
        },
      ],
      unattributed: [{ run_id: "b".repeat(32), role: "excluded", reason: "adapter_excluded" }],
    });
    expect([...marks.keys()]).toEqual([RETRY_ID, "b".repeat(32)]);
    expect(attemptLabel(marks.get(RETRY_ID))).toBe("Reintento · fila 1");
    expect(attemptLabel(marks.get("b".repeat(32)))).toBe("Excluido");
    expect(attemptLabel(undefined)).toBe("No aceptado por D05-02");
    expect(attemptMarks(null).size).toBe(0);
  });

  it("selección abierta: no hay candidato que cotejar", () => {
    expect(selectionCrossCheck(campaignRuns().runs, OPEN)).toEqual({ candidate: null, problems: [] });
  });
});

type Route = { status?: number; body: unknown };
function mockApi(routes: Record<string, Route>) {
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const url = typeof input === "string" ? input : input.toString();
    const route = routes[url.replace(/^https?:\/\/[^/]+/, "")];
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
const renderExperiments = () =>
  render(
    <MemoryRouter initialEntries={["/ml/experiments"]}>
      <App />
    </MemoryRouter>
  );

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("Experiments con la selección", () => {
  it("candidato propuesto: se marca su fila y se rotula como propuesta pendiente de D05-08", async () => {
    mockApi({
      "/api/experiments/runs": { body: campaignRuns() },
      "/api/selection": { body: CANDIDATE },
    });
    renderExperiments();
    const panel = await screen.findByTestId("experiments-selection");
    expect(panel).toHaveTextContent(CANDIDATE.candidate?.run_id as string);
    expect(panel).toHaveTextContent("Propuesta pendiente de cierre (D05-08)");
    expect(panel).toHaveTextContent("Filas de campaña comparables: 11 (mínimo 10)");
    expect(panel).not.toHaveTextContent("MODEL SELECTION CLOSED");
    const row = screen.getByTestId(`run-row-${CANDIDATE.candidate?.run_id}`);
    expect(row).toHaveTextContent("Candidato propuesto");
    expect(row).toHaveTextContent("Fila 1");
    expect(screen.getAllByText("Candidato propuesto")).toHaveLength(1);
  });

  it("los auxiliares y el training fuera de la campaña no suman a las filas aceptadas", async () => {
    mockApi({
      "/api/experiments/runs": { body: campaignRuns() },
      "/api/selection": { body: CANDIDATE },
    });
    renderExperiments();
    await screen.findByTestId("experiments-selection");
    expect(screen.getAllByText(/^Fila \d+$/)).toHaveLength(11);
    expect(screen.getByTestId("run-row-c46e4c3ab2bb4ee18c37571adbb65d92")).not.toHaveTextContent(
      /Fila \d+/
    );
  });

  it("filtro 'Campaña aceptada' deja solo los runs del ranking de la selección", async () => {
    mockApi({
      "/api/experiments/runs": { body: campaignRuns() },
      "/api/selection": { body: CANDIDATE },
    });
    renderExperiments();
    await screen.findByTestId("experiments-selection");
    expect(screen.getAllByTestId(/^run-row-/)).toHaveLength(12);
    fireEvent.change(screen.getByLabelText("Campaña"), { target: { value: "accepted" } });
    expect(screen.getAllByTestId(/^run-row-/)).toHaveLength(11);
  });

  it("selección cerrada: lo dice con la fecha", async () => {
    mockApi({
      "/api/experiments/runs": { body: campaignRuns(CLOSED) },
      "/api/selection": { body: CLOSED },
    });
    renderExperiments();
    const panel = await screen.findByTestId("experiments-selection");
    expect(panel).toHaveTextContent("MODEL SELECTION CLOSED");
    expect(screen.getByTestId(`run-row-${CLOSED.candidate?.run_id}`)).toHaveTextContent(
      "Candidato seleccionado"
    );
  });

  it("selección abierta: sin candidato todavía", async () => {
    mockApi({
      "/api/experiments/runs": { body: campaignRuns() },
      "/api/selection": { body: OPEN },
    });
    renderExperiments();
    expect(await screen.findByTestId("experiments-selection")).toHaveTextContent(
      "Todavía no hay candidato propuesto"
    );
    expect(screen.queryByText("Candidato propuesto")).not.toBeInTheDocument();
  });

  it("candidato discordante: aviso con el motivo y sin presentar la aceptación", async () => {
    const runs = campaignRuns();
    runs.runs = runs.runs.filter((r) => r.run_id !== CANDIDATE.candidate?.run_id);
    mockApi({ "/api/experiments/runs": { body: runs }, "/api/selection": { body: CANDIDATE } });
    renderExperiments();
    const alert = await screen.findByTestId("selection-mismatch");
    expect(alert).toHaveTextContent("no aparece");
    expect(screen.queryAllByText(/^Fila \d+$/)).toHaveLength(0);
    expect(screen.queryByText("Candidato propuesto")).not.toBeInTheDocument();
    expect(within(screen.getByTestId("experiments-selection")).queryByText(
      "Propuesta pendiente de cierre (D05-08)"
    )).not.toBeInTheDocument();
  });

  it("Actualizar vuelve a pedir la selección (abierta → candidato propuesto)", async () => {
    const routes: Record<string, Route> = {
      "/api/experiments/runs": { body: campaignRuns() },
      "/api/selection": { body: OPEN },
    };
    mockApi(routes);
    renderExperiments();
    expect(await screen.findByTestId("experiments-selection")).toHaveTextContent(
      "Todavía no hay candidato propuesto"
    );
    routes["/api/selection"] = { body: CANDIDATE };
    fireEvent.click(screen.getByRole("button", { name: "Actualizar" }));
    expect(
      await screen.findByText(/Propuesta pendiente de cierre \(D05-08\)/)
    ).toBeInTheDocument();
  });

  it("reintentos del expediente de D05-02: visibles con su estado y fuera del conteo", async () => {
    const runs = campaignRuns();
    const representative = CANDIDATE.ranking[0] as RankedRun;
    const retry = runFor({ ...representative, run_id: RETRY_ID });
    runs.runs.push(retry);
    mockApi({
      "/api/experiments/runs": { body: runs },
      "/api/selection": { body: CANDIDATE },
      "/api/selection/campaign": { body: dossierFor(CANDIDATE) },
    });
    renderExperiments();
    await screen.findByTestId("experiments-selection");
    const counts = await screen.findByTestId("experiments-counts");
    expect(counts).toHaveTextContent("Runs de training: 13");
    expect(counts).toHaveTextContent("Cuentan para la campaña (filas aceptadas por D05-02): 11");
    expect(counts).toHaveTextContent("Reintentos y no aceptados: 2 (no cuentan)");
    expect(counts).not.toHaveTextContent("Elegibles para campaña");
    const retryRow = await screen.findByTestId(`run-row-${RETRY_ID}`);
    expect(await within(retryRow).findByText("Reintento · fila 1")).toBeInTheDocument();
    expect(retryRow).not.toHaveTextContent(/^Fila \d+$/);
    expect(screen.getByTestId("run-row-c46e4c3ab2bb4ee18c37571adbb65d92")).toHaveTextContent(
      "No aceptado por D05-02"
    );
    expect(screen.getAllByText(/^Fila \d+$/)).toHaveLength(11);
  });

  it("sin expediente (falla /selection/campaign) los no aceptados igual no cuentan", async () => {
    const runs = campaignRuns();
    runs.runs.push(runFor({ ...(CANDIDATE.ranking[0] as RankedRun), run_id: RETRY_ID }));
    mockApi({
      "/api/experiments/runs": { body: runs },
      "/api/selection": { body: CANDIDATE },
    });
    renderExperiments();
    await screen.findByTestId("experiments-selection");
    const counts = screen.getByTestId("experiments-counts");
    expect(counts).toHaveTextContent("Cuentan para la campaña (filas aceptadas por D05-02): 11");
    expect(counts).toHaveTextContent("Reintentos y no aceptados: 2 (no cuentan)");
    expect(screen.getByTestId(`run-row-${RETRY_ID}`)).toHaveTextContent("No aceptado por D05-02");
  });

  it("selección abierta: el conteo sigue siendo el de elegibles del adaptador", async () => {
    mockApi({
      "/api/experiments/runs": { body: campaignRuns() },
      "/api/selection": { body: OPEN },
    });
    renderExperiments();
    await screen.findByTestId("experiments-selection");
    expect(screen.getByTestId("experiments-counts")).toHaveTextContent("Elegibles para campaña: 12");
    expect(screen.queryByText("No aceptado por D05-02")).not.toBeInTheDocument();
  });

  it("si /api/selection falla, el motivo sale en su panel y los runs siguen visibles", async () => {
    mockApi({
      "/api/experiments/runs": { body: campaignRuns() },
      "/api/selection": {
        status: 503,
        body: { error: "el adaptador de runs de MLflow (D04-01) aún no está integrado" },
      },
    });
    renderExperiments();
    expect(
      await screen.findByText(
        "El servidor respondió con estado 503: el adaptador de runs de MLflow (D04-01) aún no está integrado"
      )
    ).toBeInTheDocument();
    expect(screen.getAllByTestId(/^run-row-/)).toHaveLength(12);
  });
});
