/**
 * D06-05 — Evaluation con la evaluación `ready`: procedencia, umbral, métricas por clase,
 * ejemplos por crop_id y predicciones consultables/exportables.
 *
 * Las respuestas son los fixtures compartidos `contracts/p3/fixtures` (datos de ejemplo
 * etiquetados): prueban que la página muestra exactamente lo que devuelve la API y que solo
 * hace GET. No son resultados del frozen test: eso es D06-01.
 */
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { App } from "../src/App";
import type { EvaluationDetails, EvaluationPredictions } from "../src/p3/contracts";
import { fixturePayload } from "./p3-fixtures";

type Routes = Record<string, { status?: number; body: unknown }>;

const READY = fixturePayload("evaluation_response", "valid-ready");
const DETAILS = fixturePayload("evaluation_details", "valid-official") as EvaluationDetails;
const PREDICTIONS = fixturePayload(
  "evaluation_predictions",
  "valid-official"
) as EvaluationPredictions;

/** Método HTTP de cada llamada a fetch, en orden. */
let methods: string[] = [];

function mockApi(routes: Routes) {
  methods = [];
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    methods.push((init?.method ?? "GET").toUpperCase());
    const path = String(input).replace(/^https?:\/\/[^/]+/, "");
    const reply = routes[path];
    return new Response(JSON.stringify(reply?.body ?? { error: "Recurso no encontrado." }), {
      status: reply ? (reply.status ?? 200) : 404,
      headers: { "Content-Type": "application/json" },
    });
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

const readyApi = (overrides: Routes = {}) =>
  mockApi({
    "/api/evaluation": { body: READY },
    "/api/evaluation/details": { body: DETAILS },
    "/api/evaluation/predictions": { body: PREDICTIONS },
    ...overrides,
  });

const renderEvaluation = () =>
  render(
    <MemoryRouter initialEntries={["/ml/evaluation"]}>
      <App />
    </MemoryRouter>
  );

async function appears(testId: string): Promise<HTMLElement> {
  await waitFor(() => expect(screen.queryByTestId(testId)).not.toBeNull());
  return screen.getByTestId(testId);
}

const text = (element: Element | null) => element?.textContent ?? "";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("Evaluation ready: la página muestra lo que devuelve la API", () => {
  it("procedencia completa y cronología cierre → evaluación", async () => {
    readyApi();
    renderEvaluation();
    const provenance = text(await appears("evaluation-full-provenance"));
    const p = DETAILS.provenance;
    for (const value of [
      p.candidate_run_id,
      p.checkpoint_sha256,
      p.git_commit,
      p.manifest_hash,
      p.test_split_hash,
      p.dataset_version,
      String(p.n_test),
    ]) {
      expect(provenance).toContain(value);
    }
    expect(provenance).toMatch(/Selección cerrada .* → evaluada /);
  });

  it("umbral 0.85 con los conteos de la API, sin calcularlos en la página", async () => {
    readyApi();
    renderEvaluation();
    const target = text(await appears("evaluation-target"));
    expect(target).toContain(`${DETAILS.target.correct} / ${DETAILS.target.n_test}`);
    expect(target).toContain("cumple el objetivo");
    cleanup();

    const below = structuredClone(DETAILS);
    below.target = { ...below.target, correct: 50, met: false };
    readyApi({ "/api/evaluation/details": { body: { ...below, examples: { ...below.examples, n_correct: 50, n_errors: 16 } } } });
    renderEvaluation();
    expect(text(await appears("evaluation-target"))).toContain("no alcanza el objetivo");
  });

  it("métricas por clase de evaluation_response", async () => {
    readyApi();
    renderEvaluation();
    const table = text(await appears("evaluation-per-class"));
    expect(table).toContain("88.57%");
    expect(table).toContain("91.18%");
    expect(table).toContain("34");
    expect(table).toContain("32");
  });

  it("ejemplos de errores y aciertos ligados a crop_id", async () => {
    readyApi();
    renderEvaluation();
    const errors = await appears("evaluation-errors");
    const correct = screen.getByTestId("evaluation-correct");
    const ids = (element: HTMLElement) =>
      [...element.querySelectorAll("[data-crop-id]")].map((li) =>
        Number(li.getAttribute("data-crop-id"))
      );
    expect(ids(errors)).toEqual(DETAILS.examples.errors.map((e) => e.crop_id));
    expect(ids(correct)).toEqual(DETAILS.examples.correct.map((e) => e.crop_id));
    expect(text(errors)).toContain(`de ${DETAILS.examples.n_errors}`);
  });

  it("predicciones guardadas: todas, filtro de errores y exportación CSV/JSON", async () => {
    readyApi();
    renderEvaluation();
    await appears("evaluation-predictions");
    const rows = () =>
      screen.getAllByTestId("prediction-row").map((row) => Number(row.firstChild?.textContent));
    expect(rows()).toEqual(PREDICTIONS.predictions.map((s) => s.crop_id));
    fireEvent.click(screen.getByRole("checkbox", { name: "Solo errores" }));
    expect(rows()).toEqual(
      PREDICTIONS.predictions.filter((s) => s.true_class !== s.predicted_class).map((s) => s.crop_id)
    );
    expect(screen.getByTestId("export-csv").getAttribute("href")).toBe(
      "/api/evaluation/predictions?format=csv"
    );
    expect(screen.getByTestId("export-json").getAttribute("href")).toBe(
      "/api/evaluation/predictions?format=json"
    );
  });

  it("synthetic: se avisa que no es el resultado final", async () => {
    readyApi({
      "/api/evaluation": { body: fixturePayload("evaluation_response", "valid-ready-synthetic") },
      "/api/evaluation/details": { body: fixturePayload("evaluation_details", "valid-synthetic") },
    });
    renderEvaluation();
    expect(text(await appears("evaluation-not-final"))).toContain("no es el resultado final");
  });

  it.each([
    "invalid-synthetic-final",
    "invalid-local-test-namespace",
    "invalid-target-rounded",
  ])("%s: fuera de contrato, no se muestra", async (name) => {
    readyApi({ "/api/evaluation/details": { body: fixturePayload("evaluation_details", name) } });
    renderEvaluation();
    await waitFor(() =>
      expect(document.body.textContent).toContain(
        "La respuesta del servidor no tiene el formato esperado."
      )
    );
    expect(screen.queryByTestId("evaluation-target")).toBeNull();
  });

  it("run cerrado incoherente (503): error con el motivo, sin procedencia", async () => {
    readyApi({
      "/api/evaluation/details": {
        status: 503,
        body: { error: "Run cerrado incoherente: el run se entrenó con otro manifest." },
      },
    });
    renderEvaluation();
    await waitFor(() => expect(document.body.textContent).toContain("otro manifest"));
    expect(screen.queryByTestId("evaluation-full-provenance")).toBeNull();
  });
});

describe("Evaluation no reejecuta ni pide resultados antes de tiempo", () => {
  it("blocked y pending no piden detalle ni predicciones", async () => {
    for (const name of ["valid-blocked", "valid-pending"]) {
      const fetchMock = mockApi({
        "/api/evaluation": { body: fixturePayload("evaluation_response", name) },
      });
      renderEvaluation();
      await waitFor(() => expect(fetchMock).toHaveBeenCalled());
      await waitFor(() =>
        expect(
          screen.queryByTestId("p3-state-blocked") ?? screen.queryByTestId("evaluation-pending")
        ).not.toBeNull()
      );
      expect(fetchMock.mock.calls.map(([url]) => String(url))).toEqual(["/api/evaluation"]);
      cleanup();
      vi.unstubAllGlobals();
    }
  });

  it("ver y refrescar la página solo hace GET de lectura", async () => {
    const fetchMock = readyApi();
    renderEvaluation();
    await appears("evaluation-predictions");
    cleanup();
    renderEvaluation();
    await appears("evaluation-predictions");
    const urls = fetchMock.mock.calls.map(([url]) => String(url));
    expect(methods).toEqual(urls.map(() => "GET"));
    expect(new Set(urls)).toEqual(
      new Set(["/api/evaluation", "/api/evaluation/details", "/api/evaluation/predictions"])
    );
    expect(urls).toHaveLength(6);
  });
});
