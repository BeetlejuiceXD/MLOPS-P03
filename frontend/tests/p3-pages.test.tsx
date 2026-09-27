/**
 * D01-05 — Navegación y estados de las cinco páginas P3 (Training, Experiments,
 * Evaluation, Models, Inference) dentro del mismo portal de P1/P2.
 *
 * Las respuestas vienen de los fixtures compartidos `contracts/p3/fixtures`: son
 * datos de ejemplo para probar los componentes, no evidencia de integración real.
 */
import "@testing-library/jest-dom/vitest";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { App } from "../src/App";
import { fixturePayload } from "./p3-fixtures";

type Routes = Record<string, { status?: number; body: unknown }>;

function mockApi(routes: Routes) {
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const url = typeof input === "string" ? input : input.toString();
    const path = url.replace(/^https?:\/\/[^/]+/, "");
    const route = routes[path];
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

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>
  );
}

const ok = (contract: string, name: string) => ({ body: fixturePayload(contract, name) });

const TRAINING_READY: Routes = {
  "/api/releases": ok("releases_response", "valid-approved-and-rejected"),
  "/api/manifest": ok("manifest_summary", "valid-frozen"),
  "/api/training/jobs": { body: { jobs: [] } },
};

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

const PAGES = [
  { label: "Training", path: "/ml/training" },
  { label: "Experiments", path: "/ml/experiments" },
  { label: "Evaluation", path: "/ml/evaluation" },
  { label: "Models", path: "/ml/models" },
  { label: "Inference", path: "/ml/inference" },
] as const;

describe("navegación P3 en el portal existente", () => {
  it("el menú global incluye las cinco páginas junto a las de P1/P2", () => {
    mockApi({});
    renderAt("/pipeline/overview");
    const nav = screen.getByRole("navigation");
    for (const page of PAGES) {
      expect(within(nav).getByRole("link", { name: page.label })).toHaveAttribute(
        "href",
        page.path
      );
    }
    // Mismo portal: el menú de anotación y el del pipeline siguen ahí.
    expect(within(nav).getByRole("link", { name: "Tablero" })).toBeInTheDocument();
    expect(within(nav).getByRole("link", { name: "Overview" })).toBeInTheDocument();
  });

  it.each(PAGES)("el enlace $label abre su página", async ({ label, path }) => {
    mockApi({});
    renderAt("/pipeline/overview");
    fireEvent.click(within(screen.getByRole("navigation")).getByRole("link", { name: label }));
    expect(await screen.findByRole("heading", { level: 1, name: label })).toBeInTheDocument();
    expect(within(screen.getByRole("navigation")).getByRole("link", { name: label })).toHaveAttribute(
      "aria-current",
      "page"
    );
    expect(path.startsWith("/ml/")).toBe(true);
  });
});

describe("estados vacíos", () => {
  it("Training sin jobs", async () => {
    mockApi(TRAINING_READY);
    renderAt("/ml/training");
    expect(await screen.findByTestId("p3-state-empty")).toHaveTextContent(
      "Todavía no hay trabajos de entrenamiento"
    );
    // Procedencia visible aunque no haya jobs.
    expect(screen.getByText("v0.1.1")).toBeInTheDocument();
    expect(screen.getByText("p3-v0.1.1-s42")).toBeInTheDocument();
  });

  it("Experiments sin corridas", async () => {
    mockApi({ "/api/experiments/runs": ok("experiment_runs_response", "valid-empty") });
    renderAt("/ml/experiments");
    expect(await screen.findByTestId("p3-state-empty")).toHaveTextContent(
      "Todavía no hay corridas en MLflow"
    );
  });

  it("Models sin versiones", async () => {
    mockApi({ "/api/models": ok("models_response", "valid-empty") });
    renderAt("/ml/models");
    expect(await screen.findByTestId("p3-state-empty")).toHaveTextContent(
      "Todavía no hay versiones del modelo"
    );
  });
});

describe("estados bloqueados", () => {
  it("Training sin release aprobado", async () => {
    mockApi({
      ...TRAINING_READY,
      "/api/releases": ok("releases_response", "valid-none-approved"),
    });
    renderAt("/ml/training");
    expect(await screen.findByTestId("p3-state-blocked")).toHaveTextContent(
      "No hay un release aprobado"
    );
  });

  it("Training con manifest no congelado", async () => {
    mockApi({ ...TRAINING_READY, "/api/manifest": ok("manifest_summary", "valid-not-frozen") });
    renderAt("/ml/training");
    expect(await screen.findByTestId("p3-state-blocked")).toHaveTextContent(
      "El manifest 70/20/10 no está congelado"
    );
  });

  it("Evaluation antes de MODEL SELECTION CLOSED no revela nada del test", async () => {
    mockApi({ "/api/evaluation": ok("evaluation_response", "valid-blocked") });
    renderAt("/ml/evaluation");
    expect(await screen.findByTestId("p3-state-blocked")).toHaveTextContent(
      "MODEL SELECTION CLOSED"
    );
    expect(screen.queryByText(/accuracy/i)).not.toBeInTheDocument();
  });

  it("Inference sin versión publicada", async () => {
    mockApi({ "/api/models": ok("models_response", "valid-only-draft") });
    renderAt("/ml/inference");
    expect(await screen.findByTestId("p3-state-blocked")).toHaveTextContent(
      "No hay ninguna versión publicada"
    );
  });
});

describe("errores y respuestas incompatibles", () => {
  it("un error HTTP muestra el estado de error con reintento", async () => {
    mockApi({ "/api/models": { status: 500, body: { error: "Fallo interno." } } });
    renderAt("/ml/models");
    expect(await screen.findByText("El servidor respondió con estado 500.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Reintentar" })).toBeInTheDocument();
  });

  it("un endpoint que todavía no existe (404) no se confunde con datos vacíos", async () => {
    mockApi({});
    renderAt("/ml/experiments");
    expect(await screen.findByText("El servidor respondió con estado 404.")).toBeInTheDocument();
    expect(screen.queryByTestId("p3-state-empty")).not.toBeInTheDocument();
  });

  it.each([
    ["/ml/evaluation", "/api/evaluation", "evaluation_response", "invalid-matrix-sum-mismatch"],
    ["/ml/evaluation", "/api/evaluation", "evaluation_response", "invalid-evaluated-before-closed"],
    ["/ml/experiments", "/api/experiments/runs", "experiment_runs_response", "invalid-test-metric-in-run"],
    ["/ml/models", "/api/models", "models_response", "invalid-published-without-version-id"],
  ])("%s rechaza una respuesta fuera de contrato (%s → %s/%s)", async (page, api, contract, name) => {
    mockApi({ [api]: ok(contract, name) });
    renderAt(page);
    expect(
      await screen.findByText("La respuesta del servidor no tiene el formato esperado.")
    ).toBeInTheDocument();
    expect(screen.queryByTestId("p3-content")).not.toBeInTheDocument();
  });

  it("Training no muestra el formulario si una de sus tres fuentes falla", async () => {
    mockApi({ ...TRAINING_READY, "/api/manifest": { status: 503, body: { error: "x" } } });
    renderAt("/ml/training");
    expect(await screen.findByText("El servidor respondió con estado 503.")).toBeInTheDocument();
    expect(screen.queryByTestId("p3-content")).not.toBeInTheDocument();
  });
});

describe("contenido con datos válidos", () => {
  it("Experiments lista las corridas desde el contrato de MLflow", async () => {
    mockApi({ "/api/experiments/runs": ok("experiment_runs_response", "valid-two-runs") });
    renderAt("/ml/experiments");
    const content = await screen.findByTestId("p3-content");
    expect(within(content).getAllByRole("row")).toHaveLength(3); // encabezado + 2 runs
    expect(within(content).getByText("FINISHED")).toBeInTheDocument();
    expect(within(content).getByText("RUNNING")).toBeInTheDocument();
  });

  it("Evaluation con selección cerrada muestra matriz y accuracy sin redondear antes de comparar", async () => {
    mockApi({ "/api/evaluation": ok("evaluation_response", "valid-ready") });
    renderAt("/ml/evaluation");
    const content = await screen.findByTestId("p3-content");
    expect(within(content).getByText("59 / 66")).toBeInTheDocument();
    expect(within(content).getByTestId("confusion-matrix")).toBeInTheDocument();
  });

  it("Models distingue versión de modelo y versión de dataset", async () => {
    mockApi({ "/api/models": ok("models_response", "valid-published-and-draft") });
    renderAt("/ml/models");
    const content = await screen.findByTestId("p3-content");
    expect(within(content).getByText("1.0.0")).toBeInTheDocument();
    expect(within(content).getAllByText("v0.1.1").length).toBeGreaterThan(0);
    expect(within(content).getByText("published")).toBeInTheDocument();
    expect(within(content).getByText("draft")).toBeInTheDocument();
  });

  it("Inference ofrece solo versiones publicadas", async () => {
    mockApi({ "/api/models": ok("models_response", "valid-published-and-draft") });
    renderAt("/ml/inference");
    const select = await screen.findByLabelText("Versión del modelo");
    const options = within(select).getAllByRole("option").map((option) => option.textContent);
    expect(options).toEqual(["1.0.0"]);
  });

  it("Training muestra los jobs persistidos", async () => {
    mockApi({
      ...TRAINING_READY,
      "/api/training/jobs": {
        body: {
          jobs: [
            fixturePayload("training_job", "valid-running"),
            fixturePayload("training_job", "valid-queued"),
          ],
        },
      },
    });
    renderAt("/ml/training");
    const content = await screen.findByTestId("p3-content");
    await waitFor(() => expect(within(content).getByText("running")).toBeInTheDocument());
    expect(within(content).getByText("queued")).toBeInTheDocument();
    expect(within(content).getByText("3 / 30")).toBeInTheDocument();
  });
});
