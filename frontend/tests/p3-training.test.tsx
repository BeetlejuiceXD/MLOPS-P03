/**
 * D02-05 — Formulario Training y seguimiento de jobs persistentes.
 *
 * Las respuestas salen de los fixtures de `contracts/p3` (evidencia de componente). La
 * persistencia real tras recarga/reinicio la prueba el job de CI "Jobs persistentes".
 */
import "@testing-library/jest-dom/vitest";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { TrainingPage } from "../src/p3/pages/Training";
import { fixturePayload } from "./p3-fixtures";

type Reply = { status?: number; body: unknown };
type Handler = (init?: RequestInit) => Reply;
type Routes = Record<string, Reply | Handler>;

function mockApi(routes: Routes) {
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = (typeof input === "string" ? input : input.toString()).replace(
      /^https?:\/\/[^/]+/,
      ""
    );
    const method = init?.method ?? "GET";
    const route = routes[`${method} ${url}`] ?? routes[url];
    const reply = typeof route === "function" ? route(init) : route;
    if (!reply) {
      return new Response(JSON.stringify({ error: "Recurso no encontrado." }), { status: 404 });
    }
    return new Response(JSON.stringify(reply.body), {
      status: reply.status ?? 200,
      headers: { "Content-Type": "application/json" },
    });
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

const ok = (contract: string, name: string): Reply => ({ body: fixturePayload(contract, name) });
const job = (name: string) => fixturePayload("training_job", name) as Record<string, unknown>;
const DEFAULTS = fixturePayload("training_config", "valid-defaults") as Record<string, unknown>;

const SOURCES: Routes = {
  "/api/releases": ok("releases_response", "valid-approved-and-rejected"),
  "/api/manifest": ok("manifest_summary", "valid-frozen"),
};

function renderPage(pollMs = 60_000) {
  return render(
    <MemoryRouter>
      <TrainingPage pollMs={pollMs} />
    </MemoryRouter>
  );
}

function posts(fetchMock: ReturnType<typeof mockApi>, path: string) {
  return fetchMock.mock.calls.filter(
    ([url, init]) => String(url).endsWith(path) && (init as RequestInit | undefined)?.method === "POST"
  );
}

const field = (label: string) => screen.getByLabelText(label) as HTMLInputElement;

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("formulario de TrainingConfig", () => {
  it("muestra todos los campos congelados en #33 con sus defaults y la seed vacía", async () => {
    mockApi({ ...SOURCES, "/api/training/jobs": { body: { jobs: [] } } });
    renderPage();
    await screen.findByRole("button", { name: "Encolar job" });

    expect(field("Arquitectura").value).toBe("resnet18");
    expect(field("Capas entrenables").value).toBe("last_block");
    expect(field("Optimizador").value).toBe("adam");
    expect(field("Batch size").value).toBe("16");
    expect(field("Learning rate").value).toBe("0.001");
    expect(field("Weight decay").value).toBe("0.0001");
    expect(field("Épocas máximas").value).toBe("30");
    expect(field("Patience").value).toBe("5");
    expect(field("Tamaño de imagen").value).toBe("224");
    expect(field("Capas ocultas de la cabeza").value).toBe("0");
    expect(field("Dimensión oculta").value).toBe("128");
    expect(field("Dimensión oculta")).toHaveAttribute("readonly");
    expect(field("Dropout").value).toBe("0");
    expect(field("Pesos preentrenados (ImageNet)").checked).toBe(true);
    expect(field("Augmentation (solo train)").checked).toBe(true);
    expect(field("Seed").value).toBe("");
  });

  it.each([
    ["Seed", "", /seed/i],
    ["Learning rate", "0.05", /learning_rate/],
    ["Batch size", "4", /batch_size/],
    ["Tamaño de imagen", "300", /image_size/],
    ["Dropout", "0.6", /dropout/],
    ["Épocas máximas", "12.5", /max_epochs/],
  ])("%s=%s se rechaza en el portal y no se envía", async (label, value, message) => {
    const fetchMock = mockApi({ ...SOURCES, "/api/training/jobs": { body: { jobs: [] } } });
    renderPage();
    await screen.findByRole("button", { name: "Encolar job" });
    fireEvent.change(field("Seed"), { target: { value: "7" } });
    fireEvent.change(field(label), { target: { value } });

    fireEvent.click(screen.getByRole("button", { name: "Encolar job" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(message);
    expect(posts(fetchMock, "/api/training/jobs")).toHaveLength(0);
  });

  it("fail_at_epoch mayor que las épocas máximas se rechaza", async () => {
    const fetchMock = mockApi({ ...SOURCES, "/api/training/jobs": { body: { jobs: [] } } });
    renderPage();
    await screen.findByRole("button", { name: "Encolar job" });
    fireEvent.change(field("Seed"), { target: { value: "7" } });
    fireEvent.change(field("Fallar en la época (opcional)"), { target: { value: "31" } });

    fireEvent.click(screen.getByRole("button", { name: "Encolar job" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(/fail_at_epoch/);
    expect(posts(fetchMock, "/api/training/jobs")).toHaveLength(0);
  });

  it("encola una tarea controlada con el TrainingConfig completo y recarga la lista", async () => {
    let listed: unknown[] = [];
    const fetchMock = mockApi({
      ...SOURCES,
      "GET /api/training/jobs": () => ({ body: { jobs: listed } }),
      "POST /api/training/jobs": () => {
        listed = [job("valid-queued")];
        return { status: 201, body: job("valid-queued") };
      },
    });
    renderPage();
    await screen.findByRole("button", { name: "Encolar job" });
    fireEvent.change(field("Seed"), { target: { value: "7" } });

    fireEvent.click(screen.getByRole("button", { name: "Encolar job" }));

    await waitFor(() => expect(posts(fetchMock, "/api/training/jobs")).toHaveLength(1));
    const [, init] = posts(fetchMock, "/api/training/jobs")[0] as [string, RequestInit];
    const manifest = fixturePayload("manifest_summary", "valid-frozen") as {
      manifest_hash: string;
    };
    expect(JSON.parse(String(init.body))).toEqual({
      task: "controlled",
      dataset_version: "v0.1.1",
      manifest_hash: manifest.manifest_hash,
      config: DEFAULTS,
    });
    expect(await screen.findByText("queued")).toBeInTheDocument();
  });

  it("muestra el error de la API sin inventar el job", async () => {
    mockApi({
      ...SOURCES,
      "GET /api/training/jobs": { body: { jobs: [] } },
      "POST /api/training/jobs": {
        status: 400,
        body: { error: "Job inválido: config.seed: Invalid input" },
      },
    });
    renderPage();
    await screen.findByRole("button", { name: "Encolar job" });
    fireEvent.change(field("Seed"), { target: { value: "7" } });
    fireEvent.click(screen.getByRole("button", { name: "Encolar job" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("config.seed");
    expect(screen.getByTestId("p3-state-empty")).toBeInTheDocument();
  });
});

describe("entrenamiento real (compuerta)", () => {
  it.each([
    ["sin release aprobado", { "/api/releases": ok("releases_response", "valid-none-approved") }, /release aprobado/],
    ["manifest sin congelar", { "/api/manifest": ok("manifest_summary", "valid-not-frozen") }, /no está congelado/],
    ["manifest no disponible", { "/api/manifest": { status: 503, body: { error: "x" } } }, /503/],
  ])("queda deshabilitado %s, pero la tarea controlada sigue disponible", async (_c, over, reason) => {
    mockApi({ ...SOURCES, ...over, "/api/training/jobs": { body: { jobs: [] } } });
    renderPage();

    const real = await screen.findByLabelText(/Entrenamiento real/);
    expect(real).toBeDisabled();
    expect(screen.getByTestId("p3-state-blocked")).toHaveTextContent(reason);
    expect(screen.getByLabelText(/Tarea controlada/)).toBeChecked();
    expect(screen.getByRole("button", { name: "Encolar job" })).toBeEnabled();
  });

  it("con release y manifest elegibles se puede elegir; un 409 de la API se muestra", async () => {
    const fetchMock = mockApi({
      ...SOURCES,
      "GET /api/training/jobs": { body: { jobs: [] } },
      "POST /api/training/jobs": {
        status: 409,
        body: { error: "No se puede encolar un training real: manifest oficial no disponible" },
      },
    });
    renderPage();
    const real = await screen.findByLabelText(/Entrenamiento real/);
    expect(real).toBeEnabled();
    fireEvent.click(real);
    fireEvent.change(field("Seed"), { target: { value: "7" } });
    expect(screen.queryByLabelText("Fallar en la época (opcional)")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Encolar job" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("manifest oficial no disponible");
    const [, init] = posts(fetchMock, "/api/training/jobs")[0] as [string, RequestInit];
    expect(JSON.parse(String(init.body)).task).toBe("training");
  });
});

describe("seguimiento de jobs", () => {
  it("muestra estado, progreso, run de MLflow y error persistidos", async () => {
    mockApi({
      ...SOURCES,
      "/api/training/jobs": { body: { jobs: [job("valid-running"), job("valid-failed")] } },
    });
    renderPage();
    const table = await screen.findByTestId("p3-content");
    expect(within(table).getByText("3 / 30")).toBeInTheDocument();
    expect(within(table).getByText("aaaaaaaaaaaa")).toBeInTheDocument();
    expect(within(table).getByText(/CUDA no disponible/)).toBeInTheDocument();
    expect(within(table).getAllByText("controlled").length).toBeGreaterThan(0);
  });

  it("cancelar un job en ejecución llama a la API y recarga", async () => {
    let running = job("valid-running");
    const fetchMock = mockApi({
      ...SOURCES,
      "GET /api/training/jobs": () => ({ body: { jobs: [running] } }),
      "POST /api/training/jobs/1/cancel": () => {
        running = { ...running, cancel_requested: true };
        return { body: running };
      },
    });
    renderPage();
    fireEvent.click(await screen.findByRole("button", { name: "Cancelar job 1" }));

    expect(await screen.findByText("cancelación pedida")).toBeInTheDocument();
    expect(posts(fetchMock, "/api/training/jobs/1/cancel")).toHaveLength(1);
    expect(screen.queryByRole("button", { name: "Cancelar job 1" })).not.toBeInTheDocument();
  });

  it("no ofrece cancelar jobs terminados", async () => {
    mockApi({ ...SOURCES, "/api/training/jobs": { body: { jobs: [job("valid-succeeded")] } } });
    renderPage();
    await screen.findByTestId("p3-content");
    expect(screen.queryByRole("button", { name: /Cancelar job/ })).not.toBeInTheDocument();
  });

  it("muestra los logs persistidos del job", async () => {
    mockApi({
      ...SOURCES,
      "/api/training/jobs": { body: { jobs: [job("valid-running")] } },
      "/api/training/jobs/1/logs": ok("job_logs", "valid-ok"),
    });
    renderPage();
    fireEvent.click(await screen.findByRole("button", { name: "Logs del job 1" }));
    expect(await screen.findByText(/Epoch 1\/30 train_loss=0.61/)).toBeInTheDocument();
    expect(screen.getByText(/Fallo al leer un crop/)).toBeInTheDocument();
  });

  it("mientras hay jobs activos vuelve a consultar hasta ver el estado final", async () => {
    let calls = 0;
    mockApi({
      ...SOURCES,
      "/api/training/jobs": () => {
        calls += 1;
        return { body: { jobs: [calls < 3 ? job("valid-running") : job("valid-succeeded")] } };
      },
    });
    renderPage(20);
    expect(await screen.findByText("succeeded", {}, { timeout: 2000 })).toBeInTheDocument();
    expect(calls).toBeGreaterThanOrEqual(3);
  });

  it("un error al listar jobs se muestra sin ocultar el formulario", async () => {
    mockApi({ ...SOURCES, "/api/training/jobs": { status: 500, body: { error: "x" } } });
    renderPage();
    expect(await screen.findByText("El servidor respondió con estado 500.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Encolar job" })).toBeInTheDocument();
  });
});
