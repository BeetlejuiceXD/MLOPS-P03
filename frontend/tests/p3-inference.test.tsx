/**
 * D05-07 (preparación) — Página Inference conectada a la API de inferencia y a la cola.
 *
 * La clase NUNCA se calcula en el frontend: sale de `POST /api/inference` (motor de D05-04).
 * La página muestra la identidad del paquete (smoke), valida la entrada como el upload del
 * portal, permite elegir un crop (anotación) de una imagen del portal y enviar la predicción
 * a la cola de anotación sin presentarla como etiqueta humana.
 *
 * Respuestas con fixtures compartidos (`contracts/p3`): prueban la página aislada, no el
 * recorrido con el motor real, que se evidencia aparte.
 */
import "@testing-library/jest-dom/vitest";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { App } from "../src/App";
import { fixturePayload } from "./p3-fixtures";

type Handler = (init?: RequestInit) => { status?: number; body: unknown };
type Routes = Record<string, Handler>;

const ENGINE = fixturePayload("inference_engine", "valid-smoke") as {
  model: { package_id: string; mlflow_run_id: string; checkpoint_sha256: string };
};
const RESULT = fixturePayload("inference_result", "valid-ok") as Record<string, unknown>;
const QUEUED = fixturePayload("annotation_queue_item", "valid-ok") as Record<string, unknown>;
const LIST = fixturePayload("inference_list", "valid-two");

function mockApi(routes: Routes) {
  const calls: { method: string; path: string; init?: RequestInit }[] = [];
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input.toString();
    const path = url.replace(/^https?:\/\/[^/]+/, "");
    const method = init?.method ?? "GET";
    calls.push({ method, path, init });
    const handler = routes[`${method} ${path}`];
    if (!handler) {
      return new Response(JSON.stringify({ error: "Recurso no encontrado." }), { status: 404 });
    }
    const { status, body } = handler(init);
    return new Response(JSON.stringify(body), {
      status: status ?? 200,
      headers: { "Content-Type": "application/json" },
    });
  });
  vi.stubGlobal("fetch", fetchMock);
  return calls;
}

const BASE: Routes = {
  "GET /api/inference/engine": () => ({ body: ENGINE }),
  "GET /api/inference": () => ({ body: { inferences: [] } }),
};

function renderInference() {
  return render(
    <MemoryRouter initialEntries={["/ml/inference"]}>
      <App />
    </MemoryRouter>
  );
}

const file = (name: string, type: string, size = 1024) =>
  new File([new Uint8Array(size)], name, { type });

function chooseFile(f: File) {
  fireEvent.change(screen.getByLabelText("Imagen"), { target: { files: [f] } });
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("identidad del motor", () => {
  it("muestra el paquete smoke con run y checkpoint, y que no es el modelo seleccionado", async () => {
    mockApi(BASE);
    renderInference();
    const engine = await screen.findByTestId("inference-engine");
    expect(engine).toHaveTextContent(ENGINE.model.package_id);
    expect(engine).toHaveTextContent(ENGINE.model.mlflow_run_id);
    expect(engine).toHaveTextContent(ENGINE.model.checkpoint_sha256);
    expect(engine).toHaveTextContent(/smoke/i);
    expect(engine).toHaveTextContent("no es el modelo seleccionado, evaluado ni publicado");
  });

  it("motor ausente: muestra el motivo y no deja predecir", async () => {
    mockApi({
      ...BASE,
      "GET /api/inference/engine": () => ({
        status: 503,
        body: { error: "motor de inferencia (D05-04) no disponible: INFERENCE_ENGINE_URL no está configurado" },
      }),
    });
    renderInference();
    expect(
      await screen.findByText(/motor de inferencia \(D05-04\) no disponible/)
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Predecir" })).toBeDisabled();
  });
});

describe("archivo nuevo", () => {
  it.each([
    ["tipo no soportado", file("notas.txt", "text/plain"), /JPEG, PNG o WebP/],
    ["demasiado grande", file("enorme.png", "image/png", 5 * 1024 * 1024 + 1), /5 MiB/],
  ])("%s: error en la página y no llama a la API", async (_name, bad, message) => {
    const calls = mockApi(BASE);
    renderInference();
    await screen.findByTestId("inference-engine");
    chooseFile(bad);
    expect(await screen.findByTestId("inference-input-error")).toHaveTextContent(message);
    expect(screen.getByRole("button", { name: "Predecir" })).toBeDisabled();
    expect(calls.some((call) => call.method === "POST")).toBe(false);
  });

  it("envía el archivo a la API y muestra clase, probabilidades e identidad", async () => {
    let sent: FormData | undefined;
    mockApi({
      ...BASE,
      "POST /api/inference": (init) => {
        sent = init?.body as FormData;
        return { status: 201, body: RESULT };
      },
    });
    renderInference();
    await screen.findByTestId("inference-engine");
    const photo = file("perro.png", "image/png");
    chooseFile(photo);
    fireEvent.click(screen.getByRole("button", { name: "Predecir" }));
    const result = await screen.findByTestId("inference-result");
    expect(sent?.get("image")).toBe(photo);
    expect(result).toHaveTextContent("dog");
    expect(result).toHaveTextContent("94.13 %");
    expect(result).toHaveTextContent("5.87 %");
    expect(result).toHaveTextContent(ENGINE.model.package_id);
    expect(result).toHaveTextContent(ENGINE.model.mlflow_run_id);
    expect(result).toHaveTextContent(ENGINE.model.checkpoint_sha256);
    expect(result).toHaveTextContent("Sugerencia del modelo (smoke), no es una etiqueta validada");
  });

  it("un error de la API se muestra y no aparece un resultado", async () => {
    mockApi({
      ...BASE,
      "POST /api/inference": () => ({ status: 400, body: { error: "El archivo no es una imagen válida." } }),
    });
    renderInference();
    await screen.findByTestId("inference-engine");
    chooseFile(file("falsa.png", "image/png"));
    fireEvent.click(screen.getByRole("button", { name: "Predecir" }));
    expect(await screen.findByTestId("inference-error")).toHaveTextContent(
      "El archivo no es una imagen válida."
    );
    expect(screen.queryByTestId("inference-result")).not.toBeInTheDocument();
  });
});

describe("crop del portal", () => {
  it("elige una anotación de una imagen del portal y envía su id (no los píxeles)", async () => {
    let body: unknown;
    mockApi({
      ...BASE,
      "GET /api/images/42/annotations": () => ({
        body: [
          {
            id: 7,
            imageId: 42,
            categoryId: 3,
            bboxX: 10,
            bboxY: 20,
            bboxWidth: 120,
            bboxHeight: 90,
            area: 10800,
            iscrowd: false,
            category: { id: 3, name: "dog", color: "#f97316" },
          },
        ],
      }),
      "POST /api/inference": (init) => {
        body = JSON.parse(String(init?.body));
        return {
          status: 201,
          body: fixturePayload("inference_result", "valid-crop-queued"),
        };
      },
    });
    renderInference();
    await screen.findByTestId("inference-engine");
    fireEvent.click(screen.getByRole("radio", { name: "Crop del portal" }));
    fireEvent.change(screen.getByLabelText("ID de la imagen del portal"), {
      target: { value: "42" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Ver anotaciones" }));
    const select = await screen.findByLabelText("Anotación (crop)");
    fireEvent.change(select, { target: { value: "7" } });
    fireEvent.click(screen.getByRole("button", { name: "Predecir" }));
    const result = await screen.findByTestId("inference-result");
    expect(body).toEqual({ annotation_id: 7 });
    expect(result).toHaveTextContent("Crop de la anotación #7 (imagen #42)");
    // Ya estaba en la cola: lo dice y no ofrece enviarlo otra vez.
    expect(result).toHaveTextContent("En la cola de anotación: elemento #5");
    expect(screen.queryByRole("button", { name: "Enviar a anotación" })).not.toBeInTheDocument();
  });
});

describe("cola de anotación", () => {
  it("envía la predicción a la cola, muestra el elemento pendiente y no permite duplicarlo", async () => {
    const calls = mockApi({
      ...BASE,
      "POST /api/inference": () => ({ status: 201, body: RESULT }),
      "POST /api/inference/1/annotation-queue": () => ({ status: 201, body: QUEUED }),
    });
    renderInference();
    await screen.findByTestId("inference-engine");
    chooseFile(file("perro.png", "image/png"));
    fireEvent.click(screen.getByRole("button", { name: "Predecir" }));
    await screen.findByTestId("inference-result");
    fireEvent.click(screen.getByRole("button", { name: "Enviar a anotación" }));
    const queued = await screen.findByTestId("inference-queued");
    expect(queued).toHaveTextContent("elemento #3");
    expect(queued).toHaveTextContent("imagen #42");
    expect(queued).toHaveTextContent("pendiente de revisión humana");
    expect(within(queued).getByRole("link", { name: "Abrir en Anotar" })).toHaveAttribute(
      "href",
      "/annotate/42"
    );
    expect(screen.queryByRole("button", { name: "Enviar a anotación" })).not.toBeInTheDocument();
    expect(calls.filter((call) => call.path.endsWith("/annotation-queue"))).toHaveLength(1);
  });

  it("si la cola falla, muestra el error y no da por enviado", async () => {
    mockApi({
      ...BASE,
      "POST /api/inference": () => ({ status: 201, body: RESULT }),
      "POST /api/inference/1/annotation-queue": () => ({
        status: 503,
        body: { error: "No se pudo guardar el elemento de la cola de anotación." },
      }),
    });
    renderInference();
    await screen.findByTestId("inference-engine");
    chooseFile(file("perro.png", "image/png"));
    fireEvent.click(screen.getByRole("button", { name: "Predecir" }));
    await screen.findByTestId("inference-result");
    fireEvent.click(screen.getByRole("button", { name: "Enviar a anotación" }));
    expect(await screen.findByTestId("inference-error")).toHaveTextContent(
      "No se pudo guardar el elemento de la cola"
    );
    expect(screen.queryByTestId("inference-queued")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Enviar a anotación" })).toBeEnabled();
  });
});

describe("historial persistido", () => {
  it("lista las inferencias guardadas con su estado en la cola", async () => {
    mockApi({ ...BASE, "GET /api/inference": () => ({ body: LIST }) });
    renderInference();
    const history = await screen.findByTestId("inference-history");
    const crop = within(history).getByTestId("inference-row-2");
    expect(crop).toHaveTextContent("cat");
    expect(crop).toHaveTextContent("crop #7");
    expect(crop).toHaveTextContent("#5");
    const upload = within(history).getByTestId("inference-row-1");
    expect(upload).toHaveTextContent("gato.jpg");
    expect(upload).toHaveTextContent("sin enviar");
  });

  it("después de predecir, el historial se vuelve a pedir", async () => {
    const calls = mockApi({
      ...BASE,
      "POST /api/inference": () => ({ status: 201, body: RESULT }),
    });
    renderInference();
    await screen.findByTestId("inference-engine");
    chooseFile(file("perro.png", "image/png"));
    fireEvent.click(screen.getByRole("button", { name: "Predecir" }));
    await screen.findByTestId("inference-result");
    await waitFor(() =>
      expect(calls.filter((call) => call.method === "GET" && call.path === "/api/inference")).toHaveLength(2)
    );
  });
});
