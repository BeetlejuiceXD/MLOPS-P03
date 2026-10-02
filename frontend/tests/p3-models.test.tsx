/**
 * D05-06 (preparación) — Models muestra el registro de D04-06: la lista `official` (lo que
 * sirve `GET /api/models`) y, aparte y rotuladas como pruebas locales en MinIO, las
 * versiones `local_test` con namespace, semver, run, bucket/key, VersionId, SHA-256,
 * tamaño y estado; la integridad se comprueba bajo demanda contra el objeto real.
 *
 * Fixtures compartidos de `contracts/p3`: acreditan el componente, no el paquete real de
 * D05-01.
 */
import "@testing-library/jest-dom/vitest";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { App } from "../src/App";
import { fixturePayload } from "./p3-fixtures";

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

const ok = (contract: string, name: string): Route => ({ body: fixturePayload(contract, name) });
const LOCAL = ok("local_test_models_response", "valid-published-failed-draft");

function renderModels() {
  return render(
    <MemoryRouter initialEntries={["/ml/models"]}>
      <App />
    </MemoryRouter>
  );
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("Models official (D06-03): identidad exacta del objeto y su tarjeta", () => {
  it("cada versión official muestra bucket/key, VersionId y SHA-256 completos, y su tarjeta", async () => {
    mockApi({
      "/api/models": ok("models_response", "valid-published-and-draft"),
      "/api/models/local-test": ok("local_test_models_response", "valid-empty"),
    });
    renderModels();
    const row = await screen.findByTestId("official-row-1.0.0");
    expect(row).toHaveTextContent(
      "mlops-p2-prod-models-example/models/p3-cnn-classifier/1.0.0/model.pt"
    );
    expect(row).toHaveTextContent("3HL4kqtJlcpXroDTDmJ.rmSpXd3dIbrHY");
    expect(row).toHaveTextContent("f".repeat(64));
    expect(row).toHaveTextContent("models/p3-cnn-classifier/1.0.0/model_card.json");
    expect(row).toHaveTextContent("aB3.cardVersionId-01");
    expect(row).toHaveTextContent("Tarjeta verificada");
    expect(screen.getByTestId("official-row-1.1.0")).toHaveTextContent("Sin tarjeta");
  });

  it("una tarjeta todavía sin verificar no se presenta como verificada", async () => {
    mockApi({
      "/api/models": ok("models_response", "valid-draft-with-draft-card"),
      "/api/models/local-test": ok("local_test_models_response", "valid-empty"),
    });
    renderModels();
    const row = await screen.findByTestId("official-row-1.2.0");
    expect(row).toHaveTextContent("models/p3-cnn-classifier/1.2.0/model_card.json");
    expect(row).toHaveTextContent("Tarjeta draft");
    expect(row).not.toHaveTextContent("Tarjeta verificada");
  });
});

describe("Models: official y local_test separados", () => {
  it("las pruebas locales salen rotuladas aparte y nunca en la lista official", async () => {
    mockApi({
      "/api/models": ok("models_response", "valid-published-and-draft"),
      "/api/models/local-test": LOCAL,
    });
    renderModels();

    const official = await screen.findByTestId("p3-content");
    expect(within(official).getByText("1.0.0")).toBeInTheDocument();
    expect(within(official).queryByText("0.0.1")).not.toBeInTheDocument();

    const local = await screen.findByTestId("models-local-test");
    expect(local).toHaveTextContent("local_test");
    expect(local).toHaveTextContent("No es una publicación en AWS");
    const row = within(local).getByTestId("local-row-0.0.1");
    expect(row).toHaveTextContent("published");
    expect(row).toHaveTextContent("c46e4c3ab2bb4ee18c37571adbb65d92");
    expect(row).toHaveTextContent("p3-models-local/models/p3-cnn-classifier/0.0.1/model.pt");
    expect(row).toHaveTextContent("6f3c2e1a-0d4b-4c1e-9a77-2b5f1c0e8d11");
    expect(row).toHaveTextContent(
      "e93de23e2cf9e72e8efa97bde1e9fb2083402d15a422705dab80ffd11bbd5b97"
    );
    expect(row).toHaveTextContent("44781003 B");
  });

  it("una versión fallida muestra su motivo y no ofrece descarga; un borrador tampoco", async () => {
    mockApi({ "/api/models": ok("models_response", "valid-empty"), "/api/models/local-test": LOCAL });
    renderModels();
    const failed = await screen.findByTestId("local-row-0.0.2");
    expect(failed).toHaveTextContent("failed");
    expect(failed).toHaveTextContent("sha256_mismatch");
    expect(within(failed).queryByRole("link")).not.toBeInTheDocument();
    expect(within(screen.getByTestId("local-row-0.0.3")).queryByRole("link")).not.toBeInTheDocument();
    const download = within(screen.getByTestId("local-row-0.0.1")).getByRole("link", {
      name: "Descargar 0.0.1",
    });
    expect(download).toHaveAttribute("href", "/api/models/local-test/0.0.1/object");
  });

  it("official vacío no esconde las pruebas locales", async () => {
    mockApi({ "/api/models": ok("models_response", "valid-empty"), "/api/models/local-test": LOCAL });
    renderModels();
    expect(await screen.findByTestId("p3-state-empty")).toHaveTextContent(
      "Todavía no hay versiones del modelo"
    );
    expect(await screen.findByTestId("models-local-test")).toBeInTheDocument();
  });
});

describe("Models: integridad de una versión local_test", () => {
  it("Comprobar consulta el objeto por VersionId y muestra el resultado", async () => {
    const fetchMock = mockApi({
      "/api/models": ok("models_response", "valid-empty"),
      "/api/models/local-test": LOCAL,
      "/api/models/local-test/0.0.1": ok("local_test_model_detail", "valid-published-verified"),
    });
    renderModels();
    const row = await screen.findByTestId("local-row-0.0.1");
    fireEvent.click(within(row).getByRole("button", { name: "Comprobar 0.0.1" }));
    expect(await screen.findByTestId("integrity-0.0.1")).toHaveTextContent(
      "Integridad OK: el objeto por VersionId coincide con el SHA-256 y el tamaño registrados"
    );
    expect(fetchMock).toHaveBeenCalledWith("/api/models/local-test/0.0.1");
  });

  it("si el objeto ya no está, lo dice con el motivo (sin cambiar el estado registrado)", async () => {
    mockApi({
      "/api/models": ok("models_response", "valid-empty"),
      "/api/models/local-test": LOCAL,
      "/api/models/local-test/0.0.1": ok("local_test_model_detail", "valid-published-object-missing"),
    });
    renderModels();
    const row = await screen.findByTestId("local-row-0.0.1");
    fireEvent.click(within(row).getByRole("button", { name: "Comprobar 0.0.1" }));
    const integrity = await screen.findByTestId("integrity-0.0.1");
    expect(integrity).toHaveTextContent("object_missing");
    expect(row).toHaveTextContent("published");
  });

  it("storage caído al comprobar: 503 con el motivo, no 'objeto ausente'", async () => {
    mockApi({
      "/api/models": ok("models_response", "valid-empty"),
      "/api/models/local-test": LOCAL,
      "/api/models/local-test/0.0.1": {
        status: 503,
        body: { error: "Storage de modelos no disponible: connect ECONNREFUSED" },
      },
    });
    renderModels();
    const row = await screen.findByTestId("local-row-0.0.1");
    fireEvent.click(within(row).getByRole("button", { name: "Comprobar 0.0.1" }));
    expect(await screen.findByTestId("integrity-0.0.1")).toHaveTextContent(
      "El servidor respondió con estado 503: Storage de modelos no disponible: connect ECONNREFUSED"
    );
  });

  it("si falla la lista local_test, el error sale en esa sección y official sigue visible", async () => {
    mockApi({
      "/api/models": ok("models_response", "valid-published-and-draft"),
      "/api/models/local-test": {
        status: 503,
        body: { error: "Storage de modelos no disponible: connect ECONNREFUSED" },
      },
    });
    renderModels();
    expect(await screen.findByTestId("p3-content")).toBeInTheDocument();
    const local = await screen.findByTestId("models-local-test");
    expect(
      await within(local).findByText(
        "El servidor respondió con estado 503: Storage de modelos no disponible: connect ECONNREFUSED"
      )
    ).toBeInTheDocument();
  });

  it("sin pruebas locales registradas lo dice", async () => {
    mockApi({
      "/api/models": ok("models_response", "valid-empty"),
      "/api/models/local-test": ok("local_test_models_response", "valid-empty"),
    });
    renderModels();
    expect(await screen.findByTestId("models-local-test")).toHaveTextContent(
      "Todavía no hay versiones local_test"
    );
  });
});
