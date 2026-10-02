/**
 * D05-05 — Estados de la página Evaluation sobre la API protegida de D04-05.
 *
 * Las respuestas vienen de los fixtures compartidos `contracts/p3/fixtures/evaluation_response`
 * (datos de ejemplo etiquetados): prueban el render y los estados, no son resultados del
 * frozen test ni evidencia de la evaluación oficial (D06-01/D06-05).
 */
import "@testing-library/jest-dom/vitest";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { App } from "../src/App";
import { fixturePayload } from "./p3-fixtures";

type Reply = { status?: number; body: unknown };

/** `/api/evaluation` responde en orden la lista dada (la última se repite). */
function mockEvaluation(...replies: Reply[]) {
  let call = 0;
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const url = typeof input === "string" ? input : input.toString();
    const path = url.replace(/^https?:\/\/[^/]+/, "");
    if (path !== "/api/evaluation") {
      return new Response(JSON.stringify({ error: "Recurso no encontrado." }), { status: 404 });
    }
    const reply = replies[Math.min(call, replies.length - 1)] as Reply;
    call += 1;
    return new Response(JSON.stringify(reply.body), {
      status: reply.status ?? 200,
      headers: { "Content-Type": "application/json" },
    });
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

const fixture = (name: string): Reply => ({ body: fixturePayload("evaluation_response", name) });

const renderEvaluation = () =>
  render(
    <MemoryRouter initialEntries={["/ml/evaluation"]}>
      <App />
    </MemoryRouter>
  );

const requestedPaths = (fetchMock: ReturnType<typeof mockEvaluation>) =>
  fetchMock.mock.calls.map(([input]) => String(input).replace(/^https?:\/\/[^/]+/, ""));

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("Evaluation: bloqueada, pendiente, resultado y fallo son estados distintos", () => {
  it("blocked: no muestra nada del test ni pide la exportación por muestra", async () => {
    const fetchMock = mockEvaluation(fixture("valid-blocked"));
    renderEvaluation();
    expect(await screen.findByTestId("p3-state-blocked")).toHaveTextContent(
      "MODEL SELECTION CLOSED"
    );
    expect(screen.queryByTestId("p3-content")).not.toBeInTheDocument();
    expect(screen.queryByTestId("evaluation-pending")).not.toBeInTheDocument();
    expect(screen.queryByText(/accuracy/i)).not.toBeInTheDocument();
    expect(requestedPaths(fetchMock)).toEqual(["/api/evaluation"]);
  });

  it("pending: selección cerrada sin evaluación oficial, sin métricas ni matriz", async () => {
    mockEvaluation(fixture("valid-pending"));
    renderEvaluation();
    const pending = await screen.findByTestId("evaluation-pending");
    expect(pending).toHaveTextContent("La selección está cerrada");
    expect(pending).toHaveTextContent("la evaluación oficial aún no existe");
    // Identidad de la selección cerrada, para saber qué se evaluará en D06-01.
    expect(pending).toHaveTextContent("aaaaaaaa");
    expect(within(pending).getByTestId("evaluation-namespace")).toHaveTextContent("official");
    expect(screen.queryByTestId("confusion-matrix")).not.toBeInTheDocument();
    expect(screen.queryByTestId("p3-state-blocked")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Reintentar" })).not.toBeInTheDocument();
  });

  it("ready official: namespace visible y procedencia, sin aviso de prueba", async () => {
    mockEvaluation(fixture("valid-ready"));
    renderEvaluation();
    const content = await screen.findByTestId("p3-content");
    expect(within(content).getByTestId("evaluation-namespace")).toHaveTextContent("official");
    expect(within(content).getByTestId("evaluation-provenance")).toHaveTextContent("dddddddd");
    expect(within(content).getByText("59 / 66")).toBeInTheDocument();
    expect(screen.queryByTestId("evaluation-synthetic-warning")).not.toBeInTheDocument();
  });

  it("ready synthetic: se rotula como recorrido de prueba, nunca como evaluación oficial", async () => {
    mockEvaluation(fixture("valid-ready-synthetic"));
    renderEvaluation();
    const content = await screen.findByTestId("p3-content");
    expect(within(content).getByTestId("evaluation-namespace")).toHaveTextContent("synthetic");
    expect(screen.getByTestId("evaluation-synthetic-warning")).toHaveTextContent(
      "no es la evaluación oficial"
    );
  });

  it("fallo del servicio (503): error con el motivo y reintento, no pending ni blocked", async () => {
    mockEvaluation({
      status: 503,
      body: { error: "Evaluación guardada incoherente: el manifest no es el de la selección cerrada." },
    });
    renderEvaluation();
    expect(
      await screen.findByText(
        "El servidor respondió con estado 503: Evaluación guardada incoherente: el manifest no es el de la selección cerrada."
      )
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Reintentar" })).toBeInTheDocument();
    expect(screen.queryByTestId("evaluation-pending")).not.toBeInTheDocument();
    expect(screen.queryByTestId("p3-state-blocked")).not.toBeInTheDocument();
  });

  it.each([
    "invalid-ready-without-namespace",
    "invalid-ready-local-test-namespace",
    "invalid-pending-with-results",
  ])("%s: fuera de contrato, no se muestra", async (name) => {
    mockEvaluation(fixture(name));
    renderEvaluation();
    expect(
      await screen.findByText("La respuesta del servidor no tiene el formato esperado.")
    ).toBeInTheDocument();
    expect(screen.queryByTestId("p3-content")).not.toBeInTheDocument();
    expect(screen.queryByTestId("evaluation-pending")).not.toBeInTheDocument();
  });

  it("el estado sale siempre del backend: al recargar se vuelve a consultar", async () => {
    const fetchMock = mockEvaluation(
      { status: 503, body: { error: "MariaDB no responde." } },
      fixture("valid-pending")
    );
    renderEvaluation();
    fireEvent.click(await screen.findByRole("button", { name: "Reintentar" }));
    expect(await screen.findByTestId("evaluation-pending")).toBeInTheDocument();
    cleanup();

    renderEvaluation();
    expect(await screen.findByTestId("evaluation-pending")).toBeInTheDocument();
    expect(requestedPaths(fetchMock)).toEqual([
      "/api/evaluation",
      "/api/evaluation",
      "/api/evaluation",
    ]);
  });
});
