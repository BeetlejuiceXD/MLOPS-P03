/**
 * D05-05 — Estados de la página Evaluation sobre la API protegida de D04-05.
 *
 * Las respuestas vienen de los fixtures compartidos `contracts/p3/fixtures/evaluation_response`
 * (datos de ejemplo etiquetados): prueban el render y los estados, no son resultados del
 * frozen test ni evidencia de la evaluación oficial (D06-01/D06-05).
 *
 * Las comprobaciones usan `expect` de vitest (dentro de `waitFor` cuando hay que esperar al
 * fetch) en vez de `findBy*`/matchers de jest-dom, para que un estado equivocado falle por
 * aserción y no por una excepción de Testing Library.
 */
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
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

/** Espera a que aparezca el elemento y lo devuelve. */
async function appears(testId: string): Promise<HTMLElement> {
  await waitFor(() => expect(screen.queryByTestId(testId)).not.toBeNull());
  return screen.getByTestId(testId);
}

const text = (element: HTMLElement | null) => element?.textContent ?? "";
const absent = (testId: string) => expect(screen.queryByTestId(testId)).toBeNull();
const namespaceIn = (element: HTMLElement) =>
  text(element.querySelector('[data-testid="evaluation-namespace"]'));

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("Evaluation: bloqueada, pendiente, resultado y fallo son estados distintos", () => {
  it("blocked: no muestra nada del test ni pide la exportación por muestra", async () => {
    const fetchMock = mockEvaluation(fixture("valid-blocked"));
    renderEvaluation();
    expect(text(await appears("p3-state-blocked"))).toContain("MODEL SELECTION CLOSED");
    absent("p3-content");
    absent("evaluation-pending");
    expect(document.body.textContent).not.toMatch(/accuracy/i);
    expect(requestedPaths(fetchMock)).toEqual(["/api/evaluation"]);
  });

  it("pending: selección cerrada sin evaluación oficial, sin métricas ni matriz", async () => {
    mockEvaluation(fixture("valid-pending"));
    renderEvaluation();
    const pending = await appears("evaluation-pending");
    expect(text(pending)).toContain("La selección está cerrada");
    expect(text(pending)).toContain("la evaluación oficial aún no existe");
    // Identidad de la selección cerrada, para saber qué se evaluará en D06-01.
    expect(text(pending)).toContain("aaaaaaaa");
    expect(namespaceIn(pending)).toBe("official");
    absent("confusion-matrix");
    absent("p3-state-blocked");
    expect(screen.queryByRole("button", { name: "Reintentar" })).toBeNull();
  });

  it("ready official: namespace visible y procedencia, sin aviso de prueba", async () => {
    mockEvaluation(fixture("valid-ready"));
    renderEvaluation();
    const content = await appears("p3-content");
    expect(namespaceIn(content)).toBe("official");
    expect(text(screen.getByTestId("evaluation-provenance"))).toContain("dddddddd");
    expect(text(content)).toContain("59 / 66");
    absent("evaluation-synthetic-warning");
  });

  it("ready synthetic: se rotula como recorrido de prueba, nunca como evaluación oficial", async () => {
    mockEvaluation(fixture("valid-ready-synthetic"));
    renderEvaluation();
    const content = await appears("p3-content");
    expect(namespaceIn(content)).toBe("synthetic");
    expect(text(screen.queryByTestId("evaluation-synthetic-warning"))).toContain(
      "no es la evaluación oficial"
    );
  });

  it("fallo del servicio (503): error con el motivo y reintento, no pending ni blocked", async () => {
    mockEvaluation({
      status: 503,
      body: { error: "Evaluación guardada incoherente: el manifest no es el de la selección cerrada." },
    });
    renderEvaluation();
    await waitFor(() =>
      expect(document.body.textContent).toContain(
        "El servidor respondió con estado 503: Evaluación guardada incoherente: el manifest no es el de la selección cerrada."
      )
    );
    expect(screen.queryByRole("button", { name: "Reintentar" })).not.toBeNull();
    absent("evaluation-pending");
    absent("p3-state-blocked");
  });

  it.each([
    "invalid-ready-without-namespace",
    "invalid-ready-local-test-namespace",
    "invalid-pending-with-results",
  ])("%s: fuera de contrato, no se muestra", async (name) => {
    mockEvaluation(fixture(name));
    renderEvaluation();
    await waitFor(() =>
      expect(document.body.textContent).toContain(
        "La respuesta del servidor no tiene el formato esperado."
      )
    );
    absent("p3-content");
    absent("evaluation-pending");
  });

  it("el estado sale siempre del backend: al recargar se vuelve a consultar", async () => {
    const fetchMock = mockEvaluation(
      { status: 503, body: { error: "MariaDB no responde." } },
      fixture("valid-pending")
    );
    renderEvaluation();
    await waitFor(() =>
      expect(screen.queryByRole("button", { name: "Reintentar" })).not.toBeNull()
    );
    fireEvent.click(screen.getByRole("button", { name: "Reintentar" }));
    await appears("evaluation-pending");
    cleanup();

    renderEvaluation();
    await appears("evaluation-pending");
    expect(requestedPaths(fetchMock)).toEqual([
      "/api/evaluation",
      "/api/evaluation",
      "/api/evaluation",
    ]);
  });
});
