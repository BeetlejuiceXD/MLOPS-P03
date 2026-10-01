import { useCallback, useEffect, useState } from "react";
import type { ZodSchema } from "zod";

/**
 * Base URL del backend. Igual que en los clientes de `lib/api` y `api`:
 * si la variable no está definida o viene vacía, se usa el proxy `/api`
 * declarado en vite.config.ts. Una cadena vacía no activa `??`, por eso se
 * comprueba explícitamente.
 */
const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL as string | undefined;
const API_BASE_URL =
  configuredBaseUrl !== undefined && configuredBaseUrl.trim() !== "" ? configuredBaseUrl : "/api";

type FetchState<T> =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "success"; data: T };

async function errorReason(res: Response): Promise<string | null> {
  try {
    const body: unknown = await res.json();
    return body && typeof body === "object" && "error" in body && typeof body.error === "string"
      ? body.error
      : null;
  } catch {
    return null;
  }
}

/**
 * Wraps native fetch + Zod validation so no component ever trusts an
 * unvalidated backend payload. On a schema mismatch we surface a generic
 * error instead of silently rendering `undefined`/`NaN` in the UI.
 */
export function useValidatedFetch<T>(
  url: string,
  schema: ZodSchema<T>,
  options: Readonly<{ showServerReason?: boolean }> = {}
) {
  const { showServerReason = false } = options;
  const [state, setState] = useState<FetchState<T>>({ status: "loading" });

  const load = useCallback(() => {
    let cancelled = false;
    setState({ status: "loading" });

    // La URL puede venir absoluta (http...) o relativa al backend (/dashboard...).
    // Las relativas se prefijan con la base del API para pasar por el proxy.
    const requestUrl = /^https?:\/\//.test(url) ? url : `${API_BASE_URL}${url}`;

    fetch(requestUrl)
      .then(async (res) => {
        if (!res.ok) {
          // Opcional: si la API explica el motivo ({ error }), se añade al mensaje. Lo usan
          // las fuentes de Training (D03-03), cuyo 503 trae lo que publicó trainer-worker.
          const reason = showServerReason ? await errorReason(res) : null;
          throw new Error(
            `El servidor respondió con estado ${res.status}${reason ? `: ${reason}` : "."}`
          );
        }
        const json: unknown = await res.json();
        const parsed = schema.safeParse(json);
        if (!parsed.success) {
          throw new Error("La respuesta del servidor no tiene el formato esperado.");
        }
        if (!cancelled) {
          setState({ status: "success", data: parsed.data });
        }
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        const message =
          err instanceof Error ? err.message : "Error desconocido al cargar los datos.";
        setState({ status: "error", message });
      });

    return () => {
      cancelled = true;
    };
  }, [url, schema, showServerReason]);

  useEffect(() => load(), [load]);

  return { ...state, reload: load };
}
