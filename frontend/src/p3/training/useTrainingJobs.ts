import { useCallback, useEffect, useRef, useState } from "react";
import { buildApiUrl } from "@/api/client";
import { type TrainingJob, trainingJobListSchema } from "../contracts";

type State =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "success"; jobs: TrainingJob[] };

const ACTIVE = new Set(["queued", "running"]);

/**
 * D02-05 — Lista de jobs persistidos. Mientras alguno siga `queued`/`running` vuelve a
 * consultar cada `pollMs`; el estado sale siempre de la API, así que sobrevive a recargas.
 */
export function useTrainingJobs(pollMs: number) {
  const [state, setState] = useState<State>({ status: "loading" });
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const alive = useRef(true);

  const load = useCallback(async () => {
    if (timer.current) clearTimeout(timer.current);
    let next: State;
    try {
      const response = await fetch(buildApiUrl("/training/jobs"));
      if (!response.ok) throw new Error(`El servidor respondió con estado ${response.status}.`);
      const parsed = trainingJobListSchema.safeParse(await response.json());
      if (!parsed.success)
        throw new Error("La respuesta del servidor no tiene el formato esperado.");
      next = { status: "success", jobs: parsed.data.jobs };
    } catch (error) {
      next = {
        status: "error",
        message: error instanceof Error ? error.message : "Error desconocido al cargar los jobs.",
      };
    }
    if (!alive.current) return;
    setState(next);
    if (next.status === "success" && next.jobs.some((job) => ACTIVE.has(job.status))) {
      timer.current = setTimeout(() => void load(), pollMs);
    }
  }, [pollMs]);

  useEffect(() => {
    alive.current = true;
    void load();
    return () => {
      alive.current = false;
      if (timer.current) clearTimeout(timer.current);
    };
  }, [load]);

  return { state, reload: load };
}
