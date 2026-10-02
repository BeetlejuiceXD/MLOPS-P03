import { useState } from "react";
import { apiUrl } from "@/hooks/useValidatedFetch";
import { useLocalTestModels } from "../api";
import { FetchBoundary } from "../components/FetchBoundary";
import { type LocalTestModelDetail, localTestModelDetailSchema } from "../contracts";
import { dateTime } from "../format";

type Check =
  | { state: "loading" }
  | { state: "done"; detail: LocalTestModelDetail }
  | { state: "error"; message: string };

/** GET /api/models/local-test/:semver: integridad comprobada ahora por VersionId. */
async function checkIntegrity(semver: string): Promise<LocalTestModelDetail> {
  const res = await fetch(apiUrl(`/models/local-test/${semver}`));
  if (!res.ok) {
    let reason: string | null = null;
    try {
      const body: unknown = await res.json();
      if (body && typeof body === "object" && "error" in body && typeof body.error === "string") {
        reason = body.error;
      }
    } catch {
      reason = null;
    }
    throw new Error(
      `El servidor respondió con estado ${res.status}${reason ? `: ${reason}` : "."}`
    );
  }
  const parsed = localTestModelDetailSchema.safeParse(await res.json());
  if (!parsed.success) throw new Error("La respuesta del servidor no tiene el formato esperado.");
  return parsed.data;
}

function IntegrityResult({ semver, check }: Readonly<{ semver: string; check: Check }>) {
  if (check.state === "loading") {
    return <span className="text-ink-muted">Comprobando…</span>;
  }
  if (check.state === "error") {
    return (
      <span data-testid={`integrity-${semver}`} className="text-status-pending">
        {check.message}
      </span>
    );
  }
  const { integrity } = check.detail;
  return (
    <span
      data-testid={`integrity-${semver}`}
      className={integrity?.ok ? "text-status-done" : "text-status-pending"}
    >
      {integrity === null
        ? "No se audita: la versión no está published."
        : integrity.ok
          ? `Integridad OK: el objeto por VersionId coincide con el SHA-256 y el tamaño registrados (${dateTime(integrity.checked_at)}).`
          : `Integridad fallida (${integrity.reason}): ${integrity.detail} (${dateTime(integrity.checked_at)}). El registro no cambia.`}
    </span>
  );
}

/**
 * D05-06 — Pruebas locales del registro de D04-06 (namespace `local_test`, MinIO). Se
 * muestran aparte de las versiones official y nunca como publicación en AWS (D06-03).
 */
export function LocalTestModels() {
  const models = useLocalTestModels();
  const [checks, setChecks] = useState<Record<string, Check>>({});

  const runCheck = (semver: string) => {
    setChecks((current) => ({ ...current, [semver]: { state: "loading" } }));
    checkIntegrity(semver)
      .then((detail) =>
        setChecks((current) => ({ ...current, [semver]: { state: "done", detail } }))
      )
      .catch((error: unknown) =>
        setChecks((current) => ({
          ...current,
          [semver]: {
            state: "error",
            message: error instanceof Error ? error.message : "Error desconocido.",
          },
        }))
      );
  };

  return (
    <section
      data-testid="models-local-test"
      aria-label="Pruebas locales del registro"
      className="flex flex-col gap-3 rounded-2xl border border-dashed border-border-strong p-5"
    >
      <div>
        <h2 className="text-sm font-semibold text-ink">
          Pruebas locales del registro{" "}
          <span className="rounded-full bg-surface px-2 py-0.5 font-mono text-xs">local_test</span>
        </h2>
        <p className="mt-1 text-xs text-ink-muted">
          Versiones subidas al MinIO local para probar el registro (register → upload → verify). No
          es una publicación en AWS ni aparece en la lista official.
        </p>
      </div>
      <FetchBoundary results={[models]}>
        {(data) =>
          data.models.length === 0 ? (
            <p className="text-sm text-ink-muted">Todavía no hay versiones local_test.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="text-ink-muted">
                  <tr>
                    <th className="px-2 py-1">Semver</th>
                    <th className="px-2 py-1">Estado</th>
                    <th className="px-2 py-1">Run MLflow</th>
                    <th className="px-2 py-1">Bucket / key</th>
                    <th className="px-2 py-1">VersionId</th>
                    <th className="px-2 py-1">SHA-256</th>
                    <th className="px-2 py-1">Tamaño</th>
                    <th className="px-2 py-1">Verificada</th>
                    <th className="px-2 py-1" />
                  </tr>
                </thead>
                <tbody>
                  {data.models.map((model) => {
                    const check = checks[model.semver];
                    return (
                      <tr
                        key={model.semver}
                        data-testid={`local-row-${model.semver}`}
                        className="border-t border-border align-top"
                      >
                        <td className="px-2 py-1 font-semibold">{model.semver}</td>
                        <td className="px-2 py-1">
                          {model.status}
                          {model.failure_reason && (
                            <span className="mt-1 block text-status-pending">
                              {model.failure_reason}: {model.failure_detail}
                            </span>
                          )}
                        </td>
                        <td className="px-2 py-1 font-mono">{model.mlflow_run_id}</td>
                        <td className="px-2 py-1 font-mono break-all">
                          {model.s3_bucket}/{model.s3_key}
                        </td>
                        <td className="px-2 py-1 font-mono break-all">{model.version_id ?? "—"}</td>
                        <td className="px-2 py-1 font-mono break-all">{model.sha256}</td>
                        <td className="px-2 py-1">{model.size_bytes} B</td>
                        <td className="px-2 py-1">{dateTime(model.published_at)}</td>
                        <td className="px-2 py-1">
                          {model.status === "published" && (
                            <div className="flex flex-col gap-1">
                              <button
                                type="button"
                                className="text-accent underline"
                                aria-label={`Comprobar ${model.semver}`}
                                onClick={() => runCheck(model.semver)}
                              >
                                Comprobar
                              </button>
                              <a
                                className="text-accent underline"
                                aria-label={`Descargar ${model.semver}`}
                                href={apiUrl(`/models/local-test/${model.semver}/object`)}
                                download
                              >
                                Descargar
                              </a>
                            </div>
                          )}
                          {check && <IntegrityResult semver={model.semver} check={check} />}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )
        }
      </FetchBoundary>
    </section>
  );
}
