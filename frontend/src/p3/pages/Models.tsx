import { PageHeader } from "@/pipeline/components/PageHeader";
import { useModels } from "../api";
import { FetchBoundary } from "../components/FetchBoundary";
import { StatePanel } from "../components/StatePanel";
import { dateTime, shortHash } from "../format";
import { LocalTestModels } from "../models/LocalTestModels";

/**
 * D01-05 / D05-06 / D06-03 — Models: versiones del modelo y su trazabilidad. Official muestra
 * la identidad exacta del objeto (bucket/key, VersionId, SHA-256) y su tarjeta. La versión del modelo
 * (semver propio) y la del dataset (release DVC) se muestran en columnas distintas. Arriba,
 * solo `official` (GET /api/models); abajo y aparte, las pruebas locales `local_test`.
 */
export function ModelsPage() {
  const models = useModels();

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Models"
        subtitle="Versiones publicadas en AWS S3 con su run de MLflow, dataset y hash."
      />
      <FetchBoundary results={[models]}>
        {(data) =>
          data.models.length === 0 ? (
            <StatePanel variant="empty" title="Todavía no hay versiones del modelo.">
              Una versión aparece aquí al empaquetar el candidato elegido.
            </StatePanel>
          ) : (
            <div
              data-testid="p3-content"
              className="overflow-x-auto rounded-2xl border border-border"
            >
              <table className="w-full text-left text-sm">
                <thead className="bg-surface text-xs text-ink-muted">
                  <tr>
                    <th className="px-3 py-2">Versión del modelo</th>
                    <th className="px-3 py-2">Estado</th>
                    <th className="px-3 py-2">Dataset</th>
                    <th className="px-3 py-2">Run MLflow</th>
                    <th className="px-3 py-2">Bucket / key</th>
                    <th className="px-3 py-2">VersionId</th>
                    <th className="px-3 py-2">SHA-256</th>
                    <th className="px-3 py-2">Tarjeta</th>
                    <th className="px-3 py-2">Publicado</th>
                  </tr>
                </thead>
                <tbody>
                  {data.models.map((model) => (
                    <tr
                      key={model.semver}
                      data-testid={`official-row-${model.semver}`}
                      className="border-t border-border"
                    >
                      <td className="px-3 py-2 font-semibold">{model.semver}</td>
                      <td className="px-3 py-2">{model.status}</td>
                      <td className="px-3 py-2">{model.dvc_release}</td>
                      <td className="px-3 py-2 font-mono text-xs">
                        {shortHash(model.mlflow_run_id)}
                      </td>
                      <td className="px-3 py-2 font-mono text-xs break-all">
                        {model.s3_bucket}/{model.s3_key}
                      </td>
                      <td className="px-3 py-2 font-mono text-xs break-all">
                        {model.version_id ?? "—"}
                      </td>
                      <td className="px-3 py-2 font-mono text-xs break-all">{model.sha256}</td>
                      <td className="px-3 py-2 text-xs">
                        {model.model_card ? (
                          <span className="flex flex-col gap-0.5">
                            <span className="font-mono break-all">{model.model_card.s3_key}</span>
                            <span className="font-mono break-all">
                              {model.model_card.version_id ?? "—"}
                            </span>
                            <span>
                              {model.model_card.status === "published"
                                ? "Tarjeta verificada"
                                : `Tarjeta ${model.model_card.status}`}
                            </span>
                          </span>
                        ) : (
                          "Sin tarjeta"
                        )}
                      </td>
                      <td className="px-3 py-2">{dateTime(model.published_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )
        }
      </FetchBoundary>
      <LocalTestModels />
    </div>
  );
}
