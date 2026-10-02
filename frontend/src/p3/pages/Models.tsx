import { PageHeader } from "@/pipeline/components/PageHeader";
import { useModels } from "../api";
import { FetchBoundary } from "../components/FetchBoundary";
import { StatePanel } from "../components/StatePanel";
import { dateTime, shortHash } from "../format";
import { LocalTestModels } from "../models/LocalTestModels";

/**
 * D01-05 / D05-06 — Models: versiones del modelo y su trazabilidad. La versión del modelo
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
                    <th className="px-3 py-2">Objeto S3</th>
                    <th className="px-3 py-2">SHA-256</th>
                    <th className="px-3 py-2">Publicado</th>
                  </tr>
                </thead>
                <tbody>
                  {data.models.map((model) => (
                    <tr key={model.semver} className="border-t border-border">
                      <td className="px-3 py-2 font-semibold">{model.semver}</td>
                      <td className="px-3 py-2">{model.status}</td>
                      <td className="px-3 py-2">{model.dvc_release}</td>
                      <td className="px-3 py-2 font-mono text-xs">
                        {shortHash(model.mlflow_run_id)}
                      </td>
                      <td className="px-3 py-2 font-mono text-xs">{model.s3_key}</td>
                      <td className="px-3 py-2 font-mono text-xs">{shortHash(model.sha256)}</td>
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
