import { PageHeader } from "@/pipeline/components/PageHeader";
import { useModels } from "../api";
import { FetchBoundary } from "../components/FetchBoundary";
import { StatePanel } from "../components/StatePanel";

/**
 * D01-05 — Inference: solo ofrece versiones publicadas. La carga de imagen, la
 * predicción y "enviar a cola de anotación" se conectan en D06 con el motor de
 * inferencia; aquí queda la estructura y el bloqueo sin versión publicada.
 */
export function InferencePage() {
  const models = useModels();

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Inference"
        subtitle="Clasifica una imagen nueva con una versión publicada del modelo."
      />
      <FetchBoundary results={[models]}>
        {(data) => {
          const published = data.models.filter((model) => model.status === "published");
          if (published.length === 0) {
            return (
              <StatePanel variant="blocked" title="No hay ninguna versión publicada del modelo.">
                Inference solo usa versiones verificadas en AWS S3 (ver Models).
              </StatePanel>
            );
          }
          return (
            <form
              data-testid="p3-content"
              className="flex max-w-md flex-col gap-4 rounded-2xl border border-border bg-surface p-5"
              onSubmit={(event) => event.preventDefault()}
            >
              <label htmlFor="p3-model-version" className="text-sm font-medium text-ink">
                Versión del modelo
              </label>
              <select
                id="p3-model-version"
                className="rounded-lg border border-border bg-white px-3 py-2 text-sm"
                defaultValue={published[0]?.semver}
              >
                {published.map((model) => (
                  <option key={model.semver} value={model.semver}>
                    {model.semver}
                  </option>
                ))}
              </select>
              <label htmlFor="p3-image" className="text-sm font-medium text-ink">
                Imagen
              </label>
              <input id="p3-image" type="file" accept="image/jpeg,image/png" disabled />
              <p className="text-xs text-ink-muted">
                La predicción se habilita cuando el motor de inferencia esté conectado (D06).
              </p>
            </form>
          );
        }}
      </FetchBoundary>
    </div>
  );
}
