import type { ReactNode } from "react";
import { ErrorState } from "@/components/ui/ErrorState";
import { Skeleton } from "@/components/ui/Skeleton";

export type FetchResult<T> =
  | { status: "loading"; reload: () => void }
  | { status: "error"; message: string; reload: () => void }
  | { status: "success"; data: T; reload: () => void };

/**
 * D01-05 — Loading / error / éxito de una o varias llamadas validadas con Zod
 * (`useValidatedFetch`). Si cualquiera falla o no cumple su contrato, la página
 * muestra el error y no renderiza nada de los datos.
 */
export function FetchBoundary<T extends unknown[]>({
  results,
  children,
}: Readonly<{
  results: { [K in keyof T]: FetchResult<T[K]> };
  children: (...data: T) => ReactNode;
}>) {
  const list = results as readonly FetchResult<unknown>[];
  const failed = list.find((result) => result.status === "error");
  if (failed && failed.status === "error") {
    return (
      <ErrorState
        title="No se pudo cargar la información."
        message={failed.message}
        onRetry={() => {
          for (const result of list) result.reload();
        }}
      />
    );
  }
  if (list.some((result) => result.status === "loading")) {
    return (
      <div className="flex flex-col gap-4">
        <Skeleton className="h-24" />
        <Skeleton className="h-24" />
      </div>
    );
  }
  const data = list.map((result) => (result.status === "success" ? result.data : undefined));
  return <>{children(...(data as T))}</>;
}
