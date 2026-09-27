import { Navigate, Route, Routes } from "react-router-dom";
import { AnnotateScreen } from "@/components/annotate/AnnotateScreen";
import { AppLayout } from "@/components/layout/AppLayout";
import { UploadScreen } from "@/components/upload/UploadScreen";
import { EvaluationPage } from "@/p3/pages/Evaluation";
import { ExperimentsPage } from "@/p3/pages/Experiments";
import { InferencePage } from "@/p3/pages/Inference";
import { ModelsPage } from "@/p3/pages/Models";
import { TrainingPage } from "@/p3/pages/Training";
import { DashboardPage } from "@/pages/Dashboard";
import { SearchPage } from "@/pages/SearchPage";
import { AnalyzersPage } from "@/pipeline/pages/Analyzers";
import { CopilotPage } from "@/pipeline/pages/Copilot";
import { OverviewPage } from "@/pipeline/pages/Overview";
import { ProjectionsPage } from "@/pipeline/pages/Projections";
import { SettingsPage } from "@/pipeline/pages/Settings";
import { SplitsPage } from "@/pipeline/pages/Splits";
import { VersionsPage } from "@/pipeline/pages/Versions";

/** D01-05: las cinco páginas P3 viven en el mismo portal y AppLayout que P1/P2. */
const P3_ROUTES = [
  { path: "/ml/training", Page: TrainingPage },
  { path: "/ml/experiments", Page: ExperimentsPage },
  { path: "/ml/evaluation", Page: EvaluationPage },
  { path: "/ml/models", Page: ModelsPage },
  { path: "/ml/inference", Page: InferencePage },
] as const;

export function App(): JSX.Element {
  return (
    <Routes>
      <Route path="/" element={<Navigate to="/dashboard" replace />} />
      <Route
        path="/dashboard"
        element={
          <AppLayout>
            <DashboardPage />
          </AppLayout>
        }
      />
      {/* SearchPage se envuelve con AppLayout internamente (no aquí), porque
          necesita pasarle su propio contenido de filtros como sidebarExtra
          — ver SearchPage.tsx. */}
      <Route path="/search" element={<SearchPage />} />
      <Route
        path="/upload"
        element={
          <AppLayout>
            <UploadScreen />
          </AppLayout>
        }
      />
      {/* Annotate es un modo de enfoque de pantalla completa a propósito: sin
          nav global, con su propio botón "Volver". Ver GlobalNav.tsx. */}
      <Route path="/annotate/:imageId" element={<AnnotateScreen />} />

      {/* P2-14: dashboard de calidad de dataset (Frente 7) — mismo portal,
          mismo AppLayout/GlobalNav que el resto; no tiene nav ni shell
          propio (ver GlobalNav.tsx). */}
      <Route path="/pipeline" element={<Navigate to="/pipeline/overview" replace />} />
      <Route
        path="/pipeline/overview"
        element={
          <AppLayout>
            <OverviewPage />
          </AppLayout>
        }
      />
      <Route
        path="/pipeline/analyzers"
        element={
          <AppLayout>
            <AnalyzersPage />
          </AppLayout>
        }
      />
      <Route
        path="/pipeline/splits"
        element={
          <AppLayout>
            <SplitsPage />
          </AppLayout>
        }
      />
      <Route
        path="/pipeline/versions"
        element={
          <AppLayout>
            <VersionsPage />
          </AppLayout>
        }
      />
      <Route
        path="/pipeline/copilot"
        element={
          <AppLayout>
            <CopilotPage />
          </AppLayout>
        }
      />
      <Route
        path="/pipeline/settings"
        element={
          <AppLayout>
            <SettingsPage />
          </AppLayout>
        }
      />

      <Route
        path="/pipeline/projections"
        element={
          <AppLayout>
            <ProjectionsPage />
          </AppLayout>
        }
      />

      <Route path="/ml" element={<Navigate to="/ml/training" replace />} />
      {P3_ROUTES.map(({ path, Page }) => (
        <Route
          key={path}
          path={path}
          element={
            <AppLayout>
              <Page />
            </AppLayout>
          }
        />
      ))}
      <Route path="*" element={<Navigate to="/dashboard" replace />} />
    </Routes>
  );
}
