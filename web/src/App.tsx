import { type ReactNode } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "./components";
import { DataOperationsPage } from "./pages/DataOperationsPage";
import { DecisionReviewPage } from "./pages/DecisionReviewPage";
import { DecisionSavedPage } from "./pages/DecisionSavedPage";
import { ExecutionPage } from "./pages/ExecutionPage";
import { HomePage } from "./pages/HomePage";
import { RankingPage } from "./pages/RankingPage";
import { SettingsPage } from "./pages/SettingsPage";
import { StockDetailPage } from "./pages/StockDetailPage";
import { TodayPage } from "./pages/TodayPage";
import { ValidationPage } from "./pages/ValidationPage";

export function App(): ReactNode {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<HomePage />} />
        <Route path="today" element={<TodayPage />} />
        <Route path="today/review" element={<DecisionReviewPage />} />
        <Route path="today/decision" element={<DecisionSavedPage />} />
        <Route path="today/executions" element={<ExecutionPage />} />
        <Route path="ranking" element={<RankingPage />} />
        <Route path="validation" element={<ValidationPage />} />
        <Route path="validation/details" element={<ValidationPage />} />
        <Route path="settings" element={<SettingsPage />} />
        <Route path="settings/data" element={<DataOperationsPage />} />
        <Route path="settings/:section" element={<SettingsPage />} />
        <Route path="stocks/:symbol" element={<StockDetailPage />} />
        <Route path="models/:modelId" element={<ValidationPage />} />
        <Route path="*" element={<Navigate to="/today" replace />} />
      </Route>
    </Routes>
  );
}
