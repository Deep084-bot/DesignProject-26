import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AppLayout } from './components/layout/AppLayout';
import { OverviewPage } from './pages/OverviewPage';
import { NewInspectionPage } from './pages/NewInspectionPage';
import { HistoryPage } from './pages/HistoryPage';
import { InspectionDetailPage } from './pages/InspectionDetailPage';
import { InspectionProgressPage } from './pages/InspectionProgressPage';
import { NodesPage } from './pages/NodesPage';

export const App: React.FC = () => {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<AppLayout />}>
          <Route index element={<OverviewPage />} />
          <Route path="new" element={<NewInspectionPage />} />
          <Route path="inspections" element={<HistoryPage />} />
          <Route path="inspections/:id" element={<InspectionDetailPage />} />
          <Route path="inspections/:id/progress" element={<InspectionProgressPage />} />
          <Route path="nodes" element={<NodesPage />} />

          {/* Legacy & friendly redirects */}
          <Route path="history" element={<Navigate to="/inspections" replace />} />
          <Route path="inspection/:id" element={<Navigate to="/inspections/:id" replace />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
};

export default App;
