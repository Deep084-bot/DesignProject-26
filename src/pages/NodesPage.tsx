import React, { useState, useEffect, useCallback } from 'react';
import { Server, Cpu, RefreshCw, CheckCircle2, AlertCircle, Clock, Layers } from 'lucide-react';
import { fetchClusterWorkers, fetchClusterSummary, getExecutionMode } from '../services/api';
import type { ClusterWorker, ClusterSummaryResponse, ExecutionMode } from '../types/inspection';
import { Button } from '../components/common/Button';

export const NodesPage: React.FC = () => {
  const [workers, setWorkers] = useState<ClusterWorker[]>([]);
  const [summary, setSummary] = useState<ClusterSummaryResponse | null>(null);
  const [mode, setMode] = useState<ExecutionMode>(getExecutionMode());
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);

  const loadClusterData = useCallback(async () => {
    try {
      setError(null);
      const [workersRes, summaryRes] = await Promise.all([
        fetchClusterWorkers(),
        fetchClusterSummary(),
      ]);
      setWorkers(workersRes.workers);
      setSummary(summaryRes);
    } catch (err: any) {
      setError(err.message || 'Failed to query cluster state.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    let isMounted = true;

    const fetchInitial = async () => {
      try {
        setError(null);
        const [workersRes, summaryRes] = await Promise.all([
          fetchClusterWorkers(),
          fetchClusterSummary(),
        ]);
        if (isMounted) {
          setWorkers(workersRes.workers);
          setSummary(summaryRes);
        }
      } catch (err: any) {
        if (isMounted) {
          setError(err.message || 'Failed to query cluster state.');
        }
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    };

    fetchInitial();

    const handleModeChange = (e: Event) => {
      const customEvent = e as CustomEvent<ExecutionMode>;
      if (customEvent.detail) setMode(customEvent.detail);
      else setMode(getExecutionMode());
      fetchInitial();
    };

    window.addEventListener('surface-mode-changed', handleModeChange);
    return () => {
      isMounted = false;
      window.removeEventListener('surface-mode-changed', handleModeChange);
    };
  }, []);

  // Periodic polling every 3 seconds if auto-refresh is active
  useEffect(() => {
    if (!autoRefresh) return;
    const interval = setInterval(() => {
      loadClusterData();
    }, 3000);
    return () => clearInterval(interval);
  }, [autoRefresh, loadClusterData]);

  const clusterMode = summary?.mode || (mode === 'demo' ? 'local' : 'local');

  return (
    <div className="space-y-8 max-w-5xl mx-auto pb-12">
      {/* Top Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-[#E8E5DC] pb-6">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold text-[#343832] tracking-tight">Cluster Nodes</h1>
            <span
              className={`px-2.5 py-0.5 rounded-full text-xs font-semibold uppercase tracking-wider ${
                clusterMode === 'celery'
                  ? 'bg-emerald-100 text-emerald-800 border border-emerald-300'
                  : 'bg-amber-100 text-amber-800 border border-amber-300'
              }`}
            >
              {clusterMode === 'celery' ? 'Celery + Redis' : 'Local Thread Pool'}
            </span>
          </div>
          <p className="text-sm text-[#777D73] mt-1">
            Real-time monitoring of distributed worker processes and job execution queue.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <label className="flex items-center gap-2 text-xs text-[#777D73] cursor-pointer mr-2 select-none">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="rounded border-[#D5D2C7] text-[#4A5B48] focus:ring-[#9BAE98]"
            />
            Auto-refresh (3s)
          </label>
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              setLoading(true);
              loadClusterData();
            }}
            disabled={loading}
          >
            <RefreshCw className={`w-3.5 h-3.5 mr-1.5 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-lg bg-red-50 border border-red-200 text-sm text-red-700 flex items-start gap-2.5">
          <AlertCircle className="w-4 h-4 mt-0.5 shrink-0 text-red-600" />
          <div>
            <div className="font-semibold">Unable to fetch cluster metrics</div>
            <div className="text-xs text-red-600 mt-0.5">{error}</div>
          </div>
        </div>
      )}

      {/* Cluster Summary Metrics */}
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
        <div className="bg-[#FAF9F5] border border-[#E8E5DC] rounded-lg p-3.5">
          <div className="text-xs text-[#777D73] uppercase font-medium tracking-wider flex items-center gap-1.5">
            <Layers className="w-3.5 h-3.5" /> Total Jobs
          </div>
          <div className="text-xl font-bold text-[#343832] mt-1">
            {summary?.totalJobs ?? 0}
          </div>
        </div>

        <div className="bg-[#FAF9F5] border border-[#E8E5DC] rounded-lg p-3.5">
          <div className="text-xs text-amber-700 uppercase font-medium tracking-wider flex items-center gap-1.5">
            <Clock className="w-3.5 h-3.5" /> Queued
          </div>
          <div className="text-xl font-bold text-amber-700 mt-1">
            {summary?.queued ?? 0}
          </div>
        </div>

        <div className="bg-[#FAF9F5] border border-[#E8E5DC] rounded-lg p-3.5">
          <div className="text-xs text-blue-700 uppercase font-medium tracking-wider flex items-center gap-1.5">
            <Cpu className="w-3.5 h-3.5" /> Processing
          </div>
          <div className="text-xl font-bold text-blue-700 mt-1">
            {summary?.processing ?? 0}
          </div>
        </div>

        <div className="bg-[#FAF9F5] border border-[#E8E5DC] rounded-lg p-3.5">
          <div className="text-xs text-emerald-700 uppercase font-medium tracking-wider flex items-center gap-1.5">
            <CheckCircle2 className="w-3.5 h-3.5" /> Completed
          </div>
          <div className="text-xl font-bold text-emerald-700 mt-1">
            {summary?.completed ?? 0}
          </div>
        </div>

        <div className="bg-[#FAF9F5] border border-[#E8E5DC] rounded-lg p-3.5">
          <div className="text-xs text-rose-700 uppercase font-medium tracking-wider flex items-center gap-1.5">
            <AlertCircle className="w-3.5 h-3.5" /> Failed
          </div>
          <div className="text-xl font-bold text-rose-700 mt-1">
            {summary?.failed ?? 0}
          </div>
        </div>
      </div>

      {/* Worker Nodes Section */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-base font-semibold text-[#343832] flex items-center gap-2">
            <Server className="w-4 h-4 text-[#4A5B48]" />
            Active Workers ({workers.length})
          </h2>
          <span className="text-xs text-[#777D73]">
            {clusterMode === 'celery' ? 'Inspected via Celery control API' : 'Local process executor'}
          </span>
        </div>

        {clusterMode === 'local' ? (
          <div className="bg-[#FAF9F5] border border-[#E8E5DC] rounded-xl p-6">
            <div className="flex items-start gap-4">
              <div className="w-10 h-10 rounded-lg bg-[#E7EDE4] border border-[#9BAE98]/40 flex items-center justify-center text-[#4A5B48] shrink-0">
                <Cpu className="w-5 h-5" />
              </div>
              <div className="space-y-1">
                <div className="text-sm font-semibold text-[#343832] flex items-center gap-2">
                  <span>Local Thread Pool Worker</span>
                  <span className="px-2 py-0.5 rounded text-[11px] font-medium bg-emerald-50 text-emerald-700 border border-emerald-200">
                    Active
                  </span>
                </div>
                <p className="text-xs text-[#777D73] leading-relaxed">
                  The backend is operating in <code className="bg-[#EFECE4] px-1 py-0.5 rounded text-[#343832]">SURFACE_EXECUTION=local</code> mode.
                  Inspections are processed concurrently using in-process Python thread pools without requiring external Redis or Celery worker processes.
                </p>
                <div className="pt-3 text-xs text-[#777D73]">
                  To activate distributed execution across network laptops, launch Redis and export <code className="bg-[#EFECE4] px-1 py-0.5 rounded text-[#343832]">SURFACE_EXECUTION=celery</code>.
                </div>
              </div>
            </div>
          </div>
        ) : workers.length === 0 ? (
          <div className="bg-[#FAF9F5] border border-dashed border-[#D5D2C7] rounded-xl p-8 text-center space-y-3">
            <div className="w-10 h-10 rounded-full bg-amber-50 border border-amber-200 text-amber-700 mx-auto flex items-center justify-center">
              <Server className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-[#343832]">No Celery Workers Detected</h3>
              <p className="text-xs text-[#777D73] max-w-md mx-auto mt-1">
                Celery distributed mode is enabled, but no online worker nodes responded to inspect ping.
              </p>
            </div>
            <div className="bg-[#F0EFE8] border border-[#E8E5DC] rounded-lg p-3 text-left font-mono text-xs text-[#343832] max-w-lg mx-auto">
              <div># Start a worker from repo root:</div>
              <div className="text-[#4A5B48] mt-1 select-all font-semibold">
                PYTHONPATH=backend:. ./backend/venv/bin/python -m celery -A app.celery_app worker --pool=solo --loglevel=info -n worker-1@%h
              </div>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {workers.map((worker) => (
              <div
                key={worker.name}
                className="bg-[#FAF9F5] border border-[#E8E5DC] rounded-xl p-5 space-y-4 hover:border-[#D5D2C7] transition-colors"
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-2.5">
                    <div className="w-8 h-8 rounded-lg bg-emerald-50 border border-emerald-200 flex items-center justify-center text-emerald-700">
                      <Server className="w-4 h-4" />
                    </div>
                    <div>
                      <div className="text-sm font-semibold text-[#343832] font-mono">
                        {worker.name}
                      </div>
                      <div className="text-xs text-[#777D73]">Distributed worker node</div>
                    </div>
                  </div>
                  <span className="flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium bg-emerald-50 text-emerald-700 border border-emerald-200">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                    Online
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-3 text-xs border-t border-[#E8E5DC] pt-3">
                  <div>
                    <span className="text-[#777D73]">Active Jobs:</span>
                    <span className="ml-1.5 font-semibold text-[#343832]">
                      {worker.activeJobs.length}
                    </span>
                  </div>
                  <div>
                    <span className="text-[#777D73]">Tasks Completed:</span>
                    <span className="ml-1.5 font-semibold text-[#343832]">
                      {worker.tasksDone}
                    </span>
                  </div>
                </div>

                {worker.activeJobs.length > 0 && (
                  <div className="space-y-1.5 pt-1">
                    <div className="text-[11px] uppercase tracking-wider text-[#777D73] font-medium">
                      Currently Processing:
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {worker.activeJobs.map((jobId) => (
                        <span
                          key={jobId}
                          className="px-2 py-0.5 rounded bg-blue-50 border border-blue-200 text-blue-700 font-mono text-xs font-medium"
                        >
                          {jobId}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Active Worker Assignments Table */}
      {summary && Object.keys(summary.activeAssignments).length > 0 && (
        <div className="space-y-3 pt-2">
          <h2 className="text-sm font-semibold text-[#343832]">Active Job Assignments</h2>
          <div className="bg-[#FAF9F5] border border-[#E8E5DC] rounded-xl overflow-hidden">
            <table className="w-full text-left text-xs">
              <thead className="bg-[#F0EFE8] border-b border-[#E8E5DC] text-[#777D73] uppercase tracking-wider font-semibold">
                <tr>
                  <th className="py-2.5 px-4">Inspection ID</th>
                  <th className="py-2.5 px-4">Assigned Worker Node</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#E8E5DC]">
                {Object.entries(summary.activeAssignments).map(([jobId, workerName]) => (
                  <tr key={jobId} className="hover:bg-[#F5F4ED]">
                    <td className="py-2.5 px-4 font-mono font-medium text-[#343832]">{jobId}</td>
                    <td className="py-2.5 px-4 font-mono text-[#4A5B48]">{workerName}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

export default NodesPage;
