import React, { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { Check, AlertCircle, RefreshCw, Box, ArrowLeft, Clock, Server, PlayCircle } from 'lucide-react';
import {
  fetchInspectionById,
  updateInspectionProgress,
  pollInspectionStatus,
  getExecutionMode,
  getBackendBaseUrl,
} from '../services/api';
import type { Inspection } from '../types/inspection';
import { Button } from '../components/common/Button';

export const InspectionProgressPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const [inspection, setInspection] = useState<Inspection | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [transientWarning, setTransientWarning] = useState<string | null>(null);

  const mode = getExecutionMode();
  const backendBaseUrl = getBackendBaseUrl();

  const hasLoadedRef = useRef(false);

  // Primary inspection data loading and real-backend polling
  useEffect(() => {
    if (!id) return;

    let isMounted = true;

    if (mode === 'backend') {
      // Real processing mode: poll backend GET /api/inspections/{id}
      const unsubscribe = pollInspectionStatus(
        id,
        (data) => {
          if (!isMounted) return;
          hasLoadedRef.current = true;
          setInspection(data);
          setLoading(false);
          setError(null);
          setTransientWarning(null);
        },
        (err) => {
          if (!isMounted) return;
          if (!hasLoadedRef.current) {
            setError(err.message || 'Unable to connect to backend inspection status endpoint.');
            setLoading(false);
          } else {
            // Transient network retry
            setTransientWarning('Waiting for backend response (retrying connection)...');
          }
        },
        2500
      );

      return () => {
        isMounted = false;
        unsubscribe();
      };
    } else {
      // Demo mode: single fetch from session storage
      fetchInspectionById(id)
        .then((data) => {
          if (!isMounted) return;
          if (!data) {
            setError('Inspection not found in local demonstration session.');
          } else {
            setInspection(data);
          }
          setLoading(false);
        })
        .catch((err) => {
          if (!isMounted) return;
          setError(err.message || 'Failed to load local demo inspection.');
          setLoading(false);
        });

      return () => {
        isMounted = false;
      };
    }
  }, [id, mode]);

  // Demonstration progression simulator (STRICTLY ISOLATED TO DEMO MODE)
  useEffect(() => {
    // Real processing mode must NEVER use simulated progress
    if (
      mode === 'backend' ||
      !inspection ||
      !inspection.isDemoMode ||
      inspection.status === 'completed' ||
      inspection.status === 'failed'
    ) {
      return;
    }

    const timer = setTimeout(() => {
      const nextStepIndex = inspection.currentStepIndex + 1;
      const isFinished = nextStepIndex >= inspection.steps.length;

      const updated = updateInspectionProgress(inspection.id, nextStepIndex, isFinished);
      if (updated) {
        setInspection({ ...updated });
      }
    }, 2400);

    return () => clearTimeout(timer);
  }, [inspection, mode]);

  if (loading) {
    return (
      <div className="py-20 text-center text-xs text-[#777D73] space-y-2">
        <div className="w-5 h-5 border-2 border-[#9BAE98] border-t-transparent rounded-full animate-spin mx-auto" />
        <p>Connecting to inspection status service...</p>
      </div>
    );
  }

  if (error || !inspection) {
    return (
      <div className="max-w-md mx-auto py-16 text-center space-y-4">
        <div className="w-10 h-10 rounded-full bg-[#F9EFEA] text-[#8F543E] flex items-center justify-center mx-auto">
          <AlertCircle className="w-5 h-5" />
        </div>
        <div className="space-y-1">
          <h2 className="text-sm font-semibold text-[#343832]">Inspection status unavailable</h2>
          <p className="text-xs text-[#777D73] leading-relaxed">{error || 'Unknown error occurred.'}</p>
        </div>
        <div className="flex items-center justify-center gap-2 pt-2">
          <Link to="/inspections">
            <Button variant="secondary" size="sm">
              Return to inspections
            </Button>
          </Link>
          <Link to="/new">
            <Button variant="primary" size="sm">
              Start new inspection
            </Button>
          </Link>
        </div>
      </div>
    );
  }

  const isCompleted = inspection.status === 'completed';
  const isFailed = inspection.status === 'failed';
  const isQueued = inspection.status === 'queued';

  return (
    <div className="max-w-2xl mx-auto space-y-8">
      {/* Header */}
      <div className="space-y-1">
        <Link
          to="/inspections"
          className="text-xs text-[#777D73] hover:text-[#343832] inline-flex items-center gap-1 mb-2 transition-colors"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          <span>All inspections</span>
        </Link>

        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <h1 className="text-xl font-semibold text-[#343832] tracking-tight">
            {inspection.name}
          </h1>
          <span className="font-mono text-xs text-[#777D73] bg-[#F0EFE8] px-2 py-0.5 rounded border border-[#E8E5DC] self-start sm:self-auto">
            {inspection.id}
          </span>
        </div>

        <p className="text-xs text-[#777D73]">
          {isCompleted
            ? 'Surface reconstruction and anomaly detection are complete.'
            : isFailed
            ? 'Processing stopped due to an error during reconstruction.'
            : isQueued
            ? 'Task queued — waiting for an available photogrammetry processing worker.'
            : 'Processing stereo images and reconstructing 3D surface geometry.'}
        </p>

        {/* Environment Status Badge */}
        <div className="pt-1 flex items-center gap-2">
          {inspection.isDemoMode ? (
            <span className="inline-flex items-center gap-1.5 text-[11px] text-[#777D73] bg-[#F0EFE8] px-2.5 py-0.5 rounded border border-[#E8E5DC]">
              <PlayCircle className="w-3 h-3 text-[#9BA297]" />
              <span>Demonstration Mode: Local progression simulator</span>
            </span>
          ) : (
            <span className="inline-flex items-center gap-1.5 text-[11px] text-[#3F513D] bg-[#E7EDE4] px-2.5 py-0.5 rounded border border-[#D5E0D1]">
              <Server className="w-3 h-3 text-[#3F513D]" />
              <span>
                Real backend polling: <code className="font-mono">{backendBaseUrl || 'Default API'}</code>
              </span>
            </span>
          )}
        </div>

        {/* Transient Network Re-try Indicator */}
        {transientWarning && (
          <div className="pt-2 text-[11px] text-[#8F543E] flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-[#8F543E] animate-ping" />
            <span>{transientWarning}</span>
          </div>
        )}
      </div>

      {/* Failure Callout if applicable */}
      {isFailed && (
        <div className="p-4 rounded-xl bg-[#F9EFEA] border border-[#E8C9BD] space-y-2 text-xs">
          <div className="flex items-center gap-2 text-[#8F543E] font-medium">
            <AlertCircle className="w-4 h-4 flex-shrink-0" />
            <span>
              Processing failed: {inspection.failureMessage || 'The reconstruction pipeline encountered an unrecoverable error.'}
            </span>
          </div>
          <p className="text-[#8F543E]/80 pl-6 text-[11px] leading-relaxed">
            Please verify image sharpness, illumination consistency, and stereo baseline distance before retrying.
          </p>
        </div>
      )}

      {/* Queued Callout if applicable */}
      {isQueued && (
        <div className="p-4 rounded-xl bg-[#F0EFE8] border border-[#E8E5DC] space-y-1 text-xs">
          <div className="flex items-center gap-2 text-[#343832] font-medium">
            <Clock className="w-4 h-4 text-[#777D73]" />
            <span>Inspection task is queued</span>
          </div>
          <p className="text-[#777D73] pl-6 text-[11px] leading-relaxed">
            The compute server has received your stereo image pair and will begin depth estimation shortly.
          </p>
        </div>
      )}

      {/* 6-Stage Calm Step Indicator */}
      <div className="bg-white rounded-xl border border-[#E8E5DC] shadow-[0_1px_3px_rgba(0,0,0,0.02)] p-6 space-y-6">
        <h2 className="text-xs font-semibold text-[#343832] uppercase tracking-wider pb-2 border-b border-[#E8E5DC]">
          Inspection stages
        </h2>

        <div className="space-y-6">
          {inspection.steps.map((step, idx) => {
            const isStepCompleted = step.status === 'completed';
            const isStepActive = step.status === 'active';
            const isStepFailed = step.status === 'failed';

            return (
              <div key={step.id} className="flex items-start gap-4">
                {/* Step Icon / Dot */}
                <div className="flex-shrink-0 mt-0.5">
                  {isStepCompleted ? (
                    <div className="w-6 h-6 rounded-full bg-[#E7EDE4] border border-[#D5E0D1] flex items-center justify-center text-[#3F513D]">
                      <Check className="w-3.5 h-3.5" />
                    </div>
                  ) : isStepActive ? (
                    <div className="w-6 h-6 rounded-full bg-[#F0EFE8] border border-[#9BAE98] flex items-center justify-center text-[#343832]">
                      <span className="w-2 h-2 rounded-full bg-[#9BAE98] animate-pulse" />
                    </div>
                  ) : isStepFailed ? (
                    <div className="w-6 h-6 rounded-full bg-[#F9EFEA] border border-[#E8C9BD] flex items-center justify-center text-[#8F543E]">
                      <AlertCircle className="w-3.5 h-3.5" />
                    </div>
                  ) : (
                    <div className="w-6 h-6 rounded-full bg-[#F8F7F2] border border-[#E8E5DC] flex items-center justify-center text-[#9BA297] font-mono text-[11px]">
                      {idx + 1}
                    </div>
                  )}
                </div>

                {/* Step Content */}
                <div className="flex-1 space-y-0.5">
                  <div className="flex items-center justify-between">
                    <span
                      className={`text-xs font-medium ${
                        isStepCompleted
                          ? 'text-[#343832]'
                          : isStepActive
                          ? 'text-[#343832] font-semibold'
                          : isStepFailed
                          ? 'text-[#8F543E] font-semibold'
                          : 'text-[#9BA297]'
                      }`}
                    >
                      {step.label}
                    </span>
                    <div className="flex items-center gap-2">
                      {typeof step.durationSec === 'number' && (
                        <span className="text-[10px] font-mono text-[#777D73] bg-[#F0EFE8] px-1.5 py-0.5 rounded">
                          {step.durationSec.toFixed(2)}s
                        </span>
                      )}
                      {step.worker && (
                        <span
                          className="text-[10px] font-mono text-[#3F513D] bg-[#E7EDE4] px-1.5 py-0.5 rounded border border-[#D5E0D1]"
                          title={`Execution node: ${step.worker}`}
                        >
                          {step.worker}
                        </span>
                      )}
                      <span className="text-[11px] font-mono text-[#777D73]">
                        {isStepCompleted
                          ? 'Done'
                          : isStepActive
                          ? 'In progress'
                          : isStepFailed
                          ? 'Failed'
                          : 'Waiting'}
                      </span>
                    </div>
                  </div>
                  <p className="text-xs text-[#777D73] leading-relaxed">{step.description}</p>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Completion Action */}
      <div className="flex items-center justify-between pt-2">
        <Link to="/inspections" className="text-xs text-[#777D73] hover:text-[#343832]">
          Back to list
        </Link>

        {isCompleted ? (
          <Button
            variant="primary"
            size="md"
            icon={<Box className="w-4 h-4" />}
            onClick={() => navigate(`/inspections/${inspection.id}`)}
          >
            Inspect 3D Surface & Findings
          </Button>
        ) : isFailed ? (
          <Link to="/new">
            <Button variant="secondary" size="md" icon={<RefreshCw className="w-3.5 h-3.5" />}>
              Start a new inspection
            </Button>
          </Link>
        ) : (
          <div className="flex items-center gap-2 text-xs text-[#777D73]">
            <span className="w-2 h-2 rounded-full bg-[#9BAE98] animate-ping" />
            <span>
              {isQueued
                ? 'Waiting in queue...'
                : 'Processing surface geometry on pipeline...'}
            </span>
          </div>
        )}
      </div>
    </div>
  );
};
