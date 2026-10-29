import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { ArrowLeft, Box, Calendar, Server, PlayCircle, Download } from 'lucide-react';
import { fetchInspectionById } from '../services/api';
import type { Inspection } from '../types/inspection';
import { ThreeSurfaceViewer } from '../components/viewer/ThreeSurfaceViewer';
import { StatusBadge, FindingBadge } from '../components/common/Badge';
import { Button } from '../components/common/Button';

export const InspectionDetailPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [inspection, setInspection] = useState<Inspection | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedFindingId, setSelectedFindingId] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;

    fetchInspectionById(id)
      .then((data) => {
        setInspection(data);
        if (data && data.findings && data.findings.length > 0) {
          setSelectedFindingId(data.findings[0].id);
        }
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message || 'Failed to load inspection results.');
        setLoading(false);
      });
  }, [id]);

  if (loading) {
    return (
      <div className="py-24 text-center text-xs text-[#777D73] space-y-2">
        <div className="w-5 h-5 border-2 border-[#9BAE98] border-t-transparent rounded-full animate-spin mx-auto" />
        <p>Loading 3D inspection results...</p>
      </div>
    );
  }

  if (error || !inspection) {
    return (
      <div className="max-w-md mx-auto py-20 text-center space-y-4">
        <div className="w-10 h-10 rounded-full bg-[#F0EFE8] text-[#777D73] flex items-center justify-center mx-auto">
          <Box className="w-5 h-5" />
        </div>
        <div className="space-y-1">
          <h2 className="text-sm font-semibold text-[#343832]">Inspection not found</h2>
          <p className="text-xs text-[#777D73]">{error || 'The requested inspection record does not exist.'}</p>
        </div>
        <Link to="/inspections">
          <Button variant="secondary" size="sm">
            Browse all inspections
          </Button>
        </Link>
      </div>
    );
  }

  const selectedFinding =
    inspection.findings.find((f) => f.id === selectedFindingId) || inspection.findings[0];
  const crackCount = inspection.findings.filter((f) => f.type === 'potential_crack').length;

  const formatDate = (isoString: string) => {
    try {
      const d = new Date(isoString);
      return new Intl.DateTimeFormat('en-GB', {
        day: 'numeric',
        month: 'short',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      }).format(d);
    } catch {
      return isoString;
    }
  };

  return (
    <div className="space-y-6">
      {/* Top Breadcrumb & Title Bar */}
      <div className="space-y-2">
        <div className="flex items-center gap-2 text-xs text-[#777D73]">
          <Link to="/inspections" className="hover:text-[#343832] inline-flex items-center gap-1 transition-colors">
            <ArrowLeft className="w-3.5 h-3.5" />
            <span>All inspections</span>
          </Link>
          <span>/</span>
          <span className="font-mono text-[#343832]">{inspection.id}</span>
        </div>

        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <div className="flex items-center gap-2.5">
              <h1 className="text-xl md:text-2xl font-semibold text-[#343832] tracking-tight">
                {inspection.name}
              </h1>
              {inspection.isDemoMode ? (
                <span className="inline-flex items-center gap-1 font-mono text-[11px] text-[#777D73] bg-[#F0EFE8] px-2 py-0.5 rounded border border-[#E8E5DC]">
                  <PlayCircle className="w-3 h-3 text-[#9BA297]" />
                  <span>Demo record</span>
                </span>
              ) : (
                <span className="inline-flex items-center gap-1 font-mono text-[11px] text-[#3F513D] bg-[#E7EDE4] px-2 py-0.5 rounded border border-[#D5E0D1]">
                  <Server className="w-3 h-3 text-[#3F513D]" />
                  <span>Backend record</span>
                </span>
              )}
            </div>

            <div className="flex items-center gap-3 text-xs text-[#777D73] mt-1">
              <span>{inspection.surfaceType}</span>
              <span>•</span>
              <span className="flex items-center gap-1">
                <Calendar className="w-3.5 h-3.5 text-[#9BA297]" />
                {formatDate(inspection.createdAt)}
              </span>
              <span>•</span>
              <StatusBadge status={inspection.status} size="sm" />
            </div>
          </div>

          <div className="flex items-center gap-2">
            <Link to="/new">
              <Button variant="secondary" size="sm">
                New inspection
              </Button>
            </Link>
          </div>
        </div>
      </div>

      {/* Main 3D Viewer & Findings Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* 3D Canvas (Spacious Left Column) */}
        <div className="lg:col-span-8 space-y-4">
          <ThreeSurfaceViewer
            plyUrl={inspection.pointCloudUrl}
            meshUrl={inspection.meshUrl}
            meshStatus={inspection.meshStatus}
            meshError={inspection.meshError}
            findings={inspection.findings}
            selectedFindingId={selectedFindingId}
            isDemoModel={inspection.isDemoMode}
          />

          {/* Artifact Download & Geometry Action Strip */}
          {(inspection.meshUrl || inspection.pointCloudUrl || inspection.meshStatus === 'failed') && (
            <div className="flex flex-wrap items-center justify-between gap-3 p-3 rounded-xl bg-white border border-[#E8E5DC] text-xs shadow-[0_1px_3px_rgba(0,0,0,0.02)]">
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-semibold text-[#343832] uppercase tracking-wider text-[11px]">Artifacts:</span>
                <span className="text-[#777D73] font-mono text-[11px]">
                  {inspection.triangleCount ? `${inspection.triangleCount.toLocaleString()} triangles` : ''}
                  {inspection.triangleCount && inspection.pointCount ? ' • ' : ''}
                  {inspection.pointCount ? `${inspection.pointCount.toLocaleString()} points` : ''}
                </span>

                {inspection.meshStatus === 'failed' && (
                  <span
                    className="inline-flex items-center text-[10px] text-[#8F543E] bg-[#FBF0EC] px-2 py-0.5 rounded border border-[#EACEC5]"
                    title={inspection.meshError || 'Surface mesh reconstruction failed'}
                  >
                    Mesh reconstruction unavailable
                  </span>
                )}
              </div>
              <div className="flex items-center gap-2">
                {inspection.meshUrl && (
                  <a
                    href={inspection.meshUrl}
                    download={`${inspection.id}_mesh.ply`}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[11px] font-medium bg-[#E7EDE4] text-[#3F513D] hover:bg-[#d5e0d1] border border-[#D5E0D1] transition-colors"
                  >
                    <Download className="w-3.5 h-3.5" />
                    <span>Surface Mesh (.ply)</span>
                  </a>
                )}
                {inspection.pointCloudUrl && (
                  <a
                    href={inspection.pointCloudUrl}
                    download={`${inspection.id}_pointcloud.ply`}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[11px] font-medium bg-[#F0EFE8] text-[#343832] hover:bg-[#e4e1d7] border border-[#E8E5DC] transition-colors"
                  >
                    <Download className="w-3.5 h-3.5" />
                    <span>Point Cloud (.ply)</span>
                  </a>
                )}
              </div>
            </div>
          )}

          {/* Quick Summary Strip */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="bg-white p-3.5 rounded-xl border border-[#E8E5DC] text-xs space-y-1">
              <span className="text-[#777D73] block text-[11px]">Reconstruction method</span>
              <span className="font-medium text-[#343832] block truncate">{inspection.reconstructionMethod}</span>
            </div>

            <div className="bg-white p-3.5 rounded-xl border border-[#E8E5DC] text-xs space-y-1">
              <span className="text-[#777D73] block text-[11px]">Camera baseline</span>
              <span className="font-medium text-[#343832] block font-mono">
                {inspection.calibration?.baselineDistanceMm
                  ? `${inspection.calibration.baselineDistanceMm} mm`
                  : 'Uncalibrated'}
              </span>
            </div>

            <div className="bg-white p-3.5 rounded-xl border border-[#E8E5DC] text-xs space-y-1">
              <span className="text-[#777D73] block text-[11px]">Flagged regions</span>
              <span className="font-medium text-[#343832] block font-mono">
                {inspection.findings.length} total ({crackCount} potential cracks)
              </span>
            </div>

            <div className="bg-white p-3.5 rounded-xl border border-[#E8E5DC] text-xs space-y-1">
              <span className="text-[#777D73] block text-[11px]">Processing duration</span>
              <span className="font-medium text-[#343832] block font-mono">
                {inspection.processingDurationSec ? `${inspection.processingDurationSec}s` : 'Not recorded'}
              </span>
            </div>
          </div>
        </div>

        {/* Findings Panel (Right Column) */}
        <div className="lg:col-span-4 space-y-4">
          <div className="bg-white rounded-xl border border-[#E8E5DC] shadow-[0_1px_3px_rgba(0,0,0,0.02)] overflow-hidden">
            <div className="p-4 border-b border-[#E8E5DC] flex items-center justify-between">
              <div>
                <h2 className="text-xs font-semibold text-[#343832] uppercase tracking-wider">
                  Surface findings
                </h2>
                <p className="text-[11px] text-[#777D73] mt-0.5">
                  {inspection.isDemoMode
                    ? 'Simulated sample findings for interface testing.'
                    : 'Verified processing output from backend pipeline.'}
                </p>
              </div>
              <span className="text-xs font-mono font-medium text-[#777D73] bg-[#F8F7F2] px-2 py-0.5 rounded border border-[#E8E5DC]">
                {inspection.findings.length}
              </span>
            </div>

            {inspection.findings.length > 0 ? (
              <div className="divide-y divide-[#E8E5DC] max-h-[460px] overflow-y-auto">
                {inspection.findings.map((finding) => {
                  const isSelected = finding.id === selectedFindingId;
                  return (
                    <div
                      key={finding.id}
                      onClick={() => setSelectedFindingId(finding.id)}
                      className={`p-4 space-y-2 cursor-pointer transition-colors ${
                        isSelected ? 'bg-[#F0EFE8]/70 border-l-2 border-l-[#9BAE98]' : 'hover:bg-[#F8F7F2]/60'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-xs font-medium text-[#343832]">{finding.id}</span>
                          <FindingBadge type={finding.type} />
                        </div>
                        {isSelected && (
                          <span className="text-[10px] text-[#5B7358] font-medium bg-[#E7EDE4] px-1.5 py-0.5 rounded">
                            Active
                          </span>
                        )}
                      </div>

                      <p className="text-xs text-[#777D73] leading-relaxed">
                        {finding.description}
                      </p>

                      {/* Explicit Image Prediction vs Physical Geometry */}
                      <div className="pt-1 grid grid-cols-2 gap-2 text-[11px] border-t border-[#E8E5DC]/80">
                        <div>
                          <span className="text-[#9BA297] block text-[10px]">Visual confidence:</span>
                          <span className="font-mono text-[#343832] font-medium">
                            {typeof finding.confidence === 'number'
                              ? `${(finding.confidence * 100).toFixed(0)}%`
                              : 'Not evaluated'}
                          </span>
                        </div>
                        <div>
                          <span className="text-[#9BA297] block text-[10px]">Geometric deviation:</span>
                          <span className="font-mono text-[#8F543E] font-medium">
                            {typeof finding.geometricDeviationMm === 'number'
                              ? `${finding.geometricDeviationMm} mm indentation`
                              : finding.deviationUnit === 'relative'
                              ? 'Relative scale'
                              : 'Unmeasured'}
                          </span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="p-8 text-center text-xs text-[#777D73] space-y-1">
                <p className="font-medium text-[#343832]">No structural anomalies detected</p>
                <p className="leading-relaxed text-[11px]">
                  The analysis pipeline did not identify candidate cracks or surface deviations on this geometry.
                </p>
              </div>
            )}
          </div>

          {/* Active Finding Detail Card */}
          {selectedFinding && (
            <div className="bg-white p-4 rounded-xl border border-[#E8E5DC] shadow-[0_1px_3px_rgba(0,0,0,0.02)] space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-semibold text-[#343832] uppercase tracking-wider">
                  Detailed observation • {selectedFinding.id}
                </h3>
              </div>

              <div className="p-3 rounded-lg bg-[#F8F7F2] border border-[#E8E5DC] text-xs space-y-2">
                <div>
                  <span className="text-[#777D73] block text-[10px]">Classification Note</span>
                  <p className="text-[#343832] leading-relaxed mt-0.5">
                    {selectedFinding.description}
                  </p>
                </div>

                <div className="pt-2 border-t border-[#E8E5DC] text-[11px] text-[#777D73] leading-relaxed">
                  <strong className="text-[#343832]">Scientific note:</strong> Visual classifier confidence reflects image pattern correlation, whereas physical depth displacement{' '}
                  {typeof selectedFinding.geometricDeviationMm === 'number'
                    ? `(${selectedFinding.geometricDeviationMm} mm)`
                    : '(unmeasured)'}{' '}
                  represents verified 3D point cloud surface indentation.
                </div>
              </div>
            </div>
          )}

          {/* Camera Calibration Summary */}
          <div className="bg-white p-4 rounded-xl border border-[#E8E5DC] shadow-[0_1px_3px_rgba(0,0,0,0.02)] space-y-2 text-xs">
            <h3 className="font-semibold text-[#343832] text-xs uppercase tracking-wider">
              Camera calibration
            </h3>
            <div className="space-y-1 text-[#777D73]">
              <div className="flex justify-between">
                <span>File</span>
                <span className="font-mono text-[#343832]">
                  {inspection.calibration?.fileName || 'Not provided'}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Status</span>
                <span className="font-medium text-[#3F513D]">
                  {inspection.calibration?.status === 'valid'
                    ? 'Calibrated geometry'
                    : inspection.calibration?.status === 'partial'
                    ? 'Partially calibrated'
                    : 'Uncalibrated (relative depth)'}
                </span>
              </div>
              {inspection.calibration?.notes && (
                <p className="pt-1 text-[11px] text-[#9BA297] border-t border-[#E8E5DC]">
                  {inspection.calibration.notes}
                </p>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
