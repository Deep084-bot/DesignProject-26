import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import {
  ArrowLeft,
  Box,
  Calendar,
  Server,
  PlayCircle,
  Download,
  AlertTriangle,
  Image as ImageIcon,
  Activity,
} from 'lucide-react';
import { fetchInspectionById } from '../services/api';
import type { Inspection } from '../types/inspection';
import { ThreeSurfaceViewer } from '../components/viewer/ThreeSurfaceViewer';
import { SurfaceImageOverlay } from '../components/viewer/SurfaceImageOverlay';
import { StatusBadge, FindingBadge } from '../components/common/Badge';
import { Button } from '../components/common/Button';

export const InspectionDetailPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [inspection, setInspection] = useState<Inspection | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedFindingId, setSelectedFindingId] = useState<string | null>(null);
  const [activeViewMode, setActiveViewMode] = useState<'3d' | '2d'>('3d');
  const [cropLoadErrors, setCropLoadErrors] = useState<Record<string, boolean>>({});

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
  const irregularityCount = inspection.findings.filter((f) => f.type === 'surface_irregularity').length;

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

  const hasClassifierIssue =
    inspection.classifierStatus === 'unavailable' || inspection.classifierStatus === 'failed';

  const rectifiedUrl = inspection.rectifiedLeftUrl || inspection.leftImage?.url;

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

      {/* Classifier Status Notice Banner (when unavailable or failed) */}
      {hasClassifierIssue && (
        <div className="p-4 rounded-xl bg-[#FBF0EC] border border-[#EACEC5] text-xs space-y-1">
          <div className="flex items-center gap-2 text-[#8F543E] font-medium">
            <AlertTriangle className="w-4 h-4 flex-shrink-0" />
            <span>
              Neural-network crack classifier {inspection.classifierStatus === 'unavailable' ? 'unavailable' : 'failed'}
              {inspection.classifierError ? `: ${inspection.classifierError}` : ''}
            </span>
          </div>
          <p className="text-[#8F543E]/90 pl-6 text-[11px] leading-relaxed">
            Visual pattern crack classification was not evaluated for this inspection. Geometric 3D surface reconstruction and depth deviation measurements remain valid and are displayed below.
          </p>
        </div>
      )}

      {/* Main View Mode Selector Tabs */}
      <div className="flex items-center gap-2 border-b border-[#E8E5DC] pb-3 text-xs">
        <button
          type="button"
          onClick={() => setActiveViewMode('3d')}
          className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-medium transition-colors ${
            activeViewMode === '3d'
              ? 'bg-[#E7EDE4] text-[#3F513D] border border-[#D5E0D1]'
              : 'text-[#777D73] hover:text-[#343832] hover:bg-[#F0EFE8]'
          }`}
        >
          <Box className="w-3.5 h-3.5" />
          <span>3D Surface Model</span>
        </button>

        <button
          type="button"
          onClick={() => setActiveViewMode('2d')}
          className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-medium transition-colors ${
            activeViewMode === '2d'
              ? 'bg-[#E7EDE4] text-[#3F513D] border border-[#D5E0D1]'
              : 'text-[#777D73] hover:text-[#343832] hover:bg-[#F0EFE8]'
          }`}
        >
          <ImageIcon className="w-3.5 h-3.5" />
          <span>2D Rectified View & Heatmap</span>
        </button>
      </div>

      {/* Main Visualizer & Findings Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column (Primary Visual View) */}
        <div className="lg:col-span-8 space-y-4">
          {activeViewMode === '3d' ? (
            <ThreeSurfaceViewer
              plyUrl={inspection.pointCloudUrl}
              meshUrl={inspection.meshUrl}
              meshStatus={inspection.meshStatus}
              meshError={inspection.meshError}
              findings={inspection.findings}
              selectedFindingId={selectedFindingId}
              isDemoModel={inspection.isDemoMode}
            />
          ) : (
            <SurfaceImageOverlay
              imageUrl={rectifiedUrl || ''}
              heatmapUrl={inspection.deviationHeatmapUrl}
              findings={inspection.findings}
              selectedFindingId={selectedFindingId}
              onSelectFinding={(fId) => setSelectedFindingId(fId)}
              deviationSummary={inspection.deviationSummary}
            />
          )}

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
              <span className="text-[#777D73] block text-[11px]">Flagged findings</span>
              <span className="font-medium text-[#343832] block font-mono">
                {inspection.findings.length} total ({crackCount} potential, {irregularityCount} geometric)
              </span>
            </div>

            <div className="bg-white p-3.5 rounded-xl border border-[#E8E5DC] text-xs space-y-1">
              <span className="text-[#777D73] block text-[11px]">Processing duration</span>
              <span className="font-medium text-[#343832] block font-mono">
                {typeof inspection.processingDurationSec === 'number'
                  ? `${inspection.processingDurationSec.toFixed(2)}s`
                  : 'Not recorded'}
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
                    : 'Correlated visual and geometric findings.'}
                </p>
              </div>
              <span className="text-xs font-mono font-medium text-[#777D73] bg-[#F8F7F2] px-2 py-0.5 rounded border border-[#E8E5DC]">
                {inspection.findings.length}
              </span>
            </div>

            {inspection.findings.length > 0 ? (
              <div className="divide-y divide-[#E8E5DC] max-h-[520px] overflow-y-auto">
                {inspection.findings.map((finding) => {
                  const isSelected = finding.id === selectedFindingId;
                  const hasCropError = cropLoadErrors[finding.id];

                  return (
                    <div
                      key={finding.id}
                      onClick={() => setSelectedFindingId(finding.id)}
                      className={`p-4 space-y-2.5 cursor-pointer transition-colors ${
                        isSelected ? 'bg-[#F0EFE8]/70 border-l-2 border-l-[#5B7358]' : 'hover:bg-[#F8F7F2]/60'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-xs font-semibold text-[#343832]">{finding.id}</span>
                          <FindingBadge type={finding.type} />
                        </div>
                        {isSelected && (
                          <span className="text-[10px] text-[#3F513D] font-medium bg-[#E7EDE4] px-1.5 py-0.5 rounded border border-[#D5E0D1]">
                            Active
                          </span>
                        )}
                      </div>

                      {/* Finding Card Body: Thumbnail & Details */}
                      <div className="flex items-start gap-3">
                        {finding.imageCropUrl && !hasCropError ? (
                          <img
                            src={finding.imageCropUrl}
                            alt={`${finding.id} crop`}
                            onError={() =>
                              setCropLoadErrors((prev) => ({ ...prev, [finding.id]: true }))
                            }
                            className="w-14 h-14 rounded-lg object-cover border border-[#E8E5DC] flex-shrink-0 bg-[#F0EFE8]"
                          />
                        ) : (
                          <div className="w-14 h-14 rounded-lg border border-[#E8E5DC] bg-[#F8F7F2] flex items-center justify-center text-[#9BA297] flex-shrink-0">
                            <Activity className="w-5 h-5 text-[#9BA297]" />
                          </div>
                        )}

                        <p className="text-xs text-[#777D73] leading-relaxed flex-1">
                          {finding.description}
                        </p>
                      </div>

                      {/* Metric Strips: Confidence & Deviation */}
                      <div className="pt-1.5 grid grid-cols-2 gap-2 text-[11px] border-t border-[#E8E5DC]/80">
                        <div>
                          <span className="text-[#9BA297] block text-[10px]">Visual confidence:</span>
                          <span className="font-mono text-[#343832] font-medium">
                            {typeof finding.confidence === 'number'
                              ? `${(finding.confidence * 100).toFixed(0)}%`
                              : 'n/a'}
                          </span>
                        </div>
                        <div>
                          <span className="text-[#9BA297] block text-[10px]">Geometric deviation:</span>
                          <span className="font-mono text-[#8F543E] font-medium">
                            {typeof finding.geometricDeviationMm === 'number'
                              ? `${finding.geometricDeviationMm} mm`
                              : 'n/a'}
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
                <FindingBadge type={selectedFinding.type} />
              </div>

              <div className="p-3 rounded-lg bg-[#F8F7F2] border border-[#E8E5DC] text-xs space-y-2">
                <div>
                  <span className="text-[#777D73] block text-[10px]">Classification Note</span>
                  <p className="text-[#343832] leading-relaxed mt-0.5">
                    {selectedFinding.description}
                  </p>
                </div>

                <div className="pt-2 border-t border-[#E8E5DC] text-[11px] text-[#777D73] leading-relaxed">
                  <strong className="text-[#343832]">Engineering note:</strong> Potential crack classifications indicate high visual classifier probability ({typeof selectedFinding.confidence === 'number' ? `${(selectedFinding.confidence * 100).toFixed(0)}%` : 'n/a'}), whereas measured geometric deviation ({typeof selectedFinding.geometricDeviationMm === 'number' ? `${selectedFinding.geometricDeviationMm} mm` : 'n/a'}) reflects physical surface indentation.
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
