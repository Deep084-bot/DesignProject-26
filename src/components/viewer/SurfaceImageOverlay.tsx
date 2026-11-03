import React, { useState, useRef } from 'react';
import { Layers, Eye, EyeOff, Sliders, Info, Image as ImageIcon } from 'lucide-react';
import type { Finding, DeviationSummary } from '../../types/inspection';

interface SurfaceImageOverlayProps {
  imageUrl: string;
  heatmapUrl?: string | null;
  findings: Finding[];
  selectedFindingId: string | null;
  onSelectFinding: (id: string) => void;
  deviationSummary?: DeviationSummary | null;
}

export const SurfaceImageOverlay: React.FC<SurfaceImageOverlayProps> = ({
  imageUrl,
  heatmapUrl,
  findings,
  selectedFindingId,
  onSelectFinding,
  deviationSummary,
}) => {
  const [showHeatmap, setShowHeatmap] = useState<boolean>(false);
  const [heatmapOpacity, setHeatmapOpacity] = useState<number>(65);
  const [naturalDimensions, setNaturalDimensions] = useState<{ width: number; height: number } | null>(null);
  const [imageError, setImageError] = useState<boolean>(false);
  const [heatmapError, setHeatmapError] = useState<boolean>(false);

  const containerRef = useRef<HTMLDivElement>(null);

  const handleImageLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    const img = e.currentTarget;
    if (img.naturalWidth && img.naturalHeight) {
      setNaturalDimensions({
        width: img.naturalWidth,
        height: img.naturalHeight,
      });
    }
  };

  const getFindingColor = (type: Finding['type'], isSelected: boolean) => {
    switch (type) {
      case 'potential_crack':
        return {
          border: isSelected ? 'border-[#C94A29]' : 'border-[#C94A29]/80',
          bg: isSelected ? 'bg-[#C94A29]/20' : 'bg-[#C94A29]/10',
          badge: 'bg-[#C94A29] text-white',
          text: 'text-[#C94A29]',
        };
      case 'surface_irregularity':
        return {
          border: isSelected ? 'border-[#2B6CB0]' : 'border-[#2B6CB0]/80',
          bg: isSelected ? 'bg-[#2B6CB0]/20' : 'bg-[#2B6CB0]/10',
          badge: 'bg-[#2B6CB0] text-white',
          text: 'text-[#2B6CB0]',
        };
      case 'minor_anomaly':
      default:
        return {
          border: isSelected ? 'border-[#7C3AED]' : 'border-[#7C3AED]/80',
          bg: isSelected ? 'bg-[#7C3AED]/20' : 'bg-[#7C3AED]/10',
          badge: 'bg-[#7C3AED] text-white',
          text: 'text-[#7C3AED]',
        };
    }
  };

  return (
    <div className="space-y-4">
      {/* Control bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 p-3 rounded-xl bg-white border border-[#E8E5DC] text-xs shadow-[0_1px_3px_rgba(0,0,0,0.02)]">
        <div className="flex items-center gap-2">
          <span className="font-semibold text-[#343832] uppercase tracking-wider text-[11px] flex items-center gap-1.5">
            <Layers className="w-3.5 h-3.5 text-[#777D73]" />
            <span>Surface Layers:</span>
          </span>

          {heatmapUrl ? (
            <button
              type="button"
              onClick={() => setShowHeatmap(!showHeatmap)}
              className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[11px] font-medium transition-colors border ${
                showHeatmap
                  ? 'bg-[#E7EDE4] text-[#3F513D] border-[#D5E0D1]'
                  : 'bg-[#F0EFE8] text-[#777D73] border-[#E8E5DC] hover:text-[#343832]'
              }`}
            >
              {showHeatmap ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
              <span>{showHeatmap ? 'Heatmap active' : 'Overlay heatmap'}</span>
            </button>
          ) : (
            <span className="text-[11px] text-[#9BA297]">No heatmap artifact</span>
          )}
        </div>

        {/* Heatmap opacity slider */}
        {showHeatmap && heatmapUrl && (
          <div className="flex items-center gap-2.5 bg-[#F8F7F2] px-3 py-1 rounded-lg border border-[#E8E5DC]">
            <Sliders className="w-3 h-3 text-[#777D73]" />
            <span className="text-[11px] text-[#777D73]">Opacity:</span>
            <input
              type="range"
              min="10"
              max="100"
              value={heatmapOpacity}
              onChange={(e) => setHeatmapOpacity(Number(e.target.value))}
              className="w-24 h-1.5 bg-[#E8E5DC] rounded-lg appearance-none cursor-pointer accent-[#5B7358]"
            />
            <span className="font-mono text-[11px] text-[#343832] w-7 text-right">{heatmapOpacity}%</span>
          </div>
        )}
      </div>

      {/* Interactive image viewport */}
      <div
        ref={containerRef}
        className="relative w-full rounded-xl overflow-hidden border border-[#E8E5DC] bg-[#1E201D] flex items-center justify-center select-none shadow-[0_1px_3px_rgba(0,0,0,0.02)] min-h-[440px] max-h-[640px]"
      >
        {imageError ? (
          <div className="p-8 text-center text-xs text-[#9BA297] space-y-2">
            <ImageIcon className="w-8 h-8 mx-auto text-[#777D73]" />
            <p className="text-white font-medium">Rectified image artifact unavailable</p>
            <p className="text-[11px] text-[#9BA297]">
              The rectified left perspective could not be retrieved from the server.
            </p>
          </div>
        ) : (
          <div className="relative max-w-full max-h-full inline-block">
            {/* Base Rectified Left Image */}
            <img
              src={imageUrl}
              alt="Rectified left view"
              onLoad={handleImageLoad}
              onError={() => setImageError(true)}
              className="block max-w-full max-h-[620px] object-contain mx-auto"
            />

            {/* Heatmap Overlay */}
            {showHeatmap && heatmapUrl && !heatmapError && (
              <img
                src={heatmapUrl}
                alt="Deviation heatmap overlay"
                onError={() => setHeatmapError(true)}
                style={{ opacity: heatmapOpacity / 100 }}
                className="absolute inset-0 w-full h-full object-contain pointer-events-none transition-opacity duration-150"
              />
            )}

            {/* Scaled Bounding Box Overlays */}
            {naturalDimensions && (
              <div className="absolute inset-0 pointer-events-none">
                {findings.map((finding) => {
                  if (!finding.boundingBox2D) return null;
                  const { x, y, width, height } = finding.boundingBox2D;
                  const isSelected = finding.id === selectedFindingId;
                  const colors = getFindingColor(finding.type, isSelected);

                  const leftPct = (x / naturalDimensions.width) * 100;
                  const topPct = (y / naturalDimensions.height) * 100;
                  const widthPct = (width / naturalDimensions.width) * 100;
                  const heightPct = (height / naturalDimensions.height) * 100;

                  return (
                    <div
                      key={finding.id}
                      onClick={(e) => {
                        e.stopPropagation();
                        onSelectFinding(finding.id);
                      }}
                      style={{
                        left: `${leftPct}%`,
                        top: `${topPct}%`,
                        width: `${widthPct}%`,
                        height: `${heightPct}%`,
                      }}
                      className={`absolute pointer-events-auto cursor-pointer transition-all ${
                        isSelected
                          ? `border-2 ${colors.border} ${colors.bg} ring-2 ring-white/90 z-20`
                          : `border ${colors.border} ${colors.bg} hover:border-2 hover:z-10`
                      }`}
                    >
                      {/* Finding ID Tag */}
                      <span
                        className={`absolute -top-5 left-0 px-1.5 py-0.2 rounded text-[10px] font-mono font-medium whitespace-nowrap shadow-xs ${
                          isSelected ? `${colors.badge} ring-1 ring-white/50` : 'bg-black/80 text-white'
                        }`}
                      >
                        {finding.id}
                      </span>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Heatmap color key legend */}
      {showHeatmap && (
        <div className="flex flex-wrap items-center justify-between gap-3 p-3 rounded-xl bg-[#F8F7F2] border border-[#E8E5DC] text-[11px]">
          <div className="flex items-center gap-3">
            <span className="font-semibold text-[#343832]">Heatmap convention:</span>
            <span className="flex items-center gap-1.5">
              <span className="w-3 h-3 rounded-full bg-[#3B82F6]" />
              <span className="text-[#777D73]">Bulge (protrusion toward camera)</span>
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-3 h-3 rounded-full bg-[#EF4444]" />
              <span className="text-[#777D73]">Depression (indentation away from camera)</span>
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-3 h-3 rounded-full bg-black/60" />
              <span className="text-[#777D73]">Invalid / plane inliers</span>
            </span>
          </div>
          <span className="text-[#9BA297] font-mono">Calibrated reference plane</span>
        </div>
      )}

      {/* Compact Deviation Summary Table */}
      {deviationSummary && (
        <div className="bg-white p-4 rounded-xl border border-[#E8E5DC] shadow-[0_1px_3px_rgba(0,0,0,0.02)] space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-semibold text-[#343832] uppercase tracking-wider flex items-center gap-1.5">
              <Info className="w-3.5 h-3.5 text-[#5B7358]" />
              <span>Surface Deviation Summary (Stage 3)</span>
            </h3>
            {typeof deviationSummary.regionCount === 'number' && (
              <span className="text-[11px] font-mono text-[#777D73] bg-[#F8F7F2] px-2 py-0.5 rounded border border-[#E8E5DC]">
                {deviationSummary.regionCount} {deviationSummary.regionCount === 1 ? 'region' : 'regions'} detected
              </span>
            )}
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
            {typeof deviationSummary.rmseMm === 'number' && (
              <div className="p-2.5 rounded-lg bg-[#F8F7F2] border border-[#E8E5DC]/80 space-y-0.5">
                <span className="text-[10px] text-[#777D73] block uppercase tracking-wider">Surface RMSE</span>
                <span className="font-mono font-medium text-[#343832] text-sm">{deviationSummary.rmseMm} mm</span>
              </div>
            )}

            {typeof deviationSummary.sigmaMm === 'number' && (
              <div className="p-2.5 rounded-lg bg-[#F8F7F2] border border-[#E8E5DC]/80 space-y-0.5">
                <span className="text-[10px] text-[#777D73] block uppercase tracking-wider">Noise Sigma (MAD)</span>
                <span className="font-mono font-medium text-[#343832] text-sm">{deviationSummary.sigmaMm} mm</span>
              </div>
            )}

            {typeof deviationSummary.thresholdMm === 'number' && (
              <div className="p-2.5 rounded-lg bg-[#F8F7F2] border border-[#E8E5DC]/80 space-y-0.5">
                <span className="text-[10px] text-[#777D73] block uppercase tracking-wider">Cutoff Threshold</span>
                <span className="font-mono font-medium text-[#343832] text-sm">{deviationSummary.thresholdMm} mm</span>
              </div>
            )}

            {typeof deviationSummary.maxDepressionMm === 'number' && (
              <div className="p-2.5 rounded-lg bg-[#F8F7F2] border border-[#E8E5DC]/80 space-y-0.5">
                <span className="text-[10px] text-[#777D73] block uppercase tracking-wider">Max Depression</span>
                <span className="font-mono font-medium text-[#C94A29] text-sm">
                  {deviationSummary.maxDepressionMm} mm
                </span>
              </div>
            )}

            {typeof deviationSummary.maxBulgeMm === 'number' && (
              <div className="p-2.5 rounded-lg bg-[#F8F7F2] border border-[#E8E5DC]/80 space-y-0.5">
                <span className="text-[10px] text-[#777D73] block uppercase tracking-wider">Max Bulge</span>
                <span className="font-mono font-medium text-[#2B6CB0] text-sm">{deviationSummary.maxBulgeMm} mm</span>
              </div>
            )}

            {typeof deviationSummary.inlierFraction === 'number' && (
              <div className="p-2.5 rounded-lg bg-[#F8F7F2] border border-[#E8E5DC]/80 space-y-0.5">
                <span className="text-[10px] text-[#777D73] block uppercase tracking-wider">Plane Inlier Ratio</span>
                <span className="font-mono font-medium text-[#343832] text-sm">
                  {(deviationSummary.inlierFraction * 100).toFixed(1)}%
                </span>
              </div>
            )}

            {typeof deviationSummary.validPixelFraction === 'number' && (
              <div className="p-2.5 rounded-lg bg-[#F8F7F2] border border-[#E8E5DC]/80 space-y-0.5">
                <span className="text-[10px] text-[#777D73] block uppercase tracking-wider">Valid Depth Area</span>
                <span className="font-mono font-medium text-[#343832] text-sm">
                  {(deviationSummary.validPixelFraction * 100).toFixed(1)}%
                </span>
              </div>
            )}

            {deviationSummary.planeNormal && (
              <div className="p-2.5 rounded-lg bg-[#F8F7F2] border border-[#E8E5DC]/80 space-y-0.5">
                <span className="text-[10px] text-[#777D73] block uppercase tracking-wider">Fitted Plane Normal</span>
                <span className="font-mono text-[#343832] text-[11px] block truncate">
                  [{deviationSummary.planeNormal.map((n) => n.toFixed(2)).join(', ')}]
                </span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
