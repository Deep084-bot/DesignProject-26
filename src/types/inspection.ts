/**
 * Data types for Surface — 3D Structural Surface Inspection
 *
 * Clean, user-facing models representing physical surface inspections,
 * stereo image inputs, calibration parameters, and measured structural findings.
 */

export type InspectionStatus = 'completed' | 'processing' | 'failed' | 'queued';

export type FindingType = 'potential_crack' | 'surface_irregularity' | 'minor_anomaly';

export type ExecutionMode = 'backend' | 'demo';

export interface Finding {
  id: string; // e.g. "Finding 01"
  type: FindingType;
  confidence: number | null; // 0.00 to 1.00 (visual classifier confidence, or null if unclassified)
  geometricDeviationMm: number | null; // Physical 3D measurement in mm (or null if uncalibrated)
  deviationUnit: 'mm' | 'm' | 'relative' | 'uncalibrated';
  description: string;
  imageCropUrl?: string;
  boundingBox2D?: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  regionCenter3D?: [number, number, number];
}

export interface CalibrationSummary {
  fileName: string;
  baselineDistanceMm?: number;
  focalLengthPx?: number;
  imageResolution?: string;
  status: 'valid' | 'partial' | 'invalid' | 'unverified';
  notes?: string;
}

export interface PipelineStep {
  id: string;
  label: string;
  description: string;
  status: 'waiting' | 'active' | 'completed' | 'failed';
}

export interface ImageFileMeta {
  name: string;
  url: string;
  sizeBytes?: number;
  dimensions?: string; // e.g. "3840 × 2160 px"
}

export interface Inspection {
  id: string; // e.g. "INSP-2026-0042"
  name: string;
  surfaceType: string; // e.g. "Concrete retaining wall", "Bridge pier foundation"
  createdAt: string; // ISO timestamp
  status: InspectionStatus;
  currentStepIndex: number;
  steps: PipelineStep[];
  leftImage: ImageFileMeta;
  rightImage: ImageFileMeta;
  calibration: CalibrationSummary;
  reconstructionMethod: string; // e.g. "Automatic surface reconstruction"
  processingDurationSec?: number;
  pointCloudUrl?: string;
  pointCount?: number;
  meshUrl?: string;
  triangleCount?: number;
  meshStatus?: 'completed' | 'failed' | 'unavailable';
  meshError?: string;
  findings: Finding[];
  isDemoMode: boolean; // Transparent disclosure when offline or displaying sample inspections
  failureMessage?: string;
}

export interface NewInspectionInput {
  name: string;
  surfaceType: string;
  reconstructionMethod: string;
  leftImageFile: File | null;
  rightImageFile: File | null;
  calibrationFile: File | null;
}
