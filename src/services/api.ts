/**
 * API Service for Surface — 3D Structural Surface Inspection
 *
 * Manages communication with the real FastAPI photogrammetry backend when in 'backend' mode,
 * and maintains transparent isolation when in 'demo' mode.
 *
 * Real Processing Mode:
 *   - Submits multipart form data (left_image, right_image, optional calibration_file, metadata)
 *     to POST /api/inspections.
 *   - Polls GET /api/inspections/{id} for true pipeline status updates.
 *   - NEVER simulates progress timers or fabricates synthetic findings.
 *   - Propagates authentic backend error messages and pipeline failures.
 *
 * Demonstration Mode:
 *   - Uses local browser session storage with clearly disclosed sample data and simulated progression.
 */

import type { Inspection, NewInspectionInput, ExecutionMode, PipelineStep, Finding } from '../types/inspection';
import { MOCK_INSPECTIONS, DEFAULT_PIPELINE_STEPS } from './mockData';

// Configured base URL from Vite environment (e.g. "http://localhost:8000" or empty)
const ENV_API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '');

const STORAGE_KEY_MODE = 'surface_execution_mode';
const SESSION_STORAGE_KEY = 'surface_inspections_session';

/**
 * Returns whether a real backend API URL is provided in the environment.
 */
export function isBackendConfigured(): boolean {
  return Boolean(ENV_API_BASE_URL);
}

/**
 * Returns the currently configured backend base URL (if any).
 */
export function getBackendBaseUrl(): string {
  return ENV_API_BASE_URL;
}

/**
 * Returns the current execution mode: 'backend' or 'demo'.
 * Priority:
 * 1. User's explicit runtime selection stored in localStorage ('backend' | 'demo')
 * 2. Environment default: 'backend' if VITE_API_BASE_URL is set, otherwise 'demo'
 */
export function getExecutionMode(): ExecutionMode {
  try {
    const saved = localStorage.getItem(STORAGE_KEY_MODE);
    if (saved === 'backend' || saved === 'demo') {
      return saved;
    }
  } catch {
    // LocalStorage access may be restricted
  }
  return isBackendConfigured() ? 'backend' : 'demo';
}

/**
 * Sets the execution mode explicitly ('backend' or 'demo').
 */
export function setExecutionMode(mode: ExecutionMode): void {
  try {
    localStorage.setItem(STORAGE_KEY_MODE, mode);
  } catch {
    // Ignore storage errors
  }
  window.dispatchEvent(new CustomEvent('surface-mode-changed', { detail: mode }));
}

/**
 * Resolves artifact URLs (such as .ply point clouds or cropped inspection images).
 * If the URL is relative (e.g., starts with "/api/"), prefixes it with the real backend base URL.
 */
export function resolveArtifactUrl(url?: string): string | undefined {
  if (!url) return undefined;
  if (
    url.startsWith('http://') ||
    url.startsWith('https://') ||
    url.startsWith('blob:') ||
    url.startsWith('data:')
  ) {
    return url;
  }
  if (url.startsWith('/samples/')) {
    return url;
  }
  if (ENV_API_BASE_URL && url.startsWith('/')) {
    return `${ENV_API_BASE_URL}${url}`;
  }
  return url;
}

// ---------------------------------------------------------------------------
// Normalization Helpers (Supports both snake_case from FastAPI and camelCase)
// ---------------------------------------------------------------------------

function normalizeFinding(f: any): Finding {
  return {
    id: f.id || 'Finding',
    type: f.type || 'surface_irregularity',
    confidence: typeof f.confidence === 'number' ? f.confidence : null,
    geometricDeviationMm:
      typeof f.geometricDeviationMm === 'number'
        ? f.geometricDeviationMm
        : typeof f.geometric_deviation_mm === 'number'
        ? f.geometric_deviation_mm
        : null,
    deviationUnit:
      f.deviationUnit ||
      f.deviation_unit ||
      (f.geometric_deviation_mm !== undefined || f.geometricDeviationMm !== undefined
        ? 'mm'
        : 'uncalibrated'),
    description: f.description || 'Structural surface observation.',
    imageCropUrl: resolveArtifactUrl(f.imageCropUrl || f.image_crop_url),
    boundingBox2D: f.boundingBox2D || f.bounding_box_2d,
    regionCenter3D: f.regionCenter3D || f.region_center_3d,
  };
}

export function normalizeInspection(data: any, forceDemoFlag?: boolean): Inspection {
  const currentStep =
    typeof data.current_step_index === 'number'
      ? data.current_step_index
      : typeof data.currentStepIndex === 'number'
      ? data.currentStepIndex
      : 0;
  const status = data.status || 'processing';

  // Build steps if missing from backend
  let steps: PipelineStep[] = [];
  if (Array.isArray(data.steps) && data.steps.length > 0) {
    steps = data.steps.map((s: any) => ({
      id: s.id,
      label: s.label,
      description: s.description || '',
      status: s.status || 'waiting',
    }));
  } else {
    steps = DEFAULT_PIPELINE_STEPS.map((s, idx) => {
      let stepStatus: 'waiting' | 'active' | 'completed' | 'failed' = 'waiting';
      if (status === 'completed') {
        stepStatus = 'completed';
      } else if (status === 'failed') {
        if (idx < currentStep) stepStatus = 'completed';
        else if (idx === currentStep) stepStatus = 'failed';
        else stepStatus = 'waiting';
      } else {
        if (idx < currentStep) stepStatus = 'completed';
        else if (idx === currentStep) stepStatus = 'active';
        else stepStatus = 'waiting';
      }
      return { ...s, status: stepStatus };
    });
  }

  const rawFindings = Array.isArray(data.findings) ? data.findings : [];
  const findings: Finding[] = rawFindings.map(normalizeFinding);

  const isDemo =
    forceDemoFlag !== undefined
      ? forceDemoFlag
      : data.isDemoMode !== undefined
      ? Boolean(data.isDemoMode)
      : data.is_demo_mode !== undefined
      ? Boolean(data.is_demo_mode)
      : false;

  return {
    id: data.id || `INSP-${Date.now()}`,
    name: data.name || 'Structural surface inspection',
    surfaceType: data.surfaceType || data.surface_type || 'Concrete surface',
    createdAt: data.createdAt || data.created_at || new Date().toISOString(),
    status,
    currentStepIndex: currentStep,
    steps,
    leftImage: {
      name: data.leftImage?.name || data.left_image?.name || 'left_camera_view.jpg',
      url: resolveArtifactUrl(data.leftImage?.url || data.left_image?.url || '/samples/sample_left.jpg') || '',
      dimensions: data.leftImage?.dimensions || data.left_image?.dimensions,
      sizeBytes: data.leftImage?.sizeBytes || data.left_image?.size_bytes,
    },
    rightImage: {
      name: data.rightImage?.name || data.right_image?.name || 'right_camera_view.jpg',
      url: resolveArtifactUrl(data.rightImage?.url || data.right_image?.url || '/samples/sample_right.jpg') || '',
      dimensions: data.rightImage?.dimensions || data.right_image?.dimensions,
      sizeBytes: data.rightImage?.sizeBytes || data.right_image?.size_bytes,
    },
    calibration: {
      fileName: data.calibration?.fileName || data.calibration?.file_name || 'camera_calibration.json',
      baselineDistanceMm: data.calibration?.baselineDistanceMm ?? data.calibration?.baseline_distance_mm,
      focalLengthPx: data.calibration?.focalLengthPx ?? data.calibration?.focal_length_px,
      imageResolution: data.calibration?.imageResolution ?? data.calibration?.image_resolution,
      status: data.calibration?.status || 'unverified',
      notes: data.calibration?.notes,
    },
    reconstructionMethod:
      data.reconstructionMethod || data.reconstruction_method || 'Automatic surface reconstruction',
    processingDurationSec: data.processingDurationSec ?? data.processing_duration_sec,
    pointCloudUrl: resolveArtifactUrl(data.pointCloudUrl || data.point_cloud_url),
    pointCount: data.pointCount ?? data.point_count,
    meshUrl: resolveArtifactUrl(data.meshUrl || data.mesh_url),
    triangleCount: data.triangleCount ?? data.triangle_count,
    meshStatus: data.meshStatus || data.mesh_status || (data.meshUrl || data.mesh_url ? 'completed' : undefined),
    meshError: data.meshError || data.mesh_error,
    findings,
    isDemoMode: isDemo,
    failureMessage: data.failureMessage || data.failure_message,
  };
}

// ---------------------------------------------------------------------------
// Demo Session Storage
// ---------------------------------------------------------------------------

function getStoredDemoInspections(): Inspection[] {
  try {
    const raw = sessionStorage.getItem(SESSION_STORAGE_KEY);
    if (raw) {
      return JSON.parse(raw);
    }
  } catch (e) {
    console.warn('Failed to parse session inspections:', e);
  }
  return [...MOCK_INSPECTIONS];
}

function saveStoredDemoInspections(inspections: Inspection[]) {
  try {
    sessionStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(inspections));
  } catch (e) {
    console.warn('Failed to save session inspections:', e);
  }
}

// ---------------------------------------------------------------------------
// API Methods
// ---------------------------------------------------------------------------

/**
 * Fetches the list of all inspections.
 * In 'backend' mode: queries GET /api/inspections. Throws on network/server error.
 * In 'demo' mode: queries local browser session storage.
 */
export async function fetchInspections(): Promise<Inspection[]> {
  const mode = getExecutionMode();

  if (mode === 'backend') {
    if (!ENV_API_BASE_URL) {
      throw new Error(
        'Backend URL is not configured. Please set VITE_API_BASE_URL in your environment or switch to Demo Mode.'
      );
    }

    const res = await fetch(`${ENV_API_BASE_URL}/api/inspections`);
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}: ${res.statusText}`);
    }
    const data = await res.json();
    if (!Array.isArray(data)) {
      throw new Error('Invalid backend response: expected an array of inspections.');
    }
    return data.map((item) => normalizeInspection(item, false));
  }

  // Demo mode
  return getStoredDemoInspections();
}

/**
 * Fetches a single inspection by its ID.
 * In 'backend' mode: queries GET /api/inspections/{id}.
 * In 'demo' mode: searches local browser session storage.
 */
export async function fetchInspectionById(id: string): Promise<Inspection | null> {
  const mode = getExecutionMode();

  if (mode === 'backend') {
    if (!ENV_API_BASE_URL) {
      throw new Error('Backend URL is not configured.');
    }

    const res = await fetch(`${ENV_API_BASE_URL}/api/inspections/${id}`);
    if (res.status === 404) {
      return null;
    }
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}: ${res.statusText}`);
    }
    const data = await res.json();
    return normalizeInspection(data, false);
  }

  // Demo mode
  const all = getStoredDemoInspections();
  return all.find((item) => item.id === id) || null;
}

/**
 * Creates a new inspection from the upload form.
 *
 * In Real Mode:
 *   Submits multipart form data to POST /api/inspections.
 *   Throws immediately if the backend is unreachable or returns an error.
 *   Never generates fake results or simulated progress.
 *
 * In Demo Mode:
 *   Creates a session-stored demo inspection and schedules local demo progress.
 */
export async function createInspection(
  input: NewInspectionInput,
  parsedCalibration?: {
    baselineDistanceMm?: number;
    focalLengthPx?: number;
    notes?: string;
  }
): Promise<Inspection> {
  const mode = getExecutionMode();

  if (mode === 'backend') {
    if (!ENV_API_BASE_URL) {
      throw new Error(
        'Cannot submit to backend: VITE_API_BASE_URL is not set. Please configure the backend endpoint or switch to Demo mode.'
      );
    }

    if (!input.leftImageFile || !input.rightImageFile) {
      throw new Error('Both left and right stereo images are required for 3D reconstruction.');
    }

    const formData = new FormData();
    formData.append('name', input.name.trim() || 'Structural surface inspection');
    formData.append('surface_type', input.surfaceType);
    formData.append('reconstruction_method', input.reconstructionMethod);
    formData.append('left_image', input.leftImageFile);
    formData.append('right_image', input.rightImageFile);
    if (input.calibrationFile) {
      formData.append('calibration_file', input.calibrationFile);
    }

    const res = await fetch(`${ENV_API_BASE_URL}/api/inspections`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      let detail = '';
      try {
        const errJson = await res.json();
        detail = errJson.detail || errJson.message || JSON.stringify(errJson);
      } catch {
        detail = await res.text();
      }
      throw new Error(`Backend error (${res.status} ${res.statusText}): ${detail || 'Processing request rejected'}`);
    }

    const created = await res.json();
    return normalizeInspection(created, false);
  }

  // --- Demo Mode Simulation ---
  const newId = `INSP-${new Date().getFullYear()}-${Math.floor(1000 + Math.random() * 9000)}`;

  const newInspection: Inspection = {
    id: newId,
    name: input.name.trim() || 'Untitled inspection',
    surfaceType: input.surfaceType || 'Concrete structural surface',
    createdAt: new Date().toISOString(),
    status: 'processing',
    currentStepIndex: 0,
    steps: DEFAULT_PIPELINE_STEPS.map((s, idx) => ({
      ...s,
      status: idx === 0 ? 'active' : 'waiting',
    })),
    leftImage: {
      name: input.leftImageFile?.name || 'left_camera_view.jpg',
      url: input.leftImageFile ? URL.createObjectURL(input.leftImageFile) : '/samples/sample_left.jpg',
      sizeBytes: input.leftImageFile?.size || 3800000,
      dimensions: '3840 × 2160 px',
    },
    rightImage: {
      name: input.rightImageFile?.name || 'right_camera_view.jpg',
      url: input.rightImageFile ? URL.createObjectURL(input.rightImageFile) : '/samples/sample_right.jpg',
      sizeBytes: input.rightImageFile?.size || 3780000,
      dimensions: '3840 × 2160 px',
    },
    calibration: {
      fileName: input.calibrationFile?.name || 'camera_calibration.json',
      baselineDistanceMm: parsedCalibration?.baselineDistanceMm || 120.5,
      focalLengthPx: parsedCalibration?.focalLengthPx || 1420.8,
      status: input.calibrationFile ? 'valid' : 'unverified',
      notes:
        parsedCalibration?.notes ||
        (input.calibrationFile
          ? 'Parameters parsed from uploaded file.'
          : 'Default stereo geometry parameters applied.'),
    },
    reconstructionMethod: input.reconstructionMethod,
    processingDurationSec: 0,
    pointCloudUrl: '/samples/concrete_wall_surface.ply',
    pointCount: 19200,
    findings: [
      {
        id: 'Finding 01',
        type: 'potential_crack',
        confidence: 0.94,
        geometricDeviationMm: 3.42,
        deviationUnit: 'mm',
        description: 'Continuous vertical fissure with 3.42 mm out-of-plane surface indentation.',
        boundingBox2D: { x: 380, y: 220, width: 140, height: 260 },
        regionCenter3D: [35, 10, -3.8],
      },
      {
        id: 'Finding 02',
        type: 'surface_irregularity',
        confidence: 0.78,
        geometricDeviationMm: 1.15,
        deviationUnit: 'mm',
        description: 'Localised surface depression along aggregate boundary with 1.15 mm displacement.',
        boundingBox2D: { x: 620, y: 440, width: 90, height: 80 },
        regionCenter3D: [-60, -45, -1.2],
      },
    ],
    isDemoMode: true,
  };

  const stored = getStoredDemoInspections();
  saveStoredDemoInspections([newInspection, ...stored]);

  return newInspection;
}

/**
 * Polls the backend inspection status at regular intervals until completion or failure.
 * Returns an unsubscription function to stop polling when the component unmounts.
 */
export function pollInspectionStatus(
  id: string,
  onUpdate: (inspection: Inspection) => void,
  onError: (err: Error) => void,
  intervalMs = 2500
): () => void {
  let isCancelled = false;
  let timerId: ReturnType<typeof setTimeout> | null = null;

  const check = async () => {
    if (isCancelled) return;
    try {
      const inspection = await fetchInspectionById(id);
      if (isCancelled) return;

      if (!inspection) {
        onError(new Error(`Inspection ${id} was not found on the backend.`));
        return;
      }

      onUpdate(inspection);

      // Stop polling once finished or failed
      if (inspection.status === 'completed' || inspection.status === 'failed') {
        return;
      }

      timerId = setTimeout(check, intervalMs);
    } catch (err: any) {
      if (isCancelled) return;
      onError(err instanceof Error ? err : new Error(String(err)));
      timerId = setTimeout(check, intervalMs);
    }
  };

  // Trigger initial fetch immediately
  check();

  return () => {
    isCancelled = true;
    if (timerId) clearTimeout(timerId);
  };
}

/**
 * DEMO ONLY: Used for local demonstration mode progression.
 * Never called in real processing mode.
 */
export function updateInspectionProgress(
  id: string,
  stepIndex: number,
  isFinished = false,
  failureMessage?: string
): Inspection | null {
  const all = getStoredDemoInspections();
  const index = all.findIndex((i) => i.id === id);
  if (index === -1) return null;

  const inspection = { ...all[index] };
  inspection.currentStepIndex = stepIndex;

  if (failureMessage) {
    inspection.status = 'failed';
    inspection.failureMessage = failureMessage;
    inspection.steps = inspection.steps.map((s, idx) => {
      if (idx < stepIndex) return { ...s, status: 'completed' as const };
      if (idx === stepIndex) return { ...s, status: 'failed' as const };
      return { ...s, status: 'waiting' as const };
    });
  } else if (isFinished || stepIndex >= inspection.steps.length - 1) {
    inspection.status = 'completed';
    inspection.processingDurationSec = 36;
    inspection.steps = inspection.steps.map((s) => ({ ...s, status: 'completed' as const }));
  } else {
    inspection.status = 'processing';
    inspection.steps = inspection.steps.map((s, idx) => {
      if (idx < stepIndex) return { ...s, status: 'completed' as const };
      if (idx === stepIndex) return { ...s, status: 'active' as const };
      return { ...s, status: 'waiting' as const };
    });
  }

  all[index] = inspection;
  saveStoredDemoInspections(all);
  return inspection;
}
