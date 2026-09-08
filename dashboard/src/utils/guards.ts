/* Runtime shape guards for Phase 6A API payloads.
 * Malformed payloads render an explicit "malformed data" state instead of
 * crashing the page. Guards check structure only, never values. */

import type {
  ComparisonResponse,
  DetectorCapability,
  DetectorsResponse,
  RobustnessReportResponse,
} from "../types/api";

function isRecord(x: unknown): x is Record<string, unknown> {
  return typeof x === "object" && x !== null;
}

export function isDetectorCapability(x: unknown): x is DetectorCapability {
  if (!isRecord(x)) return false;
  return (
    typeof x.name === "string" &&
    typeof x.display_name === "string" &&
    typeof x.implementation_kind === "string" &&
    typeof x.compatibility === "string" &&
    typeof x.requires_config === "boolean" &&
    typeof x.supports_generation === "boolean" &&
    typeof x.supports_benchmarking === "boolean" &&
    (typeof x.tokenizer_requirements === "string" || x.tokenizer_requirements === null) &&
    Array.isArray(x.known_limitations) &&
    typeof x.description === "string"
  );
}

export function isDetectorsResponse(x: unknown): x is DetectorsResponse {
  if (!isRecord(x)) return false;
  return Array.isArray(x.detectors) && x.detectors.every(isDetectorCapability);
}

export function isRobustnessReport(x: unknown): x is RobustnessReportResponse {
  if (!isRecord(x)) return false;
  if (typeof x.schema_version !== "string") return false;
  if (typeof x.total_results !== "number") return false;
  if (!Array.isArray(x.results)) return false;
  if (!Array.isArray(x.aggregated)) return false;
  if (x.matrix !== null && x.matrix !== undefined) {
    if (!isRecord(x.matrix)) return false;
    const m = x.matrix as Record<string, unknown>;
    if (!Array.isArray(m.detectors) || !Array.isArray(m.transforms)) return false;
    if (!isRecord(m.cells)) return false;
  }
  if (!Array.isArray(x.limitations)) return false;
  if (!isRecord(x.summary)) return false;
  const s = x.summary as Record<string, unknown>;
  for (const k of ["detectors", "configs", "transforms", "text_lengths", "categories"]) {
    if (!Array.isArray(s[k])) return false;
  }
  if (!Array.isArray(x.warnings)) return false;
  return true;
}

export function isComparisonResponse(x: unknown): x is ComparisonResponse {
  if (!isRecord(x)) return false;
  return (
    typeof x.schema_version === "string" &&
    Array.isArray(x.rows) &&
    Array.isArray(x.by_model) &&
    Array.isArray(x.by_transform) &&
    Array.isArray(x.by_category) &&
    Array.isArray(x.by_length) &&
    Array.isArray(x.limitations) &&
    Array.isArray(x.warnings)
  );
}
