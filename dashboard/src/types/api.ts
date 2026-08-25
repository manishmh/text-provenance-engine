/* Types matching the provenance API contracts */

export interface HealthResponse {
  status: string;
  engine_version: string;
}

export interface ReadyResponse {
  status: string;
  engine_version: string;
  persistence: string;
  executor: string;
}

export interface MetricsResponse {
  engine_version: string;
  total_requests: number;
  successful_requests: number;
  failed_requests: number;
  total_characters: number;
  avg_duration_ms: number | null;
  by_endpoint: Record<string, number>;
  by_detector: Record<string, number>;
  by_status_code: Record<string, number>;
  jobs_queued: number;
  jobs_running: number;
  jobs_completed: number;
  jobs_failed: number;
  jobs_cancelled: number;
  jobs_cancellation_requested: number;
  configured_worker_count: number;
}

export interface DetectionResultItem {
  detector: string;
  detector_version: string;
  status: string;
  detected: boolean | null;
  implementation_kind: string;
  compatibility: string;
  score: number | null;
  threshold: number | null;
  confidence: string;
  evidence: Record<string, unknown>;
  text_requirements: Record<string, unknown>;
  limitations: string[];
  metadata: Record<string, unknown>;
}

export interface AnalyzeResponse {
  analysis_id: string;
  engine_version: string;
  status: string;
  text_stats: Record<string, unknown>;
  results: DetectionResultItem[];
  limitations: string[];
  metadata: Record<string, unknown>;
  duration_ms: number | null;
}

export interface AnalysisSummary {
  analysis_id: string;
  timestamp: string;
  engine_version: string;
  text_hash: string;
  character_count: number;
  token_count: number;
  status: string;
  detector_count: number;
}

export interface AnalysisListResponse {
  analyses: AnalysisSummary[];
  total: number;
  limit: number;
  offset: number;
}

export interface UsageResponse {
  total_requests: number;
  successful_requests: number;
  failed_requests: number;
  total_characters: number;
  by_endpoint: Record<string, number>;
  by_detector: Record<string, number>;
}

export interface AsyncAnalyzeResponse {
  job_id: string;
  status: string;
  message: string;
}

export interface JobSummary {
  job_id: string;
  status: string;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  character_count: number;
  detectors: string[];
  error_message: string | null;
  duration_ms: number | null;
  retry_count: number;
  parent_job_id: string | null;
}

export interface JobResponse {
  job_id: string;
  status: string;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  character_count: number;
  detectors: string[];
  config_path: string | null;
  error_message: string | null;
  duration_ms: number | null;
  result: Record<string, unknown> | null;
  retry_count: number;
  parent_job_id: string | null;
}

export interface JobListResponse {
  jobs: JobSummary[];
  total: number;
  limit: number;
  offset: number;
}

export interface ErrorResponse {
  detail: string;
  request_id?: string;
}

export interface DashboardConfig {
  baseUrl: string;
  apiKey: string;
}

export interface ApiKeySummary {
  key_id: string;
  name: string;
  status: string;
  created_at: string;
}

export interface ApiKeyListResponse {
  keys: ApiKeySummary[];
  total: number;
}

export interface CreateApiKeyResponse {
  key_id: string;
  name: string;
  status: string;
  created_at: string;
  key: string;
}
