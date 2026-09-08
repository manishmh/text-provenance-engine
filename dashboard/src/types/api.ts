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

/* Phase 6A: detector discovery (GET /v1/detectors) */

export interface DetectorCapability {
  name: string;
  display_name: string;
  implementation_kind: string;
  compatibility: string;
  requires_config: boolean;
  supports_generation: boolean;
  supports_benchmarking: boolean;
  tokenizer_requirements: string | null;
  known_limitations: string[];
  description: string;
}

export interface DetectorsResponse {
  detectors: DetectorCapability[];
}

/* Phase 6A: robustness benchmark artifacts */

export interface AggregatedRobustness {
  detector_name: string;
  config_identifier: string;
  transform_name: string;
  total_samples: number;
  total_baseline_detected: number;
  total_transformed_detected: number;
  total_detection_changes: number;
  mean_baseline_score: number | null;
  mean_transformed_score: number | null;
  mean_score_delta: number | null;
  robustness_rate: number;
  robustness_ci_low: number;
  robustness_ci_high: number;
  baseline_rate: number;
  baseline_ci_low: number;
  baseline_ci_high: number;
  transformed_rate: number;
  transformed_ci_low: number;
  transformed_ci_high: number;
  experiment_count: number;
  text_lengths: (number | null)[];
  seeds: number[];
}

export interface RobustnessMatrixCell {
  detector: string;
  transform: string;
  robustness_rate: number;
  robustness_ci_low: number;
  robustness_ci_high: number;
  baseline_rate: number;
  transformed_rate: number;
  mean_score_delta: number | null;
  total_samples: number;
  total_detection_changes: number;
}

export interface RobustnessMatrix {
  schema_version: string;
  detectors: string[];
  transforms: string[];
  cells: Record<string, RobustnessMatrixCell>;
  limitations: string[];
}

export interface CategoryAggregation {
  detector_name: string;
  config_identifier: string;
  category: string;
  total_samples: number;
  robustness_rate: number;
  robustness_ci_low: number;
  robustness_ci_high: number;
  mean_score_delta: number | null;
  result_count: number;
}

export interface RobustnessSummary {
  total_files_scanned: number;
  results_dir: string;
  detectors: string[];
  configs: string[];
  transforms: string[];
  text_lengths: number[];
  categories: string[];
  filters: {
    detector: string | null;
    config: string | null;
    transform: string | null;
    text_length: number | null;
    run_id?: string | null;
  };
}

export interface RobustnessWarning {
  file: string;
  error_type: string;
  message: string;
}

export interface RobustnessReportResponse {
  schema_version: string;
  total_results: number;
  detectors: string[];
  transforms: string[];
  results: Record<string, unknown>[];
  aggregated: AggregatedRobustness[];
  matrix: RobustnessMatrix | null;
  category_aggregation?: CategoryAggregation[];
  category_map?: Record<string, string>;
  limitations: string[];
  summary: RobustnessSummary;
  warnings: RobustnessWarning[];
}

export interface ComparisonRow {
  model_config: string;
  detector: string;
  transform: string;
  length: number | null;
  baseline_rate: number;
  transformed_rate: number;
  robustness_rate: number;
  robustness_ci_low: number;
  robustness_ci_high: number;
  mean_score_delta: number | null;
  n_samples: number;
  baseline_detected: number;
  transformed_detected: number;
}

export interface ComparisonGroup {
  robustness_rate: number;
  robustness_ci_low: number;
  robustness_ci_high: number;
  total_samples: number;
  [key: string]: unknown;
}

export interface ComparisonResponse {
  schema_version: string;
  run_id: string;
  benchmark_name: string;
  rows: ComparisonRow[];
  by_model: ComparisonGroup[];
  by_transform: ComparisonGroup[];
  by_category: ComparisonGroup[];
  by_length: ComparisonGroup[];
  limitations: string[];
  warnings: RobustnessWarning[];
}

export interface RobustnessFilters {
  detector?: string;
  config?: string;
  transform?: string;
  text_length?: number;
  run_id?: string;
}

/* Phase 6C: dashboard-triggered benchmark runs */

export interface BenchmarkRunConfig {
  detector: string;
  config: string | null;
  profile: string | null;
  transforms: string[] | null;
  lengths: number[];
  samples: number;
  seed: number;
}

export interface BenchmarkRunCreateInput {
  detector: string;
  config?: string | null;
  profile?: string | null;
  transforms?: string[] | null;
  lengths: number[];
  samples: number;
  seed: number;
}

export interface BenchmarkRunProgress {
  experiments_total: number;
  experiments_completed: number;
  experiments_failed: number;
  current_experiment: string | null;
}

export interface BenchmarkRunSummary {
  run_id: string;
  status: string;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  config: BenchmarkRunConfig;
  progress: BenchmarkRunProgress | null;
  error_message: string | null;
  duration_ms: number | null;
  retry_count: number;
}

export interface BenchmarkRunResult {
  experiment_id: string;
  experiments_total: number;
  experiments_completed: number;
  experiments_failed: number;
  out_dir: string;
  has_results: boolean;
  error: string | null;
}

export interface BenchmarkRunResponse extends BenchmarkRunSummary {
  out_dir: string;
  result: BenchmarkRunResult | null;
}

export interface BenchmarkRunListResponse {
  runs: BenchmarkRunSummary[];
  total: number;
  limit: number;
  offset: number;
}

export interface BenchmarkOptionDetector {
  name: string;
  display_name: string;
  implementation_kind: string;
  compatibility: string;
  requires_config: boolean;
  supports_generation: boolean;
  supports_benchmarking: boolean;
}

export interface BenchmarkOptionProfile {
  name: string;
  description: string;
  transform_names: string[];
}

export interface BenchmarkOptionTransform {
  name: string;
  category: string;
  description: string;
}

export interface BenchmarkOptions {
  detectors: BenchmarkOptionDetector[];
  profiles: BenchmarkOptionProfile[];
  transforms: BenchmarkOptionTransform[];
  constraints: {
    max_lengths_count: number;
    max_text_length: number;
    max_samples: number;
    max_transforms: number;
    notes: string;
  };
}

export interface RobustnessFocus {
  detector?: string;
  config?: string;
  runId?: string;
  token: number;
}
