/* API client for the Text Provenance Engine */

import type {
  AnalyzeResponse,
  AnalysisListResponse,
  AsyncAnalyzeResponse,
  ErrorResponse,
  JobListResponse,
  JobResponse,
  MetricsResponse,
  ReadyResponse,
  UsageResponse,
  ApiKeySummary,
  ApiKeyListResponse,
} from "../types/api";

const STATUS_MESSAGES: Record<number, string> = {
  401: "Authentication required. Please check your API key.",
  403: "Access denied. You don't have permission for this action.",
  404: "Resource not found.",
  409: "Conflict. The resource is in a state that prevents this action.",
  422: "Invalid request. Please check your input.",
  429: "Rate limit exceeded. Please wait before retrying.",
  500: "Server error. Please try again later.",
  502: "Server unavailable. Please try again later.",
  503: "Server unavailable. Please try again later.",
};

export class ApiError extends Error {
  constructor(
    public status: number,
    public detail: string,
    public requestId?: string,
    public retryAfter?: number,
  ) {
    super(detail);
    this.name = "ApiError";
  }

  get userMessage(): string {
    if (this.status === 429 && this.retryAfter) {
      return `Rate limit exceeded. Retry after ${this.retryAfter}s.`;
    }
    return STATUS_MESSAGES[this.status] || this.detail || `HTTP ${this.status} error`;
  }
}

export class ProvenanceApiClient {
  private baseUrl: string;
  private apiKey: string;

  constructor(baseUrl: string, apiKey: string) {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.apiKey = apiKey;
  }

  private async request<T>(
    method: string,
    path: string,
    body?: unknown,
  ): Promise<T> {
    const url = `${this.baseUrl}${path}`;
    const headers: Record<string, string> = {
      Accept: "application/json",
    };
    if (this.apiKey) {
      headers["X-API-Key"] = this.apiKey;
    }
    if (body !== undefined) {
      headers["Content-Type"] = "application/json";
    }

    let res: Response;
    try {
      res = await fetch(url, {
        method,
        headers,
        body: body !== undefined ? JSON.stringify(body) : undefined,
      });
    } catch (e: unknown) {
      const msg = e instanceof TypeError ? "Network error. Is the server running?" : "Request failed";
      throw new ApiError(0, msg);
    }

    const text = await res.text();
    let data: unknown;
    try {
      data = JSON.parse(text);
    } catch {
      data = { detail: text || "Unknown error" };
    }

    if (!res.ok) {
      const err = data as ErrorResponse;
      let retryAfter: number | undefined;
      const ra = res.headers.get("Retry-After");
      if (ra) {
        const n = parseInt(ra, 10);
        if (!isNaN(n)) retryAfter = n;
      }
      throw new ApiError(
        res.status,
        err.detail || `HTTP ${res.status}`,
        undefined,
        retryAfter,
      );
    }

    return data as T;
  }

  /* Public endpoints (no auth) */
  async health() {
    return this.request<{ status: string; engine_version: string }>("GET", "/health");
  }

  async ready() {
    return this.request<ReadyResponse>("GET", "/ready");
  }

  async metrics() {
    return this.request<MetricsResponse>("GET", "/metrics");
  }

  /* Analysis */
  async analyze(text: string, detectors?: string[], configPath?: string): Promise<AnalyzeResponse> {
    const body: Record<string, unknown> = { text };
    if (detectors) body.detectors = detectors;
    if (configPath) body.config_path = configPath;
    return this.request("POST", "/v1/analyze", body);
  }

  async analyzeAsync(text: string, detectors?: string[], configPath?: string): Promise<AsyncAnalyzeResponse> {
    const body: Record<string, unknown> = { text };
    if (detectors) body.detectors = detectors;
    if (configPath) body.config_path = configPath;
    return this.request("POST", "/v1/analyze/async", body);
  }

  /* Analyses */
  async listAnalyses(limit = 20, offset = 0): Promise<AnalysisListResponse> {
    return this.request("GET", `/v1/analyses?limit=${limit}&offset=${offset}`);
  }

  async getAnalysis(id: string) {
    return this.request<Record<string, unknown>>("GET", `/v1/analyses/${id}`);
  }

  /* Jobs */
  async listJobs(limit = 20, offset = 0): Promise<JobListResponse> {
    return this.request("GET", `/v1/jobs?limit=${limit}&offset=${offset}`);
  }

  async getJob(jobId: string): Promise<JobResponse> {
    return this.request("GET", `/v1/jobs/${jobId}`);
  }

  async cancelJob(jobId: string): Promise<{ job_id: string; status: string; message: string }> {
    return this.request("DELETE", `/v1/jobs/${jobId}`);
  }

  async retryJob(jobId: string, text: string, detectors?: string[], configPath?: string): Promise<AsyncAnalyzeResponse> {
    const body: Record<string, unknown> = { text };
    if (detectors) body.detectors = detectors;
    if (configPath) body.config_path = configPath;
    return this.request("POST", `/v1/jobs/${jobId}/retry`, body);
  }

  /* Usage */
  async getUsage(): Promise<UsageResponse> {
    return this.request("GET", "/v1/usage");
  }

  /* API Key Management (admin) */
  async listApiKeys(): Promise<ApiKeyListResponse> {
    return this.request("GET", "/v1/api-keys");
  }

  async createApiKey(name: string): Promise<{ key_id: string; name: string; status: string; created_at: string; key: string }> {
    return this.request("POST", "/v1/api-keys", { name });
  }

  async revokeApiKey(keyId: string): Promise<{ key_id: string; status: string }> {
    return this.request("DELETE", `/v1/api-keys/${keyId}`);
  }
}
