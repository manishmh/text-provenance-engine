import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import App from "./App";

const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

function mockJson(data: unknown, status = 200) {
  return { ok: status >= 200 && status < 300, status, text: async () => JSON.stringify(data), headers: new Headers() };
}
function mockOnce(data: unknown, status = 200) { mockFetch.mockResolvedValueOnce(mockJson(data, status)); }

beforeEach(() => {
  sessionStorage.clear();
  mockFetch.mockReset();
  mockFetch.mockImplementation(() => Promise.resolve(mockJson({ detail: "Unexpected call" }, 500)));
});

/** Init mocks: Overview child effect fires first (metrics + analyses), then App parent effect (usage). */
function mockInit() {
  mockFetch
    .mockResolvedValueOnce(mockJson({
      engine_version: "0.1.0", total_requests: 10, successful_requests: 8, failed_requests: 2,
      total_characters: 5000, avg_duration_ms: 150.5,
      by_endpoint: { "/v1/analyze": 8 }, by_detector: { unicode: 8 }, by_status_code: {},
      jobs_queued: 0, jobs_running: 0, jobs_completed: 3, jobs_failed: 1, jobs_cancelled: 0,
      jobs_cancellation_requested: 0, configured_worker_count: 2,
    }))
    .mockResolvedValueOnce(mockJson({
      analyses: [{
        analysis_id: "abc-123", timestamp: "2025-01-01T00:00:00Z", engine_version: "0.1.0",
        text_hash: "abc123def456", character_count: 100, token_count: 20, status: "ok", detector_count: 1,
      }],
      total: 1, limit: 20, offset: 0,
    }))
    .mockResolvedValueOnce(mockJson({
      total_requests: 5, successful_requests: 4, failed_requests: 1,
      total_characters: 2500, by_endpoint: { "/v1/analyze": 4 }, by_detector: { unicode: 4 },
    }));
}

async function setupDashboard() {
  const user = userEvent.setup();
  mockOnce({ status: "ok", engine_version: "0.1.0" }); // health
  mockInit(); // metrics, analyses, usage
  render(<App />);
  await user.click(screen.getByTestId("btn-connect"));
  await waitFor(() => expect(screen.getByTestId("nav-overview")).toBeInTheDocument());
  return user;
}

describe("Setup", () => {
  it("shows setup screen when no config", () => {
    render(<App />);
    expect(screen.getByTestId("input-base-url")).toBeInTheDocument();
    expect(screen.getByTestId("input-api-key")).toBeInTheDocument();
  });

  it("connects to API", async () => {
    const user = await setupDashboard();
    expect(screen.getByTestId("nav-overview")).toBeInTheDocument();
  });

  it("shows error on connection failure", async () => {
    const user = userEvent.setup();
    mockOnce({ detail: "Service unavailable" }, 503);
    render(<App />);
    await user.click(screen.getByTestId("btn-connect"));
    await waitFor(() => expect(screen.getByTestId("setup-error")).toBeInTheDocument());
  });

  it("does not expose API key in URL", async () => {
    const user = userEvent.setup();
    mockOnce({ status: "ok", engine_version: "0.1.0" });
    mockInit();
    render(<App />);
    await user.type(screen.getByTestId("input-api-key"), "secret-key-xyz");
    await user.click(screen.getByTestId("btn-connect"));
    await waitFor(() => expect(screen.getByTestId("nav-overview")).toBeInTheDocument());
    const urls = mockFetch.mock.calls.map((c: unknown[]) => String(c[0]));
    expect(urls.every((u: string) => !u.includes("secret-key"))).toBe(true);
  });

  it("disconnect returns to setup", async () => {
    const user = await setupDashboard();
    await user.click(screen.getByTestId("btn-disconnect"));
    await waitFor(() => expect(screen.getByTestId("btn-connect")).toBeInTheDocument());
  });
});

describe("Analyze", () => {
  it("performs analysis and shows results", async () => {
    const user = await setupDashboard();
    await user.click(screen.getByTestId("nav-analyze"));
    await user.type(screen.getByTestId("input-text"), "Hello world");
    mockOnce({
      analysis_id: "t1", engine_version: "0.1.0", status: "ok", duration_ms: 50,
      text_stats: { character_count: 11 },
      results: [{ detector: "unicode", detector_version: "0.1.0", status: "ok", detected: false,
        implementation_kind: "deterministic", compatibility: "ok", score: null, threshold: null,
        confidence: "none", evidence: {}, text_requirements: {}, limitations: [], metadata: {} }],
      limitations: [], metadata: {},
    });
    await user.click(screen.getByTestId("btn-analyze"));
    await waitFor(() => expect(screen.getByTestId("analyze-result")).toBeInTheDocument());
    expect(screen.getAllByTestId("detector-result")).toHaveLength(1);
  });

  it("shows error on API failure", async () => {
    const user = await setupDashboard();
    await user.click(screen.getByTestId("nav-analyze"));
    await user.type(screen.getByTestId("input-text"), "Test");
    mockOnce({ detail: "Unknown detector(s): foo" }, 422);
    await user.click(screen.getByTestId("btn-analyze"));
    await waitFor(() => expect(screen.getByTestId("analyze-error")).toBeInTheDocument());
  });

  it("disables button during analysis", async () => {
    const user = await setupDashboard();
    await user.click(screen.getByTestId("nav-analyze"));
    await user.type(screen.getByTestId("input-text"), "Test");
    let resolve: (v: unknown) => void;
    mockFetch.mockReturnValueOnce(new Promise((r) => { resolve = r; }));
    await user.click(screen.getByTestId("btn-analyze"));
    await waitFor(() => expect(screen.getByTestId("btn-analyze")).toBeDisabled());
    resolve!({ ok: true, status: 200, text: async () => JSON.stringify({ analysis_id: "x", engine_version: "0.1.0", status: "ok", text_stats: {}, results: [], limitations: [], metadata: {} }), headers: new Headers() });
    await waitFor(() => expect(screen.getByTestId("btn-analyze")).not.toBeDisabled());
  });

  it("button is disabled when text is empty", async () => {
    const user = await setupDashboard();
    await user.click(screen.getByTestId("nav-analyze"));
    expect(screen.getByTestId("btn-analyze")).toBeDisabled();
  });
});

describe("Jobs", () => {
  it("shows empty state", async () => {
    const user = await setupDashboard();
    mockOnce({ jobs: [], total: 0, limit: 20, offset: 0 });
    await user.click(screen.getByTestId("nav-jobs"));
    await waitFor(() => expect(screen.getByText(/No jobs found/)).toBeInTheDocument());
  });

  it("shows completed job", async () => {
    const user = await setupDashboard();
    mockOnce({ jobs: [{ job_id: "j1", status: "completed", created_at: "2025-01-01T00:00:00Z",
      character_count: 10, detectors: ["unicode"], retry_count: 0 }], total: 1, limit: 20, offset: 0 });
    await user.click(screen.getByTestId("nav-jobs"));
    await waitFor(() => expect(screen.getByTestId("btn-view-j1")).toBeInTheDocument());
  });

  it("shows cancel for active jobs", async () => {
    const user = await setupDashboard();
    mockOnce({ jobs: [{ job_id: "j2", status: "queued", created_at: "2025-01-01T00:00:00Z",
      character_count: 10, detectors: ["unicode"], retry_count: 0 }], total: 1, limit: 20, offset: 0 });
    await user.click(screen.getByTestId("nav-jobs"));
    await waitFor(() => expect(screen.getByTestId("btn-cancel-j2")).toBeInTheDocument());
  });

  it("shows retry for failed jobs", async () => {
    const user = await setupDashboard();
    mockOnce({ jobs: [{ job_id: "j3", status: "failed", created_at: "2025-01-01T00:00:00Z",
      character_count: 10, detectors: ["unicode"], retry_count: 0 }], total: 1, limit: 20, offset: 0 });
    await user.click(screen.getByTestId("nav-jobs"));
    await waitFor(() => expect(screen.getByTestId("btn-retry-j3")).toBeInTheDocument());
  });
});

describe("History", () => {
  it("shows analysis list", async () => {
    const user = await setupDashboard();
    mockOnce({ analyses: [{
      analysis_id: "abc-123", timestamp: "2025-01-01T00:00:00Z", engine_version: "0.1.0",
      text_hash: "abc123def456", character_count: 100, token_count: 20, status: "ok", detector_count: 1,
    }], total: 1, limit: 20, offset: 0 });
    await user.click(screen.getByTestId("nav-history"));
    await waitFor(() => expect(screen.getByTestId("btn-detail-abc-123")).toBeInTheDocument());
  });

  it("shows empty state", async () => {
    const user = await setupDashboard();
    mockOnce({ analyses: [], total: 0, limit: 20, offset: 0 });
    await user.click(screen.getByTestId("nav-history"));
    await waitFor(() => expect(screen.getByText(/No analyses found/)).toBeInTheDocument());
  });
});

describe("API Keys", () => {
  it("shows admin info without admin key", async () => {
    const user = await setupDashboard();
    mockOnce({ keys: [], total: 0 });
    await user.click(screen.getByTestId("nav-admin"));
    await waitFor(() => expect(screen.getByText(/requires an admin key/i)).toBeInTheDocument());
  });
});

describe("Error handling", () => {
  it("handles 401 gracefully", async () => {
    const user = await setupDashboard();
    mockOnce({ detail: "Unauthorized" }, 401);
    await user.click(screen.getByTestId("nav-usage"));
    await waitFor(() => expect(screen.getByTestId("nav-usage")).toBeInTheDocument());
  });
});
