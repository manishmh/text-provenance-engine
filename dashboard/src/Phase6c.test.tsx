import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import App from "./App";
import { ProvenanceApiClient } from "./api/client";

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

afterEach(() => {
  vi.restoreAllMocks();
});

function mockInit() {
  mockFetch
    .mockResolvedValueOnce(mockJson({
      engine_version: "0.1.0", total_requests: 10, successful_requests: 8, failed_requests: 2,
      total_characters: 5000, avg_duration_ms: 150.5,
      by_endpoint: {}, by_detector: {}, by_status_code: {},
      jobs_queued: 0, jobs_running: 0, jobs_completed: 0, jobs_failed: 0, jobs_cancelled: 0,
      jobs_cancellation_requested: 0, configured_worker_count: 2,
    }))
    .mockResolvedValueOnce(mockJson({ analyses: [], total: 0, limit: 20, offset: 0 }))
    .mockResolvedValueOnce(mockJson({
      total_requests: 5, successful_requests: 4, failed_requests: 1,
      total_characters: 2500, by_endpoint: {}, by_detector: {},
    }));
}

async function setupDashboard() {
  const user = userEvent.setup();
  mockOnce({ status: "ok", engine_version: "0.1.0" });
  mockInit();
  render(<App />);
  await user.click(screen.getByTestId("btn-connect"));
  await waitFor(() => expect(screen.getByTestId("nav-overview")).toBeInTheDocument());
  return user;
}

function optionsFixture() {
  return {
    detectors: [
      {
        name: "kgw", display_name: "KGW", implementation_kind: "watermark-kgw",
        compatibility: "gpt-2", requires_config: true,
        supports_generation: true, supports_benchmarking: true,
      },
      {
        name: "mystery-detector", display_name: "Mystery", implementation_kind: "watermark-x",
        compatibility: "any", requires_config: false,
        supports_generation: true, supports_benchmarking: true,
      },
    ],
    profiles: [
      { name: "all_safe", description: "safe", transform_names: ["lowercase"] },
    ],
    transforms: [
      { name: "identity", category: "baseline", description: "none" },
      { name: "lowercase", category: "casing", description: "lower" },
    ],
    constraints: {
      max_lengths_count: 10, max_text_length: 500, max_samples: 50,
      max_transforms: 50, notes: "n",
    },
  };
}

function runFixture(runId: string, status: string, overrides = {}) {
  return {
    run_id: runId,
    status,
    created_at: "2026-01-01T00:00:00",
    started_at: status === "queued" ? null : "2026-01-01T00:00:01",
    completed_at: ["completed", "failed", "cancelled"].includes(status) ? "2026-01-01T00:01:00" : null,
    config: {
      detector: "kgw", config: "configs/kgw.hf.example.json", profile: "all_safe",
      transforms: null, lengths: [50], samples: 2, seed: 42,
    },
    out_dir: `/tmp/runs/${runId}`,
    progress: status === "queued"
      ? null
      : { experiments_total: 1, experiments_completed: status === "completed" ? 1 : 0, experiments_failed: status === "failed" ? 1 : 0, current_experiment: null },
    error_message: status === "failed" ? "generation exploded" : null,
    duration_ms: ["completed", "failed"].includes(status) ? 1234 : null,
    retry_count: 0,
    result: status === "completed"
      ? { experiment_id: "abc", experiments_total: 1, experiments_completed: 1, experiments_failed: 0, out_dir: `/tmp/runs/${runId}`, has_results: true, error: null }
      : null,
    ...overrides,
  };
}

function listFixture(runs: unknown[]) {
  return { runs, total: runs.length, limit: 20, offset: 0 };
}

async function openBenchmarks(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByTestId("nav-benchmarks"));
}

describe("Benchmark client (Phase 6C)", () => {
  it("calls options/list/create/get/cancel/retry endpoints", async () => {
    mockOnce(optionsFixture());
    const client = new ProvenanceApiClient("http://x", "k");
    await client.getBenchmarkOptions();
    expect(mockFetch.mock.calls[0][0]).toBe("http://x/v1/benchmark-runs/options");

    mockOnce(listFixture([]));
    await client.listBenchmarkRuns("running");
    expect(String(mockFetch.mock.calls[1][0])).toBe("http://x/v1/benchmark-runs?status=running");

    mockOnce(runFixture("brun-1", "queued"));
    const created = await client.createBenchmarkRun({ detector: "kgw", lengths: [50], samples: 2, seed: 1 });
    expect(String(mockFetch.mock.calls[2][0])).toBe("http://x/v1/benchmark-runs");
    expect(mockFetch.mock.calls[2][1].method).toBe("POST");
    expect(created.run_id).toBe("brun-1");

    mockOnce(runFixture("brun-1", "completed"));
    await client.getBenchmarkRun("brun-1");
    expect(String(mockFetch.mock.calls[3][0])).toBe("http://x/v1/benchmark-runs/brun-1");

    mockOnce({ run_id: "brun-1", status: "cancelled", message: "x" });
    await client.cancelBenchmarkRun("brun-1");
    expect(String(mockFetch.mock.calls[4][0])).toBe("http://x/v1/benchmark-runs/brun-1/cancel");

    mockOnce(runFixture("brun-1", "queued"));
    await client.retryBenchmarkRun("brun-1");
    expect(String(mockFetch.mock.calls[5][0])).toBe("http://x/v1/benchmark-runs/brun-1/retry");
  });
});

describe("Benchmarks page", () => {
  it("lists runs from the API", async () => {
    const user = await setupDashboard();
    mockOnce(listFixture([runFixture("brun-1", "running"), runFixture("brun-2", "completed")]));
    await openBenchmarks(user);
    await waitFor(() => expect(screen.getByTestId("runs-table")).toBeInTheDocument());
    expect(screen.getByTestId("run-row-brun-1")).toBeInTheDocument();
    expect(screen.getByTestId("run-row-brun-2")).toBeInTheDocument();
    expect(screen.getByTestId("run-progress-brun-1")).toHaveTextContent("0/1");
  });

  it("shows empty state", async () => {
    const user = await setupDashboard();
    mockOnce(listFixture([]));
    await openBenchmarks(user);
    await waitFor(() => expect(screen.getByTestId("runs-empty")).toBeInTheDocument());
  });

  it("shows error and malformed states", async () => {
    const user = await setupDashboard();
    mockOnce({ detail: "boom" }, 500);
    await openBenchmarks(user);
    await waitFor(() => expect(screen.getByTestId("runs-error")).toBeInTheDocument());
  });

  it("cancel and retry buttons call endpoints", async () => {
    const user = await setupDashboard();
    mockOnce(listFixture([runFixture("brun-1", "running"), runFixture("brun-9", "failed")]));
    await openBenchmarks(user);
    await waitFor(() => expect(screen.getByTestId("runs-table")).toBeInTheDocument());

    mockOnce({ run_id: "brun-1", status: "cancellation_requested", message: "x" });
    mockOnce(listFixture([runFixture("brun-1", "running"), runFixture("brun-9", "failed")]));
    await user.click(screen.getByTestId("run-cancel-brun-1"));
    await waitFor(() => expect(
      mockFetch.mock.calls.some((c) => String(c[0]).endsWith("/brun-1/cancel")),
    ).toBe(true));

    mockOnce(runFixture("brun-9", "queued", { retry_count: 1 }));
    mockOnce(listFixture([runFixture("brun-1", "running"), runFixture("brun-9", "queued", { retry_count: 1 })]));
    await user.click(screen.getByTestId("run-retry-brun-9"));
    await waitFor(() => expect(
      mockFetch.mock.calls.some((c) => String(c[0]).endsWith("/brun-9/retry")),
    ).toBe(true));
  });
});

describe("New Benchmark workflow", () => {
  async function openForm() {
    const user = await setupDashboard();
    mockOnce(listFixture([]));
    await openBenchmarks(user);
    await waitFor(() => expect(screen.getByTestId("runs-new")).toBeInTheDocument());
    mockOnce(optionsFixture());
    await user.click(screen.getByTestId("runs-new"));
    await waitFor(() => expect(screen.getByTestId("newrun-form")).toBeInTheDocument());
    return user;
  }

  it("renders registry-driven options without hardcoded names", async () => {
    const user = await openForm();
    const select = screen.getByTestId("newrun-detector");
    expect(within(select).getByText(/mystery-detector/)).toBeInTheDocument();
    // kgw requires a config: summary previews once it is provided
    await user.clear(screen.getByTestId("newrun-config"));
    await user.type(screen.getByTestId("newrun-config"), "configs/kgw.hf.example.json");
    expect(screen.getByTestId("newrun-summary")).toHaveTextContent("kgw");
  });

  it("blocks invalid input client-side without POSTing", async () => {
    const user = await openForm();
    const callsBefore = mockFetch.mock.calls.length;
    await user.clear(screen.getByTestId("newrun-lengths"));
    await user.type(screen.getByTestId("newrun-lengths"), "abc");
    expect(screen.getByTestId("newrun-submit")).toBeDisabled();
    expect(mockFetch.mock.calls.length).toBe(callsBefore);
  });

  it("submits valid input and opens the detail view", async () => {
    const user = await openForm();
    await user.clear(screen.getByTestId("newrun-config"));
    await user.type(screen.getByTestId("newrun-config"), "configs/kgw.hf.example.json");
    mockOnce(runFixture("brun-new", "queued"));
    mockOnce(runFixture("brun-new", "queued"));
    await user.click(screen.getByTestId("newrun-submit"));
    await waitFor(() => expect(screen.getByTestId("rundetail")).toBeInTheDocument());
    expect(screen.getByTestId("rundetail-config")).toHaveTextContent("kgw");
    const posts = mockFetch.mock.calls.filter((c) => c[1] && c[1].method === "POST");
    expect(posts.some((c) => String(c[0]).endsWith("/v1/benchmark-runs"))).toBe(true);
  });

  it("shows structured backend validation errors", async () => {
    const user = await openForm();
    await user.clear(screen.getByTestId("newrun-config"));
    await user.type(screen.getByTestId("newrun-config"), "configs/kgw.hf.example.json");
    mockOnce(
      { detail: "Invalid benchmark configuration", errors: [{ field: "lengths", message: "too long" }] },
      422,
    );
    await user.click(screen.getByTestId("newrun-submit"));
    await waitFor(() => expect(screen.getByTestId("newrun-errors")).toBeInTheDocument());
    expect(screen.getByTestId("newrun-errors")).toHaveTextContent("lengths: too long");
  });
});

describe("Run detail", () => {
  async function openDetail(status: string, runId = "brun-1") {
    const user = await setupDashboard();
    mockOnce(listFixture([runFixture(runId, status)]));
    await openBenchmarks(user);
    await waitFor(() => expect(screen.getByTestId("runs-table")).toBeInTheDocument());
    mockOnce(runFixture(runId, status));
    await user.click(screen.getByTestId(`run-open-${runId}`));
    await waitFor(() => expect(screen.getByTestId("rundetail")).toBeInTheDocument());
    return user;
  }

  it("renders progress, config, and result summary", async () => {
    await openDetail("completed");
    expect(screen.getByTestId("rundetail-progressbar")).toBeInTheDocument();
    expect(screen.getByTestId("rundetail-config")).toHaveTextContent("all_safe");
    expect(screen.getByTestId("rundetail-result")).toHaveTextContent("1/1 experiment(s) completed");
    expect(screen.getByTestId("rundetail-view-results")).toBeInTheDocument();
  });

  it("shows run errors without success styling", async () => {
    await openDetail("failed");
    expect(screen.getByTestId("rundetail-run-error")).toHaveTextContent("generation exploded");
    expect(screen.getByTestId("rundetail-retry")).toBeInTheDocument();
    expect(screen.queryByTestId("rundetail-view-results")).not.toBeInTheDocument();
  });

  it("navigates to Robustness scoped to the run", async () => {
    const user = await openDetail("completed");
    const report = {
      schema_version: "provenance-robustness-v1", total_results: 1,
      detectors: ["kgw"], transforms: ["identity"], results: [],
      aggregated: [{
        detector_name: "kgw", config_identifier: "configs/kgw.hf.example.json",
        transform_name: "identity", total_samples: 2, total_baseline_detected: 2,
        total_transformed_detected: 2, total_detection_changes: 0,
        mean_baseline_score: 1, mean_transformed_score: 1, mean_score_delta: 0,
        robustness_rate: 1, robustness_ci_low: 0.34, robustness_ci_high: 1,
        baseline_rate: 1, baseline_ci_low: 0.34, baseline_ci_high: 1,
        transformed_rate: 1, transformed_ci_low: 0.34, transformed_ci_high: 1,
        experiment_count: 1, text_lengths: [50], seeds: [42],
      }],
      matrix: null, category_map: { identity: "baseline" }, limitations: [],
      summary: {
        total_files_scanned: 1, results_dir: "d", detectors: ["kgw"],
        configs: ["configs/kgw.hf.example.json"], transforms: ["identity"],
        text_lengths: [50], categories: ["baseline"],
        filters: { detector: null, config: null, transform: null, text_length: null },
      },
      warnings: [],
    };
    const comparison = {
      schema_version: "provenance-benchmark-report-v1", run_id: "api",
      benchmark_name: "c", rows: [], by_model: [], by_transform: [],
      by_category: [], by_length: [], limitations: [], warnings: [],
    };
    mockOnce(report);
    mockOnce(comparison);
    await user.click(screen.getByTestId("rundetail-view-results"));
    await waitFor(() => expect(screen.getByTestId("robustness-scope")).toBeInTheDocument());
    expect(screen.getByTestId("robustness-scope")).toHaveTextContent("brun-1");
    expect(mockFetch.mock.calls.some(
      (c) => String(c[0]).includes("/v1/robustness/results?") && String(c[0]).includes("run_id=brun-1"),
    )).toBe(true);
  });

  it("polls while active, not when terminal", async () => {
    // waitFor itself uses short-interval timers; only 3000ms polls are ours.
    const polls = () => vi.mocked(window.setInterval).mock.calls.filter((c) => c[1] === 3000);
    const setIntervalSpy = vi.spyOn(window, "setInterval");
    const user = await setupDashboard();
    mockOnce(listFixture([runFixture("brun-1", "running")]));
    await openBenchmarks(user);
    await waitFor(() => expect(screen.getByTestId("runs-table")).toBeInTheDocument());
    expect(polls().length).toBeGreaterThan(0);

    setIntervalSpy.mockClear();
    mockOnce(runFixture("brun-1", "completed"));
    await user.click(screen.getByTestId("run-open-brun-1"));
    await waitFor(() => expect(screen.getByTestId("rundetail")).toBeInTheDocument());
    // Terminal detail view must not poll; allow effects to settle.
    await new Promise((r) => setTimeout(r, 100));
    expect(polls()).toHaveLength(0);
  });
});
