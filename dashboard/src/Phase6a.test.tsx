import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
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

function cap(name: string, overrides = {}) {
  return {
    name,
    display_name: `${name} display`,
    implementation_kind: "watermark-test",
    compatibility: "test",
    requires_config: true,
    supports_generation: false,
    supports_benchmarking: true,
    tokenizer_requirements: null,
    known_limitations: ["limitation one"],
    description: `${name} description`,
    ...overrides,
  };
}

function agg(detector: string, config: string, transform: string, overrides = {}) {
  return {
    detector_name: detector,
    config_identifier: config,
    transform_name: transform,
    total_samples: 10,
    total_baseline_detected: 10,
    total_transformed_detected: 8,
    total_detection_changes: 2,
    mean_baseline_score: 10.0,
    mean_transformed_score: 8.0,
    mean_score_delta: -2.0,
    robustness_rate: 0.8,
    robustness_ci_low: 0.49,
    robustness_ci_high: 0.94,
    baseline_rate: 1.0,
    baseline_ci_low: 0.72,
    baseline_ci_high: 1.0,
    transformed_rate: 0.8,
    transformed_ci_low: 0.49,
    transformed_ci_high: 0.94,
    experiment_count: 1,
    text_lengths: [50],
    seeds: [42],
    ...overrides,
  };
}

function reportFixture(overrides = {}) {
  return {
    schema_version: "provenance-robustness-v1",
    total_results: 3,
    detectors: ["kgw", "synthid"],
    transforms: ["identity", "lowercase"],
    results: [],
    aggregated: [
      agg("kgw", "model-a", "identity", { robustness_rate: 1.0 }),
      agg("kgw", "model-a", "lowercase"),
      agg("synthid", "model-b", "identity", { robustness_rate: 1.0 }),
    ],
    matrix: null,
    category_map: { identity: "baseline", lowercase: "casing" },
    limitations: ["Test limitation: not resistant to removal."],
    summary: {
      total_files_scanned: 3,
      results_dir: "data/robustness",
      detectors: ["kgw", "synthid"],
      configs: ["model-a", "model-b"],
      transforms: ["identity", "lowercase"],
      text_lengths: [50, 100],
      categories: ["baseline", "casing"],
      filters: { detector: null, config: null, transform: null, text_length: null },
    },
    warnings: [],
    ...overrides,
  };
}

function comparisonFixture(overrides = {}) {
  const group = (extra = {}) => ({
    robustness_rate: 0.8, robustness_ci_low: 0.49, robustness_ci_high: 0.94,
    total_samples: 20, baseline_rate: 1.0, transformed_rate: 0.8, ...extra,
  });
  return {
    schema_version: "provenance-benchmark-report-v1",
    run_id: "api",
    benchmark_name: "robustness-comparison",
    rows: [],
    by_model: [group({ model_config: "model-a" }), group({ model_config: "model-b" })],
    by_transform: [group({ transform: "identity" }), group({ transform: "lowercase" })],
    by_category: [group({ category: "baseline" })],
    by_length: [group({ length: 50 })],
    limitations: ["Comparison limitation."],
    warnings: [],
    ...overrides,
  };
}

describe("API client (Phase 6A)", () => {
  it("listDetectors hits /v1/detectors", async () => {
    mockOnce({ detectors: [cap("kgw")] });
    const client = new ProvenanceApiClient("http://x", "k");
    const data = await client.listDetectors();
    expect(mockFetch.mock.calls[0][0]).toBe("http://x/v1/detectors");
    expect(data.detectors[0].name).toBe("kgw");
  });

  it("getRobustnessResults passes filters as query params", async () => {
    mockOnce(reportFixture());
    const client = new ProvenanceApiClient("http://x", "k");
    await client.getRobustnessResults({ detector: "kgw", text_length: 50 });
    const url = String(mockFetch.mock.calls[0][0]);
    expect(url).toContain("/v1/robustness/results?");
    expect(url).toContain("detector=kgw");
    expect(url).toContain("text_length=50");
  });

  it("getRobustnessComparison passes filters as query params", async () => {
    mockOnce(comparisonFixture());
    const client = new ProvenanceApiClient("http://x", "k");
    await client.getRobustnessComparison({ transform: "lowercase" });
    const url = String(mockFetch.mock.calls[0][0]);
    expect(url).toContain("/v1/robustness/comparison?");
    expect(url).toContain("transform=lowercase");
  });
});

describe("Detectors page", () => {
  async function openDetectors(payload: unknown, status = 200) {
    const user = await setupDashboard();
    // Queue before navigating: the page fetches on mount.
    mockOnce(payload, status);
    await user.click(screen.getByTestId("nav-detectors"));
    return user;
  }

  it("renders detectors from the API without hardcoded names", async () => {
    await openDetectors({ detectors: [cap("kgw"), cap("brand-new-detector", { requires_config: false })] });
    await waitFor(() => expect(screen.getByTestId("detectors-table")).toBeInTheDocument());
    // A detector the UI has never seen renders generically
    expect(screen.getByTestId("detector-row-brand-new-detector")).toBeInTheDocument();
    expect(screen.getByTestId("detector-row-kgw")).toBeInTheDocument();
  });

  it("shows empty state", async () => {
    await openDetectors({ detectors: [] });
    await waitFor(() => expect(screen.getByTestId("detectors-empty")).toBeInTheDocument());
  });

  it("shows error on API failure", async () => {
    await openDetectors({ detail: "Unauthorized" }, 401);
    await waitFor(() => expect(screen.getByTestId("detectors-error")).toBeInTheDocument());
  });

  it("shows malformed state on bad shape", async () => {
    await openDetectors({ detectors: [{ name: 123 }] });
    await waitFor(() => expect(screen.getByTestId("detectors-malformed")).toBeInTheDocument());
  });
});

describe("Robustness page", () => {
  async function openRobustness(payloadOverrides = {}, comparisonOverrides = {}) {
    const user = await setupDashboard();
    // Queue before navigating: the page fetches on mount.
    mockOnce(reportFixture(payloadOverrides));
    mockOnce(comparisonFixture(comparisonOverrides));
    await user.click(screen.getByTestId("nav-robustness"));
    await waitFor(() => expect(
      screen.queryByTestId("robustness-matrix") ||
      screen.queryByTestId("robustness-empty") ||
      screen.queryByTestId("robustness-error") ||
      screen.queryByTestId("robustness-malformed"),
    ).toBeInTheDocument());
    return user;
  }

  it("renders matrix, detail, comparison, and disclaimer", async () => {
    await openRobustness();
    expect(screen.getByTestId("robustness-matrix")).toBeInTheDocument();
    expect(screen.getByTestId("robustness-disclaimer")).toBeInTheDocument();
    // Matrix cell from server data (no hardcoded transform names needed)
    expect(screen.getByTestId("matrix-cell-kgw-model-a-lowercase")).toBeInTheDocument();
    // Detail panel distinguishes baseline detection from preservation
    const detail = screen.getByTestId("robustness-detail");
    expect(within(detail).getByText("Baseline detection")).toBeInTheDocument();
    expect(within(detail).getByText("Robustness (preserved)")).toBeInTheDocument();
    // Comparison sections render from server data
    expect(screen.getByTestId("comparison-by-model")).toBeInTheDocument();
    expect(screen.getByTestId("comparison-by-transform")).toBeInTheDocument();
    expect(screen.getByTestId("comparison-by-category")).toBeInTheDocument();
    expect(screen.getByTestId("comparison-by-length")).toBeInTheDocument();
  });

  it("filters narrow the matrix", async () => {
    const user = await openRobustness();
    await user.selectOptions(screen.getByTestId("filter-detector"), "synthid");
    await waitFor(() => expect(
      screen.queryByTestId("matrix-cell-kgw-model-a-lowercase"),
    ).not.toBeInTheDocument());
    expect(screen.getByTestId("matrix-cell-synthid-model-b-identity")).toBeInTheDocument();
  });

  it("category filter narrows transforms", async () => {
    const user = await openRobustness();
    await user.selectOptions(screen.getByTestId("filter-category"), "casing");
    await waitFor(() => expect(
      screen.queryByTestId("matrix-cell-kgw-model-a-identity"),
    ).not.toBeInTheDocument());
    expect(screen.getByTestId("matrix-cell-kgw-model-a-lowercase")).toBeInTheDocument();
  });

  it("shows empty state when no results stored", async () => {
    await openRobustness({ total_results: 0, aggregated: [] });
    expect(screen.getByTestId("robustness-empty")).toBeInTheDocument();
    // Disclaimer still shown on empty state
    expect(screen.getByTestId("robustness-disclaimer")).toBeInTheDocument();
  });

  it("shows error on API failure", async () => {
    const user = await setupDashboard();
    mockOnce({ detail: "boom" }, 500);
    mockOnce({ detail: "boom" }, 500);
    await user.click(screen.getByTestId("nav-robustness"));
    await waitFor(() => expect(screen.getByTestId("robustness-error")).toBeInTheDocument());
  });

  it("shows malformed state on bad shape", async () => {
    const user = await setupDashboard();
    mockOnce({ total_results: "three" });
    mockOnce(comparisonFixture());
    await user.click(screen.getByTestId("nav-robustness"));
    await waitFor(() => expect(screen.getByTestId("robustness-malformed")).toBeInTheDocument());
  });

  it("shows warnings for unloadable files", async () => {
    await openRobustness({ warnings: [{ file: "bad.jsonl", error_type: "ValueError", message: "bad" }] });
    expect(screen.getByTestId("robustness-warnings")).toBeInTheDocument();
    expect(screen.getByText(/bad\.jsonl/)).toBeInTheDocument();
  });
});
