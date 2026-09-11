import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { PublicSite } from "./pages/Public";
import { PricingPage } from "./pages/Pricing";
import { MethodologyPage } from "./pages/Methodology";
import { PrivacyPage, TermsPage } from "./pages/Legal";
import { ProvenanceApiClient } from "./api/client";
import type { SaaSSession } from "./hooks/useSession";
import { allowedShellPages, canEnterWorkspace } from "./utils/entitlements";
import type { Entitlements } from "./types/api";

const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

function mockJson(data: unknown, status = 200) {
  return { ok: status >= 200 && status < 300, status, text: async () => JSON.stringify(data), headers: new Headers() };
}
function mockOnce(data: unknown, status = 200) { mockFetch.mockResolvedValueOnce(mockJson(data, status)); }

function session(over: Partial<SaaSSession> = {}): SaaSSession {
  return {
    status: "anonymous",
    token: null,
    me: null,
    quota: { plan: "anonymous", limit: 2, used: 0, remaining: 2, max_chars_per_analysis: 5000 },
    authIssue: null,
    client: new ProvenanceApiClient("http://localhost:8000", ""),
    refresh: vi.fn().mockResolvedValue(undefined),
    signOut: vi.fn().mockResolvedValue(undefined),
    ...over,
  };
}

const noop = () => {};

beforeEach(() => {
  mockFetch.mockReset();
  mockFetch.mockImplementation(() => Promise.resolve(mockJson({ detail: "Unexpected call" }, 500)));
});

function renderSite(s?: SaaSSession) {
  mockOnce({ plan: "anonymous", limit: 2, used: 0, remaining: 2, max_chars_per_analysis: 5000 }); // initial quota
  render(<PublicSite session={s ?? session()} onEnterWorkspace={noop} onDeveloperSignIn={noop} />);
}

function analyzeResponse(over = {}) {
  return {
    overall_result: "no_supported_signal_detected",
    verdict: "No hidden or unusual Unicode signal was detected by the public check. Configured statistical watermark families were not tested.",
    signals_checked: [{
      detector: "unicode", display_name: "Unicode Artifact Detection",
      signal_type: "hidden_unicode_provenance", status: "not_detected",
      detected: false, confidence: "none",
      evidence: ["0 hidden or unusual Unicode artifacts found."], limitations: [],
    }],
    signals_detected: [],
    unavailable_detectors: [
      {
        detector: "kgw", display_name: "KGW Watermark Detection",
        signal_type: "statistical_watermark", classification: "benchmark_only",
        status: "not_applicable", compute_class: "cheap_configured",
        reason: "Controlled-token simulation for tests and benchmarks.",
      },
      {
        detector: "kgw-reference", display_name: "KGW Reference Detection",
        signal_type: "statistical_watermark", classification: "reference_config_specific",
        status: "unavailable_without_key_or_config", compute_class: "potentially_model_backed",
        reason: "Reference verification requires matching KGW configuration.",
      },
    ],
    character_count: 11,
    quota: { plan: "anonymous", limit: 2, used: 1, remaining: 1, max_chars_per_analysis: 5000 },
    limitations: ["The public analyzer checks hidden or unusual Unicode artifacts only."],
    disclaimer: "A clean result does not establish human authorship or the absence of every watermark.",
    upgrade_hint: "Sign in for higher daily limits.",
    ...over,
  };
}

describe("Public homepage", () => {
  it("renders hero tool and sections", async () => {
    renderSite();
    expect(screen.getByTestId("site-text")).toBeInTheDocument();
    expect(screen.getByTestId("site-analyze")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /Inspect text for hidden provenance signals/i })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /Text can carry signals your eyes never see/i })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /Start with a clear signal/i })).toBeInTheDocument();
    expect(screen.getByText("Unicode signals")).toBeInTheDocument();
    expect(screen.getAllByText("FAQ")).not.toHaveLength(0);
    await waitFor(() => expect(screen.getByTestId("site-quota")).toBeInTheDocument());
    expect(screen.getByTestId("site-quota")).toHaveTextContent("2 of 2 free analyses left today");
  });

  it("analyzes text and shows simplified result", async () => {
    const user = userEvent.setup();
    renderSite();
    await user.type(screen.getByTestId("site-text"), "Hello world");
    mockOnce(analyzeResponse());
    await user.click(screen.getByTestId("site-analyze"));
    await waitFor(() => expect(screen.getByTestId("site-result")).toBeInTheDocument());
    expect(screen.getByText(/No hidden or unusual Unicode signal/i)).toBeInTheDocument();
    expect(screen.getByText(/0 hidden or unusual Unicode artifacts found/i)).toBeInTheDocument();
    expect(screen.getByTestId("site-unavailable-detectors")).toHaveTextContent("benchmark-only");
    expect(screen.getByTestId("site-unavailable-detectors")).toHaveTextContent("configuration required");
    expect(screen.getByText(/absence of every watermark/i)).toBeInTheDocument();
    expect(screen.getByTestId("site-quota")).toHaveTextContent("1 of 2 free analyses left today");
  });

  it("renders detected Unicode separately from unavailable statistical checks", async () => {
    const user = userEvent.setup();
    renderSite();
    await user.type(screen.getByTestId("site-text"), "hello world");
    mockOnce(analyzeResponse({
      overall_result: "signal_detected",
      verdict: "Hidden or unusual Unicode provenance signals were detected. This is not an AI-authorship determination.",
      signals_detected: ["unicode"],
      signals_checked: [{
        detector: "unicode", display_name: "Unicode Artifact Detection",
        signal_type: "hidden_unicode_provenance", status: "detected",
        detected: true, confidence: "high",
        evidence: ["1 hidden or unusual Unicode artifact found."], limitations: [],
      }],
    }));
    await user.click(screen.getByTestId("site-analyze"));
    await waitFor(() => expect(screen.getByText(/hidden Unicode signal detected/i)).toBeInTheDocument());
    expect(screen.getByText(/not an AI-authorship determination/i)).toBeInTheDocument();
    expect(screen.getByTestId("site-unavailable-detectors")).toHaveTextContent("KGW Reference Detection");
  });

  it("does not advertise universal AI or watermark detection", async () => {
    renderSite();
    await waitFor(() => expect(screen.getByTestId("site-quota")).toBeInTheDocument());
    expect(screen.queryByText(/detect every watermark/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/AI-generated text detector/i)).not.toBeInTheDocument();
    expect(screen.getByText(/does not run statistical KGW or SynthID verification/i)).toBeInTheDocument();
  });

  it("keeps implementation detail behind an explicit result control", async () => {
    const user = userEvent.setup();
    renderSite();
    await user.type(screen.getByTestId("site-text"), "Hello world");
    mockOnce(analyzeResponse());
    await user.click(screen.getByTestId("site-analyze"));
    await waitFor(() => expect(screen.getByTestId("site-result")).toBeInTheDocument());
    const details = screen.getByText("View technical details").closest("details");
    expect(details).not.toHaveAttribute("open");
    await user.click(screen.getByText("View technical details"));
    expect(details).toHaveAttribute("open");
    expect(screen.getByTestId("site-unavailable-detectors")).toHaveTextContent("KGW Reference Detection");
  });

  it("opens and closes the compact navigation control", async () => {
    const user = userEvent.setup();
    renderSite();
    const menu = screen.getByRole("button", { name: "Menu" });
    expect(menu).toHaveAttribute("aria-expanded", "false");
    await user.click(menu);
    expect(menu).toHaveAttribute("aria-expanded", "true");
    await user.click(screen.getByRole("link", { name: "FAQ" }));
    expect(menu).toHaveAttribute("aria-expanded", "false");
  });

  it("shows quota exhausted state", async () => {
    const user = userEvent.setup();
    renderSite();
    await user.type(screen.getByTestId("site-text"), "Hello world");
    mockOnce({ detail: "Daily free limit reached" }, 429);
    mockOnce({ plan: "anonymous", limit: 2, used: 2, remaining: 0, max_chars_per_analysis: 5000 });
    await user.click(screen.getByTestId("site-analyze"));
    await waitFor(() => expect(screen.getByTestId("site-error")).toBeInTheDocument());
    expect(screen.getByTestId("site-error")).toHaveTextContent(/2 free analyses for today/);
  });

  it("shows error state on failure", async () => {
    const user = userEvent.setup();
    renderSite();
    await user.type(screen.getByTestId("site-text"), "Hello world");
    mockOnce({ detail: "boom" }, 500);
    await user.click(screen.getByTestId("site-analyze"));
    await waitFor(() => expect(screen.getByTestId("site-error")).toBeInTheDocument());
  });

  it("opens sign-in UI", async () => {
    const user = userEvent.setup();
    renderSite();
    await user.click(screen.getByTestId("site-get-started"));
    await waitFor(() => expect(screen.getByTestId("site-auth")).toBeInTheDocument());
  });

  it("shows a useful sign-in dialog when an OAuth callback has no session", async () => {
    renderSite(session({ authIssue: "Your sign-in session could not be restored. Please try again." }));
    await waitFor(() => expect(screen.getByTestId("site-auth")).toBeInTheDocument());
    await waitFor(() => expect(screen.getByTestId("site-auth-error")).toHaveTextContent(/session could not be restored/i));
  });

  it("asks anonymous visitors to sign in before Pro checkout", async () => {
    const user = userEvent.setup();
    renderSite();
    await user.click(screen.getByTestId("site-upgrade-pro"));
    expect(screen.getByTestId("site-auth")).toBeInTheDocument();
  });

  it("uses the authenticated backend checkout endpoint", async () => {
    const client = new ProvenanceApiClient("http://localhost:8000", "");
    mockOnce({ checkout_url: "https://checkout.test/session" });
    await expect(client.createBillingCheckout("jwt")).resolves.toEqual({ checkout_url: "https://checkout.test/session" });
    expect(mockFetch).toHaveBeenCalledWith("http://localhost:8000/v1/billing/checkout", expect.objectContaining({
      method: "POST", headers: expect.objectContaining({ Authorization: "Bearer jwt" }),
    }));
  });

  it("opens Razorpay Checkout but waits for webhook-backed Pro state", async () => {
    let checkoutHandler: (() => void) | undefined;
    class FakeRazorpay {
      constructor(options: Record<string, unknown>) { checkoutHandler = options.handler as () => void; }
      on() { return undefined; }
      open() { checkoutHandler?.(); }
    }
    Object.assign(window, { Razorpay: FakeRazorpay });
    mockOnce({ plan: "free", limit: 50, used: 0, remaining: 50, max_chars_per_analysis: 20000 });
    mockOnce({ plan: "free", subscription: null, webhook_processing: false });
    mockOnce({ provider: "razorpay", configured: true, checkout_mode: "razorpay_checkout" });
    render(<PublicSite session={session({ status: "user", token: "jwt", me: { kind: "user", auth_user_id: "u", email: "u@example.com", plan: "free", quota: { plan: "free", limit: 50, used: 0, remaining: 50, max_chars_per_analysis: 20000 }, entitlements: freeEnt } })} onEnterWorkspace={noop} onDeveloperSignIn={noop} />);
    mockOnce({ provider: "razorpay", checkout_url: null, checkout_data: { key_id: "rzp_test_public", subscription_id: "sub_1", name: "Text Provenance Engine", description: "Pro subscription" } });
    mockOnce({ plan: "free", subscription: { provider: "razorpay", status: "created", current_period_end: null, cancel_at_period_end: false }, webhook_processing: true });
    await userEvent.setup().click(await screen.findByTestId("site-upgrade-pro"));
    await waitFor(() => expect(screen.getByText(/Verifying your subscription/i)).toBeInTheDocument());
    expect(screen.queryByTestId("site-pro-state")).not.toBeInTheDocument();
    delete (window as Window & { Razorpay?: unknown }).Razorpay;
  });

  it("shows workspace entry for signed-in users", async () => {
    const onEnter = vi.fn();
    mockOnce({ plan: "anonymous", limit: 2, used: 0, remaining: 2, max_chars_per_analysis: 5000 });
    render(<PublicSite session={session({ status: "user" })} onEnterWorkspace={onEnter} onDeveloperSignIn={noop} />);
    await userEvent.setup().click(screen.getByTestId("site-open-workspace"));
    expect(onEnter).toHaveBeenCalled();
  });
});

describe("Product information pages", () => {
  it("renders a truthful methodology page", () => {
    render(<MethodologyPage />);
    expect(screen.getByTestId("methodology-page")).toBeInTheDocument();
    expect(screen.getByText(/not a general-purpose AI authorship classifier/i)).toBeInTheDocument();
    expect(screen.getByText(/No supported signal detected/i)).toBeInTheDocument();
  });

  it("keeps Pro checkout unavailable when billing status is not configured", async () => {
    mockOnce({ provider: "razorpay", configured: false, checkout_mode: null });
    render(<PricingPage session={session()} onOpenWorkspace={noop} />);
    await waitFor(() => expect(screen.getByTestId("pricing-billing-unavailable")).toBeInTheDocument());
    expect(screen.getByText(/Free analysis remains available/i)).toBeInTheDocument();
  });

  it("renders factual privacy and terms pages", () => {
    const { unmount } = render(<PrivacyPage />);
    expect(screen.getByTestId("privacy-page")).toHaveTextContent(/not stored as public analysis content/i);
    unmount();
    render(<TermsPage />);
    expect(screen.getByTestId("terms-page")).toHaveTextContent(/not universal AI detection/i);
  });
});

const anonEnt: Entitlements = {
  plan: "anonymous", max_daily_analyses: 2, max_chars_per_analysis: 5000,
  can_analyze: true,
  can_view_full_report: false, can_access_dashboard: false, can_access_advanced: false,
  can_run_benchmarks: false, can_use_api: false,
};
const freeEnt: Entitlements = { ...anonEnt, plan: "free", can_access_dashboard: true };
const proEnt: Entitlements = {
  plan: "pro", max_daily_analyses: 1000, max_chars_per_analysis: 100000,
  can_analyze: true,
  can_view_full_report: true, can_access_dashboard: true, can_access_advanced: true,
  can_run_benchmarks: true, can_use_api: true,
};

describe("Workspace gating", () => {
  it("legacy API key keeps full dashboard", () => {
    expect(allowedShellPages({ legacyApiKey: true, entitlements: null })).toHaveLength(10);
    expect(canEnterWorkspace({ legacyApiKey: true, entitlements: null })).toBe(true);
  });

  it("anonymous users cannot enter", () => {
    expect(allowedShellPages({ legacyApiKey: false, entitlements: null })).toHaveLength(0);
    expect(allowedShellPages({ legacyApiKey: false, entitlements: anonEnt })).toHaveLength(0);
    expect(canEnterWorkspace({ legacyApiKey: false, entitlements: anonEnt })).toBe(false);
  });

  it("free users get the basic workspace only", () => {
    const pages = allowedShellPages({ legacyApiKey: false, entitlements: freeEnt });
    expect(pages).toEqual(["overview", "analyze", "history", "usage", "settings"]);
    expect(canEnterWorkspace({ legacyApiKey: false, entitlements: freeEnt })).toBe(true);
  });

  it("pro users get the full workspace", () => {
    const pages = allowedShellPages({ legacyApiKey: false, entitlements: proEnt });
    expect(pages).toHaveLength(10);
    expect(pages).toContain("benchmarks");
  });
});
