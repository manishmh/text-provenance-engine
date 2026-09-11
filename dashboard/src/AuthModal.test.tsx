import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

const authMocks = vi.hoisted(() => ({
  signInWithPassword: vi.fn(),
  signUp: vi.fn(),
  signInWithOtp: vi.fn(),
  signInWithOAuth: vi.fn(),
}));

vi.mock("./lib/supabase", () => ({
  supabaseConfigured: true,
  supabase: {
    auth: authMocks,
  },
}));

import { AuthModal } from "./pages/Public";

const onClose = vi.fn();

function renderModal(onSignedIn = vi.fn().mockResolvedValue(true)) {
  render(<AuthModal onClose={onClose} onSignedIn={onSignedIn} />);
  return onSignedIn;
}

async function fillPassword(user: ReturnType<typeof userEvent.setup>) {
  await user.type(screen.getByTestId("site-auth-email"), "new@example.com");
  await user.type(screen.getByTestId("site-auth-password"), "safe-password");
}

beforeEach(() => {
  vi.clearAllMocks();
  window.history.replaceState({}, "", "/#/" );
  sessionStorage.clear();
});

describe("AuthModal", () => {
  it("takes an immediately authenticated signup into backend session completion", async () => {
    authMocks.signUp.mockResolvedValue({ data: { session: { access_token: "session" } }, error: null });
    const onSignedIn = renderModal();
    const user = userEvent.setup();
    await fillPassword(user);
    await user.click(screen.getByRole("button", { name: "Create account" }));
    await waitFor(() => expect(onSignedIn).toHaveBeenCalledTimes(1));
    expect(screen.queryByText(/You can now sign in/i)).not.toBeInTheDocument();
    expect(sessionStorage.getItem("provenance_auth_return_to")).toBe("#/app");
  });

  it("explains email verification without pretending an unauthenticated signup is signed in", async () => {
    authMocks.signUp.mockResolvedValue({ data: { session: null }, error: null });
    const onSignedIn = renderModal();
    const user = userEvent.setup();
    await fillPassword(user);
    await user.click(screen.getByRole("button", { name: "Create account" }));
    await waitFor(() => expect(screen.getByText(/Verify your email to continue/i)).toBeInTheDocument());
    expect(onSignedIn).not.toHaveBeenCalled();
  });

  it("starts Supabase-managed Google OAuth with a same-origin return URL", async () => {
    authMocks.signInWithOAuth.mockResolvedValue({ data: { url: "https://accounts.google.test/" }, error: null });
    renderModal();
    await userEvent.setup().click(screen.getByTestId("site-auth-google"));
    await waitFor(() => expect(authMocks.signInWithOAuth).toHaveBeenCalledWith({
      provider: "google",
      options: { redirectTo: `${window.location.origin}${window.location.pathname}` },
    }));
    expect(screen.getByText(/Redirecting to Google/i)).toBeInTheDocument();
  });

  it("shows a sanitized Google configuration failure", async () => {
    authMocks.signInWithOAuth.mockResolvedValue({ data: { url: null }, error: { message: "provider_not_enabled" } });
    renderModal();
    await userEvent.setup().click(screen.getByTestId("site-auth-google"));
    await waitFor(() => expect(screen.getByTestId("site-auth-error")).toHaveTextContent(/Google sign-in is unavailable/i));
    expect(screen.getByTestId("site-auth-error")).not.toHaveTextContent("provider_not_enabled");
  });

  it("shows a setup failure when authentication exists but backend sync fails", async () => {
    authMocks.signInWithPassword.mockResolvedValue({ data: { session: { access_token: "session" } }, error: null });
    const onSignedIn = renderModal(vi.fn().mockResolvedValue(false));
    const user = userEvent.setup();
    await fillPassword(user);
    await user.click(screen.getByRole("button", { name: "Sign in" }));
    await waitFor(() => expect(onSignedIn).toHaveBeenCalledTimes(1));
    expect(screen.getByTestId("site-auth-error")).toHaveTextContent(/workspace could not be set up/i);
  });
});
