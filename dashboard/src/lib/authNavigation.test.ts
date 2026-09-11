import { beforeEach, describe, expect, it } from "vitest";
import {
  authReturnToFromLocation,
  consumeAuthReturnTo,
  isSupabaseAuthCallback,
  rememberAuthReturnTo,
} from "./authNavigation";

beforeEach(() => {
  sessionStorage.clear();
  window.history.replaceState({}, "", "/#/" );
});

describe("auth return navigation", () => {
  it("uses the workspace for an ordinary public sign-in", () => {
    expect(authReturnToFromLocation()).toBe("#/app");
    rememberAuthReturnTo();
    expect(consumeAuthReturnTo()).toBe("#/app");
    expect(consumeAuthReturnTo()).toBeNull();
  });

  it("preserves a requested workspace route for OAuth/session restoration", () => {
    window.history.replaceState({}, "", "/#/app/benchmarks");
    rememberAuthReturnTo();
    expect(consumeAuthReturnTo()).toBe("#/app/benchmarks");
  });

  it("recognizes a Supabase PKCE callback even without a saved tab intent", () => {
    window.history.replaceState({}, "", "/?code=supabase-code");
    expect(isSupabaseAuthCallback()).toBe(true);
  });
});
