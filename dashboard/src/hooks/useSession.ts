import { useCallback, useEffect, useRef, useState } from "react";
import { supabase, apiBaseUrl, supabaseConfigured } from "../lib/supabase";
import { isSupabaseAuthCallback } from "../lib/authNavigation";
import { ProvenanceApiClient } from "../api/client";
import { ApiError } from "../api/client";
import type { MeResponse, QuotaInfo } from "../types/api";

export type SessionStatus = "unknown" | "anonymous" | "user";

export interface SaaSSession {
  status: SessionStatus;
  token: string | null;
  me: MeResponse | null;
  quota: QuotaInfo | null;
  /** Sanitized session/callback setup error for product UI. */
  authIssue: string | null;
  client: ProvenanceApiClient;
  /** Synchronize an existing Supabase session with the backend. */
  refresh: () => Promise<boolean>;
  signOut: () => Promise<void>;
}

const publicClient = new ProvenanceApiClient(apiBaseUrl, "");

/** SaaS session: Supabase auth state joined with backend identity/entitlements. */
export function useSaaSSession(): SaaSSession {
  const [status, setStatus] = useState<SessionStatus>("unknown");
  const [token, setToken] = useState<string | null>(null);
  const [me, setMe] = useState<MeResponse | null>(null);
  const [authIssue, setAuthIssue] = useState<string | null>(null);
  const callbackPending = useRef(isSupabaseAuthCallback());
  const refreshInFlight = useRef<Promise<boolean> | null>(null);

  const syncSession = useCallback(async (): Promise<boolean> => {
    if (!supabaseConfigured || !supabase) {
      setStatus("anonymous");
      return false;
    }
    const { data, error } = await supabase.auth.getSession();
    if (error) {
      setToken(null);
      setMe(null);
      setStatus("anonymous");
      if (callbackPending.current) {
        callbackPending.current = false;
        setAuthIssue("Your sign-in session could not be restored. Please try again.");
      }
      return false;
    }
    const t = data.session?.access_token ?? null;
    setToken(t);
    if (!t) {
      try {
        const identity = await publicClient.me();
        setMe(identity);
      } catch {
        setMe(null);
      }
      setStatus("anonymous");
      if (callbackPending.current) {
        callbackPending.current = false;
        setAuthIssue("Your sign-in session could not be restored. Please try again.");
      }
      return false;
    }
    try {
      // Provision first. This idempotent endpoint also claims anonymous
      // usage while the signed visitor cookie is still present. Fetching /me
      // afterwards gives the route guard one fully synchronized state.
      await publicClient.authSync(t);
      const identity = await publicClient.me(t);
      setMe(identity);
      setStatus(identity.kind === "user" ? "user" : "anonymous");
      callbackPending.current = false;
      setAuthIssue(null);
      return identity.kind === "user";
    } catch (error: unknown) {
      // A Supabase session alone must not unlock the workspace. Leave the
      // browser session intact for a retry, but require successful backend
      // provisioning before treating it as an application session.
      setMe(null);
      setStatus("anonymous");
      setAuthIssue(
        error instanceof ApiError && error.status === 401
          ? "Your sign-in session could not be verified. Refresh the page once, then sign in again if needed."
          : error instanceof ApiError && error.status === 503
            ? "Workspace sign-in is temporarily unavailable. Please try again shortly."
            : "Your session was restored, but your workspace could not be set up. Please try again.",
      );
      return false;
    }
  }, []);

  const refresh = useCallback(async (): Promise<boolean> => {
    // Supabase emits an auth event at the same time that password signup or
    // OAuth completion invokes this method. Share that one provisioning
    // request so /v1/auth/sync and /v1/me cannot race each other.
    if (refreshInFlight.current) return refreshInFlight.current;
    const pending = syncSession();
    refreshInFlight.current = pending;
    try {
      return await pending;
    } finally {
      if (refreshInFlight.current === pending) refreshInFlight.current = null;
    }
  }, [syncSession]);

  useEffect(() => {
    void refresh();
    if (!supabaseConfigured || !supabase) return;
    const { data: sub } = supabase.auth.onAuthStateChange(() => {
      // Avoid awaiting network work inside Supabase's synchronous listener.
      window.setTimeout(() => { void refresh(); }, 0);
    });
    return () => { sub.subscription.unsubscribe(); };
  }, [refresh]);

  const signOut = useCallback(async () => {
    if (supabase) await supabase.auth.signOut();
    setToken(null);
    setMe(null);
    setStatus("anonymous");
    setAuthIssue(null);
  }, []);

  return { status, token, me, quota: me?.quota ?? null, authIssue, client: publicClient, refresh, signOut };
}
