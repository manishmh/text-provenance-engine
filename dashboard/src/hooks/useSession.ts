import { useCallback, useEffect, useState } from "react";
import { supabase, apiBaseUrl, supabaseConfigured } from "../lib/supabase";
import { ProvenanceApiClient } from "../api/client";
import type { MeResponse, QuotaInfo } from "../types/api";

export type SessionStatus = "unknown" | "anonymous" | "user";

export interface SaaSSession {
  status: SessionStatus;
  token: string | null;
  me: MeResponse | null;
  quota: QuotaInfo | null;
  client: ProvenanceApiClient;
  refresh: () => Promise<void>;
  signOut: () => Promise<void>;
}

const publicClient = new ProvenanceApiClient(apiBaseUrl, "");

/** SaaS session: Supabase auth state joined with backend identity/entitlements. */
export function useSaaSSession(): SaaSSession {
  const [status, setStatus] = useState<SessionStatus>("unknown");
  const [token, setToken] = useState<string | null>(null);
  const [me, setMe] = useState<MeResponse | null>(null);

  const refresh = useCallback(async () => {
    if (!supabaseConfigured || !supabase) {
      setStatus("anonymous");
      return;
    }
    const { data } = await supabase.auth.getSession();
    const t = data.session?.access_token ?? null;
    setToken(t);
    try {
      const identity = await publicClient.me(t ?? undefined);
      setMe(identity);
      setStatus(identity.kind === "user" ? "user" : "anonymous");
      if (t && identity.kind === "user") {
        // Provision on the backend (idempotent) so quota migrates.
        await publicClient.authSync(t).catch(() => undefined);
        const after = await publicClient.me(t).catch(() => undefined);
        if (after) setMe(after);
      }
    } catch {
      setStatus(t ? "user" : "anonymous");
    }
  }, []);

  useEffect(() => {
    refresh();
    if (!supabaseConfigured || !supabase) return;
    const { data: sub } = supabase.auth.onAuthStateChange(() => { refresh(); });
    return () => { sub.subscription.unsubscribe(); };
  }, [refresh]);

  const signOut = useCallback(async () => {
    if (supabase) await supabase.auth.signOut();
    setToken(null);
    setMe(null);
    setStatus("anonymous");
  }, []);

  return { status, token, me, quota: me?.quota ?? null, client: publicClient, refresh, signOut };
}
