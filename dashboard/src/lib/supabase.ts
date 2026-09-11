import { createClient } from "@supabase/supabase-js";
import type { SupabaseClient } from "@supabase/supabase-js";

const url = import.meta.env.VITE_SUPABASE_URL as string | undefined;
const anonKey = import.meta.env.VITE_SUPABASE_ANON_KEY as string | undefined;

/** True when the SaaS product shell is enabled (Supabase configured). */
export const supabaseConfigured = Boolean(url && anonKey);

export const supabase: SupabaseClient | null =
  supabaseConfigured ? createClient(url as string, anonKey as string) : null;

/**
 * API base URL for the public site and SaaS workspace.
 *
 * Local development keeps the direct FastAPI origin unless explicitly
 * overridden.  Production defaults to the Vercel Function mounted below the
 * current origin, so no deployment hostname is baked into the browser bundle.
 */
const configuredApiBaseUrl = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.trim();
export const apiBaseUrl = configuredApiBaseUrl || (import.meta.env.DEV ? "http://localhost:8000" : "/api");
