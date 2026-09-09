import { createClient } from "@supabase/supabase-js";
import type { SupabaseClient } from "@supabase/supabase-js";

const url = import.meta.env.VITE_SUPABASE_URL as string | undefined;
const anonKey = import.meta.env.VITE_SUPABASE_ANON_KEY as string | undefined;

/** True when the SaaS product shell is enabled (Supabase configured). */
export const supabaseConfigured = Boolean(url && anonKey);

export const supabase: SupabaseClient | null =
  supabaseConfigured ? createClient(url as string, anonKey as string) : null;

/** API base URL for the public site (defaults to local dev server). */
export const apiBaseUrl =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) || "http://localhost:8000";
