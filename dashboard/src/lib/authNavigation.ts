/** Browser-only auth return intent helpers.
 *
 * Supabase OAuth and email confirmation return to the application origin,
 * rather than to a hash route. Persist the intended product route before
 * leaving the page and consume it only after the backend has synchronized a
 * verified Supabase session.
 */
export const AUTH_RETURN_TO_KEY = "provenance_auth_return_to";

export function authReturnToFromLocation(): string {
  if (typeof window === "undefined") return "#/app";
  const hash = window.location.hash;
  // Keep a deep workspace link or an upgrade intent. Ordinary public pages
  // intentionally land in the workspace after sign-in/sign-up.
  if (hash.startsWith("#/app") || hash.includes("upgrade=pro")) return hash;
  return "#/app";
}

export function rememberAuthReturnTo(): void {
  if (typeof window === "undefined") return;
  window.sessionStorage.setItem(AUTH_RETURN_TO_KEY, authReturnToFromLocation());
}

export function consumeAuthReturnTo(): string | null {
  if (typeof window === "undefined") return null;
  const intended = window.sessionStorage.getItem(AUTH_RETURN_TO_KEY);
  window.sessionStorage.removeItem(AUTH_RETURN_TO_KEY);
  return intended && intended.startsWith("#/") ? intended : null;
}

export function authRedirectUrl(): string {
  if (typeof window === "undefined") return "";
  // Hash fragments are not sent to the OAuth provider. The post-session
  // route is restored from sessionStorage above.
  return `${window.location.origin}${window.location.pathname}`;
}

export function oauthCallbackError(): string | null {
  if (typeof window === "undefined") return null;
  const params = new URLSearchParams(window.location.search);
  const code = params.get("error_code") || params.get("error");
  if (!code) return null;
  // Provider/query text is not suitable as UI copy. Preserve a useful,
  // stable message without exposing a raw Supabase error object.
  return "Sign-in could not be completed. Please try again or use email sign-in.";
}

/** Whether this page load is a Supabase email/OAuth callback. */
export function isSupabaseAuthCallback(): boolean {
  if (typeof window === "undefined") return false;
  const params = new URLSearchParams(window.location.search);
  // Supabase's default PKCE flow returns `code`; token_hash covers verified
  // email actions that establish a session through the client callback.
  return Boolean(params.get("code") || params.get("token_hash"));
}
