"""Public SaaS routes: anonymous analysis, quota, and auth provisioning.

These endpoints deliberately bypass the API-key model:

- ``POST /v1/public/analyze`` — anonymous (or signed-in) arbitrary-text
  analysis with server-side quota enforcement. Detector eligibility comes
  from the central registry; keyed, benchmark, and model-backed paths are
  reported but never executed here.
- ``GET /v1/public/quota`` — remaining-quota metadata for the homepage.
- ``POST /v1/auth/sync`` — post-sign-in provisioning (Supabase JWT).
- ``GET /v1/me`` — identity + entitlements for the frontend shell.

Quota is consumed only by *successful* analyses.  Validation failures
and internal errors never consume quota.  Raw text is never persisted
by these routes (only aggregate counts in ``usage_events``).
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from provenance.api.entitlements import entitlements_for_plan
from provenance.api.identity import (
    COOKIE_NAME,
    abuse_signal_hash,
    coarse_ip_prefix,
    cookie_domain,
    cookie_samesite,
    cookie_secure,
    new_visitor_id,
    sign_visitor_id,
    verify_visitor_cookie,
)
from provenance.api.middleware import rate_limit_dependency
from provenance.api.models import (
    AuthSyncResponse,
    ErrorResponse,
    MeResponse,
    PublicAnalyzeRequest,
    PublicAnalyzeResponse,
    QuotaInfo,
)
from provenance.api.state import get_repo
from provenance.api.supabase_auth import (
    OptionalSupabaseUser,
    RequireSupabaseUser,
    supabase_configured,
)

router = APIRouter()

DISCLAIMER = (
    "This public result is a limited provenance-signal scan, not a universal "
    "AI-text detector. A clean result does not establish human authorship or "
    "the absence of every watermark."
)
PUBLIC_LIMITATIONS = [
    "The public analyzer checks hidden or unusual Unicode artifacts only.",
    "KGW and SynthID evidence is meaningful only against a matching known key, parameters, variant, and tokenizer configuration.",
    "The result cannot attribute text to a model or prove AI or human authorship.",
]
UPGRADE_HINT = (
    "Sign in to use the existing detailed analysis workflow; configured "
    "watermark verification and benchmark access remain entitlement-dependent."
)
_QUOTA_EXHAUSTED_HINT = (
    "Daily free limit reached. Sign in for higher limits, or try again tomorrow."
)


def _client_ip(request: Request) -> str:
    if request.client:
        ip = request.client.host
    else:
        ip = "unknown"
    xff = request.headers.get("x-forwarded-for")
    if xff:
        ip = xff.split(",")[0].strip()
    return ip


def _abuse_hash(request: Request) -> str:
    return abuse_signal_hash(
        coarse_ip_prefix(_client_ip(request)),
        (request.headers.get("user-agent") or "")[:200],
    )


def _resolve_visitor(request: Request) -> tuple[str, bool]:
    """Return (visitor_id, is_new).  Creates a fresh ID when missing/invalid."""
    incoming = request.cookies.get(COOKIE_NAME)
    verified = verify_visitor_cookie(incoming)
    if verified:
        return verified, False
    return new_visitor_id(), True


def _set_visitor_cookie(response: Response, visitor_id: str) -> None:
    response.set_cookie(
        COOKIE_NAME,
        sign_visitor_id(visitor_id),
        max_age=365 * 24 * 3600,
        path="/",
        domain=cookie_domain(),
        secure=cookie_secure(),
        httponly=True,
        samesite=cookie_samesite(),
    )


def _quota_info(plan: str, limit: int, used: int, max_chars: int) -> QuotaInfo:
    used = min(used, limit)
    return QuotaInfo(
        plan=plan,
        limit=limit,
        used=used,
        remaining=limit - used,
        max_chars_per_analysis=max_chars,
    )


def _quota_headers(quota: QuotaInfo) -> dict[str, str]:
    return {
        "X-Quota-Plan": quota.plan,
        "X-Quota-Limit": str(quota.limit),
        "X-Quota-Used": str(quota.used),
        "X-Quota-Remaining": str(quota.remaining),
    }


def _identity_context(
    request: Request, claims: dict[str, Any] | None
) -> tuple[str | None, str | None, str, str, bool]:
    """Resolve (visitor_id, auth_user_id, plan, email, visitor_is_new)."""
    visitor_id, is_new = _resolve_visitor(request)
    repo = get_repo()
    repo.get_or_create_visitor(visitor_id, _abuse_hash(request))
    if claims is None:
        return visitor_id, None, "anonymous", "", is_new
    auth_user_id = str(claims.get("sub", ""))
    email = claims.get("email") if isinstance(claims.get("email"), str) else None
    user = repo.get_or_create_user(auth_user_id, email)
    return visitor_id, auth_user_id, str(user.get("plan", "free")), email or "", is_new


@router.get(
    "/v1/public/quota",
    response_model=QuotaInfo,
    summary="Public quota status",
    tags=["public"],
)
def public_quota(
    request: Request,
    response: Response,
    claims: dict[str, Any] | None = OptionalSupabaseUser,
    _rl: None = Depends(rate_limit_dependency),
) -> QuotaInfo:
    visitor_id, auth_user_id, plan, _email, is_new = _identity_context(request, claims)
    ent = entitlements_for_plan(plan)
    used = get_repo().count_public_uses_today(visitor_id=visitor_id, auth_user_id=auth_user_id)
    if is_new:
        _set_visitor_cookie(response, visitor_id)
    quota = _quota_info(
        plan, ent.max_daily_analyses, used, ent.max_chars_per_analysis
    )
    response.headers.update(_quota_headers(quota))
    return quota


@router.post(
    "/v1/public/analyze",
    response_model=PublicAnalyzeResponse,
    summary="Public text analysis (quota-enforced)",
    description=(
        "Analyze text with cheap detectors valid for arbitrary input and report "
        "why configured watermark detectors were not run. "
        "No API key required. Anonymous visitors are limited to "
        "PROVENANCE_PUBLIC_DAILY_LIMIT successful analyses per UTC day."
    ),
    tags=["public"],
    responses={
        413: {"model": ErrorResponse, "description": "Text exceeds the plan limit"},
        422: {"model": ErrorResponse, "description": "Invalid request"},
        429: {"model": ErrorResponse, "description": "Daily quota exhausted"},
    },
)
def public_analyze(
    payload: PublicAnalyzeRequest,
    request: Request,
    response: Response,
    claims: dict[str, Any] | None = OptionalSupabaseUser,
    _rl: None = Depends(rate_limit_dependency),
) -> PublicAnalyzeResponse:
    visitor_id, auth_user_id, plan, _email, is_new = _identity_context(request, claims)
    ent = entitlements_for_plan(plan)
    repo = get_repo()

    text = payload.text
    if len(text) > ent.max_chars_per_analysis:
        raise HTTPException(
            status_code=413,
            detail=f"Text too long ({len(text)} chars); this plan allows "
            f"{ent.max_chars_per_analysis} characters per analysis",
        )

    used = repo.count_public_uses_today(visitor_id=visitor_id, auth_user_id=auth_user_id)
    if used >= ent.max_daily_analyses:
        # NOTE: headers must ride on the exception — the injected response
        # object is discarded when raising.
        raise HTTPException(
            status_code=429,
            detail=_QUOTA_EXHAUSTED_HINT,
            headers=_quota_headers(_quota_info(
                plan, ent.max_daily_analyses, used, ent.max_chars_per_analysis
            )),
        )

    from provenance.api.service import run_public_analysis

    try:
        result_dict = run_public_analysis(text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=500, detail="Analysis failed; please try again")

    # Consume quota only now that the analysis succeeded.  A lost race
    # surfaces as 429 without exceeding the limit.
    char_count = result_dict.get("character_count", len(text))
    consumed, remaining = repo.try_consume_public_use(
        visitor_id=visitor_id,
        auth_user_id=auth_user_id,
        chars=char_count,
        limit=ent.max_daily_analyses,
    )
    if not consumed:
        raise HTTPException(
            status_code=429,
            detail=_QUOTA_EXHAUSTED_HINT,
            headers=_quota_headers(_quota_info(
                plan,
                ent.max_daily_analyses,
                ent.max_daily_analyses,
                ent.max_chars_per_analysis,
            )),
        )

    quota = _quota_info(
        plan,
        ent.max_daily_analyses,
        ent.max_daily_analyses - remaining,
        ent.max_chars_per_analysis,
    )
    response.headers.update(_quota_headers(quota))
    if is_new:
        _set_visitor_cookie(response, visitor_id)
    return PublicAnalyzeResponse(
        overall_result=result_dict["overall_result"],
        verdict=result_dict["verdict"],
        signals_checked=result_dict["signals_checked"],
        signals_detected=result_dict["signals_detected"],
        unavailable_detectors=result_dict["unavailable_detectors"],
        character_count=char_count,
        quota=quota,
        limitations=PUBLIC_LIMITATIONS,
        disclaimer=DISCLAIMER,
        upgrade_hint=UPGRADE_HINT,
    )


@router.post(
    "/v1/auth/sync",
    response_model=AuthSyncResponse,
    summary="Provision user after sign-in",
    description=(
        "Verify the Supabase session (Bearer JWT), create the application "
        "user row on first sign-in, and migrate unclaimed anonymous usage "
        "so the daily quota is not reset by signing up."
    ),
    tags=["auth"],
    responses={
        401: {"model": ErrorResponse, "description": "Invalid or missing session"},
        503: {"model": ErrorResponse, "description": "Auth not configured"},
    },
)
def auth_sync(
    request: Request,
    claims: dict[str, Any] = RequireSupabaseUser,
    _rl: None = Depends(rate_limit_dependency),
) -> AuthSyncResponse:
    auth_user_id = str(claims.get("sub", ""))
    email = claims.get("email") if isinstance(claims.get("email"), str) else None
    repo = get_repo()
    user = repo.get_or_create_user(auth_user_id, email)
    plan = str(user.get("plan", "free"))
    claimed = 0
    visitor_id = verify_visitor_cookie(request.cookies.get(COOKIE_NAME))
    if visitor_id:
        claimed = repo.claim_visitor_events(visitor_id, auth_user_id)
    ent = entitlements_for_plan(plan)
    used = repo.count_public_uses_today(visitor_id=visitor_id, auth_user_id=auth_user_id)
    return AuthSyncResponse(
        auth_user_id=auth_user_id,
        email=email,
        plan=plan,
        claimed_events=claimed,
        quota=_quota_info(
            plan, ent.max_daily_analyses, used, ent.max_chars_per_analysis
        ),
    )


@router.get(
    "/v1/me",
    response_model=MeResponse,
    summary="Current identity and entitlements",
    tags=["auth"],
)
def me(
    request: Request,
    claims: dict[str, Any] | None = OptionalSupabaseUser,
    _rl: None = Depends(rate_limit_dependency),
) -> MeResponse:
    visitor_id, auth_user_id, plan, email, _is_new = _identity_context(request, claims)
    ent = entitlements_for_plan(plan)
    used = get_repo().count_public_uses_today(visitor_id=visitor_id, auth_user_id=auth_user_id)
    return MeResponse(
        kind="user" if auth_user_id else "anonymous",
        auth_user_id=auth_user_id,
        email=email or None,
        plan=plan,
        quota=_quota_info(
            plan, ent.max_daily_analyses, used, ent.max_chars_per_analysis
        ),
        entitlements=asdict(ent),
    )


def auth_configured_payload() -> dict[str, Any]:
    """Public feature flags for the frontend shell (no secrets)."""
    return {"supabase_configured": supabase_configured()}


@router.get(
    "/v1/public/config",
    summary="Public feature flags",
    tags=["public"],
)
def public_config() -> dict[str, Any]:
    return auth_configured_payload()
