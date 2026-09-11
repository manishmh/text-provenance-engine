"""Authenticated Razorpay billing endpoints and signed webhook endpoint."""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from provenance.api.billing import (
    BillingConfigurationError, BillingService, billing_status, effective_plan,
    get_billing_provider, verify_razorpay_webhook_signature,
)
from provenance.api.models import (
    BillingCheckoutRequest, BillingCheckoutResponse,
    BillingStatusResponse, BillingSubscriptionResponse, ErrorResponse,
)
from provenance.api.supabase_auth import RequireSupabaseUser
from provenance.api.state import get_repo

router = APIRouter(prefix="/v1/billing", tags=["billing"])


def _user_from_claims(claims: dict[str, Any]) -> dict[str, Any]:
    auth_user_id = str(claims.get("sub", ""))
    if not auth_user_id:
        raise HTTPException(status_code=401, detail="Sign-in required")
    email = claims.get("email") if isinstance(claims.get("email"), str) else None
    return get_repo().get_or_create_user(auth_user_id, email)


def _service() -> BillingService:
    try:
        return BillingService(get_repo(), get_billing_provider())
    except BillingConfigurationError as exc:
        raise HTTPException(status_code=503, detail="Billing is not configured") from exc


@router.post("/checkout", response_model=BillingCheckoutResponse, responses={401: {"model": ErrorResponse}})
def checkout(payload: BillingCheckoutRequest, claims: dict[str, Any] = RequireSupabaseUser) -> BillingCheckoutResponse:
    if payload.plan != "pro":
        raise HTTPException(status_code=422, detail="Only the configured Pro plan is available")
    user = _user_from_claims(claims)
    checkout_session = _service().start_checkout(user)
    return BillingCheckoutResponse(
        provider=checkout_session.provider, checkout_url=checkout_session.checkout_url,
        checkout_data=checkout_session.checkout_data,
    )


@router.get("/status", response_model=BillingStatusResponse)
def status() -> BillingStatusResponse:
    """Safe optional-billing status for the public pricing UI."""
    return BillingStatusResponse(**billing_status())


@router.get("/subscription", response_model=BillingSubscriptionResponse, responses={401: {"model": ErrorResponse}})
def subscription(claims: dict[str, Any] = RequireSupabaseUser) -> BillingSubscriptionResponse:
    user = _user_from_claims(claims)
    stored = get_repo().get_subscription_for_user(str(user["id"]))
    # Do not expose provider customer/subscription IDs to browsers.
    safe = None
    if stored:
        safe = {
            "provider": stored["provider"], "status": stored["status"],
            "current_period_end": stored["current_period_end"],
            "cancel_at_period_end": stored["cancel_at_period_end"],
        }
    return BillingSubscriptionResponse(
        plan=effective_plan(get_repo(), user), subscription=safe,
        webhook_processing=bool(stored and stored["status"] == "pending"),
    )


@router.post("/cancel", responses={401: {"model": ErrorResponse}})
def cancel_subscription(claims: dict[str, Any] = RequireSupabaseUser) -> dict[str, str]:
    return _service().cancel(_user_from_claims(claims))


@router.post("/webhook/razorpay", status_code=200, include_in_schema=False)
async def razorpay_webhook(request: Request) -> dict[str, bool]:
    payload = await request.body()
    verify_razorpay_webhook_signature(payload, request.headers.get("x-razorpay-signature"))
    try:
        event = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise HTTPException(status_code=400, detail="Malformed Razorpay event") from None
    if not isinstance(event, dict):
        raise HTTPException(status_code=400, detail="Malformed Razorpay event")
    try:
        service = BillingService(get_repo(), get_billing_provider())
    except BillingConfigurationError as exc:
        raise HTTPException(status_code=503, detail="Razorpay billing is not configured") from exc
    processed = service.process_event(event, event_id=request.headers.get("x-razorpay-event-id"))
    return {"received": True, "processed": processed}
