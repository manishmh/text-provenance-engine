"""Razorpay recurring-subscription billing service.

The internal service normalizes provider state into the existing subscription
tables. Razorpay is the sole production payment provider for Product v2.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import HTTPException

from provenance.api.db import AnalysisRepository

logger = logging.getLogger("provenance.api.billing")
PROVIDER_RAZORPAY = "razorpay"
PRO_STATUSES = frozenset({"active"})


class BillingConfigurationError(RuntimeError):
    """Optional billing has no usable configuration for the requested provider."""


@dataclass(frozen=True)
class CheckoutSession:
    provider: str
    checkout_url: str | None = None
    checkout_data: dict[str, Any] | None = None


class BillingProvider(Protocol):
    name: str
    checkout_mode: str
    pro_plan_id: str
    def create_customer(self, *, email: str | None, auth_user_id: str) -> str | None: ...
    def create_checkout_session(self, *, customer_id: str | None, plan_id: str, auth_user_id: str) -> tuple[CheckoutSession, dict[str, Any]]: ...
    def checkout_for_existing_subscription(self, subscription_id: str) -> CheckoutSession | None: ...
    def cancel_at_period_end(self, subscription_id: str) -> dict[str, Any]: ...


class RazorpayBillingProvider:
    """Razorpay's recurring Subscription API; never creates one-time orders."""
    name = PROVIDER_RAZORPAY
    checkout_mode = "razorpay_checkout"
    _base_url = "https://api.razorpay.com/v1"
    def __init__(self) -> None:
        self.key_id = os.environ.get("RAZORPAY_KEY_ID", "").strip()
        self.key_secret = os.environ.get("RAZORPAY_KEY_SECRET", "").strip()
        self.pro_plan_id = os.environ.get("RAZORPAY_PRO_PLAN_ID", "").strip()
        try: self.total_count = int(os.environ.get("RAZORPAY_PRO_TOTAL_COUNT", "120"))
        except ValueError: self.total_count = 0
        if not all((self.key_id, self.key_secret, self.pro_plan_id)) or self.total_count < 1:
            raise BillingConfigurationError("Razorpay billing is not configured")
    def _request(self, method: str, path: str, values: dict[str, Any] | None = None) -> dict[str, Any]:
        token = base64.b64encode(f"{self.key_id}:{self.key_secret}".encode()).decode()
        request = Request(f"{self._base_url}{path}", data=json.dumps(values).encode() if values is not None else None, method=method, headers={"Authorization": f"Basic {token}", "Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=15) as response:  # nosec B310 fixed provider host
                return json.loads(response.read().decode())
        except (HTTPError, URLError, ValueError) as exc:
            logger.warning("Billing provider request failed provider=razorpay operation=%s error_type=%s", path, type(exc).__name__)
            raise HTTPException(status_code=502, detail="Billing provider is temporarily unavailable") from exc
    def create_customer(self, *, email: str | None, auth_user_id: str) -> None:
        # customer_id is populated by Razorpay after Checkout authorization.
        return None
    def _checkout(self, subscription_id: str) -> CheckoutSession:
        return CheckoutSession(provider=self.name, checkout_data={"key_id": self.key_id, "subscription_id": subscription_id, "name": "Text Provenance Engine", "description": "Pro subscription"})
    def create_checkout_session(self, *, customer_id: str | None, plan_id: str, auth_user_id: str) -> tuple[CheckoutSession, dict[str, Any]]:
        response = self._request("POST", "/subscriptions", {"plan_id": plan_id, "total_count": self.total_count, "quantity": 1, "customer_notify": True, "notes": {"auth_user_id": auth_user_id, "product_plan": "pro"}})
        subscription_id = str(response["id"])
        return self._checkout(subscription_id), {"provider_subscription_id": subscription_id, "provider_customer_id": str(response.get("customer_id") or "pending"), "status": str(response.get("status") or "created"), "current_period_start": _unix_to_iso(response.get("current_start")), "current_period_end": _unix_to_iso(response.get("current_end"))}
    def checkout_for_existing_subscription(self, subscription_id: str) -> CheckoutSession | None:
        return self._checkout(subscription_id)
    def cancel_at_period_end(self, subscription_id: str) -> dict[str, Any]:
        return self._request("POST", f"/subscriptions/{subscription_id}/cancel", {"cancel_at_cycle_end": True})


def _unix_to_iso(value: Any) -> str | None:
    return datetime.fromtimestamp(value, timezone.utc).isoformat() if isinstance(value, (int, float)) else None


def _verify_hmac(payload: bytes, signature: str | None, secret: str, provider: str) -> None:
    if not secret or not signature: raise HTTPException(status_code=400, detail=f"Invalid {provider} signature")
    expected = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature): raise HTTPException(status_code=400, detail=f"Invalid {provider} signature")


def verify_razorpay_webhook_signature(payload: bytes, signature: str | None) -> None:
    """HMAC-SHA256 of Razorpay's exact raw webhook body."""
    _verify_hmac(payload, signature, os.environ.get("RAZORPAY_WEBHOOK_SECRET", "").strip(), "Razorpay")


def subscription_grants_pro(subscription: dict[str, Any] | None) -> bool:
    if not subscription or str(subscription.get("status")) not in PRO_STATUSES: return False
    if not subscription.get("cancel_at_period_end"): return True
    try: return datetime.fromisoformat(str(subscription.get("current_period_end")).replace("Z", "+00:00")) > datetime.now(timezone.utc)
    except ValueError: return False


def effective_plan(repo: AnalysisRepository, user: dict[str, Any]) -> str:
    subscription = repo.get_subscription_for_user(str(user["id"]))
    if subscription is not None: return "pro" if subscription_grants_pro(subscription) else "free"
    return "pro" if user.get("plan") == "pro" else "free"  # test/admin compatibility only


class BillingService:
    def __init__(self, repo: AnalysisRepository, provider: BillingProvider) -> None:
        self.repo, self.provider = repo, provider
    def start_checkout(self, user: dict[str, Any]) -> CheckoutSession:
        existing = self.repo.get_subscription_for_user(str(user["id"]))
        if existing and existing.get("provider") == self.provider.name and existing.get("provider_subscription_id") and str(existing.get("status")) in {"created", "authenticated", "pending", "halted"}:
            resumed = self.provider.checkout_for_existing_subscription(str(existing["provider_subscription_id"]))
            if resumed: return resumed
        customer_id = existing.get("provider_customer_id") if existing and existing.get("provider") == self.provider.name else None
        if not customer_id or customer_id == "pending": customer_id = self.provider.create_customer(email=user.get("email"), auth_user_id=str(user["auth_user_id"]))
        checkout, details = self.provider.create_checkout_session(customer_id=customer_id, plan_id=self.provider.pro_plan_id, auth_user_id=str(user["auth_user_id"]))
        self.repo.upsert_subscription(user_id=str(user["id"]), provider=self.provider.name, provider_customer_id=str(details.get("provider_customer_id") or customer_id or "pending"), provider_subscription_id=details.get("provider_subscription_id"), provider_price_id=self.provider.pro_plan_id, status=str(details.get("status") or "pending"), current_period_start=details.get("current_period_start"), current_period_end=details.get("current_period_end"), cancel_at_period_end=False)
        return checkout
    def cancel(self, user: dict[str, Any]) -> dict[str, str]:
        sub = self.repo.get_subscription_for_user(str(user["id"]))
        if not sub or not sub.get("provider_subscription_id"): raise HTTPException(status_code=404, detail="No active subscription exists for this account")
        response = self.provider.cancel_at_period_end(str(sub["provider_subscription_id"]))
        self.repo.upsert_subscription(user_id=str(sub["user_id"]), provider=self.provider.name, provider_customer_id=str(response.get("customer_id") or sub["provider_customer_id"]), provider_subscription_id=str(response.get("id") or sub["provider_subscription_id"]), provider_price_id=sub.get("provider_price_id"), status=str(response.get("status") or sub["status"]), current_period_start=_unix_to_iso(response.get("current_start") or response.get("current_period_start")) or sub.get("current_period_start"), current_period_end=_unix_to_iso(response.get("current_end") or response.get("current_period_end")) or sub.get("current_period_end"), cancel_at_period_end=True)
        return {"status": "cancellation_scheduled"}
    def process_event(self, event: dict[str, Any], *, event_id: str | None = None) -> bool:
        return self._process_razorpay_event(event, event_id)
    def _record(self, event_id: str, event_type: str) -> bool:
        if not event_id or not event_type: raise HTTPException(status_code=400, detail="Malformed billing event")
        return self.repo.record_webhook_event(provider=self.provider.name, event_id=event_id, event_type=event_type)
    def _process_razorpay_event(self, event: dict[str, Any], header_id: str | None) -> bool:
        event_type = str(event.get("event", ""))
        if not self._record(str(header_id or ""), event_type): return False
        payload = event.get("payload", {}); sub = payload.get("subscription", {}).get("entity", {}) if isinstance(payload, dict) else {}
        if not isinstance(sub, dict) or not event_type.startswith("subscription."): return True
        customer, subscription_id = str(sub.get("customer_id") or "pending"), str(sub.get("id") or "")
        old = self.repo.get_subscription_by_provider_customer(self.provider.name, customer) if customer != "pending" else None
        if old: user_id = str(old["user_id"])
        else:
            auth = str((sub.get("notes") or {}).get("auth_user_id") or "")
            if not auth: logger.warning("Billing subscription event has no internal mapping provider=razorpay event_type=%s", event_type); return True
            user_id = str(self.repo.get_or_create_user(auth, None)["id"])
        self.repo.upsert_subscription(user_id=user_id, provider=self.provider.name, provider_customer_id=customer, provider_subscription_id=subscription_id or None, provider_price_id=str(sub.get("plan_id")) if sub.get("plan_id") else None, status=str(sub.get("status") or "unknown"), current_period_start=_unix_to_iso(sub.get("current_start")), current_period_end=_unix_to_iso(sub.get("current_end")), cancel_at_period_end=False)
        return True


_provider_override: BillingProvider | None = None
def get_billing_provider() -> BillingProvider:
    if _provider_override is not None:
        return _provider_override
    return RazorpayBillingProvider()
def billing_status() -> dict[str, Any]:
    try:
        provider = get_billing_provider(); return {"provider": provider.name, "configured": True, "checkout_mode": provider.checkout_mode}
    except BillingConfigurationError:
        return {"provider": PROVIDER_RAZORPAY, "configured": False, "checkout_mode": None}
def set_billing_provider_for_testing(provider: BillingProvider | None) -> None:
    global _provider_override; _provider_override = provider
