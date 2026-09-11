"""Razorpay recurring billing uses fakes and signed raw-body webhooks only."""
from __future__ import annotations

import hashlib
import hmac
import json
import time

import jwt
import pytest
from fastapi.testclient import TestClient

from provenance.api.app import create_app
from provenance.api.billing import (
    CheckoutSession, RazorpayBillingProvider,
    get_billing_provider, set_billing_provider_for_testing,
)

SUPABASE_URL = "https://example.supabase.co"
SUPABASE_SECRET = "razorpay-test-jwt"
WEBHOOK_SECRET = "razorpay-webhook-test"


class FakeRazorpay:
    name = "razorpay"
    checkout_mode = "razorpay_checkout"
    pro_plan_id = "plan_pro_test"
    def __init__(self): self.created = 0; self.cancelled = []
    def create_customer(self, *, email, auth_user_id): return None
    def create_checkout_session(self, *, customer_id, plan_id, auth_user_id):
        self.created += 1
        return CheckoutSession(provider="razorpay", checkout_data={"key_id": "rzp_test_public", "subscription_id": "sub_rzp_1", "name": "Text Provenance Engine", "description": "Pro subscription"}), {
            "provider_subscription_id": "sub_rzp_1", "provider_customer_id": "pending", "status": "created",
        }
    def checkout_for_existing_subscription(self, subscription_id):
        return CheckoutSession(provider="razorpay", checkout_data={"key_id": "rzp_test_public", "subscription_id": subscription_id, "name": "Text Provenance Engine", "description": "Pro subscription"})
    def cancel_at_period_end(self, subscription_id):
        self.cancelled.append(subscription_id)
        return {"id": subscription_id, "status": "active", "current_end": int(time.time()) + 86400}


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", SUPABASE_URL)
    monkeypatch.setenv("SUPABASE_JWT_SECRET", SUPABASE_SECRET)
    monkeypatch.setenv("RAZORPAY_WEBHOOK_SECRET", WEBHOOK_SECRET)
    monkeypatch.setenv("RATE_LIMIT_MAX_REQUESTS", "100000")
    fake = FakeRazorpay(); set_billing_provider_for_testing(fake)
    yield fake
    set_billing_provider_for_testing(None)


@pytest.fixture()
def client(tmp_path):
    with TestClient(create_app(db_url=f"sqlite:///{tmp_path}/razorpay.db")) as c: yield c


def token():
    now = int(time.time())
    return jwt.encode({"sub": "razorpay-user", "email": "rzp@example.com", "aud": "authenticated", "iss": f"{SUPABASE_URL}/auth/v1", "exp": now + 3600}, SUPABASE_SECRET, algorithm="HS256")


def headers(): return {"Authorization": f"Bearer {token()}"}


def sign(body: bytes, event_id="evt_rzp_1"):
    return {"x-razorpay-event-id": event_id, "x-razorpay-signature": hmac.new(WEBHOOK_SECRET.encode(), body, hashlib.sha256).hexdigest()}


def event(status: str, event_name="subscription.activated"):
    now = int(time.time())
    return {"entity": "event", "event": event_name, "payload": {"subscription": {"entity": {
        "id": "sub_rzp_1", "customer_id": "cust_rzp_1", "plan_id": "plan_pro_test", "status": status,
        "current_start": now - 60, "current_end": now + 86400,
        "notes": {"auth_user_id": "razorpay-user", "product_plan": "pro"},
    }}}}


def test_provider_factory_selects_razorpay(monkeypatch):
    set_billing_provider_for_testing(None)
    monkeypatch.setenv("RAZORPAY_KEY_ID", "rzp_test_public")
    monkeypatch.setenv("RAZORPAY_KEY_SECRET", "secret")
    monkeypatch.setenv("RAZORPAY_PRO_PLAN_ID", "plan_test")
    assert isinstance(get_billing_provider(), RazorpayBillingProvider)


def test_missing_razorpay_config_is_safe_and_does_not_break_health(client, monkeypatch):
    set_billing_provider_for_testing(None)
    for key in ("RAZORPAY_KEY_ID", "RAZORPAY_KEY_SECRET", "RAZORPAY_PRO_PLAN_ID"):
        monkeypatch.delenv(key, raising=False)
    assert client.get("/health").status_code == 200
    assert client.get("/v1/billing/status").json() == {"provider": "razorpay", "configured": False, "checkout_mode": None}
    assert client.post("/v1/billing/checkout", json={"plan": "pro"}, headers=headers()).status_code == 503


def test_razorpay_checkout_is_safe_and_reuses_pending_subscription(client, env):
    body = client.post("/v1/billing/checkout", json={"plan": "pro"}, headers=headers()).json()
    assert body == {"provider": "razorpay", "checkout_url": None, "checkout_data": {"key_id": "rzp_test_public", "subscription_id": "sub_rzp_1", "name": "Text Provenance Engine", "description": "Pro subscription"}}
    assert "secret" not in json.dumps(body).lower() and "plan_pro_test" not in json.dumps(body)
    assert client.post("/v1/billing/checkout", json={"plan": "pro"}, headers=headers()).status_code == 200
    assert env.created == 1


def test_razorpay_raw_webhook_activation_duplicate_failure_and_cancel(client, env):
    client.post("/v1/billing/checkout", json={"plan": "pro"}, headers=headers())
    raw = json.dumps(event("active"), separators=(",", ":")).encode()
    assert client.post("/v1/billing/webhook/razorpay", content=raw, headers=sign(raw)).json()["processed"] is True
    assert client.get("/v1/billing/subscription", headers=headers()).json()["plan"] == "pro"
    assert client.post("/v1/billing/webhook/razorpay", content=raw, headers=sign(raw)).json()["processed"] is False
    assert client.post("/v1/billing/cancel", headers=headers()).json() == {"status": "cancellation_scheduled"}
    assert env.cancelled == ["sub_rzp_1"]
    failed = json.dumps(event("pending", "subscription.pending"), separators=(",", ":")).encode()
    assert client.post("/v1/billing/webhook/razorpay", content=failed, headers=sign(failed, "evt_rzp_2")).status_code == 200
    assert client.get("/v1/billing/subscription", headers=headers()).json()["plan"] == "free"


def test_razorpay_invalid_signature_never_processes(client):
    raw = json.dumps(event("active")).encode()
    assert client.post("/v1/billing/webhook/razorpay", content=raw, headers={"x-razorpay-event-id": "evt_bad", "x-razorpay-signature": "bad"}).status_code == 400
