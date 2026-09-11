"""Tests for Product v2 foundation: public analyze, anonymous quota,
Supabase JWT auth, entitlements, and the SQLite URL fix. No network,
no HF downloads."""
from __future__ import annotations

import threading
import time

import jwt as pyjwt
import pytest
from fastapi.testclient import TestClient

from provenance.api.app import create_app
from provenance.api.db import create_repository

SUPABASE_URL = "https://example.supabase.co"
SUPABASE_SECRET = "test-jwt-secret-for-phase7a"


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.delenv("PROVENANCE_API_KEY", raising=False)
    monkeypatch.delenv("PROVENANCE_ADMIN_API_KEY", raising=False)
    monkeypatch.setenv("PROVENANCE_ANON_COOKIE_SECRET", "test-cookie-secret")
    monkeypatch.setenv("PROVENANCE_PUBLIC_DAILY_LIMIT", "2")
    monkeypatch.setenv("PROVENANCE_PUBLIC_MAX_CHARS", "5000")
    monkeypatch.setenv("RATE_LIMIT_MAX_REQUESTS", "100000")
    monkeypatch.setenv("SUPABASE_URL", SUPABASE_URL)
    monkeypatch.setenv("SUPABASE_JWT_SECRET", SUPABASE_SECRET)
    import provenance.api.middleware as middleware_mod
    monkeypatch.setattr(middleware_mod, "_limiter", None)


@pytest.fixture()
def client(tmp_path):
    app = create_app(db_url=f"sqlite:///{tmp_path}/test.db")
    with TestClient(app) as c:
        yield c


def _mint(sub="user-1", email="u@example.com", secret=SUPABASE_SECRET, exp_in=3600):
    now = int(time.time())
    return pyjwt.encode(
        {"sub": sub, "email": email, "aud": "authenticated",
         "iss": f"{SUPABASE_URL}/auth/v1", "iat": now, "exp": now + exp_in},
        secret, algorithm="HS256",
    )


def _visitor_cookie(client):
    for c in client.cookies.jar:
        if c.name == "pv_visitor":
            return c.value
    return None


class TestVisitorIdentity:
    def test_quota_issues_signed_httponly_cookie(self, client):
        r = client.get("/v1/public/quota")
        assert r.status_code == 200
        assert r.json() == {
            "plan": "anonymous", "limit": 2, "used": 0, "remaining": 2,
            "max_chars_per_analysis": 5000,
        }
        raw = r.headers.get("set-cookie", "")
        assert "pv_visitor=" in raw and "HttpOnly" in raw and "samesite=lax" in raw.lower()
        assert _visitor_cookie(client) is not None

    def test_tampered_cookie_rejected_and_rotated(self, client):
        client.get("/v1/public/quota")
        good = _visitor_cookie(client)
        assert good
        client.cookies.clear()
        client.cookies.set("pv_visitor", good + "tampered", domain="testserver", path="/")
        r = client.get("/v1/public/quota")
        assert r.status_code == 200
        rotated = r.headers.get("set-cookie", "")
        assert "pv_visitor=" in rotated and good + "tampered" not in rotated
        assert r.json()["used"] == 0  # attacker gains no quota history

    def test_cookie_value_is_signed(self, client):
        client.get("/v1/public/quota")
        value = _visitor_cookie(client)
        assert value and value.count(".") >= 2


class TestQuota:
    def test_two_per_day_third_blocked(self, client):
        body = {"text": "Plain text for quota testing."}
        assert client.post("/v1/public/analyze", json=body).status_code == 200
        r2 = client.post("/v1/public/analyze", json=body)
        assert r2.status_code == 200
        assert r2.json()["quota"] == {
            "plan": "anonymous", "limit": 2, "used": 2, "remaining": 0,
            "max_chars_per_analysis": 5000,
        }
        r3 = client.post("/v1/public/analyze", json=body)
        assert r3.status_code == 429
        assert r3.headers["X-Quota-Remaining"] == "0"

    def test_validation_failure_does_not_consume(self, client):
        assert client.post("/v1/public/analyze", json={"text": ""}).status_code == 422
        r = client.get("/v1/public/quota")
        assert r.json()["used"] == 0

    def test_oversize_text_rejected_without_consuming(self, client):
        r = client.post("/v1/public/analyze", json={"text": "x" * 5001})
        assert r.status_code == 413
        assert client.get("/v1/public/quota").json()["used"] == 0

    def test_internal_analysis_failure_does_not_consume(self, client, monkeypatch, caplog):
        import provenance.api.service as service

        monkeypatch.setattr(
            service,
            "run_public_analysis",
            lambda _text: (_ for _ in ()).throw(RuntimeError("test failure")),
        )
        r = client.post("/v1/public/analyze", json={"text": "private input"})
        assert r.status_code == 500
        assert client.get("/v1/public/quota").json()["used"] == 0
        messages = "\n".join(record.getMessage() for record in caplog.records)
        assert "public_analysis_failed stage=analysis error_type=RuntimeError" in messages
        assert "private input" not in messages

    def test_concurrent_requests_cannot_exceed_quota(self, tmp_path):
        from provenance.api.db import SqliteRepository
        repo = SqliteRepository(str(tmp_path / "race.db"))
        results = []

        def worker():
            results.append(repo.try_consume_public_use(
                visitor_id="v-race", auth_user_id=None, chars=5, limit=2))

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sum(1 for ok, _ in results if ok) == 2
        assert repo.count_public_uses_today(visitor_id="v-race", auth_user_id=None) == 2

    def test_public_result_has_no_secrets_or_debug(self, client):
        r = client.post("/v1/public/analyze", json={"text": "hello world"})
        assert r.status_code == 200
        body = r.json()
        assert set(body) == {
            "overall_result", "verdict", "signals_checked", "signals_detected",
            "unavailable_detectors", "character_count", "quota", "limitations",
            "disclaimer", "upgrade_hint",
        }
        for sig in body["signals_checked"]:
            assert set(sig) == {
                "detector", "display_name", "signal_type", "status", "detected",
                "confidence", "evidence", "limitations",
            }
        serialized = str(body).lower()
        for forbidden in (
            SUPABASE_SECRET.lower(), "hash_key", "raw_debug", "config_path",
            "threshold", "z_score", "token_ids",
        ):
            assert forbidden not in serialized


class TestPublicAnalysisCapability:
    def test_public_unicode_path_never_imports_reference_or_torch(self, client, monkeypatch):
        """Regression for the slim production image, which has no torch."""
        import builtins

        original_import = builtins.__import__

        def reject_heavy_import(name, *args, **kwargs):
            if name == "torch" or name.startswith("provenance.detectors.reference"):
                raise AssertionError(f"public Unicode analysis imported {name}")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", reject_heavy_import)
        response = client.post("/v1/public/analyze", json={"text": "ordinary visible text"})
        assert response.status_code == 200

    def test_arbitrary_normal_text_checks_only_public_unicode(self, client):
        body = client.post(
            "/v1/public/analyze", json={"text": "Ordinary visible text."}
        ).json()
        assert body["overall_result"] == "no_supported_signal_detected"
        assert body["signals_detected"] == []
        assert [signal["detector"] for signal in body["signals_checked"]] == ["unicode"]
        signal = body["signals_checked"][0]
        assert signal["signal_type"] == "hidden_unicode_provenance"
        assert signal["status"] == "not_detected"
        assert signal["evidence"] == ["0 hidden or unusual Unicode artifacts found."]
        assert "were not tested" in body["verdict"]

    def test_unicode_signal_text_reports_hidden_signal_not_ai_authorship(self, client):
        body = client.post(
            "/v1/public/analyze", json={"text": "hello\u200bworld"}
        ).json()
        assert body["overall_result"] == "signal_detected"
        assert body["signals_detected"] == ["unicode"]
        assert body["signals_checked"][0]["status"] == "detected"
        assert "Unicode" in body["verdict"]
        assert "AI-authorship determination" in body["verdict"]

    def test_only_cheap_arbitrary_input_path_is_executed(self, client, monkeypatch):
        import provenance.api.service as service

        calls = []
        original = service.run_analysis

        def recording_run(text, detector_names, config_path):
            calls.append((detector_names, config_path))
            return original(text, detector_names, config_path)

        monkeypatch.setattr(service, "run_analysis", recording_run)
        assert client.post(
            "/v1/public/analyze", json={"text": "public detector path"}
        ).status_code == 200
        assert calls == [(["unicode"], None)]

    def test_config_specific_and_benchmark_detectors_are_explicitly_unavailable(self, client):
        body = client.post(
            "/v1/public/analyze", json={"text": "arbitrary third-party text"}
        ).json()
        notices = {item["detector"]: item for item in body["unavailable_detectors"]}
        assert set(notices) == {"kgw", "kgw-reference", "synthid", "synthid-reference"}
        for name in ("kgw", "synthid"):
            assert notices[name]["classification"] == "benchmark_only"
            assert notices[name]["status"] == "not_applicable"
            assert notices[name]["compute_class"] == "cheap_configured"
        for name in ("kgw-reference", "synthid-reference"):
            assert notices[name]["classification"] == "reference_config_specific"
            assert notices[name]["status"] == "unavailable_without_key_or_config"
            assert notices[name]["compute_class"] == "potentially_model_backed"

    def test_clean_result_never_claims_universal_watermark_absence(self, client):
        body = client.post(
            "/v1/public/analyze", json={"text": "clean text"}
        ).json()
        assert "no hidden or unusual unicode signal" in body["verdict"].lower()
        assert "configured statistical watermark families were not tested" in body["verdict"].lower()
        assert "does not establish human authorship" in body["disclaimer"].lower()
        assert "absence of every watermark" in body["disclaimer"].lower()

    def test_registry_classifies_every_detector_for_public_use(self):
        from provenance.detectors.registry import get_registry

        caps = {cap.name: cap for cap in get_registry().capabilities()}
        assert caps["unicode"].public_classification == "publicly_usable_arbitrary_input"
        assert caps["unicode"].public_availability == "available"
        assert caps["unicode"].compute_class == "cheap_deterministic"
        assert {caps[name].public_classification for name in ("kgw", "synthid")} == {
            "benchmark_only"
        }
        assert {
            caps[name].public_classification
            for name in ("kgw-reference", "synthid-reference")
        } == {"reference_config_specific"}


class TestAuth:
    def test_modern_supabase_jwks_session_does_not_require_legacy_secret(self, monkeypatch):
        """Modern Supabase signing keys are verified from the project JWKS."""
        import provenance.api.supabase_auth as auth

        class SigningKey:
            key = "public-key"

        class FakeJwks:
            def get_signing_key_from_jwt(self, token):
                assert token == "modern-token"
                return SigningKey()

        monkeypatch.delenv("SUPABASE_JWT_SECRET", raising=False)
        monkeypatch.setattr(auth, "_jwks_client", lambda base_url: FakeJwks())
        monkeypatch.setattr(auth.jwt, "get_unverified_header", lambda _token: {"alg": "RS256"})
        expected = {"sub": "modern-user", "aud": "authenticated", "iss": f"{SUPABASE_URL}/auth/v1", "exp": 2_000_000_000}

        def decode(token, key, **kwargs):
            assert token == "modern-token" and key == "public-key"
            assert kwargs["algorithms"] == ["RS256"]
            assert kwargs["issuer"] == expected["iss"] and kwargs["audience"] == "authenticated"
            return expected

        monkeypatch.setattr(auth.jwt, "decode", decode)
        assert auth.verify_supabase_token("modern-token") == expected

    def test_sync_provisions_user_and_claims_events(self, client):
        client.post("/v1/public/analyze", json={"text": "first anonymous use"})
        r = client.post("/v1/auth/sync", headers={"Authorization": f"Bearer {_mint()}"})
        assert r.status_code == 200
        body = r.json()
        assert body["auth_user_id"] == "user-1" and body["plan"] == "free"
        assert body["claimed_events"] == 1
        # Quota not reset: still 1 used, now against the free plan limit.
        assert body["quota"]["used"] == 1 and body["quota"]["plan"] == "free"

    def test_sync_requires_bearer(self, client):
        assert client.post("/v1/auth/sync").status_code == 401
        assert client.post(
            "/v1/auth/sync", headers={"Authorization": "Bearer garbage"}).status_code == 401

    def test_expired_and_wrong_secret_rejected(self, client):
        assert client.post(
            "/v1/auth/sync",
            headers={"Authorization": f"Bearer {_mint(exp_in=-10)}"}).status_code == 401
        assert client.post(
            "/v1/auth/sync",
            headers={"Authorization": f"Bearer {_mint(secret='wrong')}"}).status_code == 401

    def test_me_anonymous_and_user(self, client):
        anon = client.get("/v1/me").json()
        assert anon["kind"] == "anonymous" and anon["plan"] == "anonymous"
        assert anon["entitlements"]["can_access_dashboard"] is False
        user = client.get("/v1/me", headers={"Authorization": f"Bearer {_mint()}"}).json()
        assert user["kind"] == "user" and user["plan"] == "free"
        assert user["entitlements"]["can_access_dashboard"] is True
        assert user["entitlements"]["can_run_benchmarks"] is False

    def test_pro_plan_unlocks_advanced(self, tmp_path):
        db_url = f"sqlite:///{tmp_path}/pro.db"
        app = create_app(db_url=db_url)
        with TestClient(app) as c:
            c.post("/v1/auth/sync", headers={"Authorization": f"Bearer {_mint()}"})
            repo = create_repository(db_url)
            assert repo.set_user_plan("missing", "pro") is None
            assert repo.set_user_plan("user-1", "pro")["plan"] == "pro"
            repo.close()
            me = c.get("/v1/me", headers={"Authorization": f"Bearer {_mint()}"}).json()
            assert me["plan"] == "pro"
            assert me["entitlements"]["can_run_benchmarks"] is True
            assert me["entitlements"]["can_use_api"] is True


class TestEntitlements:
    def test_plan_matrix(self):
        from provenance.api.entitlements import entitlements_for_plan
        anon = entitlements_for_plan("anonymous")
        assert (anon.max_daily_analyses, anon.can_use_api) == (2, False)
        free = entitlements_for_plan("free")
        assert free.can_access_dashboard and not free.can_access_advanced
        pro = entitlements_for_plan("pro")
        assert pro.can_run_benchmarks and pro.can_use_api and pro.can_view_full_report
        assert entitlements_for_plan("nonsense").plan == "anonymous"

    def test_anonymous_and_free_sessions_cannot_execute_benchmarks(self, client):
        payload = {
            "detector": "kgw-reference",
            "config": "configs/kgw.reference.example.json",
            "profile": "all_safe",
            "lengths": [50],
            "samples": 1,
            "seed": 42,
        }
        assert client.post("/v1/benchmark-runs", json=payload).status_code == 401
        response = client.post(
            "/v1/benchmark-runs",
            json=payload,
            headers={"Authorization": f"Bearer {_mint()}"},
        )
        assert response.status_code == 403
        assert response.json()["detail"] == "Your plan does not include benchmark execution"


class TestSqliteUrlParsing:
    def test_relative_sqlite_url_resolves_relative(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        repo = create_repository("sqlite:///data/provenance.db")
        assert not str(repo._db_path).startswith("/data")
        assert str(repo._db_path).endswith("data/provenance.db")
        repo.close()

    def test_absolute_sqlite_url_resolves_absolute(self, tmp_path):
        target = tmp_path / "abs.db"
        repo = create_repository(f"sqlite:///{target}")
        assert str(repo._db_path) == str(target)
        repo.close()

    def test_documented_env_example_url(self, tmp_path, monkeypatch):
        from pathlib import Path
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("DATABASE_URL", "sqlite:///data/provenance.db")
        repo = create_repository()
        assert repo._db_path == Path("data/provenance.db")
        repo.close()
