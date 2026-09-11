"""Phase 7E production-hardening regression tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from provenance.api.app import create_app


def test_production_config_requires_secure_cookie_postgres_and_https_origin(monkeypatch):
    monkeypatch.setenv("PROVENANCE_ENVIRONMENT", "production")
    monkeypatch.setenv("PROVENANCE_ANON_COOKIE_SECRET", "x" * 32)
    monkeypatch.setenv("PROVENANCE_ANON_COOKIE_SECURE", "1")
    monkeypatch.setenv("DATABASE_URL", "postgresql://example")
    monkeypatch.setenv("CORS_ORIGINS", "https://app.example.com")
    monkeypatch.setenv("SUPABASE_URL", "https://project.supabase.co")
    monkeypatch.setenv("SUPABASE_JWT_SECRET", "y" * 32)
    from provenance.api.config import validate_config
    validate_config()

    monkeypatch.setenv("CORS_ORIGINS", "*")
    from provenance.api.config import ConfigError
    with pytest.raises(ConfigError, match="wildcard"):
        validate_config()


def test_same_site_none_requires_secure_cookie(monkeypatch):
    monkeypatch.setenv("PROVENANCE_ENVIRONMENT", "development")
    monkeypatch.setenv("PROVENANCE_COOKIE_SAMESITE", "none")
    monkeypatch.setenv("PROVENANCE_ANON_COOKIE_SECURE", "0")
    from provenance.api.config import ConfigError, validate_config
    with pytest.raises(ConfigError, match="requires"):
        validate_config()


def test_forwarded_for_is_ignored_without_trusted_proxy(monkeypatch):
    scope = {"type": "http", "method": "GET", "path": "/", "headers": [(b"x-forwarded-for", b"198.51.100.1")], "client": ("127.0.0.1", 1234)}
    request = Request(scope)
    from provenance.api.middleware import request_client_ip
    monkeypatch.setenv("PROVENANCE_TRUST_PROXY", "0")
    assert request_client_ip(request) == "127.0.0.1"
    monkeypatch.setenv("PROVENANCE_TRUST_PROXY", "1")
    assert request_client_ip(request) == "198.51.100.1"


def test_declared_oversized_body_is_rejected_before_route_work(tmp_path, monkeypatch):
    monkeypatch.setenv("PROVENANCE_MAX_REQUEST_BODY_BYTES", "1024")
    app = create_app(db_url=f"sqlite:///{tmp_path / 'test.db'}")
    with TestClient(app) as client:
        response = client.post("/v1/public/analyze", content=b"x" * 1025)
    assert response.status_code == 413
    assert response.json()["detail"] == "Request body exceeds the service limit"
