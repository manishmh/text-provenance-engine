"""Centralized entitlement / quota model for the SaaS product.

All plan-gating decisions flow through this module — never scattered
``if premium`` conditionals at call sites.  Plans:

- ``anonymous`` — cookied visitor, no account.  2 public analyses/day.
- ``free`` — authenticated account, no subscription.  Higher API-less
  limits plus a limited workspace (no benchmarks/robustness).
- ``pro`` — paid (or manually granted).  Full workspace + API + benchmarks.

Paid state is stored on the ``app_users`` row (``plan`` column) so future
payment webhooks only need to flip that column.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Entitlements:
    plan: str
    max_daily_analyses: int
    max_chars_per_analysis: int
    can_view_full_report: bool
    can_access_dashboard: bool  # any workspace access
    can_access_advanced: bool  # benchmarks, robustness, jobs, detectors, API keys
    can_run_benchmarks: bool
    can_use_api: bool


def _int_env(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, str(default)).strip()))
    except (ValueError, AttributeError):
        return default


def public_daily_limit() -> int:
    """Anonymous free analyses per UTC day (default 2)."""
    return _int_env("PROVENANCE_PUBLIC_DAILY_LIMIT", 2)


def public_max_chars() -> int:
    """Max input characters for the public endpoint (default 5000)."""
    return _int_env("PROVENANCE_PUBLIC_MAX_CHARS", 5000)


def entitlements_for_plan(plan: str) -> Entitlements:
    """Return the entitlement set for a plan name (unknown → anonymous)."""
    if plan == "pro":
        return Entitlements(
            plan="pro",
            max_daily_analyses=_int_env("PROVENANCE_PRO_DAILY_LIMIT", 1000),
            max_chars_per_analysis=_int_env("PROVENANCE_PRO_MAX_CHARS", 100000),
            can_view_full_report=True,
            can_access_dashboard=True,
            can_access_advanced=True,
            can_run_benchmarks=True,
            can_use_api=True,
        )
    if plan == "free":
        return Entitlements(
            plan="free",
            max_daily_analyses=_int_env("PROVENANCE_FREE_DAILY_LIMIT", 50),
            max_chars_per_analysis=_int_env("PROVENANCE_FREE_MAX_CHARS", 20000),
            can_view_full_report=True,
            can_access_dashboard=True,
            can_access_advanced=False,
            can_run_benchmarks=False,
            can_use_api=False,
        )
    return Entitlements(
        plan="anonymous",
        max_daily_analyses=public_daily_limit(),
        max_chars_per_analysis=public_max_chars(),
        can_view_full_report=False,
        can_access_dashboard=False,
        can_access_advanced=False,
        can_run_benchmarks=False,
        can_use_api=False,
    )
