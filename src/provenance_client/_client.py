"""ProvenanceClient — typed Python client for the Text Provenance Engine API.

Uses only the standard library ``urllib.request`` for HTTP.  No external
dependencies required.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from provenance_client._exceptions import (
    ProvenanceAPIError,
    TimeoutError,
    raise_for_status,
)
from provenance_client._models import (
    AnalyzeResult,
    AsyncAnalyzeResult,
    CancelResult,
    JobResult,
    JobSummary,
    UsageResult,
)


class ProvenanceClient:
    """Client for the Text Provenance Engine API.

    Parameters
    ----------
    base_url:
        Base URL of the API server (e.g. ``http://localhost:8000``).
    api_key:
        API key for authentication (sent as ``X-API-Key`` header).
    timeout:
        Default request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str = "",
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """Make an HTTP request and return (json_body, response_headers).

        Raises the appropriate ProvenanceAPIError subclass on failure.
        """
        url = self._url(path)
        data = json.dumps(body).encode("utf-8") if body is not None else None

        headers: dict[str, str] = {"Accept": "application/json"}
        if self._api_key:
            headers["X-API-Key"] = self._api_key
        if data is not None:
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=data, headers=headers, method=method)

        effective_timeout = timeout if timeout is not None else self._timeout

        try:
            with urllib.request.urlopen(req, timeout=effective_timeout) as resp:
                resp_body = resp.read().decode("utf-8")
                resp_headers = dict(resp.headers)
                status_code = resp.status
        except urllib.error.HTTPError as exc:
            resp_body = exc.read().decode("utf-8", errors="replace")
            resp_headers = dict(exc.headers) if exc.headers else {}
            status_code = exc.code
            # Parse error detail
            detail = resp_body
            try:
                err_json = json.loads(resp_body)
                detail = err_json.get("detail", resp_body)
            except (json.JSONDecodeError, AttributeError):
                pass
            request_id = resp_headers.get("X-Request-ID")
            retry_after = None
            if status_code == 429:
                ra = resp_headers.get("Retry-After")
                if ra:
                    try:
                        retry_after = int(ra)
                    except ValueError:
                        pass
            raise_for_status(status_code, detail, request_id=request_id, retry_after=retry_after)
        except urllib.error.URLError as exc:
            raise ProvenanceAPIError(
                f"Connection error: {exc.reason}",
                status_code=0,
                detail=str(exc.reason),
            ) from exc
        except TimeoutError:
            raise
        except Exception as exc:
            if "timed out" in str(exc).lower():
                raise TimeoutError() from exc
            raise ProvenanceAPIError(
                f"Request failed: {exc}",
                status_code=0,
                detail=str(exc),
            ) from exc

        # Parse successful response
        try:
            result = json.loads(resp_body) if resp_body else {}
        except json.JSONDecodeError:
            result = {}

        # Raise if status is not 2xx (shouldn't happen if HTTPError was caught)
        if status_code >= 300:
            detail = result.get("detail", resp_body) if isinstance(result, dict) else resp_body
            request_id = resp_headers.get("X-Request-ID")
            raise_for_status(status_code, detail, request_id=request_id)

        return result, resp_headers

    # -------------------------------------------------------------------
    # Analysis
    # -------------------------------------------------------------------

    def analyze(
        self,
        text: str,
        *,
        detectors: list[str] | None = None,
        config_path: str | None = None,
    ) -> AnalyzeResult:
        """Run synchronous text analysis.

        Parameters
        ----------
        text:
            UTF-8 text to analyze (1–200,000 characters).
        detectors:
            List of detector names.  Defaults to ``["unicode"]``.
            Supported: ``unicode``, ``kgw``, ``kgw-reference``,
            ``synthid``, ``synthid-reference``.
        config_path:
            Path to detector configuration JSON.  Required for watermark
            detectors (kgw, synthid, etc.).

        Returns
        -------
        AnalyzeResult with detector findings, text stats, and an analysis_id.
        """
        body: dict[str, Any] = {"text": text}
        if detectors is not None:
            body["detectors"] = detectors
        if config_path is not None:
            body["config_path"] = config_path
        data, _ = self._request("POST", "/v1/analyze", body=body)
        return AnalyzeResult.from_dict(data)

    # -------------------------------------------------------------------
    # Async analysis
    # -------------------------------------------------------------------

    def analyze_async(
        self,
        text: str,
        *,
        detectors: list[str] | None = None,
        config_path: str | None = None,
    ) -> AsyncAnalyzeResult:
        """Submit text for background analysis.

        Returns immediately with a job_id.  Use :meth:`get_job` or
        :meth:`wait_for_job` to poll for the result.

        Parameters
        ----------
        text:
            UTF-8 text to analyze.
        detectors:
            Detector names (defaults to ``["unicode"]``).
        config_path:
            Path to detector configuration JSON (required for watermark detectors).

        Returns
        -------
        AsyncAnalyzeResult with job_id and initial status.
        """
        body: dict[str, Any] = {"text": text}
        if detectors is not None:
            body["detectors"] = detectors
        if config_path is not None:
            body["config_path"] = config_path
        data, _ = self._request("POST", "/v1/analyze/async", body=body)
        return AsyncAnalyzeResult.from_dict(data)

    # -------------------------------------------------------------------
    # Jobs
    # -------------------------------------------------------------------

    def get_job(self, job_id: str) -> JobResult:
        """Get job status and result.

        Parameters
        ----------
        job_id:
            The job ID returned by :meth:`analyze_async`.

        Returns
        -------
        JobResult with status, result (if completed), or error_message (if failed).
        """
        data, _ = self._request("GET", f"/v1/jobs/{job_id}")
        return JobResult.from_dict(data)

    def list_jobs(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[JobSummary], int]:
        """List jobs for the calling API key.

        Parameters
        ----------
        limit:
            Maximum number of jobs to return (default 20).
        offset:
            Pagination offset (default 0).

        Returns
        -------
        Tuple of (list of JobSummary, total count).
        """
        path = f"/v1/jobs?limit={limit}&offset={offset}"
        data, _ = self._request("GET", path)
        jobs = [JobSummary.from_dict(j) for j in data.get("jobs", [])]
        return jobs, data.get("total", 0)

    def cancel_job(self, job_id: str) -> CancelResult:
        """Cancel a queued or running job.

        Parameters
        ----------
        job_id:
            The job ID to cancel.

        Returns
        -------
        CancelResult with the new status.

        Raises
        ------
        ConflictError
            If the job is already completed or failed.
        """
        data, _ = self._request("DELETE", f"/v1/jobs/{job_id}")
        return CancelResult.from_dict(data)

    def retry_job(
        self,
        job_id: str,
        text: str,
        *,
        detectors: list[str] | None = None,
        config_path: str | None = None,
    ) -> AsyncAnalyzeResult:
        """Retry a failed job.

        Creates a new job linked to the original.  Only ``failed`` jobs
        can be retried.

        Parameters
        ----------
        job_id:
            The failed job ID to retry.
        text:
            The text to re-analyze (raw text is not stored, so you must re-submit).
        detectors:
            Detector names (falls back to the original job's detectors).
        config_path:
            Path to detector configuration JSON.

        Returns
        -------
        AsyncAnalyzeResult with the new job_id.
        """
        body: dict[str, Any] = {"text": text}
        if detectors is not None:
            body["detectors"] = detectors
        if config_path is not None:
            body["config_path"] = config_path
        data, _ = self._request("POST", f"/v1/jobs/{job_id}/retry", body=body)
        return AsyncAnalyzeResult.from_dict(data)

    def wait_for_job(
        self,
        job_id: str,
        *,
        poll_interval: float = 2.0,
        timeout: float = 300.0,
    ) -> JobResult:
        """Poll a job until it reaches a terminal state.

        Parameters
        ----------
        job_id:
            The job ID to poll.
        poll_interval:
            Seconds between polls (default 2.0).  Must be >= 0.5.
        timeout:
            Maximum seconds to wait before raising TimeoutError (default 300).

        Returns
        -------
        JobResult in a terminal state (completed, failed, or cancelled).

        Raises
        ------
        TimeoutError
            If the job does not reach a terminal state within *timeout* seconds.
        """
        poll_interval = max(0.5, poll_interval)
        deadline = time.monotonic() + timeout

        while True:
            job = self.get_job(job_id)
            if job.is_terminal:
                return job

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Job {job_id} did not complete within {timeout}s "
                    f"(last status: {job.status})"
                )
            time.sleep(min(poll_interval, remaining))

    # -------------------------------------------------------------------
    # Usage
    # -------------------------------------------------------------------

    def get_usage(self) -> UsageResult:
        """Get usage statistics for the calling API key.

        Returns
        -------
        UsageResult with aggregate counts and breakdowns.
        """
        data, _ = self._request("GET", "/v1/usage")
        return UsageResult.from_dict(data)

    # Detector discovery
    # -------------------------------------------------------------------

    def list_detectors(self) -> list[dict[str, Any]]:
        """List supported detectors with capability metadata.

        Returns
        -------
        List of detector capability dicts with name, implementation_kind,
        compatibility, requires_config, and other metadata.
        """
        data, _ = self._request("GET", "/v1/detectors")
        return data.get("detectors", [])
