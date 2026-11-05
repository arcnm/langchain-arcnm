"""HTTP calls to the ARCNM REST API. One small client, no LangChain imports."""

from __future__ import annotations

import time
from pathlib import Path
from typing import IO, Any

import httpx

DEFAULT_BASE_URL = "https://api.arcnm.io"
API_PREFIX = "/api/v1"
USER_AGENT = "langchain-arcnm/0.1.0"

#: A calculation in one of these states will not change on its own. ``blocked``
#: means it was created but parked because the plan has no calculations left.
TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled", "timed_out", "blocked"})


class ArcnmAPIError(Exception):
    """An ARCNM API call failed.

    ``code`` is the stable ``error.code`` of the response; branch on it, never
    on the message. ``request_id`` is what ARCNM support asks for.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: str | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.request_id = request_id


def normalize_base_url(value: str) -> str:
    """The API origin, without a trailing slash or the ``/api/v1`` prefix."""
    url = httpx.URL(value.strip())
    if url.scheme not in ("http", "https") or not url.host:
        raise ValueError("The ARCNM base URL must start with http:// or https://.")
    if url.userinfo:
        raise ValueError("Pass the API key as api_key, not inside the base URL.")
    origin = str(url.copy_with(path="/", query=None, fragment=None)).rstrip("/")
    return origin


def _failure(response: httpx.Response) -> ArcnmAPIError:
    body: Any = None
    if response.headers.get("content-type", "").startswith("application/json"):
        try:
            body = response.json()
        except ValueError:
            body = None
    envelope = body.get("error") if isinstance(body, dict) else None
    code = envelope.get("code") if isinstance(envelope, dict) else None
    message = envelope.get("message") if isinstance(envelope, dict) else None
    if message is None and isinstance(body, dict) and isinstance(body.get("detail"), str):
        message = body["detail"]
    request_id = body.get("request_id") if isinstance(body, dict) else None
    text = message or response.reason_phrase or "request failed"
    if code:
        text = f"{code}: {text}"
    if request_id:
        text = f"{text} (request_id {request_id})"
    return ArcnmAPIError(
        f"ARCNM returned HTTP {response.status_code}. {text}",
        status_code=response.status_code,
        code=code if isinstance(code, str) else None,
        request_id=request_id if isinstance(request_id, str) else None,
    )


class ArcnmClient:
    """A thin, synchronous client for the endpoints the tools call."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        org_slug: str | None = None,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise ValueError(
                "No ARCNM API key. Set ARCNM_API_KEY, or pass api_key. "
                "Create one in ARCNM under Settings, then API keys."
            )
        headers = {"X-API-Key": api_key, "Accept": "application/json", "User-Agent": USER_AGENT}
        if org_slug:
            headers["X-Org-Slug"] = org_slug
        self._http = httpx.Client(
            base_url=normalize_base_url(base_url) + API_PREFIX,
            headers=headers,
            timeout=timeout,
            transport=transport,
        )

    def __enter__(self) -> ArcnmClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        files: dict[str, tuple[str, IO[bytes]]] | None = None,
    ) -> Any:
        query = {k: v for k, v in (params or {}).items() if v is not None}
        form = {k: str(v) for k, v in (data or {}).items() if v is not None}
        try:
            response = self._http.request(
                method, path, params=query, json=json, data=form or None, files=files
            )
        except httpx.HTTPError as error:
            raise ArcnmAPIError(f"Could not reach ARCNM: {error}") from error
        if response.is_error:
            raise _failure(response)
        if not response.content:
            return {}
        return response.json()

    # -- the calls ---------------------------------------------------------

    def upload_and_quote(
        self,
        cad_path: Path,
        *,
        part_number: str,
        drawing_path: Path | None = None,
        lot_size: int | None = None,
        annual_volume: int | None = None,
        costing_environment_id: str | None = None,
    ) -> dict[str, Any]:
        fields = {
            "part_number": part_number,
            "lot_size": lot_size,
            "annual_volume": annual_volume,
            "costing_environment_id": costing_environment_id,
        }
        with cad_path.open("rb") as cad:
            files: dict[str, tuple[str, IO[bytes]]] = {"cad_file": (cad_path.name, cad)}
            if drawing_path is None:
                return self.request(  # type: ignore[no-any-return]
                    "POST", "/calculations/upload-and-quote", data=fields, files=files
                )
            with drawing_path.open("rb") as drawing:
                files["drawing_file"] = (drawing_path.name, drawing)
                return self.request(  # type: ignore[no-any-return]
                    "POST", "/calculations/upload-and-quote", data=fields, files=files
                )

    def quote(
        self,
        part_revision_id: str,
        *,
        lot_size: int | None = None,
        annual_volume: int | None = None,
        costing_environment_id: str | None = None,
    ) -> dict[str, Any]:
        body = {
            "part_revision_id": part_revision_id,
            "lot_size": lot_size,
            "annual_volume": annual_volume,
            "costing_environment_id": costing_environment_id,
        }
        return self.request(  # type: ignore[no-any-return]
            "POST", "/calculations/quote", json={k: v for k, v in body.items() if v is not None}
        )

    def get_calculation(self, calculation_id: str) -> dict[str, Any]:
        return self.request(  # type: ignore[no-any-return]
            "GET",
            f"/calculations/{calculation_id}",
            params={"include_analytics": "false"},
        )

    def wait_for_calculation(
        self, calculation: dict[str, Any], *, timeout: float, poll_interval: float
    ) -> dict[str, Any]:
        """Poll until the calculation is terminal or ``timeout`` seconds pass."""
        deadline = time.monotonic() + timeout
        while not is_terminal(calculation) and time.monotonic() < deadline:
            time.sleep(min(poll_interval, max(0.0, deadline - time.monotonic())))
            calculation = self.get_calculation(str(calculation["id"]))
        return calculation

    def list_calculations(
        self, *, status: str | None = None, part_id: str | None = None, limit: int = 10
    ) -> dict[str, Any]:
        return self.request(  # type: ignore[no-any-return]
            "GET",
            "/calculations",
            params={"status_filter": status, "part_id": part_id, "limit": limit},
        )

    def get_cost_sheet(self, calculation_id: str, *, quantity: int | None = None) -> dict[str, Any]:
        return self.request(  # type: ignore[no-any-return]
            "GET", f"/calculations/{calculation_id}/cost-sheet", params={"quantity": quantity}
        )

    def cancel_calculation(self, calculation_id: str) -> dict[str, Any]:
        return self.request(  # type: ignore[no-any-return]
            "POST", f"/calculations/{calculation_id}/cancel"
        )

    def list_environments(self, *, limit: int = 20) -> list[dict[str, Any]]:
        return self.request(  # type: ignore[no-any-return]
            "GET", "/environments/", params={"limit": limit}
        )

    def get_entitlement(self) -> dict[str, Any]:
        return self.request("GET", "/calculations/entitlement")  # type: ignore[no-any-return]

    def list_parts(self, *, query: str | None = None, limit: int = 10) -> dict[str, Any]:
        return self.request(  # type: ignore[no-any-return]
            "GET", "/parts/", params={"q": query, "limit": limit}
        )

    def list_part_revisions(self, part_id: str, *, limit: int = 10) -> dict[str, Any]:
        return self.request(  # type: ignore[no-any-return]
            "GET", f"/parts/{part_id}/revisions", params={"limit": limit, "order": "desc"}
        )


def is_terminal(calculation: dict[str, Any]) -> bool:
    if calculation.get("terminal") is True:
        return True
    return calculation.get("status") in TERMINAL_STATUSES
