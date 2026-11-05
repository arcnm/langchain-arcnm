"""LangChain tools for the ARCNM REST API.

Each tool takes its connection settings from the environment unless you pass
them: ``ARCNM_API_KEY`` (required), ``ARCNM_BASE_URL`` and ``ARCNM_ORG_SLUG``.
Tool names match the ARCNM MCP server's, so an agent sees the same verbs on
either surface.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool, ToolException
from langchain_core.utils import from_env, secret_from_env
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from langchain_arcnm._client import DEFAULT_BASE_URL, ArcnmAPIError, ArcnmClient, is_terminal

#: The calculation fields a model needs to answer "what does it cost, and can
#: I trust it". The full record, with the analytics, is far larger.
CALCULATION_FIELDS = (
    "id",
    "status",
    "terminal",
    "part_id",
    "part_number",
    "part_revision_id",
    "costing_environment_id",
    "lot_size",
    "annual_volume",
    "currency",
    "unit_cost",
    "setup_cost",
    "total_cost",
    "offer_price",
    "unit_time_s",
    "total_time_s",
    "confidence",
    "price_completeness",
    "needs_human_review",
    "review_reasons",
    "coverage",
    "error",
    "queue_position",
    "eta_seconds",
    "created_at",
    "finished_at",
)


def _calculation(record: dict[str, Any]) -> dict[str, Any]:
    return {key: record[key] for key in CALCULATION_FIELDS if record.get(key) is not None}


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


class ArcnmBaseTool(BaseTool):
    """Connection settings and error handling shared by every ARCNM tool."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    api_key: SecretStr = Field(default_factory=secret_from_env("ARCNM_API_KEY", default=""))
    """ARCNM API key. Reads need ``parts:read``; starting or cancelling a
    calculation needs ``parts:write``."""
    base_url: str = Field(default_factory=from_env("ARCNM_BASE_URL", default=DEFAULT_BASE_URL))
    """API origin. Change it only when you were given a different one."""
    org_slug: str | None = Field(default_factory=from_env("ARCNM_ORG_SLUG", default=None))
    """Organization to act in, for a key that belongs to several."""
    request_timeout: float = 60.0
    """Seconds one HTTP request may take."""
    transport: httpx.BaseTransport | None = Field(default=None, exclude=True, repr=False)
    """Custom HTTP transport, for tests and proxies."""

    handle_tool_error: bool | str | Any = True
    """Return an API error to the model as the tool result, so it can recover."""

    def _client(self) -> ArcnmClient:
        return ArcnmClient(
            self.api_key.get_secret_value(),
            base_url=self.base_url,
            org_slug=self.org_slug,
            timeout=self.request_timeout,
            transport=self.transport,
        )

    def _call(self, fn: Any) -> str:
        try:
            with self._client() as client:
                return _dump(fn(client))
        except (ArcnmAPIError, ValueError, OSError) as error:
            raise ToolException(str(error)) from error


class _WaitingTool(ArcnmBaseTool):
    wait_timeout: float = 120.0
    """Seconds to wait for a started calculation to finish before returning.
    0 returns at once with the queued calculation."""
    poll_interval: float = 2.0
    """Seconds between status checks while waiting."""

    def _settle(self, client: ArcnmClient, started: dict[str, Any]) -> dict[str, Any]:
        if self.wait_timeout > 0 and not is_terminal(started):
            started = client.wait_for_calculation(
                started, timeout=self.wait_timeout, poll_interval=self.poll_interval
            )
        result = _calculation(started)
        if not is_terminal(started):
            result["next_step"] = (
                "Still running. Call get_calculation with this id and wait=true for the result."
            )
        return result


# -- upload_and_quote ---------------------------------------------------------


class UploadAndQuoteInput(BaseModel):
    file_path: str = Field(description="Path to the CAD file, for example bracket.step.")
    part_number: str | None = Field(
        default=None,
        description="Part number to file it under. Defaults to the file name in upper case. "
        "A part number that already exists gets a new revision.",
    )
    drawing_path: str | None = Field(
        default=None, description="Path to a technical drawing (PDF) for the same part."
    )
    lot_size: int | None = Field(default=None, ge=1, description="Pieces per order.")
    annual_volume: int | None = Field(default=None, ge=1, description="Pieces per year.")
    costing_environment_id: str | None = Field(
        default=None,
        description="Costing environment to price with. Omit it for the organization's default.",
    )


class UploadAndQuoteTool(_WaitingTool):
    """Upload a CAD file and price it. Uses one calculation from the plan."""

    name: str = "upload_and_quote"
    description: str = (
        "Upload a CAD file (STEP, IGES, DXF and similar) and calculate its manufacturing "
        "should-cost: unit cost, setup cost, total for the lot, and cycle time. Uses one "
        "calculation from the plan, so call check_quota first when quoting many parts. "
        "Returns the calculation, finished when it was fast enough."
    )
    args_schema: type[BaseModel] = UploadAndQuoteInput

    root_dir: str | None = None
    """When set, file paths are read relative to this directory and may not leave it."""

    def _resolve(self, value: str) -> Path:
        path = Path(value).expanduser()
        if self.root_dir is not None:
            root = Path(self.root_dir).expanduser().resolve()
            path = (root / path).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f"{value} is outside the directory this tool may read.")
        if not path.is_file():
            raise ValueError(f"No such file: {value}")
        return path

    def _run(
        self,
        file_path: str,
        part_number: str | None = None,
        drawing_path: str | None = None,
        lot_size: int | None = None,
        annual_volume: int | None = None,
        costing_environment_id: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        def call(client: ArcnmClient) -> dict[str, Any]:
            cad = self._resolve(file_path)
            drawing = self._resolve(drawing_path) if drawing_path else None
            started = client.upload_and_quote(
                cad,
                part_number=part_number or cad.stem.upper(),
                drawing_path=drawing,
                lot_size=lot_size,
                annual_volume=annual_volume,
                costing_environment_id=costing_environment_id,
            )
            return self._settle(client, started)

        return self._call(call)


# -- quote_part ---------------------------------------------------------------


class QuotePartInput(BaseModel):
    part_revision_id: str = Field(description="Part revision to price, from list_parts.")
    lot_size: int | None = Field(default=None, ge=1, description="Pieces per order.")
    annual_volume: int | None = Field(default=None, ge=1, description="Pieces per year.")
    costing_environment_id: str | None = Field(
        default=None,
        description="Costing environment to price with. Omit it for the organization's default.",
    )


class QuotePartTool(_WaitingTool):
    """Price a part revision ARCNM already has. Uses one calculation from the plan."""

    name: str = "quote_part"
    description: str = (
        "Price a part revision that is already in ARCNM, for another lot size or another "
        "costing environment, without uploading the file again. Uses one calculation from "
        "the plan. Returns the calculation, finished when it was fast enough."
    )
    args_schema: type[BaseModel] = QuotePartInput

    def _run(
        self,
        part_revision_id: str,
        lot_size: int | None = None,
        annual_volume: int | None = None,
        costing_environment_id: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        def call(client: ArcnmClient) -> dict[str, Any]:
            started = client.quote(
                part_revision_id,
                lot_size=lot_size,
                annual_volume=annual_volume,
                costing_environment_id=costing_environment_id,
            )
            return self._settle(client, started)

        return self._call(call)


# -- get_calculation ----------------------------------------------------------


class GetCalculationInput(BaseModel):
    calculation_id: str = Field(description="Calculation id.")
    wait: bool = Field(
        default=False, description="Wait for a queued or running calculation to finish."
    )


class GetCalculationTool(_WaitingTool):
    """Read one calculation: status, costs, times and confidence."""

    name: str = "get_calculation"
    description: str = (
        "Read one calculation: its status, unit cost, setup cost, total cost, cycle time, "
        "confidence and anything that needs a human to review. Set wait=true to wait for "
        "a calculation that is still running. Free."
    )
    args_schema: type[BaseModel] = GetCalculationInput

    def _run(
        self,
        calculation_id: str,
        wait: bool = False,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        def call(client: ArcnmClient) -> dict[str, Any]:
            record = client.get_calculation(calculation_id)
            if not wait:
                return _calculation(record)
            return self._settle(client, record)

        return self._call(call)


# -- list_calculations --------------------------------------------------------


class ListCalculationsInput(BaseModel):
    status: str | None = Field(
        default=None,
        description="Only this status: queued, running, succeeded, failed, cancelled, "
        "timed_out or blocked.",
    )
    part_id: str | None = Field(default=None, description="Only calculations of this part.")
    limit: int = Field(default=10, ge=1, le=50, description="How many, newest first.")


class ListCalculationsTool(ArcnmBaseTool):
    """List recent calculations, newest first."""

    name: str = "list_calculations"
    description: str = (
        "List recent calculations, newest first, with their status and costs. Filter by "
        "status or by part. Free."
    )
    args_schema: type[BaseModel] = ListCalculationsInput

    def _run(
        self,
        status: str | None = None,
        part_id: str | None = None,
        limit: int = 10,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        def call(client: ArcnmClient) -> dict[str, Any]:
            page = client.list_calculations(status=status, part_id=part_id, limit=limit)
            return {
                "items": [_calculation(row) for row in page.get("items", [])],
                "has_more": page.get("has_more", False),
            }

        return self._call(call)


# -- get_cost_sheet -----------------------------------------------------------


class GetCostSheetInput(BaseModel):
    calculation_id: str = Field(description="Id of a succeeded calculation.")
    quantity: int | None = Field(
        default=None,
        ge=1,
        description="Lot size to break the cost down at. Omit it for the lot the calculation "
        "was priced at.",
    )


class GetCostSheetTool(ArcnmBaseTool):
    """The itemised cost sheet of a calculation, at any lot size."""

    name: str = "get_cost_sheet"
    description: str = (
        "The itemised cost sheet of a succeeded calculation: one line per cost category "
        "(material, machining, setup, finishing and so on) with the time and rate behind "
        "each. Pass quantity to see the same part at another lot size without a new "
        "calculation. Free."
    )
    args_schema: type[BaseModel] = GetCostSheetInput

    def _run(
        self,
        calculation_id: str,
        quantity: int | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        return self._call(lambda client: client.get_cost_sheet(calculation_id, quantity=quantity))


# -- cancel_calculation -------------------------------------------------------


class CancelCalculationInput(BaseModel):
    calculation_id: str = Field(description="Id of a queued or running calculation.")


class CancelCalculationTool(ArcnmBaseTool):
    """Stop a calculation that is still queued or running."""

    name: str = "cancel_calculation"
    description: str = (
        "Stop a calculation that is still queued or running. A finished calculation cannot "
        "be cancelled."
    )
    args_schema: type[BaseModel] = CancelCalculationInput

    def _run(
        self, calculation_id: str, run_manager: CallbackManagerForToolRun | None = None
    ) -> str:
        return self._call(lambda client: _calculation(client.cancel_calculation(calculation_id)))


# -- list_environments --------------------------------------------------------

ENVIRONMENT_FIELDS = ("id", "name", "description", "currency", "region", "is_default")


class ListEnvironmentsInput(BaseModel):
    limit: int = Field(default=20, ge=1, le=100, description="How many, newest first.")


class ListEnvironmentsTool(ArcnmBaseTool):
    """List the costing environments a calculation can price with."""

    name: str = "list_environments"
    description: str = (
        "List the organization's costing environments: the machines, rates and materials a "
        "calculation prices with. Use an id as costing_environment_id. Free."
    )
    args_schema: type[BaseModel] = ListEnvironmentsInput

    def _run(self, limit: int = 20, run_manager: CallbackManagerForToolRun | None = None) -> str:
        def call(client: ArcnmClient) -> list[dict[str, Any]]:
            rows = client.list_environments(limit=limit)
            return [{k: row[k] for k in ENVIRONMENT_FIELDS if k in row} for row in rows]

        return self._call(call)


# -- check_quota --------------------------------------------------------------

QUOTA_FIELDS = ("plan", "included", "used", "remaining", "over_quota", "window_end", "guidance")


class CheckQuotaInput(BaseModel):
    pass


class CheckQuotaTool(ArcnmBaseTool):
    """How many calculations are left in this period."""

    name: str = "check_quota"
    description: str = (
        "How many calculations the plan has left in this period, and when it resets. Call it "
        "before quoting many parts. Free."
    )
    args_schema: type[BaseModel] = CheckQuotaInput

    def _run(self, run_manager: CallbackManagerForToolRun | None = None) -> str:
        def call(client: ArcnmClient) -> dict[str, Any]:
            quota = client.get_entitlement()
            return {k: quota[k] for k in QUOTA_FIELDS if quota.get(k) is not None}

        return self._call(call)


# -- list_parts ---------------------------------------------------------------


class ListPartsInput(BaseModel):
    query: str | None = Field(default=None, description="Text the part number contains.")
    limit: int = Field(default=10, ge=1, le=50, description="How many to return.")


class ListPartsTool(ArcnmBaseTool):
    """Find parts already in ARCNM."""

    name: str = "list_parts"
    description: str = (
        "Find parts already in ARCNM by part number. Returns each part with its id; pass "
        "that id to list_part_revisions for the revision quote_part prices. Free."
    )
    args_schema: type[BaseModel] = ListPartsInput

    def _run(
        self,
        query: str | None = None,
        limit: int = 10,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        def call(client: ArcnmClient) -> dict[str, Any]:
            page = client.list_parts(query=query, limit=limit)
            return {"data": page.get("data", []), "has_more": page.get("has_more", False)}

        return self._call(call)


# -- list_part_revisions ------------------------------------------------------


class ListPartRevisionsInput(BaseModel):
    part_id: str = Field(description="Part id, from list_parts.")
    limit: int = Field(default=10, ge=1, le=50, description="How many, newest first.")


class ListPartRevisionsTool(ArcnmBaseTool):
    """The revisions of one part, newest first."""

    name: str = "list_part_revisions"
    description: str = (
        "List the revisions of one part, newest first. Each revision id can be priced "
        "again with quote_part. Free."
    )
    args_schema: type[BaseModel] = ListPartRevisionsInput

    def _run(
        self,
        part_id: str,
        limit: int = 10,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        def call(client: ArcnmClient) -> dict[str, Any]:
            page = client.list_part_revisions(part_id, limit=limit)
            return {"data": page.get("data", []), "has_more": page.get("has_more", False)}

        return self._call(call)
