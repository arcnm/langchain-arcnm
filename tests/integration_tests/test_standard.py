"""LangChain's standard integration tests, against the live ARCNM API.

Needs ARCNM_API_KEY. These use no calculations; the precheck stores
shaft_thread.step until it expires. BILLABLE=1 also quotes it, which uses one.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from langchain_core.tools import BaseTool
from langchain_tests.integration_tests import ToolsIntegrationTests

from langchain_arcnm import (
    CheckQuotaTool,
    GetCalculationTool,
    GetCostSheetTool,
    ListCalculationsTool,
    ListEnvironmentsTool,
    ListPartsTool,
    PrecheckPartTool,
    UploadAndQuoteTool,
)

pytestmark = pytest.mark.skipif(
    not os.environ.get("ARCNM_API_KEY"), reason="ARCNM_API_KEY is not set"
)

PART = Path(__file__).with_name("shaft_thread.step")


class _Live(ToolsIntegrationTests):
    tool_class: type[BaseTool]
    example: dict[str, Any] = {}

    @property
    def tool_constructor(self) -> type[BaseTool]:
        return self.tool_class

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {"handle_tool_error": False}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return self.example


class TestCheckQuota(_Live):
    tool_class = CheckQuotaTool


class TestListCalculations(_Live):
    tool_class = ListCalculationsTool
    example = {"limit": 3}


class TestListEnvironments(_Live):
    tool_class = ListEnvironmentsTool
    example = {"limit": 3}


class TestListParts(_Live):
    tool_class = ListPartsTool
    example = {"limit": 3}


def test_quota_reports_remaining() -> None:
    quota = json.loads(CheckQuotaTool(handle_tool_error=False).invoke({}))
    assert "remaining" in quota


def test_cost_sheet_of_a_past_calculation() -> None:
    listed = json.loads(
        ListCalculationsTool(handle_tool_error=False).invoke({"status": "succeeded", "limit": 1})
    )
    if not listed["items"]:
        pytest.skip("no succeeded calculation to read")
    sheet = json.loads(
        GetCostSheetTool(handle_tool_error=False).invoke(
            {"calculation_id": listed["items"][0]["id"], "quantity": 25}
        )
    )
    assert sheet["point"]["quantity"] == 25
    assert sheet["lines"] and all("bucket" in line for line in sheet["lines"])
    assert "label_id" not in json.dumps(sheet)


def test_precheck_a_single_part() -> None:
    answer = json.loads(
        PrecheckPartTool(handle_tool_error=False).invoke(
            {"file_path": str(PART), "part_number": "LANGCHAIN-SHAFT-THREAD"}
        )
    )
    assert answer["status"] == "ready", answer
    assert answer["kind"] == "part", answer
    assert answer["quota"] is not None


@pytest.mark.billable
@pytest.mark.skipif(not os.environ.get("BILLABLE"), reason="set BILLABLE=1 to use a calculation")
def test_quote_a_part_end_to_end() -> None:
    started = json.loads(
        UploadAndQuoteTool(handle_tool_error=False, wait_timeout=300).invoke(
            {"file_path": str(PART), "part_number": "LANGCHAIN-SHAFT-THREAD", "lot_size": 50}
        )
    )
    calculation = json.loads(
        GetCalculationTool(handle_tool_error=False, wait_timeout=300).invoke(
            {"calculation_id": started["id"], "wait": True}
        )
    )
    assert calculation["status"] == "succeeded", calculation
    assert calculation["unit_cost"] > 0

    sheet = json.loads(
        GetCostSheetTool(handle_tool_error=False).invoke(
            {"calculation_id": started["id"], "quantity": 500}
        )
    )
    assert sheet["point"]["quantity"] == 500
    assert sheet["lines"] and all("amount" in line for line in sheet["lines"])
