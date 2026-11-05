"""LangChain's standard unit tests, once per tool."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool
from langchain_tests.unit_tests import ToolsUnitTests

from langchain_arcnm import (
    CancelCalculationTool,
    CheckQuotaTool,
    GetCalculationTool,
    GetCostSheetTool,
    ListCalculationsTool,
    ListEnvironmentsTool,
    ListPartRevisionsTool,
    ListPartsTool,
    PrecheckPartTool,
    QuotePartTool,
    UploadAndQuoteTool,
)

CALC_ID = "6f1c2d3e-0000-4000-8000-000000000001"


class _Base(ToolsUnitTests):
    tool_class: type[BaseTool]
    example: dict[str, Any] = {}

    @property
    def tool_constructor(self) -> type[BaseTool]:
        return self.tool_class

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {"api_key": "test-key"}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return self.example

    @property
    def init_from_env_params(self) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
        return (
            {"ARCNM_API_KEY": "env-key", "ARCNM_BASE_URL": "https://api.example.test"},
            {},
            {"api_key": "env-key", "base_url": "https://api.example.test"},
        )


class TestUploadAndQuote(_Base):
    tool_class = UploadAndQuoteTool
    example = {"file_path": "bracket.step", "lot_size": 50}


class TestPrecheckPart(_Base):
    tool_class = PrecheckPartTool
    example = {"file_path": "gearbox.step", "part_number": "GB-100"}


class TestQuotePart(_Base):
    tool_class = QuotePartTool
    example = {"part_revision_id": CALC_ID, "lot_size": 10}


class TestGetCalculation(_Base):
    tool_class = GetCalculationTool
    example = {"calculation_id": CALC_ID, "wait": True}


class TestListCalculations(_Base):
    tool_class = ListCalculationsTool
    example = {"status": "succeeded", "limit": 5}


class TestGetCostSheet(_Base):
    tool_class = GetCostSheetTool
    example = {"calculation_id": CALC_ID, "quantity": 100}


class TestCancelCalculation(_Base):
    tool_class = CancelCalculationTool
    example = {"calculation_id": CALC_ID}


class TestListEnvironments(_Base):
    tool_class = ListEnvironmentsTool
    example = {"limit": 5}


class TestCheckQuota(_Base):
    tool_class = CheckQuotaTool


class TestListParts(_Base):
    tool_class = ListPartsTool
    example = {"query": "BRACKET"}


class TestListPartRevisions(_Base):
    tool_class = ListPartRevisionsTool
    example = {"part_id": CALC_ID}
