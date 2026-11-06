"""Every ARCNM tool, configured once."""

from __future__ import annotations

from typing import Any

import httpx
from langchain_core.tools import BaseTool, BaseToolkit
from langchain_core.utils import from_env, secret_from_env
from pydantic import ConfigDict, Field, SecretStr

from langchain_arcnm._client import DEFAULT_BASE_URL
from langchain_arcnm.tools import (
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


class ArcnmToolkit(BaseToolkit):
    """The ARCNM tools, sharing one API key and base URL.

    Setup:
        .. code-block:: bash

            pip install -U langchain-arcnm
            export ARCNM_API_KEY="..."

    Instantiate:
        .. code-block:: python

            from langchain_arcnm import ArcnmToolkit

            toolkit = ArcnmToolkit()
            tools = toolkit.get_tools()

    ``read_only=True`` leaves out the four tools that store a file or start
    or stop a calculation, so an agent can read results but never use the
    plan. Those four need a key with ``parts:write``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    api_key: SecretStr = Field(default_factory=secret_from_env("ARCNM_API_KEY", default=""))
    base_url: str = Field(default_factory=from_env("ARCNM_BASE_URL", default=DEFAULT_BASE_URL))
    org_slug: str | None = Field(default_factory=from_env("ARCNM_ORG_SLUG", default=None))
    request_timeout: float = 60.0
    wait_timeout: float = 120.0
    poll_interval: float = 2.0
    root_dir: str | None = None
    """Directory upload_and_quote may read files from. None allows any path."""
    read_only: bool = False
    transport: httpx.BaseTransport | None = Field(default=None, exclude=True, repr=False)

    def get_tools(self) -> list[BaseTool]:
        common: dict[str, Any] = {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "org_slug": self.org_slug,
            "request_timeout": self.request_timeout,
            "transport": self.transport,
        }
        waiting: dict[str, Any] = {
            **common,
            "wait_timeout": self.wait_timeout,
            "poll_interval": self.poll_interval,
        }
        tools: list[BaseTool] = []
        if not self.read_only:
            tools += [
                PrecheckPartTool(**common, root_dir=self.root_dir),
                UploadAndQuoteTool(**waiting, root_dir=self.root_dir),
                QuotePartTool(**waiting),
                CancelCalculationTool(**common),
            ]
        tools += [
            GetCalculationTool(**waiting),
            ListCalculationsTool(**common),
            GetCostSheetTool(**common),
            ListPartsTool(**common),
            ListPartRevisionsTool(**common),
            ListEnvironmentsTool(**common),
            CheckQuotaTool(**common),
        ]
        return tools
