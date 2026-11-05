"""What each tool sends and returns, against a mock ARCNM API."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from langchain_arcnm import (
    ArcnmToolkit,
    GetCalculationTool,
    GetCostSheetTool,
    PrecheckPartTool,
    QuotePartTool,
    UploadAndQuoteTool,
)
from langchain_arcnm._client import normalize_base_url

CALC_ID = "6f1c2d3e-0000-4000-8000-000000000001"

Handler = Callable[[httpx.Request], httpx.Response]


def _recording(handler: Handler) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        request.read()
        seen.append(request)
        return handler(request)

    return httpx.MockTransport(record), seen


def _calc(status: str, **extra: Any) -> dict[str, Any]:
    return {"id": CALC_ID, "status": status, "terminal": status == "succeeded", **extra}


@pytest.fixture
def cad(tmp_path: Path) -> Path:
    path = tmp_path / "bracket.step"
    path.write_bytes(b"ISO-10303-21;\nEND-ISO-10303-21;\n")
    return path


def test_upload_and_quote_uploads_then_waits(cad: Path) -> None:
    polls = iter(["running", "succeeded"])

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(202, json=_calc("queued", queue_position=1))
        status = next(polls)
        return httpx.Response(
            200,
            json=_calc(
                status,
                unit_cost=12.34,
                currency="EUR",
                analytics={"big": "x" * 1000},
            ),
        )

    transport, seen = _recording(handler)
    tool = UploadAndQuoteTool(api_key="k", transport=transport, poll_interval=0)
    result = json.loads(tool.invoke({"file_path": str(cad), "lot_size": 50}))

    post = seen[0]
    assert post.url == "https://api.arcnm.io/api/v1/calculations/upload-and-quote"
    assert post.headers["X-API-Key"] == "k"
    body = post.content.decode()
    assert 'name="part_number"\r\n\r\nBRACKET' in body
    assert 'name="lot_size"\r\n\r\n50' in body
    assert 'filename="bracket.step"' in body
    assert "costing_environment_id" not in body
    assert [r.url.path for r in seen[1:]] == [f"/api/v1/calculations/{CALC_ID}"] * 2
    assert seen[1].url.params["include_analytics"] == "false"
    assert result == {
        "id": CALC_ID,
        "status": "succeeded",
        "terminal": True,
        "currency": "EUR",
        "unit_cost": 12.34,
    }


def test_still_running_after_wait_names_the_next_step() -> None:
    transport, seen = _recording(lambda r: httpx.Response(202, json=_calc("queued")))
    tool = QuotePartTool(api_key="k", transport=transport, wait_timeout=0)
    result = json.loads(tool.invoke({"part_revision_id": "rev-1"}))

    assert len(seen) == 1
    assert json.loads(seen[0].content) == {"part_revision_id": "rev-1"}
    assert result["status"] == "queued"
    assert "get_calculation" in result["next_step"]


def test_blocked_is_terminal_without_polling() -> None:
    transport, seen = _recording(lambda r: httpx.Response(202, json=_calc("blocked")))
    tool = QuotePartTool(api_key="k", transport=transport, poll_interval=0)
    result = json.loads(tool.invoke({"part_revision_id": "rev-1"}))
    assert len(seen) == 1
    assert result["status"] == "blocked"
    assert "next_step" not in result


def test_get_calculation_without_wait_reads_once() -> None:
    transport, seen = _recording(lambda r: httpx.Response(200, json=_calc("running")))
    tool = GetCalculationTool(api_key="k", transport=transport)
    result = json.loads(tool.invoke({"calculation_id": CALC_ID}))
    assert len(seen) == 1
    assert result["status"] == "running"


def test_api_error_reaches_the_model_with_code_and_request_id() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            402,
            json={
                "error": {"code": "quota_exhausted", "message": "No calculations left."},
                "request_id": "req_123",
            },
        )

    transport, _ = _recording(handler)
    tool = QuotePartTool(api_key="k", transport=transport)
    result = tool.invoke({"part_revision_id": "rev-1"})
    assert "HTTP 402" in result
    assert "quota_exhausted: No calculations left." in result
    assert "req_123" in result


def test_api_error_raises_when_asked() -> None:
    transport, _ = _recording(lambda r: httpx.Response(404, json={"detail": "Not found"}))
    tool = GetCostSheetTool(api_key="k", transport=transport, handle_tool_error=False)
    with pytest.raises(Exception, match="HTTP 404. Not found"):
        tool.invoke({"calculation_id": CALC_ID})


def test_cost_sheet_passes_quantity() -> None:
    transport, seen = _recording(lambda r: httpx.Response(200, json={"cost_sheet": []}))
    GetCostSheetTool(api_key="k", transport=transport).invoke(
        {"calculation_id": CALC_ID, "quantity": 200}
    )
    assert seen[0].url.path == f"/api/v1/calculations/{CALC_ID}/cost-sheet"
    assert seen[0].url.params["quantity"] == "200"


def test_root_dir_keeps_reads_inside_it(tmp_path: Path, cad: Path) -> None:
    outside = tmp_path.parent / "secret.step"
    outside.write_bytes(b"x")
    transport, seen = _recording(lambda r: httpx.Response(202, json=_calc("succeeded")))
    tool = UploadAndQuoteTool(api_key="k", transport=transport, root_dir=str(tmp_path))

    refused = tool.invoke({"file_path": "../secret.step"})
    assert "outside the directory" in refused
    assert seen == []

    accepted = json.loads(tool.invoke({"file_path": "bracket.step"}))
    assert accepted["status"] == "succeeded"


def test_missing_file_is_reported_not_sent() -> None:
    transport, seen = _recording(lambda r: httpx.Response(500))
    result = UploadAndQuoteTool(api_key="k", transport=transport).invoke(
        {"file_path": "/no/such/part.step"}
    )
    assert "No such file" in result
    assert seen == []


def test_missing_api_key_says_where_to_get_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARCNM_API_KEY", raising=False)
    result = GetCalculationTool().invoke({"calculation_id": CALC_ID})
    assert "ARCNM_API_KEY" in result


def test_org_slug_header() -> None:
    transport, seen = _recording(lambda r: httpx.Response(200, json=_calc("succeeded")))
    GetCalculationTool(api_key="k", org_slug="acme", transport=transport).invoke(
        {"calculation_id": CALC_ID}
    )
    assert seen[0].headers["X-Org-Slug"] == "acme"


def test_toolkit_read_only_cannot_start_or_stop() -> None:
    full = {t.name for t in ArcnmToolkit(api_key="k").get_tools()}
    read_only = {t.name for t in ArcnmToolkit(api_key="k", read_only=True).get_tools()}
    writes = {"precheck_part", "upload_and_quote", "quote_part", "cancel_calculation"}
    assert writes <= full
    assert read_only == full - writes


def test_toolkit_passes_settings_to_every_tool() -> None:
    tools = ArcnmToolkit(api_key="k", base_url="https://api.example.test").get_tools()
    assert len({t.name for t in tools}) == len(tools)
    for tool in tools:
        assert tool.api_key.get_secret_value() == "k"  # type: ignore[attr-defined]
        assert tool.base_url == "https://api.example.test"  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("https://api.arcnm.io", "https://api.arcnm.io"),
        ("https://api.arcnm.io/", "https://api.arcnm.io"),
        ("https://api.arcnm.io/api/v1", "https://api.arcnm.io"),
        (" http://localhost:8000 ", "http://localhost:8000"),
    ],
)
def test_normalize_base_url(value: str, expected: str) -> None:
    assert normalize_base_url(value) == expected


@pytest.mark.parametrize(
    "value", ["api.arcnm.io", "ftp://api.arcnm.io", "https://key@api.arcnm.io"]
)
def test_normalize_base_url_refuses(value: str) -> None:
    with pytest.raises(ValueError):
        normalize_base_url(value)


async def test_async_invoke_runs_the_sync_tool() -> None:
    transport, _ = _recording(lambda r: httpx.Response(200, json=_calc("succeeded")))
    tool = GetCalculationTool(api_key="k", transport=transport)
    result = json.loads(await tool.ainvoke({"calculation_id": CALC_ID}))
    assert result["status"] == "succeeded"


def test_cost_sheet_keeps_the_figures_and_drops_the_page_detail() -> None:
    line = {
        "bucket": "machining",
        "amount": 41.2,
        "exact": 41.2034,
        "time_s": 312.0,
        "time_basis": "per_part",
        "rate": {"value": 78.0, "unit": "eur_per_h", "currency": "EUR", "burden": 9.0},
        "label_id": "costsheet.machining",
        "formula_id": "costsheet.formula.time_rate",
        "evidence": {"source": "x" * 500},
        "children": [{"bucket": "machining.roughing", "amount": 20.1, "label_id": "y"}],
    }
    response = {
        "basis": "curve",
        "rebuilt": False,
        "cost_sheet": {"currency": "EUR", "lot_size": 50, "total": 41.2, "lines": [line]},
        "point": {"quantity": 200, "unit_cost": 38.9, "offer_price": 44.7, "cost_sheet": {}},
    }
    transport, _ = _recording(lambda r: httpx.Response(200, json=response))
    result = json.loads(
        GetCostSheetTool(api_key="k", transport=transport).invoke(
            {"calculation_id": CALC_ID, "quantity": 200}
        )
    )
    assert result == {
        "basis": "curve",
        "currency": "EUR",
        "lot_size": 50,
        "total": 41.2,
        "rebuilt": False,
        "point": {"quantity": 200, "unit_cost": 38.9, "offer_price": 44.7},
        "lines": [
            {
                "bucket": "machining",
                "amount": 41.2,
                "time_s": 312.0,
                "time_basis": "per_part",
                "rate": {"value": 78.0, "unit": "eur_per_h", "currency": "EUR"},
                "children": [{"bucket": "machining.roughing", "amount": 20.1}],
            }
        ],
    }


PRECHECK_ID = "7a2b3c4d-0000-4000-8000-000000000002"


def _precheck(status: str, **extra: Any) -> dict[str, Any]:
    return {"id": PRECHECK_ID, "status": status, "kind": "unknown", "rows": [], **extra}


def test_precheck_sends_the_file_and_waits_for_the_answer(cad: Path) -> None:
    answers = iter(
        [
            _precheck("received"),
            _precheck(
                "ready",
                kind="assembly",
                per_part=True,
                totals={"dispatched": 2, "metered_max": 2},
                quota={"remaining_before": 5, "remaining_after": 3},
                rows=[
                    {
                        "position": 1,
                        "label": "Housing",
                        "qty_per": 1,
                        "fate": "calculate",
                        "sourcing": "make",
                        "placements": [{"x": 1}] * 40,
                    },
                    {
                        "position": 2,
                        "label": "M6 screw",
                        "qty_per": 8,
                        "fate": "catalogue",
                        "sourcing": "catalogue",
                    },
                ],
            ),
        ]
    )
    transport, seen = _recording(lambda r: httpx.Response(200, json=next(answers)))
    tool = PrecheckPartTool(api_key="k", transport=transport, poll_interval=0)
    result = json.loads(tool.invoke({"file_path": str(cad), "part_number": "GB-100"}))

    post = seen[0]
    assert (post.method, post.url.path) == ("POST", "/api/v1/prechecks")
    assert post.headers["Prefer"] == "wait=5"
    body = post.content.decode()
    assert 'name="cad_file"; filename="bracket.step"' in body
    assert 'name="part_number"\r\n\r\nGB-100' in body
    assert seen[1].url.path == f"/api/v1/prechecks/{PRECHECK_ID}"
    assert result["kind"] == "assembly"
    assert result["quota"] == {"remaining_before": 5, "remaining_after": 3}
    assert result["rows"][0] == {
        "position": 1,
        "label": "Housing",
        "qty_per": 1,
        "fate": "calculate",
        "sourcing": "make",
    }
    assert "next_step" not in result


def test_precheck_still_reading_says_so(cad: Path) -> None:
    transport, seen = _recording(lambda r: httpx.Response(202, json=_precheck("received")))
    tool = PrecheckPartTool(api_key="k", transport=transport, wait_timeout=0)
    result = json.loads(tool.invoke({"file_path": str(cad)}))
    assert len(seen) == 1
    assert seen[0].headers["Prefer"] == "wait=0"
    assert "Still being read" in result["next_step"]


def test_precheck_without_an_answer_points_to_pricing(cad: Path) -> None:
    transport, _ = _recording(
        lambda r: httpx.Response(202, json=_precheck("failed", error_code="unreadable"))
    )
    result = json.loads(
        PrecheckPartTool(api_key="k", transport=transport).invoke({"file_path": str(cad)})
    )
    assert result["error_code"] == "unreadable"
    assert "upload_and_quote" in result["next_step"]


def test_precheck_keeps_reads_inside_root_dir(tmp_path: Path, cad: Path) -> None:
    transport, seen = _recording(lambda r: httpx.Response(500))
    tool = PrecheckPartTool(api_key="k", transport=transport, root_dir=str(tmp_path))
    assert "outside the directory" in tool.invoke({"file_path": "../x.step"})
    assert seen == []
