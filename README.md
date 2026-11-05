# langchain-arcnm

**Manufacturing should-cost from a CAD file, for your LangChain agent.**

[ARCNM](https://arcnm.io) turns a STEP file into an itemised manufacturing cost
for machined and sheet-metal parts: unit cost, setup cost, cycle time, and the
cost sheet behind them. This package gives a LangChain or LangGraph agent the
ARCNM REST API as tools, so it can price a part, explain what drives the cost,
and compare lot sizes.

```bash
pip install -U langchain-arcnm
```

## Setup

1. In ARCNM, open Settings, then API keys, then New key. ARCNM shows the key once.
2. Export it:

```bash
export ARCNM_API_KEY="..."
```

Reads need the `parts:read` scope. Starting or cancelling a calculation needs
`parts:write`. The Free plan includes 10 calculations a month, no card required.

## Use it in an agent

```python
from langchain.agents import create_agent
from langchain_arcnm import ArcnmToolkit

agent = create_agent(
    "anthropic:claude-sonnet-5-5",
    tools=ArcnmToolkit(root_dir="./cad").get_tools(),
)
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What does bracket.step cost at a lot of 50, and what drives it?",
            }
        ]
    }
)
print(result["messages"][-1].content)
```

## Call a tool directly

```python
from langchain_arcnm import UploadAndQuoteTool

quote = UploadAndQuoteTool()
print(quote.invoke({"file_path": "bracket.step", "lot_size": 50}))
```

Every tool returns JSON as a string. An API error comes back as the tool
result, with its stable error code and request id, so the agent can react to
it; set `handle_tool_error=False` to have it raised instead.

## Tools

| Tool | What it does | Uses a calculation |
|---|---|---|
| `upload_and_quote` | Uploads a CAD file, and a drawing if you have one, and prices it. | Yes |
| `quote_part` | Prices a part revision ARCNM already has, at another lot size or in another costing environment. | Yes |
| `cancel_calculation` | Stops a calculation that is still queued or running. | No |
| `get_calculation` | Reads one calculation: status, costs, cycle time, confidence. `wait=true` waits for it to finish. | No |
| `list_calculations` | Lists recent calculations, by status or part. | No |
| `get_cost_sheet` | The itemised cost sheet, at the priced lot size or any other. | No |
| `list_parts` | Finds parts by part number. | No |
| `list_part_revisions` | Lists a part's revisions, for `quote_part`. | No |
| `list_environments` | Lists the costing environments (machines, rates, materials) a calculation can price with. | No |
| `check_quota` | How many calculations are left in this period. | No |

The names match the [ARCNM MCP server](https://github.com/arcnm/arcnm-mcp), so
an agent sees the same verbs on either surface.

`upload_and_quote`, `quote_part` and `get_calculation` with `wait=true` wait up
to `wait_timeout` seconds (default 120) for the result. A calculation still
running after that comes back with its id, and the agent can ask again with
`get_calculation`.

## Configuration

Pass these to `ArcnmToolkit` or to any single tool.

| Argument | Default | |
|---|---|---|
| `api_key` | `ARCNM_API_KEY` | Your ARCNM API key. |
| `base_url` | `ARCNM_BASE_URL`, else `https://api.arcnm.io` | Change it only when you were given a different origin. |
| `org_slug` | `ARCNM_ORG_SLUG` | The organization to act in, for a key that belongs to several. |
| `root_dir` | none | The directory `upload_and_quote` may read files from. |
| `read_only` | `False` | Toolkit only. Leaves out the tools that start or stop a calculation. |
| `wait_timeout` | `120` | Seconds to wait for a started calculation. `0` returns at once. |
| `poll_interval` | `2` | Seconds between status checks while waiting. |
| `request_timeout` | `60` | Seconds one HTTP request may take. |

## Security

`upload_and_quote` reads the file path the model gives it and sends that file
to ARCNM. Set `root_dir` so it can only read the folder you meant, and use
`read_only=True` for an agent that should never start a calculation. A key with
only `parts:read` cannot start one either.

## Development

```bash
uv sync --all-groups
make test              # unit tests, no network
make integration_test  # against the live API; needs ARCNM_API_KEY, uses no calculations
make lint
```

`make integration_test BILLABLE=1` also quotes `tests/integration_tests/shaft_thread.step`,
which uses one calculation.

`arcnm-contract.json` lists every ARCNM operation and field this package relies
on. ARCNM checks it against every change to its API before that change merges.
It is not part of the published package.

## Links

* [ARCNM docs](https://arcnm.io/docs)
* [ARCNM API examples](https://github.com/arcnm/arcnm-examples)
* [LangChain tools](https://docs.langchain.com/oss/python/langchain/tools)

## License

MIT
