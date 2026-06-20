"""Smoke tests for the LSTNet MCP server (tool registration + offline call)."""
from __future__ import annotations

import asyncio
import json

from lstnet.mcp_server import list_sites, mcp


def test_tools_registered():
    for name in ("list_sites", "compute_lst", "validate_csv"):
        tool = asyncio.run(mcp.get_tool(name))
        assert tool.name == name


def test_list_sites_returns_25_sites():
    sites = list_sites()
    assert len(sites) == 25
    assert all({"name", "network", "lon", "lat"} <= set(s) for s in sites)


def test_list_sites_via_mcp_protocol_is_json():
    # The path an agent client uses: call_tool -> ToolResult.content[0].text (JSON).
    result = asyncio.run(mcp.call_tool("list_sites", {}))
    text = result.content[0].text
    data = json.loads(text)
    assert len(data) == 25
    assert data[0]["name"] in {s["name"] for s in list_sites()}
