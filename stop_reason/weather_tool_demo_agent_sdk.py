"""
Same weather tool-use example as weather_tool_demo.py, but built on the
Claude Agent SDK (claude_agent_sdk) instead of the raw Claude API.

Key difference from the API version:
  - With the raw Messages API, YOU own the loop: you read `stop_reason`
    on every response ("tool_use" vs "end_turn"), decide when to call
    the tool, and manually append assistant/tool_result messages.
  - With the Agent SDK, the Claude Code harness owns that loop. You
    register a tool and call query(); the SDK drives the tool_use ->
    execute -> continue cycle internally and hands you a stream of
    higher-level messages instead of raw API responses.

Because of that, `stop_reason` is not something you branch on here --
the SDK does not surface it as a decision point in your code. It still
exists deeper in the stack (each underlying Messages API turn still has
one), and this demo logs it wherever the SDK exposes it on an
AssistantMessage, purely for comparison with the manual-loop version.
The one place `stop_reason` IS a first-class, user-facing signal is on
the final `ResultMessage` (e.g. "end_turn" vs "error_max_turns"),
which reports why the whole agent turn ended, not why one API call did.

Tool registration uses an in-process MCP server (@tool + create_sdk_mcp_server)
running inside this same Python process -- no subprocess, no IPC.
"""

import asyncio
import json
import logging

import truststore

# Use the OS certificate store (not just the certifi bundle) for outbound
# HTTPS calls -- required behind corporate TLS-inspecting proxies where the
# proxy's root CA is trusted by Windows but not by certifi.
truststore.inject_into_ssl()

import httpx
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
    query,
    tool,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stop_reason_demo_agent_sdk")

# WMO weather-code -> human-readable condition (subset covering common codes).
# https://open-meteo.com/en/docs#weathervariables
_WMO_CONDITIONS = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    61: "light rain", 63: "moderate rain", 65: "heavy rain",
    71: "light snow", 73: "moderate snow", 75: "heavy snow",
    80: "light rain showers", 81: "moderate rain showers", 82: "violent rain showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with heavy hail",
}


@tool(
    "get_weather",
    "Get the current weather for a city. Call this when the user asks about "
    "current conditions or temperature in a specific location.",
    {"location": str, "unit": str},
)
async def get_weather(args: dict) -> dict:
    """Agent SDK tool implementation -- same Open-Meteo lookup as the API demo."""
    location = args["location"]
    unit = args.get("unit") or "fahrenheit"

    async with httpx.AsyncClient(timeout=10.0) as http:
        geo = (
            await http.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": location, "count": 1},
            )
        ).json()
        results = geo.get("results")
        if not results:
            return {
                "content": [{"type": "text", "text": f"Location not found: {location}"}],
                "is_error": True,
            }

        place = results[0]
        lat, lon = place["latitude"], place["longitude"]
        resolved_name = ", ".join(
            part for part in (place.get("name"), place.get("admin1"), place.get("country")) if part
        )

        forecast = (
            await http.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,weather_code",
                    "temperature_unit": "fahrenheit" if unit == "fahrenheit" else "celsius",
                },
            )
        ).json()

    current = forecast["current"]
    condition = _WMO_CONDITIONS.get(current["weather_code"], f"code {current['weather_code']}")
    result = {
        "location": resolved_name,
        "unit": unit,
        "temperature": current["temperature_2m"],
        "condition": condition,
    }
    log.info("executing tool get_weather(%s) -> %s", args, result)
    return {"content": [{"type": "text", "text": json.dumps(result)}]}


def log_message(turn: int, message) -> None:
    """Log whatever the Agent SDK yields, for comparison with the API version's
    per-response stop_reason logging."""
    if isinstance(message, AssistantMessage):
        log.info(
            "turn %d | AssistantMessage model=%s stop_reason=%s",
            turn,
            message.model,
            message.stop_reason,
        )
        for i, block in enumerate(message.content):
            if isinstance(block, TextBlock):
                log.info("  content[%d] text=%r", i, block.text)
            elif isinstance(block, ToolUseBlock):
                log.info(
                    "  content[%d] tool_use name=%s id=%s input=%s",
                    i,
                    block.name,
                    block.id,
                    json.dumps(block.input),
                )
            elif isinstance(block, ToolResultBlock):
                log.info(
                    "  content[%d] tool_result tool_use_id=%s is_error=%s",
                    i,
                    block.tool_use_id,
                    block.is_error,
                )
    elif isinstance(message, ResultMessage):
        log.info(
            "turn %d | ResultMessage subtype=%s stop_reason=%s num_turns=%d is_error=%s",
            turn,
            message.subtype,
            message.stop_reason,
            message.num_turns,
            message.is_error,
        )
    elif isinstance(message, SystemMessage):
        log.info("turn %d | SystemMessage subtype=%s", turn, message.subtype)
    else:
        log.info("turn %d | %s", turn, type(message).__name__)


async def run_agent_sdk_query(user_message: str) -> str:
    weather_server = create_sdk_mcp_server(
        name="weather",
        version="1.0.0",
        tools=[get_weather],
    )

    options = ClaudeAgentOptions(
        mcp_servers={"weather": weather_server},
        allowed_tools=["mcp__weather__get_weather"],
        system_prompt="You are a helpful weather assistant.",
        max_turns=5,
    )

    log.info("user: %s", user_message)

    final_text = ""
    turn = 0
    async for message in query(prompt=user_message, options=options):
        turn += 1
        log_message(turn, message)
        if isinstance(message, ResultMessage) and message.result:
            final_text = message.result

    log.info("final answer: %s", final_text)
    return final_text


def main() -> None:
    asyncio.run(run_agent_sdk_query("What's the weather like in Bengaluru right now?"))


if __name__ == "__main__":
    main()
