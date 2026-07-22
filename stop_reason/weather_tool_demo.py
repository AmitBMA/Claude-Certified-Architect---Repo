"""
Demonstrates how `stop_reason` drives the Claude Agent SDK tool-use loop.

Walks through a weather tool-use example end to end:
  - stop_reason == "tool_use"  -> Claude wants to call a tool
  - stop_reason == "end_turn"  -> Claude has a final answer, loop ends
  - how the message list is appended each turn (assistant content with
    tool_use blocks, followed by a user message with tool_result blocks)

Logging uses each SDK response's `.model_dump()` (the SDK's own
pydantic-based parser) so every logged event reflects the exact
shape the API returned, not a hand-rolled summary.
"""

import json
import logging
import os

import truststore

# Use the OS certificate store (not just the certifi bundle) for outbound
# HTTPS calls -- required behind corporate TLS-inspecting proxies where the
# proxy's root CA is trusted by Windows but not by certifi.
truststore.inject_into_ssl()

import httpx
from anthropic import AnthropicBedrock

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stop_reason_demo")

# This environment authenticates to Claude via Amazon Bedrock (see
# CLAUDE_CODE_USE_BEDROCK / AWS_PROFILE in the shell env) rather than a
# first-party ANTHROPIC_API_KEY, so we use the Bedrock client and the
# cross-region inference profile ID already configured for this account.
MODEL = os.environ.get("ANTHROPIC_DEFAULT_OPUS_MODEL", "eu.anthropic.claude-opus-4-8")

WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get the current weather for a city. Call this when the user asks about current conditions or temperature in a specific location.",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City and state/country, e.g. San Francisco, CA",
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature unit",
            },
        },
        "required": ["location"],
    },
}

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


def get_weather(location: str, unit: str = "fahrenheit") -> dict:
    """Tool implementation. Calls Open-Meteo (free, no API key) for real data."""
    with httpx.Client(timeout=10.0) as http:
        geo = http.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1},
        ).json()
        results = geo.get("results")
        if not results:
            return {"location": location, "error": "location not found"}

        place = results[0]
        lat, lon = place["latitude"], place["longitude"]
        resolved_name = ", ".join(
            part for part in (place.get("name"), place.get("admin1"), place.get("country")) if part
        )

        forecast = http.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,weather_code",
                "temperature_unit": "fahrenheit" if unit == "fahrenheit" else "celsius",
            },
        ).json()

    current = forecast["current"]
    condition = _WMO_CONDITIONS.get(current["weather_code"], f"code {current['weather_code']}")
    return {
        "location": resolved_name,
        "unit": unit,
        "temperature": current["temperature_2m"],
        "condition": condition,
    }


def log_response(turn: int, response) -> None:
    """Log the SDK response using its own parsed representation."""
    dumped = response.model_dump()
    log.info(
        "turn %d | id=%s stop_reason=%s usage=%s",
        turn,
        dumped["id"],
        dumped["stop_reason"],
        dumped["usage"],
    )
    for i, block in enumerate(dumped["content"]):
        if block["type"] == "text":
            log.info("  content[%d] text=%r", i, block["text"])
        elif block["type"] == "tool_use":
            log.info(
                "  content[%d] tool_use name=%s id=%s input=%s",
                i,
                block["name"],
                block["id"],
                json.dumps(block["input"]),
            )


def run_agentic_loop(client: AnthropicBedrock, user_message: str) -> str:
    messages = [{"role": "user", "content": user_message}]
    log.info("user: %s", user_message)

    turn = 0
    while True:
        turn += 1
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            tools=[WEATHER_TOOL],
            messages=messages,
        )
        log_response(turn, response)

        if response.stop_reason == "tool_use":
            # Append the full assistant turn (all blocks, not just tool_use)
            # so the tool_use block's id lines up with the tool_result below.
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                log.info("executing tool %s(%s)", block.name, block.input)
                if block.name == "get_weather":
                    result = get_weather(**block.input)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result),
                        }
                    )
                else:
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": f"Unknown tool: {block.name}",
                            "is_error": True,
                        }
                    )

            # All tool_result blocks for this turn go back in a single
            # user message.
            messages.append({"role": "user", "content": tool_results})
            continue

        if response.stop_reason == "end_turn":
            final_text = "".join(b.text for b in response.content if b.type == "text")
            log.info("final answer: %s", final_text)
            return final_text

        if response.stop_reason == "max_tokens":
            log.warning("response truncated at max_tokens; consider raising max_tokens")
            return "".join(b.text for b in response.content if b.type == "text")

        if response.stop_reason == "refusal":
            log.warning("model declined to respond (stop_reason=refusal)")
            return ""

        log.warning("unhandled stop_reason=%s", response.stop_reason)
        return ""


def main() -> None:
    client = AnthropicBedrock()
    run_agentic_loop(client, "What's the weather like in Bengaluru right now?")


if __name__ == "__main__":
    main()
