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

# Mock weather data so this demo runs with no external API dependency.
FAKE_WEATHER = {
    "san francisco": {"temp_f": 61, "condition": "foggy"},
    "tokyo": {"temp_f": 72, "condition": "clear"},
}


def get_weather(location: str, unit: str = "fahrenheit") -> dict:
    """Tool implementation. In a real app this would call a weather API."""
    key = location.lower()
    data = next(
        (v for k, v in FAKE_WEATHER.items() if k in key),
        {"temp_f": 70, "condition": "unknown"},
    )
    temp = data["temp_f"] if unit == "fahrenheit" else round((data["temp_f"] - 32) * 5 / 9)
    return {"location": location, "unit": unit, "temperature": temp, "condition": data["condition"]}


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


def run_agentic_loop(client: Anthropic, user_message: str) -> str:
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
    run_agentic_loop(client, "What's the weather like in San Francisco right now?")


if __name__ == "__main__":
    main()
