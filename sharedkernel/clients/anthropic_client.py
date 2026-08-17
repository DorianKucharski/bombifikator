from __future__ import annotations

import json
from typing import Any

import anthropic

from sharedkernel.logger import get_logger

LOGGER = get_logger("clients.anthropic")


class StructuredExtractionClient:
    def __init__(self, api_key: str, model: str, max_tokens: int, effort: str) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._effort = effort

    def extract(self, system_prompt: str, user_prompt: str, json_schema: dict[str, Any]) -> dict[str, Any] | None:
        response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
                output_config={
                    "effort": self._effort,
                    "format": {"type": "json_schema", "schema": json_schema},
                },
                messages=[{"role": "user", "content": user_prompt}],
        )
        if response.stop_reason == "refusal":
            LOGGER.warning("extraction refused: category=%s",
                           response.stop_details.category if response.stop_details else None)
            return None
        if response.stop_reason == "max_tokens":
            LOGGER.warning("extraction truncated at max_tokens=%d", self._max_tokens)
            return None
        text = next((block.text for block in response.content if block.type == "text"), None)
        if text is None:
            return None
        return json.loads(text)
