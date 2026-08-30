from __future__ import annotations

from typing import Any

from reference_builder.reference_models import UNKNOWN_LABEL

CLUSTER_LABELING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["slug", "confidence", "reasoning"],
    "properties": {
        "slug": {
            "type": "string",
            "description": f"Slug of the matching character from the catalogue, or {UNKNOWN_LABEL} when the grid "
                           f"shows no catalogue character, mixes several characters or shows no character at all.",
        },
        "confidence": {
            "type": "number",
            "description": "Confidence between 0 and 1 that every tile in the grid shows the named character.",
        },
        "reasoning": {
            "type": "string",
            "description": "One short Polish sentence naming the visual features that decided the match.",
        },
    },
}
