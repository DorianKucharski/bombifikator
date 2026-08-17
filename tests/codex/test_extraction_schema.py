from __future__ import annotations

import unittest
from typing import Any

from codex.character_models import Build, Sex
from codex.extraction_schema import CHARACTER_EXTRACTION_SCHEMA

_UNSUPPORTED_KEYWORDS = frozenset({
    "minimum", "maximum", "multipleOf", "minLength", "maxLength", "minItems", "maxItems", "pattern",
})


def _walk_objects(node: Any):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk_objects(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_objects(item)


class TestExtractionSchema(unittest.TestCase):

    def test_every_object_forbids_additional_properties(self):
        for node in _walk_objects(CHARACTER_EXTRACTION_SCHEMA):
            if node.get("type") == "object":
                self.assertIs(False, node.get("additionalProperties"), node.get("properties", {}).keys())

    def test_every_object_requires_all_of_its_properties(self):
        for node in _walk_objects(CHARACTER_EXTRACTION_SCHEMA):
            if node.get("type") == "object":
                self.assertEqual(sorted(node["properties"]), sorted(node["required"]))

    def test_schema_avoids_keywords_structured_outputs_reject(self):
        for node in _walk_objects(CHARACTER_EXTRACTION_SCHEMA):
            self.assertEqual(set(), _UNSUPPORTED_KEYWORDS & set(node))

    def test_enums_stay_in_sync_with_the_domain_model(self):
        properties = CHARACTER_EXTRACTION_SCHEMA["properties"]

        self.assertEqual([member.value for member in Sex], properties["sex"]["enum"])
        self.assertEqual([member.value for member in Build],
                         properties["appearance"]["properties"]["build"]["enum"])


if __name__ == "__main__":
    unittest.main()
