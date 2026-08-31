"""Tests for structural obfuscation functions."""

import json
import random

from redact.jailbreak.obfuscation.structural import (
    get_structural_functions,
    to_json,
    to_markdown,
    to_xml,
)

SAMPLE = "test prompt"


class TestToJson:
    def test_contains_prompt(self):
        random.seed(42)
        result, info = to_json(SAMPLE)
        assert SAMPLE in result

    def test_valid_json_embedded(self):
        random.seed(42)
        result, _ = to_json(SAMPLE)
        # Extract JSON portion after "CRITICAL SYSTEM DATA:\n"
        json_part = result.split("CRITICAL SYSTEM DATA:\n", 1)[1]
        parsed = json.loads(json_part)
        assert parsed["content_payload"]["instruction_set"] == SAMPLE

    def test_info_has_priority_and_format(self):
        random.seed(42)
        _, info = to_json(SAMPLE)
        assert "priority=" in info
        assert "format=" in info


class TestToXml:
    def test_contains_prompt(self):
        random.seed(42)
        result, _ = to_xml(SAMPLE)
        assert SAMPLE in result

    def test_xml_tags_present(self):
        random.seed(42)
        result, _ = to_xml(SAMPLE)
        assert "<payload>" in result
        assert "</payload>" in result
        assert "<action>" in result

    def test_info_has_policy_id(self):
        random.seed(42)
        _, info = to_xml(SAMPLE)
        assert "policy_id=" in info


class TestToMarkdown:
    def test_contains_prompt(self):
        result, _ = to_markdown(SAMPLE)
        assert SAMPLE in result

    def test_has_markdown_headers(self):
        result, _ = to_markdown(SAMPLE)
        assert "# System Documentation Update" in result
        assert "## Section" in result


class TestAllStructural:
    def test_all_return_tuple(self):
        random.seed(42)
        for fn in get_structural_functions():
            result = fn(SAMPLE)
            assert isinstance(result, tuple) and len(result) == 2

    def test_count(self):
        assert len(get_structural_functions()) == 3

    def test_prompt_preserved(self):
        random.seed(42)
        for fn in get_structural_functions():
            result, _ = fn(SAMPLE)
            assert SAMPLE in result
