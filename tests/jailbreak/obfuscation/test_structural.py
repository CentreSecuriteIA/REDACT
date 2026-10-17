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

    def test_non_ascii_is_kept_as_it_is(self):
        prompt = "don’t do it 😀"
        result, _ = to_json(prompt, rng=random.Random(0))
        assert prompt in result


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


class TestTemplatesComeFromConfig:
    def test_xml_renders_the_config_template(self, monkeypatch):
        import redact.jailbreak.obfuscation.structural as S

        cfg = S._TEMPLATES["to_xml"]
        monkeypatch.setattr(S.random, "choice", lambda seq: seq[0])
        result, info = to_xml(SAMPLE)
        expected = cfg["template"].format_map(
            {"policy_id": cfg["policy_ids"][0], "prompt": SAMPLE})
        assert result == expected
        assert info == f"policy_id={cfg['policy_ids'][0]}"

    def test_markdown_renders_the_config_template(self):
        import redact.jailbreak.obfuscation.structural as S

        result, info = to_markdown(SAMPLE)
        assert result == S._TEMPLATES["to_markdown"]["template"].format_map(
            {"prompt": SAMPLE})
        assert info == ""
