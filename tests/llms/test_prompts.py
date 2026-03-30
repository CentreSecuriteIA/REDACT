"""Tests for prompt loading, template rendering, and message building."""

import pytest

from redact.llms.prompts import load_prompt, render_template, build_messages


class TestLoadPrompt:
    def test_loads_existing_prompt(self):
        config = load_prompt("constitution/generation", "benign")
        assert "system_prompt" in config
        assert "template" in config
        assert "seed_fields" in config

    def test_missing_pipeline_raises(self):
        with pytest.raises(FileNotFoundError):
            load_prompt("nonexistent_pipeline", "nonexistent_cat")


class TestRenderTemplate:
    def test_basic_rendering(self):
        result = render_template("Hello {name}, you have {count} items", name="Alice", count="5")
        assert result == "Hello Alice, you have 5 items"

    def test_missing_key_raises(self):
        with pytest.raises(KeyError):
            render_template("{missing_key}")

    def test_no_placeholders(self):
        assert render_template("plain text") == "plain text"


class TestBuildMessages:
    def test_structure(self, sample_prompt_config):
        msgs = build_messages(
            sample_prompt_config,
            num_samples="5",
            topic="safety",
        )
        assert msgs[0]["role"] == "system"
        assert msgs[-1]["role"] == "user"

    def test_system_prompt_rendered(self, sample_prompt_config):
        # System prompt has no placeholders in fixture, but template does
        msgs = build_messages(
            sample_prompt_config,
            num_samples="3",
            topic="privacy",
        )
        assert "3" in msgs[-1]["content"]
        assert "privacy" in msgs[-1]["content"]

    def test_few_shot_included(self):
        config = {
            "system_prompt": "System",
            "template": "User msg",
            "few_shot_examples": [
                {"role": "user", "content": "example q"},
                {"role": "assistant", "content": "example a"},
            ],
        }
        msgs = build_messages(config)
        assert len(msgs) == 4  # system + 2 few-shot + user
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"

    def test_few_shot_excluded(self):
        config = {
            "system_prompt": "System",
            "template": "User msg",
            "few_shot_examples": [
                {"role": "user", "content": "example"},
            ],
        }
        msgs = build_messages(config, few_shot=False)
        assert len(msgs) == 2  # system + user only

    def test_user_input_appended(self, sample_prompt_config):
        msgs = build_messages(
            sample_prompt_config,
            user_input="extra text",
            num_samples="1",
            topic="x",
        )
        assert "extra text" in msgs[-1]["content"]

    def test_no_system_prompt(self):
        config = {"template": "Hello"}
        msgs = build_messages(config)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
