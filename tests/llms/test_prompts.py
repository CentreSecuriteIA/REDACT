"""Tests for prompt loading, template rendering, and message building."""

import json

import pytest

from redact.llms.prompts import (
    PromptTemplate,
    build_messages,
    load_prompt,
    render_template,
    scaffold_prompt_tree,
)


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


class TestPromptTemplate:
    """The reusable two-stage object build_messages() is a one-shot wrapper
    around — the case build_messages() alone can't express: build once,
    call many times with a fixed system prompt.
    """

    def test_system_prompt_rendered_once_at_construction(self):
        config = {"system_prompt": "You check {category}.", "template": "Sample: {sample}"}
        tmpl = PromptTemplate(config, category="privacy")
        assert tmpl.system_prompt == "You check privacy."

    def test_call_renders_template_independently_of_construction_kwargs(self):
        config = {"system_prompt": "Checker for {category}.", "template": "{sample}"}
        tmpl = PromptTemplate(config, category="privacy")
        first = tmpl(sample="apple")
        second = tmpl(sample="banana")
        # Same system prompt both times — not re-rendered per call.
        assert first[0] == second[0] == {"role": "system", "content": "Checker for privacy."}
        assert first[-1]["content"] == "apple"
        assert second[-1]["content"] == "banana"

    def test_two_arg_checker_shape(self):
        # The shape content_moderation/checker.py's builders use: construct
        # once, wrap in a 2-arg (original, sample) adapter matching
        # check_sample()'s contract.
        config = {
            "system_prompt": "Compare.",
            "template": "INPUT:\n{input_prompt}\n\nOUTPUT:\n{output_response}",
        }
        tmpl = PromptTemplate(config)

        def checker(original: str, sample: str) -> list[dict]:
            return tmpl(input_prompt=original, output_response=sample)

        msgs = checker("the input", "the output")
        assert msgs == [
            {"role": "system", "content": "Compare."},
            {"role": "user", "content": "INPUT:\nthe input\n\nOUTPUT:\nthe output"},
        ]

    def test_few_shot_examples_included_by_default(self):
        config = {
            "system_prompt": "S",
            "template": "T",
            "few_shot_examples": [{"role": "user", "content": "ex"}],
        }
        tmpl = PromptTemplate(config)
        msgs = tmpl()
        assert msgs == [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "ex"},
            {"role": "user", "content": "T"},
        ]

    def test_few_shot_excluded(self):
        config = {
            "system_prompt": "S",
            "template": "T",
            "few_shot_examples": [{"role": "user", "content": "ex"}],
        }
        tmpl = PromptTemplate(config, few_shot=False)
        assert tmpl() == [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "T"},
        ]

    def test_no_build_kwargs_leaves_system_prompt_unrendered(self):
        # A template with no build-time placeholders shouldn't require any.
        config = {"system_prompt": "Fixed system prompt.", "template": "{sample}"}
        tmpl = PromptTemplate(config)
        assert tmpl.system_prompt == "Fixed system prompt."

    def test_format_style_appends_instruction_and_sets_extraction_style(self):
        config = {"system_prompt": "Base prompt.", "template": "Generate."}
        tmpl = PromptTemplate(config, format_style="numbered", num_samples="3")
        assert tmpl.system_prompt.startswith("Base prompt.")
        assert "numbered list" in tmpl.system_prompt.lower()
        assert "3" in tmpl.system_prompt
        assert tmpl.extraction_style == "numbered"

    def test_no_format_style_leaves_extraction_style_none(self):
        config = {"system_prompt": "S", "template": "T"}
        tmpl = PromptTemplate(config)
        assert tmpl.extraction_style is None
        assert tmpl.system_prompt == "S"

    def test_format_style_defaults_num_samples_when_absent(self):
        config = {"system_prompt": "S", "template": "T"}
        tmpl = PromptTemplate(config, format_style="delimiter")  # no num_samples kwarg
        assert "---" in tmpl.system_prompt


class TestBuildMessagesFormatStyle:
    def test_format_style_threaded_through(self):
        config = {"system_prompt": "S.", "template": "T"}
        msgs = build_messages(config, format_style="delimiter", num_samples="2")
        assert "---" in msgs[0]["content"]

    def test_default_omits_format_instruction(self):
        config = {"system_prompt": "S.", "template": "T"}
        msgs = build_messages(config)
        assert msgs[0]["content"] == "S."


class TestScaffoldPromptTree:
    def _write_source(self, root):
        d = root / "input" / "quality_check"
        d.mkdir(parents=True)
        (d / "template.json").write_text(
            '{"category": "quality_check", "pipeline": "input", '
            '"system_prompt": "REAL secret system prompt for {Category}.", '
            '"template": "REAL secret template: {sample}", '
            '"seed_fields": ["Category", "sample"], '
            '"few_shot_examples": [{"role": "user", "content": "REAL example"}], '
            '"metadata": {"version": "1.0", "notes": "internal notes"}}',
            encoding="utf-8",
        )
        return d / "template.json"

    def test_mirrors_directory_structure(self, tmp_path):
        source = tmp_path / "source"
        self._write_source(source)
        target = tmp_path / "target"

        written = scaffold_prompt_tree(target, source_dir=source)

        assert written == 1
        assert (target / "input" / "quality_check" / "template.json").exists()

    def test_redacts_text_preserves_structure(self, tmp_path):
        source = tmp_path / "source"
        self._write_source(source)
        target = tmp_path / "target"
        scaffold_prompt_tree(target, source_dir=source)

        stub = json.loads((target / "input" / "quality_check" / "template.json").read_text())
        assert stub["category"] == "quality_check"
        assert stub["pipeline"] == "input"
        assert stub["seed_fields"] == ["Category", "sample"]
        assert "REAL secret" not in stub["system_prompt"]
        assert "REAL secret" not in stub["template"]
        assert "{Category}" in stub["system_prompt"]
        assert "{sample}" in stub["system_prompt"]
        assert stub["few_shot_examples"] == []

    def test_placeholder_round_trips_through_build_messages(self, tmp_path):
        # The whole point: a scaffolded file must not KeyError just because
        # nobody has filled it in yet.
        source = tmp_path / "source"
        self._write_source(source)
        target = tmp_path / "target"
        scaffold_prompt_tree(target, source_dir=source)

        stub = json.loads((target / "input" / "quality_check" / "template.json").read_text())
        msgs = build_messages(stub, Category="violence", sample="text")
        assert len(msgs) == 2  # system + user; few-shot cleared
