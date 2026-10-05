"""Tests for prompt loading, template rendering, and message building."""

import json
import logging
import pathlib
import string

import pytest

from redact.llms.prompting import (
    PromptTemplate,
    build_messages,
    load_prompt,
    report_prompt_sources,
    resolve_prompt,
    scaffold_prompt_tree,
)


class TestLoadPrompt:
    def test_loads_existing_prompt(self):
        config = load_prompt("constitution", "benign")
        assert "system_prompt" in config
        assert "template" in config
        assert "seed_fields" in config

    def test_missing_pipeline_raises(self):
        with pytest.raises(FileNotFoundError):
            load_prompt("nonexistent_pipeline", "nonexistent_cat")


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

    def test_system_prompt_is_rendered_once_and_reused(self):
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

    def test_format_style_requires_num_samples(self):
        """It used to default to 1, which asks the model for one sample while
        the template asks for fifteen — a mismatch that only surfaces later as
        a short extraction. Omitting format_style is how you say "just one"."""
        config = {"system_prompt": "S", "template": "T"}
        with pytest.raises(ValueError, match="num_samples"):
            PromptTemplate(config, format_style="delimiter")


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
            '{"system_prompt": "REAL secret system prompt for {Category}.", '
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

        written = scaffold_prompt_tree(target, source_dir=source, mode="empty")

        assert written == 1
        assert (target / "input" / "quality_check" / "template.json").exists()

    def test_redacts_text_preserves_structure(self, tmp_path):
        source = tmp_path / "source"
        self._write_source(source)
        target = tmp_path / "target"
        scaffold_prompt_tree(target, source_dir=source, mode="empty")

        stub = json.loads((target / "input" / "quality_check" / "template.json").read_text())
        # seed_fields survives — it is the prompt's contract with callers.
        assert stub["seed_fields"] == ["Category", "sample"]
        assert "REAL secret" not in stub["system_prompt"]
        assert "REAL secret" not in stub["template"]
        # Each field keeps its own placeholders, and only those.
        assert "{Category}" in stub["system_prompt"]
        assert "{sample}" not in stub["system_prompt"]
        assert "{sample}" in stub["template"]
        assert "{Category}" not in stub["template"]
        # Neither real examples nor the real version may leak.
        assert stub["few_shot_examples"] == []
        assert stub["metadata"]["version"] == "0.0"
        assert "internal notes" not in json.dumps(stub)

    def test_placeholder_round_trips_through_build_messages(self, tmp_path):
        # The whole point: a scaffolded file must not KeyError just because
        # nobody has filled it in yet.
        source = tmp_path / "source"
        self._write_source(source)
        target = tmp_path / "target"
        scaffold_prompt_tree(target, source_dir=source, mode="empty")

        stub = json.loads((target / "input" / "quality_check" / "template.json").read_text())
        msgs = build_messages(stub, Category="violence", sample="text")
        assert len(msgs) == 2  # system + user; few-shot cleared


class TestMissingPlaceholders:
    """Agent finding #8: rendering used to be guarded by `if text and kwargs`,
    so zero kwargs sent the literal `{Category}` to the model while partial
    kwargs raised a bare KeyError — two outcomes for one mistake, one silent."""

    CONFIG = {"system_prompt": "Judge {Category}.", "template": "Sample: {sample}"}

    def test_system_prompt_placeholder_with_no_kwargs_raises(self):
        with pytest.raises(ValueError, match=r"system_prompt is missing \{Category\}"):
            PromptTemplate(self.CONFIG)

    def test_template_placeholder_with_no_kwargs_raises(self):
        tmpl = PromptTemplate(self.CONFIG, Category="violence")
        with pytest.raises(ValueError, match=r"template is missing \{sample\}"):
            tmpl()

    def test_partial_kwargs_raise_the_same_way(self):
        with pytest.raises(ValueError, match="Category"):
            PromptTemplate(self.CONFIG, something_else="x")

    def test_no_placeholders_needs_no_kwargs(self):
        msgs = PromptTemplate({"system_prompt": "S", "template": "T"})()
        assert [m["content"] for m in msgs] == ["S", "T"]

    def test_every_missing_placeholder_is_named_at_once(self):
        """format_map stops at the first missing key, so reporting only that
        one sends someone round a fix-run-fail loop per placeholder."""
        cfg = {
            "system_prompt": "{Category} {subcategory} {entry_type}",
            "template": "T",
        }
        with pytest.raises(ValueError) as exc:
            PromptTemplate(cfg)
        for name in ("Category", "subcategory", "entry_type"):
            assert f"{{{name}}}" in str(exc.value)

    def test_supplied_placeholders_are_not_reported_missing(self):
        cfg = {
            "system_prompt": "{Category} {subcategory} {entry_type}",
            "template": "T",
        }
        with pytest.raises(ValueError) as exc:
            PromptTemplate(cfg, Category="v", subcategory="s")
        msg = str(exc.value)
        assert "{entry_type}" in msg
        assert "{Category}" not in msg

    def test_a_repeated_placeholder_is_named_once(self):
        cfg = {"system_prompt": "{A} then {A} then {B}", "template": "T"}
        with pytest.raises(ValueError) as exc:
            PromptTemplate(cfg)
        assert str(exc.value).count("{A}") == 1

    def test_malformed_template_is_reported_as_malformed(self):
        """A missing key hit before the bad braces used to be reported as the
        problem — a wrong diagnosis, since supplying it fixes nothing and the
        next call dies on the braces anyway."""
        cfg = {"system_prompt": "{Category} and {unclosed", "template": "T"}
        with pytest.raises(ValueError, match="not a valid template") as exc:
            PromptTemplate(cfg)
        assert "Pass Category=" not in str(exc.value)

    def test_malformed_before_any_placeholder_also_names_the_field(self):
        """format_map raises ValueError (not KeyError) here, so this path has
        to be caught separately or Python's own message escapes with no clue
        which prompt field it came from."""
        cfg = {"system_prompt": "{unclosed and {Category}", "template": "T"}
        with pytest.raises(ValueError, match="Prompt system_prompt is not a valid"):
            PromptTemplate(cfg)

    def test_malformed_template_field_is_named_too(self):
        cfg = {"system_prompt": "S", "template": "{unclosed and {x}"}
        with pytest.raises(ValueError, match="Prompt template is not a valid"):
            PromptTemplate(cfg)()


def test_scaffold_mirrors_every_real_prompt_shape(tmp_path):
    """The scaffold documents the schema by example, so a stub's key set must
    equal its source's — otherwise it teaches a shape that doesn't exist.
    Regression: few_shot_examples was assigned unconditionally, giving the
    format_instructions/ snippets a field no real one has.
    """
    real = pathlib.Path(__file__).resolve().parents[2] / "src" / "redact" / "prompts"
    scaffold_prompt_tree(tmp_path, mode="empty")
    mismatches = []
    for stub_file in sorted(tmp_path.rglob("*.json")):
        rel = stub_file.relative_to(tmp_path)
        source_keys = set(json.loads((real / rel).read_text(encoding="utf-8")))
        stub_keys = set(json.loads(stub_file.read_text(encoding="utf-8")))
        if source_keys != stub_keys:
            mismatches.append((str(rel), sorted(stub_keys ^ source_keys)))
    assert not mismatches, f"scaffold shape drifted: {mismatches}"


class TestScaffoldModes:
    """Two modes because they answer different questions: 'edit the real one'
    vs 'what fields does this shape have'."""

    def test_copy_is_the_default_and_writes_real_text(self, tmp_path):
        written = scaffold_prompt_tree(tmp_path)
        assert written > 0
        real = pathlib.Path(__file__).resolve().parents[2] / "src" / "redact" / "prompts"
        for stub in tmp_path.rglob("*.json"):
            source = real / stub.relative_to(tmp_path)
            assert stub.read_bytes() == source.read_bytes()

    def test_empty_writes_todos_instead(self, tmp_path):
        scaffold_prompt_tree(tmp_path, mode="empty")
        one = json.loads(
            (tmp_path / "input" / "quality_check" / "template.json").read_text()
        )
        assert one["system_prompt"].startswith("TODO")

    def test_unknown_mode_raises(self, tmp_path):
        with pytest.raises(ValueError, match="copy.*empty"):
            scaffold_prompt_tree(tmp_path, mode="partial")


class TestPromptOverlay:
    """prompt_dir is an overlay, not a replacement. The point is that copying
    one file to change one checker doesn't fork the other 33 — which would
    silently miss every later improvement to them."""

    def _override(self, root, pipeline, category, text):
        d = root / pipeline / category
        d.mkdir(parents=True)
        (d / "template.json").write_text(
            json.dumps({"system_prompt": text, "template": "T", "seed_fields": []}),
            encoding="utf-8",
        )

    def test_override_wins_for_that_prompt(self, tmp_path):
        self._override(tmp_path, "input", "quality_check", "MINE")
        assert load_prompt("input", "quality_check", prompt_dir=tmp_path)[
            "system_prompt"
        ] == "MINE"

    def test_unoverridden_prompts_still_come_from_the_library(self, tmp_path):
        self._override(tmp_path, "input", "quality_check", "MINE")
        other = load_prompt("output", "generation", prompt_dir=tmp_path)
        assert other["system_prompt"] != "MINE"
        _, is_override = resolve_prompt("output", "generation", prompt_dir=tmp_path)
        assert is_override is False

    def test_empty_override_dir_changes_nothing(self, tmp_path):
        assert load_prompt("input", "quality_check", prompt_dir=tmp_path) == load_prompt(
            "input", "quality_check"
        )

    def test_missing_everywhere_names_both_places_searched(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Looked in"):
            load_prompt("input", "no_such_prompt", prompt_dir=tmp_path)

    def test_report_flags_a_typo_that_overrides_nothing(self, tmp_path, caplog):
        self._override(tmp_path, "input", "quality_check", "MINE")
        self._override(tmp_path, "input", "quality_chek", "TYPO")
        with caplog.at_level(logging.WARNING, logger="redact.llms.prompting.prompts"):
            summary = report_prompt_sources(tmp_path)
        assert summary["overridden"] == ["input/quality_check"]
        assert summary["unused"] == ["input/quality_chek"]
        assert "never loaded" in caplog.text


class TestPromptTemplateLoad:
    """load() is the normal entry point: resolve + read + construct in one
    step, keeping the path that load_prompt()+build_messages() threw away."""

    def test_loads_a_packaged_prompt_and_records_provenance(self):
        tmpl = PromptTemplate.load(
            "input", "quality_check",
            Category="violence", subcategory="", entry_type="harmful",
        )
        assert tmpl.name == "input/quality_check"
        assert tmpl.is_override is False
        assert tmpl.source.name.endswith(".json")
        assert "violence" in tmpl.system_prompt

    def test_an_override_is_used_and_flagged(self, tmp_path):
        d = tmp_path / "input" / "quality_check"
        d.mkdir(parents=True)
        (d / "template.json").write_text(
            json.dumps({"system_prompt": "MINE", "template": "T"}), encoding="utf-8"
        )
        tmpl = PromptTemplate.load("input", "quality_check", prompt_dir=tmp_path)
        assert tmpl.is_override is True
        assert tmpl.source.parent == d
        assert tmpl.system_prompt == "MINE"

    def test_render_error_names_the_file_and_the_origin(self, tmp_path):
        """The reason provenance is carried at all: realistic failures are
        user overrides, and a traceback alone does not say which file."""
        d = tmp_path / "input" / "quality_check"
        d.mkdir(parents=True)
        (d / "template.json").write_text(
            json.dumps({"system_prompt": "S", "template": "{Severity}"}),
            encoding="utf-8",
        )
        tmpl = PromptTemplate.load("input", "quality_check", prompt_dir=tmp_path)
        with pytest.raises(ValueError) as exc:
            tmpl()
        msg = str(exc.value)
        assert "input/quality_check" in msg
        assert "override" in msg
        assert "template.json" in msg
        assert "{Severity}" in msg

    def test_a_bare_dict_still_works_without_provenance(self):
        """Constructing from a dict stays supported; the error just cannot
        name a file."""
        tmpl = PromptTemplate({"system_prompt": "S", "template": "{x}"})
        assert tmpl.name is None
        with pytest.raises(ValueError, match=r"^Prompt template is missing"):
            tmpl()

    def test_missing_prompt_raises_before_construction(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            PromptTemplate.load("input", "no_such_prompt", prompt_dir=tmp_path)


class TestTemplateKwargs:
    def test_a_placeholder_used_in_both_fields_is_passed_once(self):
        config = {"system_prompt": "Cat {Category}",
                  "template": "Check {sample} for {Category}"}
        tmpl = PromptTemplate(config, Category="violence")
        assert tmpl(sample="s")[-1]["content"] == "Check s for violence"
        assert tmpl(sample="s", Category="x")[-1]["content"] == "Check s for x"

    def test_format_instruction_honours_the_override_dir(self, tmp_path):
        d = tmp_path / "format_instructions" / "numbered"
        d.mkdir(parents=True)
        (d / "template.json").write_text(
            json.dumps({"instruction": "\n\nMINE {num_samples}"}), encoding="utf-8"
        )
        tmpl = PromptTemplate.load("output", "generation", prompt_dir=tmp_path,
                                   format_style="numbered", num_samples=3)
        assert tmpl.system_prompt.endswith("MINE 3")


def _write(root, name, config=None, filename="template.json", raw=None):
    """Write one prompt file under ``root/name`` and return its path."""
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    if raw is not None:
        path.write_bytes(raw)
    else:
        config = config or {"system_prompt": "MINE", "template": "T"}
        path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _fields(text):
    """Root placeholder names in a format string."""
    return {
        name.split(".")[0].split("[")[0]
        for _, name, _, _ in string.Formatter().parse(text or "")
        if name is not None
    }


PACKAGED = report_prompt_sources(None)["packaged"]


class TestPromptResolution:
    def test_a_single_json_file_is_the_prompt_whatever_its_name(self, tmp_path):
        _write(tmp_path, "input/quality_check", filename="mine.json")
        path, is_override = resolve_prompt("input", "quality_check", tmp_path)
        assert (path.name, is_override) == ("mine.json", True)

    def test_template_json_wins_among_several(self, tmp_path):
        _write(tmp_path, "input/quality_check", {"system_prompt": "OLD"}, "old.json")
        _write(tmp_path, "input/quality_check")
        config = load_prompt("input", "quality_check", prompt_dir=tmp_path)
        assert config["system_prompt"] == "MINE"

    def test_several_json_files_without_template_json_fall_back(self, tmp_path):
        _write(tmp_path, "input/quality_check", filename="a.json")
        _write(tmp_path, "input/quality_check", filename="b.json")
        _, is_override = resolve_prompt("input", "quality_check", tmp_path)
        assert is_override is False

    def test_nested_category_is_overridden_alone(self, tmp_path):
        _write(tmp_path, "input/generation/from_constitution/long")
        nested = "generation/from_constitution/"
        config = load_prompt("input", nested + "long", tmp_path)
        assert config["system_prompt"] == "MINE"
        assert resolve_prompt("input", nested + "short", tmp_path)[1] is False

    @pytest.mark.parametrize("kind", ["missing", "file"])
    def test_prompt_dir_that_is_not_a_directory(self, tmp_path, caplog, kind):
        prompt_dir = tmp_path / "nope"
        if kind == "file":
            prompt_dir.write_text("x", encoding="utf-8")
        assert resolve_prompt("input", "quality_check", prompt_dir)[1] is False
        with caplog.at_level(logging.WARNING, logger="redact.llms.prompting.prompts"):
            summary = report_prompt_sources(prompt_dir)
        assert "not a directory" in caplog.text
        assert summary["overridden"] == summary["unused"] == []
        assert summary["packaged"] == PACKAGED

    @pytest.mark.parametrize("pipeline, category", [
        ("input", "../quality_check"),
        ("..", "quality_check"),
        ("input", "a\\..\\b"),
        ("input", "/etc/prompts"),
        ("input", "C:\\prompts"),
    ])
    def test_a_name_outside_the_tree_is_rejected(self, tmp_path, pipeline, category):
        with pytest.raises(ValueError, match="relative path"):
            resolve_prompt(pipeline, category, tmp_path)


class TestPromptFileErrors:
    LOADERS = [
        lambda root: load_prompt("input", "quality_check", prompt_dir=root),
        lambda root: PromptTemplate.load("input", "quality_check", prompt_dir=root),
    ]

    def test_a_utf8_bom_is_accepted(self, tmp_path):
        raw = b"\xef\xbb\xbf" + json.dumps({"system_prompt": "MINE"}).encode()
        _write(tmp_path, "input/quality_check", raw=raw)
        config = load_prompt("input", "quality_check", prompt_dir=tmp_path)
        assert config == {"system_prompt": "MINE"}

    @pytest.mark.parametrize("load", LOADERS, ids=["load_prompt", "template_load"])
    @pytest.mark.parametrize("raw, reason", [
        (b'{"system_prompt": "S",}', "not valid UTF-8 JSON"),
        ('{"system_prompt": "caf\xe9"}'.encode("latin-1"), "not valid UTF-8 JSON"),
        (b'["a list"]', "JSON object"),
    ], ids=["malformed", "wrong-encoding", "not-an-object"])
    def test_an_unreadable_file_is_named(self, tmp_path, load, raw, reason):
        path = _write(tmp_path, "input/quality_check", raw=raw)
        with pytest.raises(ValueError, match=reason) as exc:
            load(tmp_path)
        assert str(path) in str(exc.value)
        assert "override" in str(exc.value)


class TestReportPromptSources:
    LOGGER = "redact.llms.prompting.prompts"

    def test_no_override_directory_lists_the_packaged_prompts(self):
        summary = report_prompt_sources(None)
        assert summary["overridden"] == summary["unused"] == []
        assert "input/generation/from_constitution/long" in summary["packaged"]
        assert len(summary["packaged"]) == len(set(summary["packaged"]))

    def test_every_packaged_name_resolves(self):
        for name in PACKAGED:
            assert resolve_prompt(*name.split("/", 1))[1] is False

    def test_several_json_files_are_not_reported_as_an_override(self, tmp_path, caplog):
        _write(tmp_path, "input/quality_check", filename="a.json")
        _write(tmp_path, "input/quality_check", filename="b.json")
        with caplog.at_level(logging.WARNING, logger=self.LOGGER):
            summary = report_prompt_sources(tmp_path)
        assert summary["overridden"] == []
        assert summary["unused"] == ["input/quality_check"]
        assert "ignored override" in caplog.text

    def test_a_prompt_with_several_files_is_listed_once(self, tmp_path):
        _write(tmp_path, "input/quality_check")
        _write(tmp_path, "input/quality_check", filename="old.json")
        summary = report_prompt_sources(tmp_path)
        assert summary["overridden"] == ["input/quality_check"]
        assert summary["unused"] == []

    def test_report_matches_resolution_for_a_case_different_directory(self, tmp_path):
        # Loaded on a case-insensitive filesystem, not loaded on others.
        _write(tmp_path, "Input/Quality_Check")
        _, loaded = resolve_prompt("input", "quality_check", tmp_path)
        summary = report_prompt_sources(tmp_path)
        assert summary["overridden"] == (["input/quality_check"] if loaded else [])
        assert summary["unused"] == ([] if loaded else ["Input/Quality_Check"])

    def test_log_says_an_override_needs_a_stage_that_passes_prompt_dir(
        self, tmp_path, caplog
    ):
        _write(tmp_path, "input/quality_check")
        with caplog.at_level(logging.INFO, logger=self.LOGGER):
            report_prompt_sources(tmp_path)
        assert "stages that pass prompt_dir" in caplog.text
        assert "override: input/quality_check" in caplog.text

    def test_many_overrides_are_counted_not_listed(self, tmp_path, caplog):
        scaffold_prompt_tree(tmp_path)
        with caplog.at_level(logging.INFO, logger=self.LOGGER):
            summary = report_prompt_sources(tmp_path)
        assert summary["overridden"] == PACKAGED
        assert f"({len(PACKAGED)} overrides;" in caplog.text
        assert "override: input/quality_check" not in caplog.text


class TestRenderErrors:
    @pytest.mark.parametrize("text", ['Reply {"verdict": "yes"}', "{0}", "{}"])
    def test_a_field_that_is_not_a_name_is_reported_as_braces(self, text):
        with pytest.raises(ValueError, match="not a valid template") as exc:
            PromptTemplate({"system_prompt": text}, verdict="x")
        assert "doubled" in str(exc.value)
        assert "is missing" not in str(exc.value)

    @pytest.mark.parametrize("text, value", [
        ("{a.nope}", "x"),
        ("{a[0]}", 5),
        ("{a[9]}", [1]),
        ("{a:>10}", [1]),
        ("{a:d}", "x"),
    ])
    def test_a_supplied_field_that_cannot_be_rendered(self, text, value):
        with pytest.raises(ValueError, match="system_prompt could not be rendered"):
            PromptTemplate({"system_prompt": text}, a=value)

    def test_a_field_nested_in_a_format_spec_is_named(self):
        with pytest.raises(ValueError, match=r"is missing \{b\}"):
            PromptTemplate({"system_prompt": "{a:{b}}"}, a="x")


class TestMessageAssembly:
    CONFIG = {
        "system_prompt": "S",
        "template": "",
        "few_shot_examples": [{"role": "user", "content": "ex {Category}"}],
    }

    def test_few_shot_examples_are_not_rendered(self):
        msgs = build_messages(self.CONFIG, Category="violence")
        assert msgs[-1] == {"role": "user", "content": "ex {Category}"}

    def test_few_shot_examples_are_not_shared_between_calls(self):
        tmpl = PromptTemplate(self.CONFIG)
        tmpl()[-1]["content"] += " EDITED"
        assert tmpl()[-1]["content"] == "ex {Category}"
        assert self.CONFIG["few_shot_examples"][0]["content"] == "ex {Category}"

    def test_empty_template_is_omitted(self):
        assert PromptTemplate({"system_prompt": "S", "template": ""})() == [
            {"role": "system", "content": "S"}
        ]

    def test_build_messages_renders_the_system_prompt_too(self):
        config = {"system_prompt": "Judge {Category}.", "template": "{Category}: {s}"}
        msgs = build_messages(config, Category="violence", s="x")
        assert [m["content"] for m in msgs] == ["Judge violence.", "violence: x"]


class TestFormatStyleValidation:
    CONFIG = {"system_prompt": "S", "template": "T"}

    @pytest.mark.parametrize("count", ["abc", "4.0", "", 3.9, None, 0, -1, "0", True])
    def test_num_samples_must_be_a_positive_integer(self, count):
        with pytest.raises(ValueError, match="num_samples must be an integer >= 1"):
            PromptTemplate(self.CONFIG, format_style="numbered", num_samples=count)

    @pytest.mark.parametrize("count", [3, "3"])
    def test_an_integer_or_a_string_of_one_is_accepted(self, count):
        tmpl = PromptTemplate(self.CONFIG, format_style="numbered", num_samples=count)
        assert "exactly 3 samples" in tmpl.system_prompt

    def test_unknown_style_is_reported_before_num_samples(self):
        with pytest.raises(ValueError, match="Unknown format style 'bogus'"):
            PromptTemplate(self.CONFIG, format_style="bogus")

    def test_empty_style_means_no_style(self):
        tmpl = PromptTemplate(self.CONFIG, format_style="")
        assert tmpl.extraction_style is None
        assert tmpl.system_prompt == "S"

    def test_null_system_prompt_takes_a_format_instruction(self):
        config = {"system_prompt": None, "template": None}
        msgs = build_messages(config, format_style="numbered", num_samples=2)
        assert [m["role"] for m in msgs] == ["system"]


class TestEmptyScaffoldRenders:
    """An ``--empty`` stub must render wherever its source does."""

    @pytest.fixture(scope="class")
    def scaffold(self, tmp_path_factory):
        root = tmp_path_factory.mktemp("scaffold")
        scaffold_prompt_tree(root, mode="empty")
        return root

    @pytest.mark.parametrize("name", PACKAGED)
    def test_each_text_field_keeps_only_its_own_placeholders(self, scaffold, name):
        pipeline, category = name.split("/", 1)
        source = load_prompt(pipeline, category)
        stub = load_prompt(pipeline, category, prompt_dir=scaffold)
        assert set(stub) == set(source)
        for key in ("system_prompt", "template", "instruction"):
            if key in source:
                assert stub[key].startswith("TODO")
                assert _fields(stub[key]) == _fields(source[key])

    @pytest.mark.parametrize(
        "name", [n for n in PACKAGED if not n.startswith("format_instructions/")]
    )
    def test_stub_renders_one_shot_and_in_two_stages(self, scaffold, name):
        pipeline, category = name.split("/", 1)
        source = load_prompt(pipeline, category)
        build = dict.fromkeys(_fields(source["system_prompt"]), "B")
        call = dict.fromkeys(_fields(source["template"]), "C")

        two_stage = PromptTemplate.load(
            pipeline, category, prompt_dir=scaffold, **build
        )(**call)
        stub = load_prompt(pipeline, category, prompt_dir=scaffold)
        one_shot = build_messages(stub, **{**call, **build})

        assert [m["role"] for m in two_stage] == ["system", "user"]
        assert [m["role"] for m in one_shot] == ["system", "user"]
        assert "{" not in two_stage[0]["content"] + two_stage[1]["content"]

    def test_the_checker_builders_accept_it(self, scaffold):
        from redact.content_moderation.checker import (
            build_category_checker,
            build_output_quality_checker,
            build_paraphrase_checker,
            build_quality_checker,
        )

        checkers = [
            build_quality_checker("violence", prompt_dir=scaffold),
            build_output_quality_checker("violence", prompt_dir=scaffold),
            build_paraphrase_checker(prompt_dir=scaffold),
            build_category_checker("violence", prompt_dir=scaffold),
        ]
        for checker in checkers:
            assert [m["role"] for m in checker("a", "b")] == ["system", "user"]

    def test_the_format_instruction_stubs_render(self, scaffold):
        tmpl = PromptTemplate.load(
            "output", "generation", prompt_dir=scaffold,
            format_style="numbered", num_samples=3,
        )
        assert tmpl.system_prompt.endswith("Placeholders: 3")
