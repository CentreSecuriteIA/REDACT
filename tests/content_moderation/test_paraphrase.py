"""Tests for paraphrase_batch, the paraphrase prompts and the check's gates."""

import hashlib
import json
import logging

import pytest

import redact.content_moderation.paraphrase as CP
from redact.content_moderation.paraphrase import (
    GATES,
    artifact_gate,
    check_paraphrase_prompt,
    gate_reason,
    language_gate,
    length_gate,
    paraphrase_batch,
    paraphrase_prompt,
    script_gate,
    split_gate,
    squeezed,
)
from redact.jailbreak.obfuscation import langid
from tests.conftest import MockBackend, make_client


class TestParaphraseBatch:
    def test_returns_model_outputs_in_order(self):
        backend = MockBackend(["p1", "p2", "p3"])
        result = paraphrase_batch(make_client(backend, "model"), ["t1", "t2", "t3"])
        assert result == ["p1", "p2", "p3"]
        assert len(backend.calls) == 3

    def test_empty_list(self):
        assert paraphrase_batch(make_client(MockBackend()), []) == []

    def test_the_sample_is_stripped(self):
        backend = MockBackend("p")
        paraphrase_batch(make_client(backend, "venice-paraphraser"), [" a b \n"])
        assert backend.calls[0]["messages"][-1]["content"] == "TEXT:\na b"


def _prompt(root, name: str, system: str) -> None:
    """Write ``paraphrase/<name>/template.json`` under ``root``."""
    folder = root / "paraphrase" / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "template.json").write_text(
        json.dumps({"system_prompt": system, "template": "{sample}"}), encoding="utf-8")


class TestParaphrasePrompt:
    """``paraphrase/generation/<model>`` is a model's own prompt."""

    def test_a_model_with_a_prompt_of_its_own_gets_it(self, tmp_path):
        _prompt(tmp_path, "generation/mock-model", "OWN")
        backend = MockBackend("p")
        paraphrase_batch(make_client(backend, "mock-model"), ["t"], prompt_dir=tmp_path)
        assert backend.calls[0]["system_prompt"] == "OWN"
        template = paraphrase_prompt("mock-model", tmp_path)
        assert (template.name, template.is_override) == (
            "paraphrase/generation/mock-model", True)

    def test_a_model_without_one_gets_the_generic_prompt(self, tmp_path):
        _prompt(tmp_path, "generation/mock-model", "OWN")
        template = paraphrase_prompt("venice-paraphraser", tmp_path)
        assert (template.name, template.is_override) == ("paraphrase/generation", False)

    def test_a_generic_override_does_not_reach_a_model_with_its_own(self, tmp_path):
        _prompt(tmp_path, "generation", "GENERIC")
        _prompt(tmp_path, "generation/mock-model", "OWN")
        assert paraphrase_prompt("mock-model", tmp_path).system_prompt == "OWN"
        assert paraphrase_prompt("venice-paraphraser", tmp_path).system_prompt == (
            "GENERIC")


_SEP = "\n\n### Rewritten:\n\n"


def _card_instruction() -> str:
    from redact.llms.prompting import load_prompt

    return load_prompt("paraphrase", "generation/humanizer-12b")["system_prompt"]


def _folded(draft: str) -> str:
    """The humanizer's user message for ``draft``, as the stage builds it."""
    backend = MockBackend("p")
    paraphrase_batch(make_client(backend, "humanizer-12b"), [draft])
    (message,) = backend.calls[0]["messages"]
    assert message["role"] == "user" and "system_prompt" not in backend.calls[0]
    return message["content"]


def _render_chat_template(messages: list[dict]) -> str:
    """The entry's chat_template, rendered as transformers renders one."""
    jinja2 = pytest.importorskip("jinja2")
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    from redact.llms.model_config import get_model_config

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = raise_exception
    template = env.from_string(get_model_config("humanizer-12b").vllm.chat_template)
    return template.render(messages=messages, bos_token="<bos>",
                           add_generation_prompt=True)


class TestThePool:
    def test_the_distribution_is_the_humanizer_alone(self):
        assert CP.paraphrase_pool("distribution") == ["humanizer-12b"]
        assert CP.paraphrase_pool(None) == ["humanizer-12b"]

    def test_the_stand_in_is_still_a_name(self):
        assert CP.paraphrase_pool("venice-paraphraser") == ["venice-paraphraser"]


class TestHumanizerPrompt:
    """The packaged prompt renders the model card's string, byte for byte."""

    def test_the_prompt_for_x_hashes_to_the_cards_fingerprint(self):
        assert hashlib.sha256(_folded("X").encode()).hexdigest()[:16] == (
            "cc51d66b4c593fbe")

    def test_the_draft_is_stripped_between_the_instruction_and_the_separator(self):
        assert _folded("  some draft \n") == (
            _card_instruction() + "\n\n" + "some draft" + _SEP)

    def test_a_generic_override_does_not_reach_it(self, tmp_path):
        folder = tmp_path / "paraphrase" / "generation"
        folder.mkdir(parents=True)
        (folder / "template.json").write_text(
            json.dumps({"system_prompt": "GENERIC", "template": "{sample}"}),
            encoding="utf-8")
        template = paraphrase_prompt("humanizer-12b", tmp_path)
        assert (template.name, template.is_override) == (
            "paraphrase/generation/humanizer-12b", False)

    def test_the_check_passes_the_packaged_prompt(self):
        check_paraphrase_prompt("humanizer-12b")

    def test_the_check_stops_an_override_with_another_separator(self, tmp_path):
        from redact.llms.prompting import load_prompt

        broken = load_prompt("paraphrase", "generation/humanizer-12b")
        broken["template"] = "{sample}\n\nRewritten:\n\n"
        folder = tmp_path / "paraphrase" / "generation" / "humanizer-12b"
        folder.mkdir(parents=True)
        (folder / "template.json").write_text(json.dumps(broken), encoding="utf-8")
        with pytest.raises(ValueError, match="cc51d66b4c593fbe") as raised:
            check_paraphrase_prompt("humanizer-12b", tmp_path)
        assert "override" in str(raised.value)
        assert str(folder / "template.json") in str(raised.value)

    def test_an_override_without_the_key_is_unchecked_with_a_warning(
            self, tmp_path, caplog):
        folder = tmp_path / "paraphrase" / "generation" / "humanizer-12b"
        folder.mkdir(parents=True)
        (folder / "template.json").write_text(
            json.dumps({"system_prompt": "Reword this.", "template": "{sample}"}),
            encoding="utf-8")
        with caplog.at_level(logging.WARNING):
            check_paraphrase_prompt("humanizer-12b", tmp_path)
        (warning,) = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert "format is not checked" in warning.getMessage()

    def test_a_prompt_without_the_key_or_a_model_not_registered_is_not_checked(
            self, caplog):
        with caplog.at_level(logging.WARNING):
            check_paraphrase_prompt("venice-paraphraser")
            check_paraphrase_prompt("no-such-model")
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_an_empty_scaffold_of_it_fails_the_check(self, tmp_path):
        """--empty keeps the key and replaces the text with a TODO."""
        from redact.llms.prompting import scaffold_prompt_tree

        scaffold_prompt_tree(tmp_path, mode="empty")
        with pytest.raises(ValueError, match="does not render"):
            check_paraphrase_prompt("humanizer-12b", tmp_path)

    def test_the_chat_template_gives_bos_and_the_folded_prompt(self):
        content = _folded("A short draft to rewrite, of a dozen words or so here.")
        assert _render_chat_template([{"role": "user", "content": content}]) == (
            "<bos>" + content)

    @pytest.mark.parametrize("messages", [
        [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}],
        [{"role": "system", "content": "s"}, {"role": "user", "content": "a"}],
        [{"role": "system", "content": "s"}],
    ])
    def test_the_chat_template_refuses_anything_but_one_user_message(self, messages):
        jinja2 = pytest.importorskip("jinja2")
        with pytest.raises(jinja2.exceptions.TemplateError, match="one user message"):
            _render_chat_template(messages)


def _words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


class TestReplyCap:
    """About 2.5 times the original's tokens, plus slack, rounded up to a
    power of two from 64, at most the ceiling."""

    def test_a_short_text_gets_the_floor(self):
        assert (CP.reply_cap("x" * 6, 2048), CP.REPLY_FLOOR_TOKENS) == (64, 64)

    def test_a_text_above_the_floor_gets_its_need_rounded_up(self):
        # 60 characters, 20 tokens: 2.5 x 20 + 32 = 82, rounded up to 128.
        assert [CP.reply_cap("x" * 60, 2048)] == [128]
        # 600 characters, 200 tokens: 532, rounded up to 1024.
        assert [CP.reply_cap("x" * 600, 2048)] == [1024]

    def test_the_caps_are_few(self):
        caps = {CP.reply_cap("x" * n, 2048) for n in range(0, 3000)}
        assert caps == {64, 128, 256, 512, 1024, 2048}

    def test_a_ceiling_under_the_power_of_two_wins(self):
        assert [CP.reply_cap("x" * 10_000, 2000)] == [2000]

    def test_a_long_text_gets_the_ceiling(self):
        assert [CP.reply_cap("x" * 10_000, 2048)] == [2048]


class TestSkipReason:
    def test_a_text_under_min_words_is_short(self):
        assert CP.skip_reason(_words(11), min_words=12) == "short"

    def test_a_text_of_min_words_is_sent(self):
        assert CP.skip_reason(_words(12), min_words=12) == ""

    def test_zero_sends_every_text(self):
        assert CP.skip_reason("", min_words=0) == ""

    def test_runs_of_spaces_and_newlines_are_one_break(self):
        assert CP.skip_reason("one  two\n\nthree \t four", min_words=4) == ""
        assert CP.skip_reason("one  two\n\nthree", min_words=4) == "short"

    def test_the_default_is_twelve(self):
        assert (CP.MIN_WORDS, CP.DEFAULT_TARGET) == (12, "inputs")


SPLIT = "split into lines"


class TestGates:
    def test_squeezed_takes_the_layout_out_of_a_table_and_a_rule_line(self):
        table = "| name | size |\n|------|:----:|\n| alpha | 2 |"
        assert squeezed(table) == "name size alpha 2"
        ruled = "Title\n=====\n\nsome   body\ttext\n---\n"
        assert squeezed(ruled) == "Title some body text"

    def test_length_inside_the_band_passes(self):
        assert length_gate("a" * 100, "b" * 100) == ""
        assert length_gate("a" * 100, "b" * 170) == ""   # 1.5x + 20, the edge
        assert length_gate("a" * 100, "b" * 50) == ""    # 0.5x

    def test_length_above_the_band_fails(self):
        assert length_gate("a" * 100, "b" * 171) == "length ratio 1.71"
        # A short original has the slack: 10 characters may become 35.
        assert length_gate("a" * 10, "b" * 35) == ""
        assert length_gate("a" * 10, "b" * 36) == "length ratio 3.60"

    def test_length_under_the_band_fails(self):
        assert length_gate("a" * 100, "b" * 49) == "length ratio 0.49"

    def test_length_is_read_squeezed(self):
        table = "| a | b |\n|---|---|\n| c | d |"
        assert length_gate("a b c d", table) == ""

    def test_length_against_an_empty_squeezed_original(self):
        assert length_gate("|---|", "x" * 21) == "length 21 against an empty original"
        assert length_gate("", "x" * 20) == ""

    def test_a_short_single_line_split_into_lines_fails(self):
        assert split_gate(_words(35), _words(17) + "\n" + _words(18)) == SPLIT

    def test_a_longer_single_line_may_be_split(self):
        assert split_gate(_words(36), _words(18) + "\n" + _words(18)) == ""

    def test_a_multi_line_original_may_be_split(self):
        assert split_gate("line one\nline two", "line one\nline\ntwo") == ""

    def test_blank_lines_do_not_count(self):
        assert split_gate("one short line", "\n  one short line \n\n   \n") == ""
        assert split_gate("one short line", "one short\n   \nline") == SPLIT

    def test_latin_to_han_fails(self):
        reason = script_gate("Could you suggest a few easy dinners?",
                             "你能推荐几道简单的晚餐吗？")
        assert reason == "script changed: Latin -> Han"

    def test_latin_to_latin_passes(self):
        assert script_gate("Could you suggest dinners?", "Can you name dinners?") == ""

    def test_a_han_name_in_a_long_latin_text_passes(self):
        text = "Please write a short note to my colleague about the meeting next week"
        assert script_gate(text, text + " with 王小明") == ""

    def test_cyrillic_to_latin_fails(self):
        reason = script_gate("Привет, как дела у тебя сегодня?",
                             "Hello, how are you today?")
        assert reason == "script changed: Cyrillic -> Latin"

    def test_the_reason_names_the_script_the_original_lacks(self):
        # Most letters stay Latin; the Cyrillic ones are over 10%.
        reason = script_gate("hello there my friend", "hello there my friend привет")
        assert reason == "script changed: Latin -> Cyrillic"

    def test_a_text_without_letters_passes(self):
        assert script_gate("Could you suggest dinners?", "") == ""
        assert script_gate("1234 + 5678 = ?", "Сколько будет?") == ""

    @pytest.fixture()
    def labels(self, monkeypatch):
        """Patch GlotLID: ``labels[text] = (label, probability)``."""
        found: dict = {}
        monkeypatch.setattr(langid, "label_probability", found.get)
        return found

    def test_different_languages_both_sure_fail(self, labels):
        labels.update({"hello": ("eng_Latn", 0.95), "hallo": ("deu_Latn", 0.95)})
        assert language_gate("hello", "hallo") == "language eng_Latn -> deu_Latn"

    def test_an_unsure_label_passes(self, labels):
        labels.update({"hello": ("eng_Latn", 0.95), "hallo": ("deu_Latn", 0.6)})
        assert language_gate("hello", "hallo") == ""

    def test_the_same_language_passes(self, labels):
        labels.update({"hello": ("eng_Latn", 0.95), "hi": ("eng_Latn", 0.99)})
        assert language_gate("hello", "hi") == ""

    def test_without_a_label_the_language_gate_passes(self, labels):
        labels.update({"hello": ("eng_Latn", 0.95)})
        assert language_gate("hello", "hallo") == ""

    def test_without_glotlid_the_language_gate_passes(self):
        assert language_gate("hello there", "hallo da") == ""

    def test_an_html_tag_added_fails(self):
        assert artifact_gate("Some text", "<p>Some text</p>") == "artifact <p>"

    def test_a_rule_line_inside_a_text_fails(self):
        assert artifact_gate("A title and a body",
                             "A title\n=====\nand a body") == "artifact ====="

    def test_a_tag_the_original_has_too_passes(self):
        assert artifact_gate("<p>Some text</p>", "<p>Some words</p>") == ""

    def test_a_tag_outside_the_list_passes(self):
        assert artifact_gate("Some code", "<pre>Some code</pre>") == ""

    def test_the_first_gate_that_fails_names_the_reason(self):
        original = "Could you suggest some easy dinners"
        # Too long and split: the length gate comes first.
        assert gate_reason(original, original + "\n" + original * 2).startswith(
            "length ratio")
        # Split and another script: the split gate comes first.
        han = "你能推荐几道简单又快手的晚餐吗\n适合忙碌的工作日晚上做的那种"
        assert gate_reason(original, han) == SPLIT
        assert gate_reason(original, "Can you suggest some simple dinners?") == ""

    def test_the_gates_and_their_thresholds(self):
        assert [gate.__name__ for gate in GATES] == [
            "length_gate", "split_gate", "script_gate", "language_gate",
            "artifact_gate"]
        assert (CP.LENGTH_HIGH, CP.LENGTH_SLACK, CP.LENGTH_LOW) == (1.5, 20, 0.5)
        assert (CP.SPLIT_WORDS, CP.LANGID_PROBABILITY) == (35, 0.9)
        assert (CP.SCRIPT_SHARE, CP.SCRIPT_RARE) == (0.10, 0.01)
