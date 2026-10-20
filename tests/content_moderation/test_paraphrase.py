"""Tests for paraphrase_batch and the paraphrase check's gates."""

import pytest

import redact.content_moderation.paraphrase as CP
from redact.content_moderation.paraphrase import (
    GATES,
    artifact_gate,
    gate_reason,
    language_gate,
    length_gate,
    paraphrase_batch,
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


def _words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


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
