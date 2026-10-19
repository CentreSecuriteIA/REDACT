"""Tests for the sensitive-word techniques."""

import pytest

from redact.jailbreak.obfuscation.encoding import to_base64
from redact.jailbreak.obfuscation.sensitive_words import (
    _sensitive_char_finish,
    _sensitive_encode_finish,
    _sensitive_separator_finish,
    to_sensitive_words_variables,
    to_synonym_substitution,
)
from redact.llm_pipeline import drive_sync


def _run(technique, prompt: str, replies: list[str]):
    """``technique`` on ``prompt``, its model answering ``replies`` in turn."""
    answers = iter(replies)
    return drive_sync(technique(prompt, gen_model="m"), lambda request: next(answers))


class TestSensitiveEncodeFinish:
    def test_an_encoded_word_with_backslashes_is_inserted_as_it_is(self):
        """A unicode-escaped word used to fail in re.sub: "bad escape \\u"."""

        result, info = _sensitive_encode_finish(
            "say apple now", lambda w: ("\\u0061pple", ""), "unicode", ["apple"])
        assert result == "say \\u0061pple now"
        assert info == "encoding=unicode;detected_words=apple"

    def test_a_base64_word_breaks_no_line(self):
        result, _ = _sensitive_encode_finish(
            "how to grow an apple at home", to_base64, "base64", ["apple"])
        assert result == "how to grow an YXBwbGU= at home"


class TestSeparatorFinishes:
    WORD = "cafe" + chr(0x301)      # the accent a combining mark

    def test_a_fixed_separator_keeps_a_mark_with_its_letter(self):
        result, _ = _sensitive_char_finish(
            f"a {self.WORD} here", "-", "hyphen", [self.WORD])
        assert result == "a c-a-f-" + "e" + chr(0x301) + " here"

    def test_a_drawn_separator_keeps_a_mark_with_its_letter(self):
        import random

        result, info = _sensitive_separator_finish(
            f"a {self.WORD} here", [self.WORD], random.Random(0))
        sep = info.split(";")[1].removeprefix("sep=")
        assert result == f"a {sep.join(['c', 'a', 'f', 'e' + chr(0x301)])} here"


def test_the_substitutions_are_one_field():
    _, info = _run(to_synonym_substitution, "an apple and a pear",
                   ["apple pear", "fruit", "produce"])
    assert info == "substitutions=apple->fruit,pear->produce"


def test_a_bulleted_word_list_marks_those_words_only():
    text, info = _run(to_sensitive_words_variables,
                      "a well-known apple-growing guide to pears", ["- apple\n- pears"])
    assert text.startswith("a well-known VAR_1-growing guide to VAR_2")
    assert info == "char_sep=variables;detected_words=apple pears"


def test_a_labelled_word_list_is_rejected():
    from tests.jailbreak.obfuscation.test_typos import row_of_replies

    row = row_of_replies(to_sensitive_words_variables, ["Harmful words: apple"],
                         prompt="how to grow an apple")
    assert (row["accepted"], row["jailbreak"]) == (False, "how to grow an apple")
    assert "technique=to_sensitive_words_variables" in row["reasoning"]
    assert "malformed word list" in row["reasoning"]


# --- no word marked --------------------------------------------------------------

_HEAVY = ["to_sensitive_words_encode_base64", "to_sensitive_words_encode_unicode",
          "to_sensitive_words_encode_ascii"]


def _technique(name: str):
    from redact.jailbreak.obfuscation.sensitive_words import (
        get_sensitive_words_functions,
    )

    return {fn.__name__: fn for fn in get_sensitive_words_functions()}[name]


def test_the_heavy_word_encodings_are_those_the_spec_tags_heavy():
    from redact.jailbreak.obfuscation.sensitive_words import (
        get_sensitive_words_functions,
    )

    assert sorted(fn.__name__ for fn in get_sensitive_words_functions()
                  if fn.encode_weight == "heavy") == sorted(_HEAVY)


@pytest.mark.parametrize("name", _HEAVY)
def test_a_heavy_encoding_with_no_word_marked_discards_the_prompt(name):
    """Encoding the whole prompt would be a heavy full-text encoding, which
    the spec keeps away from fsh and dap."""
    from redact.jailbreak.obfuscation.sensitive_words import NO_SENSITIVE_WORD

    prompt = "how to grow an apple"
    assert _run(_technique(name), prompt, ["none"]) == (prompt, NO_SENSITIVE_WORD)
    text, info = _run(_technique(name), prompt, ["apple"])
    assert text != prompt and info.endswith("detected_words=apple")


@pytest.mark.parametrize(("name", "expected"), [
    ("to_sensitive_words_encode_rot13", "ubj gb tebj na nccyr"),
    ("to_sensitive_words_encode_leetspeak_basic", "h0w 70 gr0w 4n 4ppl3"),
    ("to_sensitive_words_split", "h o w t o g r o w a n a p p l e"),
    ("to_sensitive_words_variables", "how to grow an apple"),
])
def test_the_other_techniques_keep_their_fallback(name, expected):
    text, info = _run(_technique(name), "how to grow an apple", ["none"])
    assert text == expected
    assert info.endswith("detected_words=")


def test_a_heavy_discard_rejects_the_row_naming_the_technique():
    from tests.jailbreak.obfuscation.test_typos import row_of_replies

    row = row_of_replies(_technique("to_sensitive_words_encode_base64"), ["none"],
                         prompt="how to grow an apple")
    assert (row["accepted"], row["jailbreak"]) == (False, "how to grow an apple")
    assert "technique=to_sensitive_words_encode_base64" in row["reasoning"]
    assert "no sensitive word" in row["reasoning"]
