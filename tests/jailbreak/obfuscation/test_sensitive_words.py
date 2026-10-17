"""Tests for the sensitive-word techniques."""

from redact.jailbreak.obfuscation.encoding import to_base64
from redact.jailbreak.obfuscation.sensitive_words import (
    _sensitive_encode_finish,
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


def test_the_substitutions_are_one_field():
    _, info = _run(to_synonym_substitution, "an apple and a pear",
                   ["apple pear", "fruit", "produce"])
    assert info == "substitutions=apple->fruit,pear->produce"
