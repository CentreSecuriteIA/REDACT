"""Tests for the sensitive-word techniques."""

from redact.jailbreak.obfuscation.sensitive_words import _sensitive_encode_finish


class TestSensitiveEncodeFinish:
    def test_an_encoded_word_with_backslashes_is_inserted_as_it_is(self):
        """A unicode-escaped word used to fail in re.sub: "bad escape \\u"."""

        result, info = _sensitive_encode_finish(
            "say apple now", lambda w: ("\\u0061pple", ""), "unicode", ["apple"])
        assert result == "say \\u0061pple now"
        assert info == "encoding=unicode;detected_words=apple"
