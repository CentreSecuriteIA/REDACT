"""Tests for encoding/cyphering obfuscation functions."""

import base64
import codecs

from redact.jailbreak.obfuscation.encoding import (
    to_base64,
    to_rot13,
    to_leetspeak,
    to_morse,
    to_braille,
    get_encoding_functions,
    LEET_MAP,
)


SAMPLE = "Hello World"


class TestBase64:
    def test_roundtrip(self):
        encoded, info = to_base64(SAMPLE)
        decoded = base64.b64decode(encoded).decode()
        assert decoded == SAMPLE

    def test_returns_tuple(self):
        result = to_base64(SAMPLE)
        assert isinstance(result, tuple) and len(result) == 2


class TestRot13:
    def test_roundtrip(self):
        encoded, _ = to_rot13(SAMPLE)
        decoded = codecs.encode(encoded, "rot_13")
        assert decoded == SAMPLE

    def test_letters_changed(self):
        encoded, _ = to_rot13("abc")
        assert encoded == "nop"


class TestLeetspeak:
    def test_known_replacements(self):
        result, _ = to_leetspeak("aeiost")
        assert result == "431057"

    def test_preserves_unmapped(self):
        result, _ = to_leetspeak("xyz")
        assert result == "xyz"

    def test_preserves_length(self):
        result, _ = to_leetspeak(SAMPLE)
        assert len(result) == len(SAMPLE)


class TestMorse:
    def test_known_output(self):
        result, _ = to_morse("SOS")
        assert result == "... --- ..."

    def test_space_is_slash(self):
        result, _ = to_morse("A B")
        assert "/" in result


class TestBraille:
    def test_known_output(self):
        result, _ = to_braille("ab")
        assert result == "\u2801\u2803"

    def test_space_preserved(self):
        result, _ = to_braille("a b")
        assert " " in result


class TestAllEncodings:
    def test_all_return_tuple_with_info(self):
        for fn in get_encoding_functions():
            result = fn(SAMPLE)
            assert isinstance(result, tuple), f"{fn.__name__} must return tuple"
            assert len(result) == 2

    def test_get_encoding_functions_count(self):
        assert len(get_encoding_functions()) == 5

    def test_output_differs_from_input(self):
        for fn in get_encoding_functions():
            result, _ = fn(SAMPLE)
            assert result != SAMPLE, f"{fn.__name__} output should differ"
