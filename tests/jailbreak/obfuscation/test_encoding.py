"""Tests for encoding/cyphering obfuscation functions."""

import base64
import codecs

from redact.jailbreak.obfuscation.encoding import (
    get_encoding_functions,
    to_ascii_ordinal,
    to_base64,
    to_braille,
    to_leetspeak,
    to_leetspeak_advanced,
    to_leetspeak_basic,
    to_leetspeak_intermediate,
    to_morse,
    to_rot13,
    to_rot18,
    to_rot47,
    to_separator,
    to_unicode_escape,
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


class TestRot18:
    def test_letters_and_digits(self):
        # ROT13 on letters, ROT5 on digits
        encoded, _ = to_rot18("abc123")
        assert encoded == "nop678"

    def test_roundtrip(self):
        encoded, _ = to_rot18(SAMPLE)
        decoded, _ = to_rot18(encoded)
        assert decoded == SAMPLE


class TestRot47:
    def test_roundtrip(self):
        encoded, _ = to_rot47(SAMPLE)
        decoded, _ = to_rot47(encoded)
        assert decoded == SAMPLE

    def test_changes_printable(self):
        encoded, _ = to_rot47("Hello")
        assert encoded != "Hello"


class TestUnicodeEscape:
    def test_format(self):
        encoded, _ = to_unicode_escape("AB")
        assert encoded == "\\u0041\\u0042"


class TestAsciiOrdinal:
    def test_format(self):
        encoded, _ = to_ascii_ordinal("AB")
        assert encoded == "65 66"


class TestSeparator:
    def test_inserts_separator(self):
        result, info = to_separator("abc")
        # one separator char between each character
        assert len(result) == 5
        assert info.startswith("sep=")


class TestLeetspeak:
    def test_known_replacements(self):
        result, _ = to_leetspeak("aeiost")
        assert result == "431057"

    def test_basic_alias(self):
        assert to_leetspeak("aeiost") == to_leetspeak_basic("aeiost")

    def test_preserves_unmapped(self):
        result, _ = to_leetspeak("xyz")
        assert result == "xyz"

    def test_preserves_length(self):
        result, _ = to_leetspeak(SAMPLE)
        assert len(result) == len(SAMPLE)

    def test_intermediate_adds_substitutions(self):
        # 'b' is unmapped in basic but '8' in intermediate
        assert to_leetspeak_basic("b")[0] == "b"
        assert to_leetspeak_intermediate("b")[0] == "8"

    def test_advanced_multichar(self):
        # advanced map has multi-character substitutions (e.g. h -> |-|)
        result, _ = to_leetspeak_advanced("h")
        assert result == "|-|"


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
        assert len(get_encoding_functions()) == 12

    def test_output_differs_from_input(self):
        for fn in get_encoding_functions():
            result, _ = fn(SAMPLE)
            assert result != SAMPLE, f"{fn.__name__} output should differ"
