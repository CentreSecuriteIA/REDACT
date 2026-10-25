"""Tests for encoding/cyphering obfuscation functions."""

import base64
import codecs
import html
import random
import re
import unicodedata
import urllib.parse

import pytest

from redact.jailbreak.obfuscation.encoding import (
    _LOOKALIKE_MAP,
    _MATH_STYLES,
    _ZERO_WIDTH,
    _clusters,
    get_encoding_functions,
    to_ascii_ordinal,
    to_atbash,
    to_base32,
    to_base64,
    to_base85,
    to_binary,
    to_braille,
    to_caesar_cipher,
    to_fullwidth,
    to_hex,
    to_html_entities,
    to_leetspeak_advanced,
    to_leetspeak_basic,
    to_leetspeak_intermediate,
    to_lookalike_letters,
    to_math_styled,
    to_morse,
    to_reversed_characters,
    to_reversed_word_order,
    to_reversed_words,
    to_rot13,
    to_rot18,
    to_rot47,
    to_separator,
    to_unicode_escape,
    to_url_encoding,
    to_zero_width_noise,
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

    def test_a_long_prompt_encodes_to_one_line(self):
        prompt = "a long prompt " * 15
        encoded, _ = to_base64(prompt)
        assert "\n" not in encoded
        assert base64.b64decode(encoded).decode() == prompt


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

    def test_only_ascii_digits_rotate(self):
        assert to_rot18("CO₂ in m² at ① and 5") == ("PB₂ va z² ng ① naq 0", "")


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


    def test_a_character_above_the_basic_plane_decodes_back(self):
        encoded, _ = to_unicode_escape("a😀")
        assert encoded.endswith("U0001f600")
        assert codecs.decode(encoded, "unicode_escape") == "a😀"


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


    def test_a_mark_or_a_joined_sequence_stays_with_its_character(self):
        accented = "e" + chr(0x301)     # e, then a combining acute accent
        family = chr(0x1F468) + chr(0x200D) + chr(0x1F469)   # joined by a ZWJ
        text, info = to_separator(f"{accented}{family}!", rng=random.Random(0))
        assert text.split(info.removeprefix("sep=")) == [accented, family, "!"]


THUMB = chr(0x1F44D)
TONE = chr(0x1F3FD)                       # a skin-tone modifier
FRANCE = chr(0x1F1EB) + chr(0x1F1F7)      # regional indicators F, R
GERMANY = chr(0x1F1E9) + chr(0x1F1EA)     # regional indicators D, E
SPAIN_ITALY = [chr(0x1F1EA) + chr(0x1F1F8), chr(0x1F1EE) + chr(0x1F1F9)]
KOREAN = unicodedata.normalize("NFD", "한국어")      # eight jamo, three syllables


class TestClusters:
    def test_a_skin_tone_stays_with_its_emoji(self):
        assert _clusters(f"a{THUMB}{TONE}b") == ["a", THUMB + TONE, "b"]

    def test_the_two_letters_of_a_flag_stay_together_and_the_next_flag_starts_anew(self):
        assert _clusters(FRANCE + GERMANY) == [FRANCE, GERMANY]
        assert _clusters("x" + "".join(SPAIN_ITALY)) == ["x", *SPAIN_ITALY]

    def test_a_third_regional_indicator_starts_a_cluster(self):
        assert _clusters(FRANCE + FRANCE[0]) == [FRANCE, FRANCE[0]]

    def test_the_jamo_of_a_decomposed_syllable_stay_together(self):
        assert len(KOREAN) == 8     # noqa: PLR2004
        assert ["".join(unicodedata.normalize("NFD", syllable))
                for syllable in "한국어"] == _clusters(KOREAN)

    def test_a_syllable_with_two_trailing_consonants_is_one_cluster(self):
        text = chr(0x1103) + chr(0x1161) + chr(0x11AF) + chr(0x11A8)
        assert _clusters(text) == [text]

    def test_a_vowel_or_a_lead_after_a_trailing_consonant_starts_a_cluster(self):
        text = unicodedata.normalize("NFD", "각아")
        assert _clusters(text) == [text[:3], text[3:]]

    @pytest.mark.parametrize("text", [
        THUMB + TONE, FRANCE + GERMANY, KOREAN, "e" + chr(0x301) + "x",
        chr(0x1F468) + chr(0x200D) + chr(0x1F469), "plain text", "",
    ])
    def test_the_clusters_give_the_text_back(self, text):
        assert "".join(_clusters(text)) == text

    def test_a_flag_reversed_is_the_same_flag(self):
        assert to_reversed_characters(f"a{FRANCE}b")[0] == f"b{FRANCE}a"

    def test_an_emoji_with_its_tone_reverses_as_one_character(self):
        assert to_reversed_characters(f"a{THUMB}{TONE}b")[0] == f"b{THUMB}{TONE}a"

    def test_a_separator_goes_between_clusters_not_inside_them(self):
        text, info = to_separator(f"{THUMB}{TONE}{FRANCE}{KOREAN}", rng=random.Random(0))
        sep = info.removeprefix("sep=")
        assert text.split(sep) == [THUMB + TONE, FRANCE,
                                   *_clusters(KOREAN)]

    def test_zero_width_noise_never_cuts_a_cluster(self):
        text = f"a{THUMB}{TONE}{FRANCE}{KOREAN}"
        for seed in range(20):
            encoded, _ = to_zero_width_noise(text, rng=random.Random(seed))
            assert [c for c in _clusters(encoded) if c not in _ZERO_WIDTH] == _clusters(text)


LONE_SURROGATE = chr(0xD800)


class TestEdgeInputs:
    """What the encodings do with text no pipeline row holds in practice, so a
    change to it is a decision."""

    @pytest.mark.parametrize("fn", [to_base64, to_hex, to_binary, to_base32,
                                    to_base85, to_url_encoding],
                             ids=lambda fn: fn.__name__)
    def test_a_lone_surrogate_stops_a_byte_codec_with_a_unicode_error(self, fn):
        """No replacement character is written in its place: the unit gets an
        error row, which names the codec and the position."""
        with pytest.raises(UnicodeEncodeError, match="surrogates not allowed"):
            fn("a" + LONE_SURROGATE)

    @pytest.mark.parametrize("fn", [to_html_entities, to_unicode_escape,
                                    to_ascii_ordinal],
                             ids=lambda fn: fn.__name__)
    def test_a_lone_surrogate_is_written_as_ascii_by_a_code_point_encoding(self, fn):
        assert fn("a" + LONE_SURROGATE)[0].isascii()

    def test_html_entities_read_back_by_html5_but_for_nul_and_the_c1_controls(self):
        """``html.unescape`` reads ``&#0;`` as U+FFFD and 27 of ``&#128;`` to
        ``&#159;`` as their Windows-1252 characters (``&#133;`` is an ellipsis),
        so those characters do not come back as written; every other one does,
        also the 5 C1 codes Windows-1252 leaves undefined."""
        undefined = (0x81, 0x8D, 0x8F, 0x90, 0x9D)
        text = SAMPLE + " caf" + chr(0xE9) + chr(0xA0) + chr(0x1F600) + "\t\n"
        text += "".join(map(chr, undefined))
        assert html.unescape(to_html_entities(text)[0]) == text
        for code in (0, *(c for c in range(0x80, 0xA0) if c not in undefined)):
            assert html.unescape(to_html_entities(chr(code))[0]) != chr(code), code

    @pytest.mark.parametrize(("text", "back"), [
        (chr(0x301) + "ab", "a" + chr(0x301) + "b"),     # a mark with no letter before it
        ("ab" + chr(0x200D), "b" + chr(0x200D) + "a"),   # a joiner with nothing after it
    ], ids=["leading-mark", "trailing-joiner"])
    def test_reversing_twice_is_not_the_identity_for_a_mark_or_joiner_at_the_edge(
            self, text, back):
        """A mark stays with the letter before it and a joiner binds the
        character after it, so a mark that starts the text lands on the letter
        that follows it on the second pass, and a joiner that ends it binds the
        letter that comes before it."""
        once = to_reversed_characters(text)[0]
        assert to_reversed_characters(once)[0] == back != text


class TestLeetspeak:
    def test_known_replacements(self):
        result, _ = to_leetspeak_basic("aeiost")
        assert result == "431057"

    def test_preserves_unmapped(self):
        result, _ = to_leetspeak_basic("xyz")
        assert result == "xyz"

    def test_preserves_length(self):
        result, _ = to_leetspeak_basic(SAMPLE)
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


class TestReversal:
    def test_characters_reversed(self):
        assert to_reversed_characters("abc")[0] == "cba"

    def test_a_base_letter_keeps_the_marks_after_it(self):
        accented = "e" + chr(0x301)      # e, then a combining acute accent
        # the accent reverses with its letter, not to the string's front
        assert to_reversed_characters(accented + "a")[0] == "a" + accented

    def test_each_word_is_reversed_in_place(self):
        assert to_reversed_words("Hello World")[0] == "olleH dlroW"

    def test_word_order_reversed(self):
        assert to_reversed_word_order("one two three")[0] == "three two one"

    def test_each_is_its_own_inverse(self):
        for fn in (to_reversed_characters, to_reversed_words,
                   to_reversed_word_order):
            assert fn(fn(SAMPLE)[0])[0] == SAMPLE, fn.__name__


class TestLookalikeLetters:
    def test_a_fraction_of_the_letters_become_cyrillic_lookalikes(self):
        encoded, _ = to_lookalike_letters(SAMPLE, rng=random.Random(0))
        assert encoded != SAMPLE
        # every change is a mapped letter swapped for its look-alike
        back = {v: k for k, v in _LOOKALIKE_MAP.items()}
        assert "".join(back.get(c, c) for c in encoded) == SAMPLE

    def test_a_fixed_seed_is_deterministic(self):
        a = to_lookalike_letters(SAMPLE, rng=random.Random(7))[0]
        b = to_lookalike_letters(SAMPLE, rng=random.Random(7))[0]
        assert a == b

    def test_a_text_with_no_mapped_letter_is_unchanged(self):
        assert to_lookalike_letters("123 ---")[0] == "123 ---"


class TestZeroWidthNoise:
    def test_stripping_the_zero_width_characters_recovers_the_text(self):
        encoded, _ = to_zero_width_noise(SAMPLE, rng=random.Random(0))
        assert encoded != SAMPLE
        assert "".join(c for c in encoded if c not in _ZERO_WIDTH) == SAMPLE

    def test_each_zero_width_character_stands_alone_for_a_later_cut(self):
        """A joiner would bind the character after it into one cluster, so a
        separator or a reversal run on the noised text would not split there."""
        for seed in range(20):
            encoded, _ = to_zero_width_noise(SAMPLE, rng=random.Random(seed))
            assert [c for c in _clusters(encoded) if c not in _ZERO_WIDTH] == list(SAMPLE)

    def test_a_fixed_seed_is_deterministic(self):
        a = to_zero_width_noise(SAMPLE, rng=random.Random(7))[0]
        b = to_zero_width_noise(SAMPLE, rng=random.Random(7))[0]
        assert a == b


class TestCodecs:
    def test_hex_roundtrip(self):
        assert bytes.fromhex(to_hex(SAMPLE)[0]).decode() == SAMPLE

    def test_binary_roundtrip(self):
        bits = to_binary(SAMPLE)[0].split()
        assert bytes(int(b, 2) for b in bits).decode() == SAMPLE

    def test_url_roundtrip(self):
        assert urllib.parse.unquote(to_url_encoding(SAMPLE)[0]) == SAMPLE

    def test_url_encoding_leaves_only_letters_digits_and_the_unreserved_marks(self):
        text = "a/b c?d=e&f~g_h-i.j é"
        encoded, _ = to_url_encoding(text)
        assert encoded == "a%2Fb%20c%3Fd%3De%26f~g_h-i.j%20%C3%A9"
        assert urllib.parse.unquote(encoded) == text

    def test_html_entities_roundtrip(self):
        encoded, _ = to_html_entities(SAMPLE)
        assert "".join(chr(int(m)) for m in re.findall(r"&#(\d+);", encoded)) == SAMPLE

    def test_base32_roundtrip(self):
        assert base64.b32decode(to_base32(SAMPLE)[0]).decode() == SAMPLE

    def test_base85_roundtrip(self):
        assert base64.b85decode(to_base85(SAMPLE)[0]).decode() == SAMPLE


class TestCaesarAndAtbash:
    def test_caesar_shifts_without_stating_the_key_in_the_text(self):
        encoded, info = to_caesar_cipher(SAMPLE, rng=random.Random(1))
        shift = int(info.removeprefix("shift="))
        assert shift != 13      # that is to_rot13      # noqa: PLR2004
        assert str(shift) not in encoded
        back = "".join(
            chr((ord(c) - 97 - shift) % 26 + 97) if "a" <= c <= "z"
            else chr((ord(c) - 65 - shift) % 26 + 65) if "A" <= c <= "Z" else c
            for c in encoded)
        assert back == SAMPLE

    def test_caesar_is_deterministic_for_a_seed(self):
        a = to_caesar_cipher(SAMPLE, rng=random.Random(3))
        b = to_caesar_cipher(SAMPLE, rng=random.Random(3))
        assert a == b

    def test_atbash_is_its_own_inverse(self):
        encoded, _ = to_atbash(SAMPLE)
        assert encoded != SAMPLE
        assert to_atbash(encoded)[0] == SAMPLE


class TestStyledUnicode:
    def test_fullwidth_roundtrip(self):
        encoded, _ = to_fullwidth("Ab9 ok")
        assert encoded != "Ab9 ok"
        back = "".join(chr(ord(c) - 0xFEE0) if 0xFF01 <= ord(c) <= 0xFF5E else c
                       for c in encoded)
        assert back == "Ab9 ok"

    def test_math_styled_names_its_style_and_maps_back(self):
        encoded, info = to_math_styled(SAMPLE, rng=random.Random(2))
        style = info.removeprefix("math_style=")
        upper, lower = _MATH_STYLES[style]
        back = "".join(
            chr(65 + ord(c) - upper) if upper <= ord(c) < upper + 26
            else chr(97 + ord(c) - lower) if lower <= ord(c) < lower + 26 else c
            for c in encoded)
        assert back == SAMPLE

    def test_math_styled_is_deterministic_for_a_seed(self):
        a = to_math_styled(SAMPLE, rng=random.Random(5))
        b = to_math_styled(SAMPLE, rng=random.Random(5))
        assert a == b


class TestAllEncodings:
    def test_all_return_tuple_with_info(self):
        for fn in get_encoding_functions():
            result = fn(SAMPLE)
            assert isinstance(result, tuple), f"{fn.__name__} must return tuple"
            assert len(result) == 2

    def test_get_encoding_functions_count(self):
        assert len(get_encoding_functions()) == 27

    def test_output_differs_from_input(self):
        for fn in get_encoding_functions():
            result, _ = fn(SAMPLE)
            assert result != SAMPLE, f"{fn.__name__} output should differ"


#: A greeting and a count to three in ten scripts other than Latin, with the
#: script's own digits where it has them: one for each script a translation
#: technique writes (:data:`SCRIPT_OF`).
NON_LATIN = {
    "Bengali": "নমস্কার বন্ধু এক দুই তিন ১২৩",
    "Thai": "สวัสดี ครับ หนึ่ง สอง สาม ๑๒๓",
    "Japanese": "こんにちは 一 二 三",
    "Arabic": "مرحبا واحد اثنان ثلاثة ١٢٣",
    "Russian": "привет один два три",
    "Greek": "γεια σου ένα δύο τρία",
    "Han": "你好 朋友 一 二 三",
    "Devanagari": "नमस्ते दोस्त एक दो तीन १२३",
    "Ethiopic": "ሰላም ጓደኛ አንድ ሁለት ሦስት ፩፪፫",
    "Khmer": "សួស្តី មិត្ត មួយ ពីរ បី ១២៣",
}

#: The ISO 15924 code of each sample's script, as a translation technique's
#: ``script`` names it.
SCRIPT_OF = {
    "Bengali": "Beng", "Thai": "Thai", "Japanese": "Jpan", "Arabic": "Arab",
    "Russian": "Cyrl", "Greek": "Grek", "Han": "Hani", "Devanagari": "Deva",
    "Ethiopic": "Ethi", "Khmer": "Khmr",
}


def _keeps_its_characters(fn, text: str) -> bool:
    """Whether ``fn`` gives back every non-ASCII character of ``text``, in
    order, and no other, without rejecting it: what it changed, if
    anything, is ASCII only."""
    import inspect

    from redact.jailbreak.technique import is_discarded
    from redact.llm_pipeline import drive_sync

    if inspect.isgeneratorfunction(fn):
        result, info = drive_sync(fn(text), lambda request: "none")
    else:
        result, info = fn(text)

    def beyond_ascii(s: str) -> list[str]:
        return [c for c in s if not c.isascii()]

    return beyond_ascii(result) == beyond_ascii(text) and not is_discarded(info)


def _leaves_unchanged(fn, text: str) -> bool:
    """Whether ``fn`` gives ``text`` back unchanged, and does not reject it.

    A sensitive-word technique's model marks no word, so it encodes the
    whole text, or rejects it when its encoding is heavy.
    """
    import inspect

    from redact.jailbreak.technique import is_discarded
    from redact.llm_pipeline import drive_sync

    if inspect.isgeneratorfunction(fn):
        result, info = drive_sync(fn(text), lambda request: "none")
    else:
        result, info = fn(text)
    return result == text and not is_discarded(info)


class TestLatinOnly:
    """``needs_latin`` marks what changes only ASCII characters, which the rules
    keep from a translation into another script. It is read off the
    techniques' behaviour on the samples above, for the families that hold
    such a technique: ``encode`` (with the sensitive-word encodings) and
    ``noise``. "Latin" is ASCII: the samples hold no ASCII punctuation or
    digit, which ROT47 would change in a text of another script."""

    @staticmethod
    def _encodings() -> list:
        from redact.jailbreak.pool import all_techniques

        return [fn for fn in all_techniques()
                if {"encode", "noise"} & set(fn.families)]

    def test_the_samples_cover_every_script_a_translation_writes(self):
        """So a language added in a script with no sample here fails until it
        has one."""
        from redact.jailbreak.pool import all_techniques

        written = {fn.script for fn in all_techniques()
                   if "translation" in fn.families}
        assert set(SCRIPT_OF) == set(NON_LATIN)
        assert written - {"Latn"} == set(SCRIPT_OF.values())

    def test_a_latin_only_technique_encodes_no_character_of_another_script(self):
        """Every tagged technique gives each sample's characters back, in
        their order, and adds none of its own but ASCII ones: morse's
        spaces and ``/``."""
        tagged = [fn for fn in self._encodings() if fn.needs_latin]
        assert len(tagged) == 28     # noqa: PLR2004
        for fn in tagged:
            for language, text in NON_LATIN.items():
                assert _keeps_its_characters(fn, text), (fn.__name__, language)

    def test_every_technique_that_leaves_them_unchanged_is_marked(self):
        """And every marked one leaves them unchanged but morse, which
        spaces their characters out, as it spaces its codes."""
        unchanged = {fn.__name__ for fn in self._encodings()
                     if all(_leaves_unchanged(fn, text) for text in NON_LATIN.values())}
        tagged = {fn.__name__ for fn in self._encodings() if fn.needs_latin}
        assert "to_braille" in unchanged
        assert unchanged == tagged - {"to_morse"}

    def test_morse_spaces_another_script_out_without_encoding_it(self):
        from redact.jailbreak.obfuscation.encoding import to_morse

        assert to_morse("привет один") == ("п р и в е т / о д и н", "")
        assert not _leaves_unchanged(to_morse, NON_LATIN["Russian"])
