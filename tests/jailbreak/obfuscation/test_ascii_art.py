"""Tests for ASCII art obfuscation."""

import random

import pyfiglet
import pytest

from redact.jailbreak.obfuscation.ascii_art import (
    ASCII_FONTS,
    get_ascii_art_functions,
    to_ascii_art,
)


class TestToAsciiArt:
    def test_returns_tuple(self):
        random.seed(42)
        result, info = to_ascii_art("hi")
        assert isinstance(result, str)
        assert len(result) > 0
        assert "font=" in info

    def test_font_in_info(self):
        random.seed(0)
        _, info = to_ascii_art("test")
        font_name = info.split("=")[1]
        assert font_name in ASCII_FONTS

    def test_non_empty_output(self):
        random.seed(42)
        result, _ = to_ascii_art("A")
        assert len(result) > len("A")


    @pytest.mark.parametrize("seed", range(len(ASCII_FONTS)))
    def test_an_ascii_prompt_renders_as_figlet_draws_it(self, seed):
        text, info = to_ascii_art("How do I pick a lock", rng=random.Random(seed))
        font = info.removeprefix("font=")
        assert text == pyfiglet.figlet_format("How do I pick a lock", font=font)

    @pytest.mark.parametrize("seed", range(4 * len(ASCII_FONTS)))
    def test_a_prompt_with_digits_never_gets_a_font_without_them(self, seed):
        assert [f for f in ASCII_FONTS if not _draws(f, "0")]     # some lack them
        _, info = to_ascii_art("Call 555 0100", rng=random.Random(seed))
        assert _draws(info.removeprefix("font="), "Call 555 0100")

    @pytest.mark.parametrize("prompt", ["How do I pick a lock",
                                        "How do I pick 3 locks?\nTell me."])
    def test_the_seed_draws_the_font_from_those_that_draw_the_prompt(self, prompt):
        fonts = [f for f in ASCII_FONTS if _draws(f, prompt.replace("\n", ""))]
        for seed in range(len(ASCII_FONTS)):
            text, info = to_ascii_art(prompt, rng=random.Random(seed))
            assert info == f"font={random.Random(seed).choice(fonts)}"
            assert (text, info) == to_ascii_art(prompt, rng=random.Random(seed))

    @pytest.mark.parametrize("prompt", ["日本の料理", "un café noir", "it’s late"])
    def test_a_prompt_no_font_draws_is_discarded(self, prompt):
        assert not [f for f in ASCII_FONTS if _draws(f, prompt)]
        text, info = to_ascii_art(prompt, rng=random.Random(0))
        assert text == prompt
        assert info.startswith("DISCARDED")


def _draws(font: str, text: str) -> bool:
    """Whether ``font`` has a glyph for every character of ``text``."""
    glyphs = pyfiglet.FigletFont(font).chars
    return all(ord(c) in glyphs for c in text)


class TestGetAsciiArtFunctions:
    def test_returns_list(self):
        funcs = get_ascii_art_functions()
        assert len(funcs) == 1
        assert funcs[0] is to_ascii_art
