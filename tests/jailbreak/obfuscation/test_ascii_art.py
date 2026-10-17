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

    @pytest.mark.parametrize("prompt", ["日本の料理", "un café noir"])
    def test_letters_the_font_lacks_discard_the_prompt(self, prompt):
        text, info = to_ascii_art(prompt, rng=random.Random(0))
        assert text == prompt
        assert info.startswith("DISCARDED") and "font" in info


class TestGetAsciiArtFunctions:
    def test_returns_list(self):
        funcs = get_ascii_art_functions()
        assert len(funcs) == 1
        assert funcs[0] is to_ascii_art
