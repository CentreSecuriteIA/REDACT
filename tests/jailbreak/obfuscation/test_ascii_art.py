"""Tests for ASCII art obfuscation."""

import random

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


class TestGetAsciiArtFunctions:
    def test_returns_list(self):
        funcs = get_ascii_art_functions()
        assert len(funcs) == 1
        assert funcs[0] is to_ascii_art
