"""ASCII-art technique: the prompt drawn in a random pyfiglet font.

``info`` is ``font=<font>``. The font is drawn from those with a glyph for
every character of the prompt, since a font leaves out a character it lacks;
a prompt no font draws whole is discarded.
"""

import random
from collections.abc import Callable
from functools import cache

import pyfiglet

from redact.jailbreak.technique import DISCARDED

ASCII_FONTS = [
    # Classic fonts
    "standard", "slant", "big", "small",
    # Stylized and "Funky" fonts
    "block", "bubble", "digital", "isometric1", "isometric2",
    "isometric3", "isometric4", "gothic",
    # Specialized & Decorative
    "banner", "alphabet", "alligator", "dotmatrix",
]


def to_ascii_art(prompt: str, *, rng: random.Random | None = None) -> tuple[str, str]:
    """Convert text to ASCII art in a random font that draws all of it."""
    needed = {ord(c) for c in prompt} - {ord("\n")}     # a newline breaks the line
    fonts = [font for font in ASCII_FONTS if needed <= _glyphs(font)]
    if not fonts:
        return prompt, f"{DISCARDED}; reason=no font has every character"
    font = (rng or random).choice(fonts)
    return pyfiglet.Figlet(font=font).renderText(prompt), f"font={font}"


@cache
def _glyphs(font: str) -> frozenset[int]:
    """The code points ``font`` has a glyph for."""
    return frozenset(pyfiglet.FigletFont(font).chars)


def get_ascii_art_functions() -> list[Callable]:
    """Return all ASCII art technique functions."""
    return [to_ascii_art]
