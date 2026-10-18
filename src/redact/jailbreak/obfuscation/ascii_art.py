"""ASCII-art technique: the prompt drawn in a random pyfiglet font.

``info`` is ``font=<font>``. The font is drawn from those with a glyph for
every character of the prompt, since a font leaves out a character it lacks,
and whose drawing is at most ``max_chars`` long; a prompt no font draws whole
is discarded, and so is one every such font draws longer than the cap.
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

# The longest drawing kept, in characters. A font draws a prompt character in
# about 6 (digital), 15 (bubble), 20 (small), 26 to 30 (alphabet, standard,
# slant), 37 (big), 46 to 49 (banner, gothic), 68 (block, alligator), 120
# (dotmatrix) or 130 (isometric) characters, so at 5,000 a 33-character
# prompt is drawn in any font, a 150-character one in the six smallest, a
# 300-character one in digital and bubble, and an 850-character one in none.
MAX_RENDERED_CHARS = 5000


def to_ascii_art(
    prompt: str, *, rng: random.Random | None = None,
    max_chars: int = MAX_RENDERED_CHARS,
) -> tuple[str, str]:
    """Convert text to ASCII art in a random font that draws all of it within
    ``max_chars`` characters."""
    needed = {ord(c) for c in prompt} - {ord("\n")}     # a newline breaks the line
    drawn = {font: pyfiglet.Figlet(font=font).renderText(prompt)
             for font in ASCII_FONTS if needed <= _glyphs(font)}
    if not drawn:
        return prompt, f"{DISCARDED}; reason=no font has every character"
    fonts = [font for font, art in drawn.items() if len(art) <= max_chars]
    if not fonts:
        shortest = min(len(art) for art in drawn.values())
        return prompt, (f"{DISCARDED}; reason=too long; "
                        f"rendered={shortest}; max_chars={max_chars}")
    font = (rng or random).choice(fonts)
    return drawn[font], f"font={font}"


@cache
def _glyphs(font: str) -> frozenset[int]:
    """The code points ``font`` has a glyph for."""
    return frozenset(pyfiglet.FigletFont(font).chars)


def get_ascii_art_functions() -> list[Callable]:
    """Return all ASCII art technique functions."""
    return [to_ascii_art]
