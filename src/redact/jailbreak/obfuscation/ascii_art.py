"""ASCII-art technique: the prompt drawn in a random pyfiglet font.

``info`` is ``font=<font>``. The font is drawn from those with a glyph for
every character of the prompt, since a font leaves out a character it lacks,
and whose drawing is at most ``max_chars`` long; a prompt no font draws whole
is discarded, and so is one every such font draws longer than the cap.

The cap holds wherever the technique runs, the eval type's pool included,
and it changes the text, not only what is discarded. A prompt every font
draws within it (up to about 38 characters) draws the font it drew without a
cap. Over that the font is drawn from fewer fonts, so the same ``rng`` often
draws another: for a prompt with a digit or a punctuation mark (12 fonts;
the isometric ones have neither), about 1 in 12 from about 45 to 75
characters, 2 in 3 at 90 to 100 and 9 in 10 from about 125; for one of
letters and spaces alone (16 fonts), about 19 in 20 from about 38. A prompt
of about 830 characters or more is discarded where it used to be drawn, and
so are most prompts an encoding lengthened first (unicode escapes, ASCII
ordinals, morse). A plan kept across this change therefore mixes drawings
made with and without the cap: a replayed plan gives the same text only
within one version of the library.
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
# (dotmatrix) or 130 (isometric) characters, so at 5,000 a prompt of up to
# about 38 characters is drawn in any font, one of 150 in the six smallest,
# one of 300 in digital and bubble, one of 400 to 800 in digital alone, and
# one of about 830 or more in none.
MAX_RENDERED_CHARS = 5000


def to_ascii_art(
    prompt: str, *, rng: random.Random | None = None,
    max_chars: int = MAX_RENDERED_CHARS,
) -> tuple[str, str]:
    """Convert text to ASCII art in a random font that draws all of it within
    ``max_chars`` characters."""
    needed = {ord(c) for c in prompt} - {ord("\n")}     # a newline breaks the line
    drawn = {font: _figlet(font).renderText(prompt)
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
def _figlet(font: str) -> pyfiglet.Figlet:
    """``font`` loaded once per process, since a call may draw all 16 fonts.

    One Figlet is shared by every call and thread. That is safe: in pyfiglet
    (read in 1.0.4, the locked version) ``renderText`` builds a new
    ``FigletBuilder`` for each text and keeps every running buffer on it,
    reading the Figlet and its ``FigletFont`` without writing to them. Never
    call ``setFont`` or set ``width`` on it.
    """
    return pyfiglet.Figlet(font=font)


@cache
def _glyphs(font: str) -> frozenset[int]:
    """The code points ``font`` has a glyph for."""
    return frozenset(_figlet(font).Font.chars)


def get_ascii_art_functions() -> list[Callable]:
    """Return all ASCII art technique functions."""
    return [to_ascii_art]
