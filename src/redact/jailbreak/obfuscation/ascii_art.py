"""ASCII-art technique: the prompt drawn in a random pyfiglet font.

``info`` is ``font=<font>``. A prompt with a letter the font has no glyph for
is discarded, since the font would leave it out.
"""

import random
from collections.abc import Callable

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
    """Convert text to ASCII art using a random font."""
    font = (rng or random).choice(ASCII_FONTS)
    figlet = pyfiglet.Figlet(font=font)
    if any(c.isalpha() and ord(c) not in figlet.Font.chars for c in prompt):
        return prompt, f"{DISCARDED}; reason=letters the font {font} lacks"
    return figlet.renderText(prompt), f"font={font}"


def get_ascii_art_functions() -> list[Callable]:
    """Return all ASCII art technique functions."""
    return [to_ascii_art]
