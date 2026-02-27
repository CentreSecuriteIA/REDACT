"""ASCII art obfuscation via pyfiglet.

Pure transform: str -> (str, str). No LLM dependency.
Ported from reference obfuscation.py lines 577-596.
"""

import random

import pyfiglet

ASCII_FONTS = [
    # Classic fonts
    "standard", "slant", "big", "small",
    # Stylized and "Funky" fonts
    "block", "bubble", "digital", "isometric1", "isometric2",
    "isometric3", "isometric4", "gothic",
    # Specialized & Decorative
    "banner", "alphabet", "alligator", "dotmatrix",
]


def to_ascii_art(prompt: str) -> tuple[str, str]:
    """Convert text to ASCII art using a random font."""
    font = random.choice(ASCII_FONTS)
    return pyfiglet.figlet_format(prompt, font=font), f"font={font}"


def get_ascii_art_functions() -> list:
    """Return all ASCII art technique functions."""
    return [to_ascii_art]
