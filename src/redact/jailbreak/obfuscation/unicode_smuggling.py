"""Invisible-Unicode smuggling: the prompt carried in characters a reader
does not see, after a visible benign carrier.

Three techniques hide text in code points that render as nothing:

- :func:`to_variation_selector_smuggling` encodes the prompt's UTF-8 bytes as
  Unicode *variation selectors* (``U+FE00``-``U+FE0F`` for a byte under 16,
  ``U+E0100``-``U+E01EF`` for the rest), appended after a visible carrier
  (one of the ``vs_carrier`` pool). The selectors are invisible, so a reader
  sees only the carrier; the bytes decode back to the exact prompt
  (:func:`decode_variation_selectors`). Any text encodes, so there is no
  applicability.
- :func:`to_invisible_suffix` leaves the prompt visible and appends a run of
  random invisible variation selectors: the static form, a classifier reading
  the raw characters meets noise a reader never sees. Stripping the trailing
  selectors (:func:`strip_trailing_selectors`) gives the prompt back.
- :func:`to_unicode_tag_smuggling` encodes the prompt as Unicode *tag*
  characters (``U+E0020``-``U+E007E``, the invisible shadows of printable
  ASCII ``0x20``-``0x7E``), appended after a visible carrier (one of the
  ``tag_carrier`` pool). Only printable ASCII has a tag character, so it is
  drawn only for a prompt that is all printable ASCII (:func:`is_taggable`);
  the tags decode back to the exact prompt (:func:`decode_tags`).

The carriers hold no instruction and no ``{prompt}`` slot: the prompt is the
invisible payload. ``info`` is ``carrier=<variant>`` for the two carried
forms and ``""`` for the suffix.
"""

import random
from collections.abc import Callable

from redact.jailbreak.technique import applicable_when

from ..directives import draw_variant, read_pools, uses_pool

_FILE = "obfuscation/unicode_smuggling_templates.json"
_POOLS: dict[str, list[dict]] = read_pools(_FILE)

#: The low variation selectors (VS1-VS16), one per byte value 0-15.
_VS_LOW = 0xFE00
#: The supplementary variation selectors (VS17-VS256), for byte values 16-255.
_VS_HIGH = 0xE0100
#: The base of the Unicode tag characters: tag(c) = U+E0000 + ord(c).
_TAG_BASE = 0xE0000
#: The printable-ASCII range that has a tag character (TAG SPACE..TAG TILDE).
_TAG_LOW, _TAG_HIGH = 0x20, 0x7E


def _byte_to_selector(value: int) -> str:
    """The variation selector that stands for byte ``value`` (0-255)."""
    return chr(_VS_LOW + value) if value < 16 else chr(_VS_HIGH + value - 16)  # noqa: PLR2004


def _selector_to_byte(char: str) -> int | None:
    """The byte a variation selector stands for, or None when ``char`` is not
    one."""
    point = ord(char)
    if _VS_LOW <= point <= _VS_LOW + 15:
        return point - _VS_LOW
    if _VS_HIGH <= point <= _VS_HIGH + 239:
        return point - _VS_HIGH + 16
    return None


def encode_variation_selectors(text: str) -> str:
    """``text`` as a run of variation selectors, one per UTF-8 byte."""
    return "".join(_byte_to_selector(byte) for byte in text.encode("utf-8"))


def decode_variation_selectors(text: str) -> str:
    """The prompt hidden in the trailing variation selectors of ``text``: the
    maximal run of selectors at its end, decoded from UTF-8. The carrier
    before them is ignored, so a carrier that itself held a selector would not
    disturb the payload (the carriers hold none)."""
    tail: list[int] = []
    for char in reversed(text):
        byte = _selector_to_byte(char)
        if byte is None:
            break
        tail.append(byte)
    tail.reverse()
    return bytes(tail).decode("utf-8")


def strip_trailing_selectors(text: str) -> str:
    """``text`` without its trailing run of variation selectors."""
    end = len(text)
    while end > 0 and _selector_to_byte(text[end - 1]) is not None:
        end -= 1
    return text[:end]


def is_taggable(text: str) -> bool:
    """Whether every character of ``text`` is printable ASCII (``0x20``-
    ``0x7E``), the only characters a tag character can stand for; a non-empty
    such text is the only kind tag smuggling is drawn for."""
    return bool(text) and all(_TAG_LOW <= ord(char) <= _TAG_HIGH for char in text)


def _char_to_tag(char: str) -> str:
    """The tag character for a printable-ASCII ``char``, else ``char`` itself
    (so a direct call on other text leaves it visible; the plan draws the
    technique only for all-taggable text)."""
    return chr(_TAG_BASE + ord(char)) if _TAG_LOW <= ord(char) <= _TAG_HIGH else char


def encode_tags(text: str) -> str:
    """``text`` with each printable-ASCII character as its tag character."""
    return "".join(_char_to_tag(char) for char in text)


def decode_tags(text: str) -> str:
    """The prompt hidden in the trailing tag characters of ``text``: the
    maximal run of tag characters at its end, each mapped back to its ASCII
    character."""
    tail: list[str] = []
    for char in reversed(text):
        point = ord(char)
        if not (_TAG_BASE + _TAG_LOW <= point <= _TAG_BASE + _TAG_HIGH):
            break
        tail.append(chr(point - _TAG_BASE))
    tail.reverse()
    return "".join(tail)


@uses_pool(_FILE, "vs_carrier", (), "carrier", False)
def to_variation_selector_smuggling(
    prompt: str, *, rng: random.Random | None = None,
) -> tuple[str, str]:
    """Hide the prompt as variation selectors after a visible carrier emoji."""
    carrier = draw_variant(_POOLS["vs_carrier"], rng)
    return carrier["template"] + encode_variation_selectors(prompt), \
        f"carrier={carrier['name']}"


def to_invisible_suffix(
    prompt: str, *, rng: random.Random | None = None,
) -> tuple[str, str]:
    """Leave the prompt visible and append a run of random invisible variation
    selectors."""
    rng = rng or random
    count = rng.randint(8, 48)  # noqa: PLR2004
    suffix = "".join(_byte_to_selector(rng.randrange(256)) for _ in range(count))
    return prompt + suffix, ""


@applicable_when(lambda text, entry_type: is_taggable(text))
@uses_pool(_FILE, "tag_carrier", (), "carrier", False)
def to_unicode_tag_smuggling(
    prompt: str, *, rng: random.Random | None = None,
) -> tuple[str, str]:
    """Hide the prompt as Unicode tag characters after a visible carrier
    sentence (drawn only for an all-printable-ASCII prompt)."""
    carrier = draw_variant(_POOLS["tag_carrier"], rng)
    return carrier["template"] + encode_tags(prompt), f"carrier={carrier['name']}"


def get_unicode_smuggling_functions() -> list[Callable]:
    """Return all invisible-Unicode smuggling technique functions."""
    return [
        to_variation_selector_smuggling,
        to_invisible_suffix,
        to_unicode_tag_smuggling,
    ]
