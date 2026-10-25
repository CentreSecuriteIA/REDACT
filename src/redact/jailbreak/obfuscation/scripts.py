"""Share of a text's letters written in a given script (ISO 15924), for the
code gates of code-switching and dialect techniques. Pure, no model and no
GlotLID: a letter is counted by the Unicode block it falls in.
"""

#: Letter ranges per ISO 15924 script code the gates need. Jpan is the union
#: of the three below it (Hiragana, Katakana, Han).
_RANGES: dict[str, list[tuple[int, int]]] = {
    "Latn": [(0x41, 0x5A), (0x61, 0x7A), (0xC0, 0x24F), (0x1E00, 0x1EFF)],
    "Cyrl": [(0x400, 0x52F)],
    "Grek": [(0x370, 0x3FF), (0x1F00, 0x1FFF)],
    "Arab": [(0x600, 0x6FF), (0x750, 0x77F), (0x8A0, 0x8FF),
             (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
    "Deva": [(0x900, 0x97F)],
    "Beng": [(0x980, 0x9FF)],
    "Hani": [(0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF)],
    "Hira": [(0x3040, 0x309F)],
    "Kana": [(0x30A0, 0x30FF)],
    "Ethi": [(0x1200, 0x137F)],
    "Thai": [(0x0E00, 0x0E7F)],
    "Khmr": [(0x1780, 0x17FF)],
}
_RANGES["Jpan"] = _RANGES["Hira"] + _RANGES["Kana"] + _RANGES["Hani"]


def _in_ranges(code_point: int, ranges: list[tuple[int, int]]) -> bool:
    return any(low <= code_point <= high for low, high in ranges)


def in_script(char: str, script: str) -> bool:
    """Whether ``char`` is a letter of ``script`` (an ISO 15924 code of
    :data:`_RANGES`). A script with no range here matches nothing."""
    return _in_ranges(ord(char), _RANGES.get(script, []))


def script_share(text: str, script: str) -> float:
    """The share of ``text``'s letters that are written in ``script`` (0 when
    it has no letter).

    A letter is a character for which ``str.isalpha`` is true; marks and
    digits do not count. Han characters are letters, so a Chinese or Japanese
    share is well defined.
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(in_script(c, script) for c in letters) / len(letters)
