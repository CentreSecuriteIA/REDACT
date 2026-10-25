"""Rule-based transliteration: English written phonetically in another
script (Cyrillic, Greek, Devanagari or Arabic), by a fixed letter and
digraph table.

Each technique reads the Latin letters of the prompt and writes an
approximate phonetic rendering in its script, leaving word boundaries,
digits, punctuation and any character of another script as they were. It is
a one-way, lossy, phonetic rendering: several Latin letters map to one
target grapheme and the target scripts drop vowels or case, so it is **not**
invertible, and it is not a translation (the words stay English, only the
letters change). The tables are deterministic, so a technique is pure (no
model call) and applying it twice changes nothing after the first pass.

Tagged ``needs_latin``: it changes only Latin letters, so a text already in
another script is left as it is, and the family rules keep it from every
translation (a translated text is no longer Latin English to render). Drawn
only for a prompt that holds a Latin letter (:func:`_has_latin`).

``info`` is empty.
"""

from collections.abc import Callable

from redact.jailbreak.technique import applicable_when

#: One table per script: the Latin letters and digraphs it maps to, longest
#: key first so a digraph wins over its letters. Every entry is lower case;
#: the transform lower-cases a Latin letter before the lookup (the target
#: scripts below are unicameral, Greek and Cyrillic kept lower here), so the
#: output carries no case. The renderings are approximate and phonetic;
#: native-reader correctness is not claimed (see the module docstring).
_TABLES: dict[str, dict[str, str]] = {
    "cyrillic": {
        "shch": "щ", "sch": "щ",
        "ch": "ч", "sh": "ш", "zh": "ж", "kh": "х", "ts": "ц",
        "ya": "я", "yu": "ю", "yo": "ё", "ye": "е",
        "a": "а", "b": "б", "c": "к", "d": "д", "e": "е", "f": "ф",
        "g": "г", "h": "х", "i": "и", "j": "дж", "k": "к", "l": "л",
        "m": "м", "n": "н", "o": "о", "p": "п", "q": "к", "r": "р",
        "s": "с", "t": "т", "u": "у", "v": "в", "w": "в", "x": "кс",
        "y": "й", "z": "з",
    },
    "greek": {
        "ch": "χ", "ph": "φ", "th": "θ", "ps": "ψ", "ks": "ξ",
        "a": "α", "b": "β", "c": "κ", "d": "δ", "e": "ε", "f": "φ",
        "g": "γ", "h": "η", "i": "ι", "j": "ι", "k": "κ", "l": "λ",
        "m": "μ", "n": "ν", "o": "ο", "p": "π", "q": "κ", "r": "ρ",
        "s": "σ", "t": "τ", "u": "υ", "v": "β", "w": "ω", "x": "ξ",
        "y": "υ", "z": "ζ",
    },
    "devanagari": {
        "chh": "छ",
        "ch": "च", "sh": "श", "th": "थ", "ph": "फ", "kh": "ख",
        "gh": "घ", "bh": "भ", "dh": "ध", "jh": "झ",
        "a": "अ", "b": "ब", "c": "क", "d": "द", "e": "ए", "f": "फ़",
        "g": "ग", "h": "ह", "i": "इ", "j": "ज", "k": "क", "l": "ल",
        "m": "म", "n": "न", "o": "ओ", "p": "प", "q": "क़", "r": "र",
        "s": "स", "t": "त", "u": "उ", "v": "व", "w": "व", "x": "क्स",
        "y": "य", "z": "ज़",
    },
    "arabic": {
        "sh": "ش", "ch": "تش", "th": "ث", "kh": "خ", "gh": "غ",
        "dh": "ذ",
        "a": "ا", "b": "ب", "c": "ك", "d": "د", "e": "ي", "f": "ف",
        "g": "غ", "h": "ه", "i": "ي", "j": "ج", "k": "ك", "l": "ل",
        "m": "م", "n": "ن", "o": "و", "p": "ب", "q": "ق", "r": "ر",
        "s": "س", "t": "ت", "u": "و", "v": "ف", "w": "و", "x": "كس",
        "y": "ي", "z": "ز",
    },
}

#: The longest key in any table: how far ahead to look for a digraph.
_MAX_KEY = max(len(key) for table in _TABLES.values() for key in table)


def _has_latin(text: str) -> bool:
    """Whether ``text`` holds an ASCII letter: the only characters a
    transliteration changes, so it is drawn only for a text that has one."""
    return any(c.isascii() and c.isalpha() for c in text)


def _transliterate(text: str, table: dict[str, str]) -> str:
    """``text`` with each run of ASCII letters mapped through ``table``,
    longest key first; every other character (space, digit, punctuation, a
    character of another script) stays. Deterministic and idempotent: the
    output holds no ASCII letter, so a second pass changes nothing."""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if not (c.isascii() and c.isalpha()):
            out.append(c)
            i += 1
            continue
        matched = False
        for size in range(_MAX_KEY, 0, -1):
            key = text[i:i + size].lower()
            if len(key) == size and key.isascii() and key.isalpha() and key in table:
                out.append(table[key])
                i += size
                matched = True
                break
        if not matched:        # a letter the table lacks: drop it, as it has no sound
            i += 1
    return "".join(out)


def _make_fn(slug: str, script_name: str) -> Callable:
    """The technique that writes the prompt in ``script_name`` (a key of
    :data:`_TABLES`), named ``to_<slug>_transliteration``, drawn only for a
    prompt with a Latin letter."""

    @applicable_when(lambda text, entry_type: _has_latin(text))
    def fn(prompt: str) -> tuple[str, str]:
        return _transliterate(prompt, _TABLES[script_name]), ""

    fn.__name__ = fn.__qualname__ = f"to_{slug}_transliteration"
    fn.__doc__ = f"Write the prompt phonetically in {script_name} script."
    return fn


to_cyrillic_transliteration = _make_fn("cyrillic", "cyrillic")
to_greek_transliteration = _make_fn("greek", "greek")
to_devanagari_transliteration = _make_fn("devanagari", "devanagari")
to_arabic_transliteration = _make_fn("arabic", "arabic")


def get_transliteration_functions() -> list[Callable]:
    """Return all rule-based transliteration technique functions."""
    return [
        to_cyrillic_transliteration,
        to_greek_transliteration,
        to_devanagari_transliteration,
        to_arabic_transliteration,
    ]
