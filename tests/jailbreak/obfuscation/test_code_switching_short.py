"""The code-switching gate on short texts, and the one-word prompt.

The Latin-script gate (Spanish, French, German, Portuguese, Swahili) labels
word windows with GlotLID and needs at least one window in the second
language and one in English. A text of four words or fewer used to be one
window, which cannot be both, so it was always rejected; it is now read in
windows of half its words (one word each for 2 or 3 words, two for 4). A text
of more than four words keeps its four-word windows: the differential test
below holds the new gate to a frozen copy of the old one, decision for
decision and window for window. A prompt of one word is refused before any
call, and the plan does not draw a code-switch for it.

GlotLID is faked throughout (a table from window to label), as the other
language-check tests do, so the words of a text are placeholders: the labels
are what each test sets.
"""

import itertools
import random

import pandas as pd
import pytest

import redact.jailbreak  # noqa: F401  (tags the techniques)
from redact.dataset import Manifest
from redact.jailbreak.obfuscation import code_switching as cs
from redact.jailbreak.plan import plan_run
from redact.jailbreak.technique import applies, is_discarded
from redact.llm_pipeline.drive import drive_sync

_FNS = {fn.__name__: fn for fn in cs.get_code_switching_functions()}
_LATIN = [row for row in cs.LANGUAGES if row["script"] == "Latn"]
_ENGLISH = "eng_Latn"
#: A Latin label no code-switch row uses: a window in a third language.
_OTHER = "ita_Latn"
_LATIN_IDS = [row["slug"] for row in _LATIN]


def _stub(monkeypatch, label_of, available=True):
    """GlotLID faked: ``label_of(window)`` is the label; every window asked
    is recorded, in order."""
    asked: list[str] = []

    def detect(text):
        asked.append(text)
        return label_of(text)

    monkeypatch.setattr(cs.langid, "available", lambda: available)
    monkeypatch.setattr(cs.langid, "detect", detect)
    return asked


def _by_word(row: dict):
    """A detector for one-word windows: a word starting ``s`` is the second
    language, ``e`` English, anything else a third language."""
    def label_of(window):
        return {"s": row["label"], "e": _ENGLISH}.get(window[0], _OTHER)
    return label_of


# --- the windows ----------------------------------------------------------


@pytest.mark.parametrize(("words", "windows"), [
    (0, []),
    (1, ["w0"]),
    (2, ["w0", "w1"]),
    (3, ["w0", "w1", "w2"]),
    (4, ["w0 w1", "w1 w2", "w2 w3"]),
    (5, ["w0 w1 w2 w3", "w1 w2 w3 w4"]),
    (7, ["w0 w1 w2 w3", "w1 w2 w3 w4", "w2 w3 w4 w5", "w3 w4 w5 w6"]),
])
def test_the_gate_reads_half_the_words_up_to_four_and_four_beyond(words, windows):
    text = " ".join(f"w{i}" for i in range(words))
    assert cs._gate_windows(text) == windows


def test_every_text_of_two_to_four_words_has_at_least_two_windows():
    for n in range(2, cs.WINDOW_WORDS + 1):
        assert len(cs._gate_windows(" ".join(["w"] * n))) >= 2  # noqa: PLR2004


# --- the rule at each short length ----------------------------------------


@pytest.mark.parametrize("row", _LATIN, ids=_LATIN_IDS)
@pytest.mark.parametrize("words", ["s1", "e1"])
def test_one_word_is_rejected_whatever_its_label(monkeypatch, row, words):
    """One word is in one language: the single window fails one side."""
    _stub(monkeypatch, _by_word(row))
    expected = "no english left" if words.startswith("s") else f"no {row['name']}"
    assert cs._latin_mix_reason(words, row["label"], row["name"]) == expected


@pytest.mark.parametrize("row", _LATIN, ids=_LATIN_IDS)
@pytest.mark.parametrize(("text", "expected"), [
    # Two words: one label per word.
    ("s1 e1", None),
    ("e1 s1", None),
    ("s1 s2", "no english left"),
    ("e1 e2", "no {name}"),
    ("s1 x1", "no english left"),
    # Three words: one label per word.
    ("s1 s2 e1", None),
    ("e1 s1 e2", None),
    ("s1 x1 x2", "no english left"),
    ("x1 x2 e1", "no {name}"),
])
def test_two_and_three_words_are_labeled_word_by_word(monkeypatch, row, text, expected):
    asked = _stub(monkeypatch, _by_word(row))
    reason = cs._latin_mix_reason(text, row["label"], row["name"])
    assert reason == (expected.format(name=row["name"]) if expected else None)
    assert asked == text.split()


@pytest.mark.parametrize("row", _LATIN, ids=_LATIN_IDS)
@pytest.mark.parametrize(("labels", "expected"), [
    (("second", "other", "english"), None),
    (("english", "english", "second"), None),
    (("second", "second", "second"), "no english left"),
    (("english", "english", "english"), "no {name}"),
    (("other", "other", "other"), "no {name}"),
])
def test_four_words_are_labeled_in_two_word_windows(monkeypatch, row, labels, expected):
    text = "a1 a2 b1 b2"
    windows = ["a1 a2", "a2 b1", "b1 b2"]
    names = {"second": row["label"], "english": _ENGLISH, "other": _OTHER}
    table = dict(zip(windows, (names[label] for label in labels), strict=True))
    asked = _stub(monkeypatch, table.__getitem__)
    reason = cs._latin_mix_reason(text, row["label"], row["name"])
    assert reason == (expected.format(name=row["name"]) if expected else None)
    assert asked == windows


def test_without_glotlid_the_short_gate_is_skipped_as_the_long_one(monkeypatch):
    _stub(monkeypatch, lambda window: None, available=False)
    spanish = _LATIN[0]
    for text in ("s1", "s1 e1", "s1 s2 s3", "s1 s2 s3 s4", "s1 s2 s3 s4 s5"):
        assert cs._latin_mix_reason(text, spanish["label"], spanish["name"]) is None


@pytest.mark.parametrize("row", _LATIN, ids=_LATIN_IDS)
def test_the_whole_gate_passes_a_short_mix_and_names_the_side_missing(
        monkeypatch, row):
    """Through ``_code_gate``: a prompt under ``MIN_BAND_CHARS`` skips the
    length band, so the mix alone decides."""
    _stub(monkeypatch, _by_word(row))
    original = "e1 e2 e3"
    assert len(original) < cs.MIN_BAND_CHARS
    assert cs._code_gate("s1 e2 e3", original, row) == ""
    assert cs._code_gate("e1 e2 e3", original, row) == f"no {row['name']}"
    assert cs._code_gate("s1 s2 s3", original, row) == "no english left"


# --- unchanged from five words: a differential against the old gate -------


def _old_windows(text: str, size: int = 4) -> list[str]:
    """The windows of the gate as it was (2220e5e), frozen here."""
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)] if words else []
    return [" ".join(words[i:i + size]) for i in range(len(words) - size + 1)]


def _old_code_gate(text: str, original: str, language: dict) -> str:
    """The Latin path of ``_code_gate`` as it was (2220e5e), frozen here:
    empty, the length band, then GlotLID over four-word windows."""
    if not text.strip():
        return "empty"
    if len(original) >= 20:  # noqa: PLR2004 - MIN_BAND_CHARS then
        ratio = len(text) / len(original)
        if not 0.5 <= ratio <= 2.0:  # noqa: PLR2004 - LENGTH_RATIO_BAND then
            return f"length_ratio={ratio:.2f}"
    name, label = language["name"], language["label"]
    if not cs.langid.available():
        return ""
    windows = _old_windows(text)
    if not windows:
        return f"no {name}"
    labels = [cs.langid.detect(window) for window in windows]
    n = len(windows)
    second = sum(lbl == label for lbl in labels)
    english = sum(lbl == _ENGLISH for lbl in labels)
    if second < 1 or second / n < 0.1:  # noqa: PLR2004 - MIN_WINDOW_SHARE then
        return f"no {name}"
    if english < 1 or english / n < 0.1:  # noqa: PLR2004
        return "no english left"
    return ""


def _cases(row: dict):
    """``(text, {window: label})`` for every texts of 5 to 8 words with each
    of its four-word windows labeled every way (the second language,
    English, a third language, no label), then random ones of 9 to 40
    words, some with repeated words, some with a rare second language so
    the share threshold decides."""
    alphabet = [row["label"], _ENGLISH, _OTHER, None]
    for n in range(5, 9):
        text = " ".join(f"t{i}" for i in range(n))
        windows = _old_windows(text)
        for labels in itertools.product(alphabet, repeat=len(windows)):
            yield text, dict(zip(windows, labels, strict=True))
    rng = random.Random(f"code-switch-{row['slug']}")
    for _ in range(600):
        n = rng.randint(9, 40)
        vocabulary = rng.choice([n, 3])  # unique words, or few repeated
        text = " ".join(f"t{rng.randrange(vocabulary)}" for _ in range(n))
        weights = rng.choice([[1, 1, 1, 1], [1, 12, 1, 1], [12, 1, 1, 1]])
        yield text, {window: rng.choices(alphabet, weights)[0]
                     for window in _old_windows(text)}


@pytest.mark.parametrize("row", _LATIN, ids=_LATIN_IDS)
def test_from_five_words_the_gate_decides_as_before_window_for_window(
        monkeypatch, row):
    outcomes = set()
    share_decided = 0
    for text, table in _cases(row):
        asked = _stub(monkeypatch, table.__getitem__)
        before = _old_code_gate(text, text, row)
        old_asked = list(asked)
        asked.clear()
        assert cs._code_gate(text, text, row) == before, text
        assert asked == old_asked, text
        outcomes.add(before)
        labels = [table[window] for window in old_asked]
        if before and labels.count(row["label"]) >= 1 and labels.count(_ENGLISH) >= 1:
            share_decided += 1
    # Every outcome is reached, and some by the share threshold alone.
    assert outcomes == {"", f"no {row['name']}", "no english left"}
    assert share_decided


def test_before_the_change_no_text_of_four_words_or_fewer_could_pass(monkeypatch):
    """The old gate read one window: no labeling passes it; the new one
    passes a mix at every length from two words."""
    spanish = _LATIN[0]
    for n in range(1, cs.WINDOW_WORDS + 1):
        text = " ".join(f"t{i}" for i in range(n))
        for label in (spanish["label"], _ENGLISH, _OTHER, None):
            _stub(monkeypatch, lambda window, label=label: label)
            assert _old_code_gate(text, text, spanish) != ""
    for text in ("s1 e1", "s1 e1 e2", "s1 s2 e1 e2"):
        _stub(monkeypatch, lambda window: {"s": spanish["label"]}.get(
            window[0], _ENGLISH))
        assert cs._code_gate(text, text, spanish) == ""


# --- the one-word prompt --------------------------------------------------


class _Fake:
    """The generation model writes ``reply``; the check model says yes."""

    def __init__(self, reply: str):
        self.reply, self.calls = reply, 0

    def __call__(self, request) -> str:
        self.calls += 1
        return "Yes, it means the same." if request.model == "chk" else self.reply


def _run(fn, prompt: str, reply: str) -> tuple[str, str, int]:
    fake = _Fake(reply)
    text, info = drive_sync(fn(prompt, gen_model="gen", check_model="chk"), fake)
    return text, info, fake.calls


@pytest.mark.parametrize("name", sorted(_FNS))
@pytest.mark.parametrize("prompt", ["", "   ", "Hello", "  Hello\n"])
def test_a_prompt_of_one_word_is_refused_before_any_call(name, prompt):
    language = next(row["name"] for row in cs.LANGUAGES
                    if f"to_code_switch_{row['slug']}" == name)
    text, info, calls = _run(_FNS[name], prompt, "never asked")
    assert calls == 0
    assert text == prompt
    assert info == f"DISCARDED; reason=too short to mix; language={language}"
    assert is_discarded(info)


@pytest.mark.parametrize("row", _LATIN, ids=_LATIN_IDS)
@pytest.mark.parametrize(("prompt", "reply"), [
    ("e1 e2", "s1 e2"),
    ("e1 e2 e3", "s1 s2 e3"),
    ("e1 e2 e3 e4", "s1 s2 e3 e4"),
])
def test_a_short_mix_is_accepted_end_to_end(monkeypatch, row, prompt, reply):
    """Generation, the gate (which the old one always failed here), the
    check: two calls, the rewrite kept."""
    def label_of(window):
        firsts = {word[0] for word in window.split()}
        if firsts == {"s"}:
            return row["label"]
        return _ENGLISH if firsts == {"e"} else _OTHER

    _stub(monkeypatch, label_of)
    text, info, calls = _run(_FNS[f"to_code_switch_{row['slug']}"], prompt, reply)
    assert (text, info, calls) == (reply, f"language={row['name']}", 2)


def test_a_one_word_rewrite_of_a_longer_prompt_is_retried_then_discarded(monkeypatch):
    """The gate's reason goes back as feedback, so the model is asked again;
    no check call is made for a rewrite the gate fails."""
    spanish = _LATIN[0]
    _stub(monkeypatch, _by_word(spanish))
    text, info, calls = _run(_FNS["to_code_switch_spanish"], "e1 e2", "s1")
    assert calls == 2  # noqa: PLR2004 - two generations, no check
    assert info.startswith("DISCARDED; reason=code switch failed; language=Spanish")
    assert info.endswith("feedback=no english left")


# --- the applicability ----------------------------------------------------


@pytest.mark.parametrize("name", sorted(_FNS))
def test_every_code_switch_applies_only_to_a_prompt_it_can_mix(name, monkeypatch):
    """The predicate is ``can_mix``, which the technique checks itself: drawn
    for a prompt exactly when the technique would call the model on it."""
    fn = _FNS[name]
    assert hasattr(fn, "applicable")
    _stub(monkeypatch, lambda window: None, available=False)
    for words in range(6):
        prompt = " ".join(["word"] * words)
        for entry_type in ("harmful", "dual_use_harmful", "dual_use_benign", "benign"):
            assert applies(fn, prompt, entry_type) is (words >= cs.MIN_MIX_WORDS)
        _, info, calls = _run(fn, prompt, "word palabra")
        assert (calls > 0) is applies(fn, prompt, "harmful")
        assert ("too short to mix" in info) is (calls == 0)


@pytest.mark.parametrize("mode", [{"exact_techniques": 1}, {}],
                         ids=["count", "budget"])
def test_the_plan_draws_no_code_switch_for_a_one_word_input(tmp_path, mode):
    """Beside a technique of another family that applies to every input, a
    one-word input gets that one, and the code-switches go to the others."""
    def to_other(prompt, **kwargs):
        return f"{prompt} other", ""

    to_other.families, to_other.layer = ["beta"], "obfuscation"
    to_other.within_layer_order, to_other.complexity = 0, 0
    pool = [*_FNS.values(), to_other]
    texts = [("Hello" if i % 2 else f"a longer input number {i}") for i in range(40)]
    inputs = pd.DataFrame({
        "sample_id": [f"u{i:02d}" for i in range(len(texts))],
        "prompt": texts,
        "category": ["Cyber"] * len(texts),
        "entry_type": ["harmful"] * len(texts),
    })
    manifest = tmp_path / "jb.manifest.jsonl"
    plan_run(inputs, pool, manifest_path=manifest, seed=7, sample_kwargs=mode,
             verbose=False)
    text_of = dict(zip(inputs["sample_id"], texts, strict=True))
    drawn = [(text_of[row["sample_id"]], name)
             for row in Manifest(manifest).load() for name in row["combination"]]
    switched = {text for text, name in drawn if name in _FNS}
    assert switched and "Hello" not in switched
