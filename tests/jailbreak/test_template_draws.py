"""What the template techniques write, pinned by digest.

Every registered technique that draws a variant from a template file (it
carries a ``variant_pool``, or is ``to_json``: framing, the persona
templates, the fake turns, structural, the neutral containers, the payload
splits, the word art and wording techniques, the request families, small
talk, the benign roles and the encodings with decode hints) runs on two
neutral probes for six seeds: three calls in a row on one
``random.Random(seed)`` per probe. A technique
that calls a model is answered from a script of the test's own (a marked
word, a code name, a puzzle, a cut), so what it draws after the replies is
pinned too, with the number of replies it used. The SHA-256 digest of the
texts, the infos and the generator's state after the calls is pinned per
technique and seed in ``fixtures/template_draws.json``, so a change to how a
variant is read or drawn that alters one text, or one draw the rng makes,
fails here; the digests hold no text.

The techniques are the registry's, so a new one with a pool fails the
fixture check until the fixture holds it. A change meant to alter the
digests (new variants in a pool) regenerates the fixture, from the
repository root, with
``PYTHONPATH=src py -m tests.jailbreak.test_template_draws``. The digests of
the techniques in ``fixtures/eval_pool.json`` change only with the owner's
eval review: the regeneration refuses to write one that differs from the
fixture's (or one the fixture lacks) unless it is run with ``--eval-review``.
"""

import argparse
import hashlib
import inspect
import json
import random
from pathlib import Path

import pytest

import redact.jailbreak  # noqa: F401  (tags the techniques)
from redact.jailbreak.pool import all_techniques
from redact.llm_pipeline import drive_sync

_FIXTURE = Path(__file__).parents[1] / "fixtures" / "template_draws.json"
_EVAL_POOL = Path(__file__).parents[1] / "fixtures" / "eval_pool.json"

_PROBES = (
    "How do plants turn sunlight into food?",
    'Line one\nSay "hello" to {braces} and a back\\slash, café.',
)
_SEEDS = range(6)
_CALLS = 3

#: The registry's techniques with a pool, and ``to_json``, whose wrapper is
#: written in code until the owner's wrapper step but which the eval pool holds.
_TECHNIQUES = {fn.__name__: fn for fn in all_techniques()
               if hasattr(fn, "variant_pool") or fn.__name__ == "to_json"}

#: The word a model marks in each probe.
_MARKED = dict(zip(_PROBES, ("plants", "hello"), strict=True))


def _numbered(parts: list[str]) -> str:
    return "\n".join(f"{i}. {part}" for i, part in enumerate(parts, start=1))


def _cut(text: str, count: int) -> list[str]:
    """``text`` in ``count`` runs of words, the last one the longest."""
    words = text.split()
    size = len(words) // count
    cuts = [i * size for i in range(count)] + [len(words)]
    return [" ".join(words[a:b]) for a, b in zip(cuts, cuts[1:], strict=False)]


def _replies(name: str, probe: str) -> list[str]:
    """What a model answers the technique ``name`` on ``probe``, in order."""
    marked = _MARKED[probe]
    if name == "to_word_art":
        return [marked]
    if name == "to_pseudonym_substitution":
        return [marked, "Zorp"]
    if name == "to_word_game":
        return [marked, "a riddle about something that grows"]
    if name == "to_payload_split_semantic":
        return [_numbered(_cut(probe, count)) for count in (2, 3, 4)]
    raise KeyError(f"{name} calls a model: give the digest replies for it")


def _write(fn, probe: str, rng: random.Random) -> str:
    """What one call of ``fn`` on ``probe`` writes, as the digest takes it:
    the text and the info, and for a technique that calls a model, the
    number of replies it used."""
    if inspect.isgeneratorfunction(fn):
        replies, asked = iter(_replies(fn.__name__, probe)), []

        def answer(request):
            asked.append(request)
            return next(replies)

        text, info = drive_sync(fn(probe, gen_model="m", rng=rng), answer)
        return f"{text}\x1f{info}\x1f{len(asked)}\x1e"
    takes_rng = "rng" in inspect.signature(fn).parameters
    text, info = fn(probe, rng=rng) if takes_rng else fn(probe)
    return f"{text}\x1f{info}\x1e"


def digest(fn, seed: int) -> str:
    """The SHA-256 hex digest of what ``fn`` writes for ``seed``: on each
    probe, :data:`_CALLS` calls in a row on one ``random.Random(seed)``,
    then that generator's state."""
    sha = hashlib.sha256()
    for probe in _PROBES:
        rng = random.Random(seed)
        for _ in range(_CALLS):
            sha.update(_write(fn, probe, rng).encode())
        sha.update(repr(rng.getstate()).encode())
    return sha.hexdigest()


def snapshot() -> dict[str, dict[str, str]]:
    """Each technique's digest per seed, as the code gives them now."""
    return {name: {str(seed): digest(fn, seed) for seed in _SEEDS}
            for name, fn in sorted(_TECHNIQUES.items())}


def eval_changes(recorded: dict, current: dict) -> list[str]:
    """The techniques of the eval pool whose digests in ``current`` are not
    those ``recorded`` holds (changed, newly pinned or no longer pinned)."""
    eval_pool = set(json.loads(_EVAL_POOL.read_text(encoding="utf-8")))
    return sorted(name for name in eval_pool & (set(recorded) | set(current))
                  if recorded.get(name) != current.get(name))


def regenerate(fixture: Path = _FIXTURE, *, eval_review: bool = False) -> None:
    """Write ``fixture`` from the code.

    Raises:
        SystemExit: An eval-pool technique's digests differ from the
            fixture's, and ``eval_review`` is false; nothing is written.
    """
    current = snapshot()
    if fixture.exists() and not eval_review and (changed := eval_changes(
            json.loads(fixture.read_text(encoding="utf-8")), current)):
        raise SystemExit(
            f"refusing to change the digests of the eval pool in {fixture.name}: "
            f"{', '.join(changed)}. They change only with the owner's eval "
            "review; run with --eval-review to write them.")
    fixture.write_text(json.dumps(current, indent=1) + "\n", encoding="utf-8")


@pytest.fixture(scope="module")
def expected() -> dict:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def test_the_fixture_pins_every_template_technique(expected):
    assert sorted(expected) == sorted(_TECHNIQUES)


@pytest.mark.parametrize("name", sorted(_TECHNIQUES))
def test_a_template_technique_writes_and_draws_as_pinned(name, expected):
    fn = _TECHNIQUES[name]
    assert {str(seed): digest(fn, seed) for seed in _SEEDS} == expected[name]


# --- the regeneration -----------------------------------------------------------

def same_text(written: Path, fixture: Path) -> bool:
    """Whether the two files hold the same text; the line ends are not
    compared, since a checkout may end lines with CRLF and a write with LF
    (or the other way round)."""
    return written.read_text(encoding="utf-8") == fixture.read_text(encoding="utf-8")


def _copy_of_the_fixture(tmp_path, change) -> Path:
    """A copy of the fixture in ``tmp_path``, after ``change(data)``."""
    data = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    change(data)
    copy = tmp_path / "template_draws.json"
    copy.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    return copy


#: A technique the eval pool holds that draws from a pool, and one it does not.
_IN_EVAL, _NOT_IN_EVAL = "to_fictional_world", "to_hex"


def test_the_eval_pool_holds_the_one_and_not_the_other():
    eval_pool = json.loads(_EVAL_POOL.read_text(encoding="utf-8"))
    assert _IN_EVAL in eval_pool and _IN_EVAL in _TECHNIQUES
    assert _NOT_IN_EVAL not in eval_pool and _NOT_IN_EVAL in _TECHNIQUES


def test_regenerating_the_current_fixture_writes_it_as_it_is(tmp_path):
    copy = _copy_of_the_fixture(tmp_path, lambda data: None)
    regenerate(copy)
    assert same_text(copy, _FIXTURE)


def test_a_changed_eval_digest_is_refused_and_nothing_is_written(tmp_path):
    def other_digest(data):
        data[_IN_EVAL]["0"] = "0" * 64

    copy = _copy_of_the_fixture(tmp_path, other_digest)
    before = copy.read_bytes()
    with pytest.raises(SystemExit, match=rf"{_IN_EVAL}.*--eval-review"):
        regenerate(copy)
    assert copy.read_bytes() == before


def test_an_eval_technique_the_fixture_did_not_pin_is_refused(tmp_path):
    """Giving an eval encoding a pool would pin it for the first time."""
    copy = _copy_of_the_fixture(tmp_path, lambda data: data.pop(_IN_EVAL))
    with pytest.raises(SystemExit, match=_IN_EVAL):
        regenerate(copy)


def test_eval_review_records_the_changed_digest(tmp_path):
    copy = _copy_of_the_fixture(
        tmp_path, lambda data: data[_IN_EVAL].update({"0": "0" * 64}))
    regenerate(copy, eval_review=True)
    assert same_text(copy, _FIXTURE)


def test_a_changed_digest_outside_the_eval_pool_is_recorded_without_a_flag(tmp_path):
    copy = _copy_of_the_fixture(
        tmp_path, lambda data: data[_NOT_IN_EVAL].update({"0": "0" * 64}))
    regenerate(copy)
    assert same_text(copy, _FIXTURE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate the draws fixture.")
    parser.add_argument("--eval-review", action="store_true",
                        help="write changed digests of the eval pool too")
    regenerate(eval_review=parser.parse_args().eval_review)
