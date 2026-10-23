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
eval review.
"""

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


@pytest.fixture(scope="module")
def expected() -> dict:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def test_the_fixture_pins_every_template_technique(expected):
    assert sorted(expected) == sorted(_TECHNIQUES)


@pytest.mark.parametrize("name", sorted(_TECHNIQUES))
def test_a_template_technique_writes_and_draws_as_pinned(name, expected):
    fn = _TECHNIQUES[name]
    assert {str(seed): digest(fn, seed) for seed in _SEEDS} == expected[name]


if __name__ == "__main__":
    _FIXTURE.write_text(json.dumps(snapshot(), indent=1) + "\n", encoding="utf-8")
