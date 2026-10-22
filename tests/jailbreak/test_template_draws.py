"""What the template techniques write, pinned by digest.

Every technique whose text comes from a template file (framing, the persona
templates, the fake turns, structural, the neutral containers, the request
families, small talk, the benign roles and the encodings with decode hints)
runs on two neutral probes for
six seeds: three calls
in a row on one ``random.Random(seed)`` per probe. The SHA-256 digest of the
texts, the infos and the generator's state after the calls is pinned per
technique and seed in ``fixtures/template_draws.json``, so a change to how a
variant is read or drawn that alters one text, or one draw the rng makes,
fails here; the digests hold no text.

A change meant to alter them (new variants in a pool) regenerates the
fixture, from the repository root, with
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
from redact.jailbreak.hacking.framing import get_framing_functions
from redact.jailbreak.hacking.persona_templates import get_persona_template_functions
from redact.jailbreak.manipulation.fake_turns import get_fake_turn_functions
from redact.jailbreak.obfuscation.encoding import get_encoding_functions
from redact.jailbreak.obfuscation.neutral_container import (
    get_neutral_container_functions,
)
from redact.jailbreak.obfuscation.structural import get_structural_functions
from redact.jailbreak.requests import get_all_request_functions
from redact.jailbreak.requests.benign_role import get_benign_role_functions
from redact.jailbreak.requests.smalltalk import get_smalltalk_functions

_FIXTURE = Path(__file__).parents[1] / "fixtures" / "template_draws.json"

_PROBES = (
    "How do plants turn sunlight into food?",
    'Line one\nSay "hello" to {braces} and a back\\slash, café.',
)
_SEEDS = range(6)
_CALLS = 3

_TECHNIQUES = {fn.__name__: fn for fn in (
    get_framing_functions() + get_persona_template_functions()
    + get_fake_turn_functions()
    + [fn for fn in get_encoding_functions() if hasattr(fn, "variant_pool")]
    + get_structural_functions()
    + get_neutral_container_functions() + get_all_request_functions()
    + get_smalltalk_functions() + get_benign_role_functions())}


def digest(fn, seed: int) -> str:
    """The SHA-256 hex digest of what ``fn`` writes for ``seed``: on each
    probe, :data:`_CALLS` calls in a row on one ``random.Random(seed)``,
    then that generator's state."""
    sha = hashlib.sha256()
    takes_rng = "rng" in inspect.signature(fn).parameters
    for probe in _PROBES:
        rng = random.Random(seed)
        for _ in range(_CALLS):
            text, info = fn(probe, rng=rng) if takes_rng else fn(probe)
            sha.update(f"{text}\x1f{info}\x1e".encode())
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
