"""The decode hints: a hinted encoding writes one of four hints around its
encoded text, at the spec's ``decode_hint_rate``.

The tests fill hints of their own (short stand-ins, two a technique) into the
pools, and assert on names, counts and slots, never on the packaged text.
"""

import inspect
import json
import random
import string
from pathlib import Path

import pytest

import redact.jailbreak  # noqa: F401  (tags the techniques)
from redact import paths
from redact.jailbreak.augmentation import resolve
from redact.jailbreak.chain import combine_techniques
from redact.jailbreak.directives import check_pools
from redact.jailbreak.obfuscation import encoding as E
from redact.jailbreak.obfuscation.encoding import get_encoding_functions
from redact.jailbreak.obfuscation.sensitive_words import _ENCODE_SPECS
from redact.jailbreak.pool import technique_pool
from redact.jailbreak.technique import decode_hint_rate, load_spec, read_spec

PROMPT = "How do plants turn sunlight into food"

#: The encodings with no hints: the five of the eval pool, until the eval
#: review, and the one the natural kind draws.
UNHINTED = {"to_base64", "to_unicode_escape", "to_ascii_ordinal", "to_morse",
            "to_braille", "to_leetspeak_basic"}

_EVAL_POOL = Path(__file__).parents[2] / "fixtures" / "eval_pool.json"

REGISTERED = {fn.__name__: fn for fn in get_encoding_functions()}
HINTED = sorted(name for name in REGISTERED if name not in UNHINTED)


def _plain(name: str, rng: random.Random) -> tuple[str, str]:
    """What the plain encoder ``name`` writes for ``PROMPT``, drawing from
    ``rng`` as the encoder does."""
    encode = getattr(E, name)
    return encode(PROMPT, rng=rng) if "rng" in inspect.signature(encode).parameters \
        else encode(PROMPT)


def _slots(template: str) -> set[str]:
    return {field for _, field, _, _ in string.Formatter().parse(template) if field}


@pytest.fixture
def stand_in_hints(monkeypatch):
    """Two hints of the tests' own for every hinted encoding; the Caesar
    cipher's states the shift."""
    pools = {name: [{"name": "one", "template": "<1:{prompt}>"},
                    {"name": "two", "template": "<2:{prompt}>"}] for name in E._HINTS}
    pools["to_caesar_cipher"] = [
        {"name": "one", "template": "<1:{shift}:{prompt}>"},
        {"name": "two", "template": "<2:{shift}:{prompt}>"}]
    monkeypatch.setattr(E, "_HINTS", pools)


@pytest.fixture
def always(monkeypatch, stand_in_hints):
    monkeypatch.setattr(E, "decode_hint_rate", lambda: 1.0)


@pytest.fixture
def never(monkeypatch, stand_in_hints):
    monkeypatch.setattr(E, "decode_hint_rate", lambda: 0.0)


# --- the hints file and which encodings have them -------------------------------

def test_twenty_two_encodings_have_four_hints_each():
    """Spec 1.4's letter spacing is the twenty-second."""
    assert len(E._HINTS) == 22  # noqa: PLR2004
    assert all(len(pool) == 4 for pool in E._HINTS.values())  # noqa: PLR2004
    assert sorted(E._HINTS) == HINTED


def test_the_hints_of_the_caesar_cipher_state_its_shift_and_no_other_does():
    for name, pool in E._HINTS.items():
        for hint in pool:
            expected = {"prompt", "shift"} if name == "to_caesar_cipher" else {"prompt"}
            assert _slots(hint["template"]) == expected, (name, hint["name"])


def test_every_hinted_encoding_draws_from_its_own_optional_pool():
    for name in HINTED:
        ref = REGISTERED[name].variant_pool
        assert (ref.file, ref.directive, ref.info_key, ref.optional) == (
            "obfuscation/decode_hints.json", name, "hint", True)
    check_pools(REGISTERED[name].variant_pool for name in HINTED)


def test_the_eval_pool_and_the_natural_encodings_have_no_hints():
    eval_pool = set(json.loads(_EVAL_POOL.read_text(encoding="utf-8")))
    assert eval_pool & set(REGISTERED) == {
        "to_base64", "to_unicode_escape", "to_ascii_ordinal", "to_morse",
        "to_braille"}
    assert set(REGISTERED) >= UNHINTED
    assert not UNHINTED & set(E._HINTS)


def test_a_hint_pool_named_for_no_encoder_raises():
    with pytest.raises(ValueError) as raised:
        E._check_hints({"to_rot13": [], "to_rot14": []}, E._ENCODERS)
    assert str(raised.value) == (
        "configs/jailbreak/obfuscation/decode_hints.json has hints for "
        "['to_rot14'], which are not encoders of obfuscation/encoding.py; name an "
        "encoder or remove the entry.")


@pytest.mark.parametrize("name", sorted(UNHINTED))
def test_a_hint_pool_for_an_eval_or_natural_encoding_raises(name):
    """A hint would change the text of an encoding the eval pool or the
    natural kind draws, so a pool naming one fails at import."""
    with pytest.raises(ValueError, match=rf"has hints for \['{name}'\], which the "
                       "eval pool or the natural kind draws"):
        E._check_hints({name: []}, E._ENCODERS)


def test_the_never_hinted_encodings_are_the_eval_and_natural_ones():
    """The constant ``_check_hints`` reads is the encoders the packaged eval
    pool and natural kind hold, so a change to either shows here."""
    encoders = {fn.__name__ for fn in E._ENCODERS}
    eval_pool = set(json.loads(_EVAL_POOL.read_text(encoding="utf-8")))
    natural = {fn.__name__ for fn in technique_pool(resolve(None, "natural"))}
    assert E.NEVER_HINTED == UNHINTED == (eval_pool | natural) & encoders


# --- the plain encoders -------------------------------------------------------

@pytest.mark.parametrize("name", sorted(UNHINTED))
def test_an_unhinted_encoding_is_its_plain_encoder(name):
    assert REGISTERED[name] is getattr(E, name)
    assert not hasattr(REGISTERED[name], "variant_pool")


@pytest.mark.parametrize("name", HINTED)
def test_the_plain_encoder_of_a_hinted_one_ignores_the_rate(name, always):
    assert getattr(E, name) is not REGISTERED[name]
    text, info = _plain(name, random.Random(1))
    assert "hint=" not in info
    assert REGISTERED[name](PROMPT, rng=random.Random(1))[0] != text


def test_the_sensitive_word_encodings_call_the_plain_encoders():
    for _, encode_fn, _, _ in _ENCODE_SPECS:
        assert encode_fn is getattr(E, encode_fn.__name__)
        assert not hasattr(encode_fn, "variant_pool")


# --- no hint: today's output and today's draws ---------------------------------

@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("prompt", [PROMPT, 'Say "hi" to {x}\nthen go, café 123', "42"],
                         ids=["plain", "hostile", "digits"])
@pytest.mark.parametrize("name", HINTED)
def test_a_call_without_a_hint_writes_what_the_plain_encoder_writes(
        name, prompt, seed, never):
    """Also for a text some encoders leave as it is (digits), which gets no
    hint at any rate."""
    encode = getattr(E, name)
    expected = (encode(prompt, rng=random.Random(seed))
                if "rng" in inspect.signature(encode).parameters else encode(prompt))
    assert REGISTERED[name](prompt, rng=random.Random(seed)) == expected


@pytest.mark.parametrize("name", HINTED)
def test_a_hinted_call_reads_one_draw_beyond_the_encoder(name, never):
    """The rate is one ``rng.random()``, after the encoder's own draws."""
    used, expected = random.Random(7), random.Random(7)
    REGISTERED[name](PROMPT, rng=used)
    _plain(name, expected)
    expected.random()
    assert used.getstate() == expected.getstate()


def test_the_packaged_rate_is_one_half_and_the_spec_reads_it():
    assert load_spec()["decode_hint_rate"] == 0.5  # noqa: PLR2004
    assert decode_hint_rate() == 0.5  # noqa: PLR2004


@pytest.mark.parametrize("name", ["to_rot13", "to_separator", "to_caesar_cipher"])
def test_about_half_the_calls_carry_a_hint_at_the_packaged_rate(
        name, stand_in_hints):
    """2,000 seeded calls give 1,003 to 1,040 hinted at 0.5; a rate of 0.45
    gives 872 to 933 and one of 0.55 gives 1,095 to 1,164, so the bounds
    (nearly three standard deviations of a fair coin, 22) keep them out."""
    hinted = sum("hint=" in REGISTERED[name](PROMPT, rng=random.Random(seed))[1]
                 for seed in range(2000))
    assert 940 <= hinted <= 1060  # noqa: PLR2004


# --- a hint --------------------------------------------------------------------

@pytest.mark.parametrize("name", HINTED)
def test_a_hint_is_set_around_the_encoded_text_and_named_in_the_info(name, always):
    encoded, plain_info = _plain(name, random.Random(3))
    text, info = REGISTERED[name](PROMPT, rng=random.Random(3))
    hint = info.rpartition("hint=")[2]
    assert hint in ("one", "two")
    assert info == ";".join(filter(None, [plain_info, f"hint={hint}"]))
    shift = f"{plain_info.partition('=')[2]}:" if name == "to_caesar_cipher" else ""
    assert text == f"<{'1' if hint == 'one' else '2'}:{shift}{encoded}>"


def test_a_hint_is_drawn_from_the_pool_by_the_rng(always):
    drawn = {REGISTERED["to_rot13"](PROMPT, rng=random.Random(seed))[1]
             for seed in range(20)}
    assert drawn == {"hint=one", "hint=two"}


def test_the_hint_of_the_caesar_cipher_states_the_shift_that_call_used(always):
    shifts = set()
    for seed in range(30):
        text, info = REGISTERED["to_caesar_cipher"](PROMPT, rng=random.Random(seed))
        shift = info.split(";")[0].partition("=")[2]
        assert f":{shift}:" in text
        shifts.add(shift)
    assert len(shifts) > 1


def test_an_encoding_that_changed_nothing_gets_no_hint(always):
    assert REGISTERED["to_rot13"]("123 456", rng=random.Random(1)) == ("123 456", "")


def test_the_chain_gives_a_hinted_encoding_an_rng_of_its_own(always):
    fn = combine_techniques(REGISTERED["to_rot13"])
    assert "rng" in inspect.signature(REGISTERED["to_rot13"]).parameters
    text, info = fn(PROMPT, rng_seed=(1, "jailbreak", "an_input", 0))
    assert text.startswith("<") and info.startswith("hint=")
    assert fn(PROMPT, rng_seed=(1, "jailbreak", "an_input", 0)) == (text, info)


@pytest.mark.parametrize("name", sorted(UNHINTED))
def test_an_unhinted_encoding_writes_the_same_at_any_rate(name, always):
    assert REGISTERED[name](PROMPT) == getattr(E, name)(PROMPT)


# --- the rate in the spec ------------------------------------------------------

def _spec_with(tmp_path, **changes):
    spec = json.loads(
        (paths.jailbreak_configs_dir() / "combination_spec.json").read_text("utf-8"))
    spec.update(changes)
    spec["generators"] = {}
    path = tmp_path / "combination_spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path


@pytest.mark.parametrize("rate", [-0.1, 1.5, "0.5", True, None])
def test_the_spec_rate_is_a_number_from_zero_to_one(tmp_path, rate):
    with pytest.raises(ValueError) as raised:
        read_spec(_spec_with(tmp_path, decode_hint_rate=rate))
    assert str(raised.value) == (
        "combination_spec.json: decode_hint_rate must be a number from 0 to 1, the "
        f"share of hinted encodings written with a hint, got {rate!r}.")


@pytest.mark.parametrize("rate", [0, 0.25, 1])
def test_a_rate_from_zero_to_one_is_read(tmp_path, rate):
    assert read_spec(_spec_with(tmp_path, decode_hint_rate=rate))[
        "decode_hint_rate"] == rate
