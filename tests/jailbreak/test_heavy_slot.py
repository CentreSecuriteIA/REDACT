"""The heavy full-text encodings' dealing slot.

A budget draw's obfuscations (phase 4) deal the ten heavy full-text
encodings under a name of their own, ``HEAVY_ENCODINGS``, beside the
``encode`` family, at the spec's ``heavy_encoding_weight`` times encode's
weight: dealt as encode's members, they were drawn 8 to 16 times in 3,000
units. A count draw (the eval type, the natural kind, the first two
escalation rounds) deals them as encode's members, as before.
"""

import json
import random
from collections import Counter

import pytest

import redact.jailbreak  # noqa: F401  (tags the techniques)
import redact.jailbreak.plan as JP
import redact.jailbreak.technique as T
from redact import paths
from redact.dataset.dealing import Dealer
from redact.jailbreak.augmentation import resolve
from redact.jailbreak.plan import assign_combination
from redact.jailbreak.pool import all_techniques
from redact.jailbreak.sampling import HEAVY_ENCODINGS, deal_key, sample_combination
from redact.jailbreak.technique import heavy_encoding_weight, load_spec, read_spec
from tests.jailbreak import simulate

#: The ten: first family encode, encode_weight heavy.
HEAVY = {"to_base64", "to_unicode_escape", "to_ascii_ordinal", "to_morse", "to_braille",
         "to_hex", "to_binary", "to_html_entities", "to_base32", "to_base85"}


@pytest.fixture
def without_the_slot(monkeypatch):
    """The packaged spec with ``heavy_encoding_weight`` left out."""
    spec = {key: value for key, value in load_spec().items()
            if key != "heavy_encoding_weight"}
    monkeypatch.setattr(T, "load_spec", lambda: spec)


def _at_weight(monkeypatch, weight: float) -> None:
    spec = {**load_spec(), "heavy_encoding_weight": weight}
    monkeypatch.setattr(T, "load_spec", lambda: spec)


# --- the weight in the spec ----------------------------------------------------

def test_the_packaged_weight_is_one_quarter_and_the_spec_reads_it():
    """0.25 since spec 1.4: 0.2 drew the ten 30 or 31 times in 3,000 default
    attack units once the four language-mix families joined the obfuscation
    picks, where the weight was set for about 40; 0.25 draws 37 to 40
    (seeds 42 to 46, in one, three or ten groups)."""
    assert load_spec()["heavy_encoding_weight"] == 0.25  # noqa: PLR2004
    assert heavy_encoding_weight() == 0.25  # noqa: PLR2004


def _spec_with(tmp_path, **changes):
    spec = json.loads(
        (paths.jailbreak_configs_dir() / "combination_spec.json").read_text("utf-8"))
    spec.update(changes)
    spec["generators"] = {}
    path = tmp_path / "combination_spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path


@pytest.mark.parametrize("weight", [0, -0.1, 1.5, "0.2", True, None])
def test_the_spec_weight_is_a_number_above_zero_and_at_most_one(tmp_path, weight):
    with pytest.raises(ValueError) as raised:
        read_spec(_spec_with(tmp_path, heavy_encoding_weight=weight))
    assert str(raised.value) == (
        "combination_spec.json: heavy_encoding_weight must be a number above 0 and "
        "at most 1, the weight the heavy full-text encodings are dealt at beside "
        f"the encode family in a budget draw, got {weight!r}; leave it out to deal "
        "them as encode's members.")


@pytest.mark.parametrize("weight", [0.05, 0.2, 1])
def test_a_weight_above_zero_and_at_most_one_is_read(tmp_path, weight):
    assert read_spec(_spec_with(tmp_path, heavy_encoding_weight=weight))[
        "heavy_encoding_weight"] == weight


def test_a_spec_without_the_weight_is_read(tmp_path):
    path = _spec_with(tmp_path)
    spec = json.loads(path.read_text("utf-8"))
    del spec["heavy_encoding_weight"]
    path.write_text(json.dumps(spec), encoding="utf-8")
    assert "heavy_encoding_weight" not in read_spec(path)


# --- what the slot holds -------------------------------------------------------

def test_the_slot_holds_the_ten_heavy_full_text_encodings():
    """Not the three heavy ``to_sensitive_words_encode_*``, which encode only
    the words a model marks and are dealt under ``sensitive_words``."""
    every = all_techniques()
    assert {fn.__name__ for fn in every
            if deal_key(fn, heavy_apart=True) == HEAVY_ENCODINGS} == HEAVY
    heavy_words = [fn for fn in every if fn.families[0] == "sensitive_words"
                   and getattr(fn, "encode_weight", None) == "heavy"]
    assert len(heavy_words) == 3  # noqa: PLR2004
    assert {deal_key(fn, heavy_apart=True) for fn in heavy_words} == {"sensitive_words"}


def _base64():
    return {fn.__name__: fn for fn in all_techniques()}["to_base64"]


def test_a_heavy_encoding_is_encode_s_unless_the_slot_is_asked_for():
    assert deal_key(_base64()) == "encode"
    assert deal_key(_base64(), heavy_apart=True) == HEAVY_ENCODINGS


def test_a_spec_without_the_weight_has_no_slot(without_the_slot):
    assert deal_key(_base64(), heavy_apart=True) == "encode"


def test_the_slot_takes_the_spec_s_weight_times_encode_s(monkeypatch):
    """The plan's dealers weigh the slot at 0.25, or 0.25 times the kind's
    ``encode`` weight."""
    weights = []

    class Recording(Dealer):
        def __init__(self, given=None, key=()):
            weights.append(dict(given or {}))
            super().__init__(given, key)

    monkeypatch.setattr(JP, "Dealer", Recording)
    simulate.plan_rows(simulate.inputs(4), all_techniques())
    simulate.plan_rows(simulate.inputs(4), all_techniques(),
                       family_weights={"encode": 2})
    assert [w[HEAVY_ENCODINGS] for w in weights] == [0.25, 0.5]


def test_family_weights_cannot_name_the_slot():
    """It is not a family: the kind's ``family_weights`` refuses it, as any
    name the kind does not draw; weight ``encode`` to move it."""
    with pytest.raises(ValueError, match="names 'encode:heavy', which the kind does "
                                         "not draw from"):
        resolve({"adversarial": {"family_weights": {HEAVY_ENCODINGS: 1}}},
                "adversarial")


def test_a_plan_made_directly_cannot_weight_the_slot(tmp_path):
    """``plan_run``, called without the section, raises rather than replace
    the weight with the spec's, before it writes anything."""
    manifest = tmp_path / "jb.manifest.jsonl"
    with pytest.raises(ValueError, match="family_weights names 'encode:heavy', the "
                                         "heavy full-text encodings' slot"):
        JP.plan_run(simulate.inputs(4), all_techniques(), manifest_path=manifest,
                    family_weights={"encode": 2, HEAVY_ENCODINGS: 1}, verbose=False)
    assert list(tmp_path.iterdir()) == []


# --- budget draws and count draws ------------------------------------------------

def _dealt(n: int, **settings) -> tuple[Dealer, Counter]:
    """Deal ``n`` units of one group from every technique; the dealer and the
    techniques drawn."""
    dealer = Dealer({HEAVY_ENCODINGS: 0.2}, key=(1, "Cyber", "harmful", 0))
    drawn: Counter = Counter()
    for position in range(1, n + 1):
        dealer.position = position
        drawn.update(assign_combination(f"unit{position}", all_techniques(), seed=1,
                                        dealer=dealer, **settings))
    return dealer, drawn


def test_a_budget_draw_deals_and_counts_the_heavy_encodings_apart():
    """The request obfuscation is off here, so every encode draw is phase
    4's: the slot counts the heavy ones, ``encode`` the others."""
    dealer, drawn = _dealt(600)
    family = {fn.__name__: fn.families[0] for fn in all_techniques()}
    heavy = sum(drawn[name] for name in HEAVY)
    assert heavy > 0
    assert dealer.counts[HEAVY_ENCODINGS] == heavy
    assert dealer.counts["encode"] == sum(
        count for name, count in drawn.items()
        if family[name] == "encode" and name not in HEAVY)


def test_a_count_draw_counts_the_heavy_encodings_as_encode_s():
    dealer, drawn = _dealt(300, exact_techniques=2)
    assert sum(drawn[name] for name in HEAVY) > 0
    assert HEAVY_ENCODINGS not in dealer.counts


def _undealt_draws() -> list[list[str]]:
    """200 budget draws without a dealer, as a direct call makes them, under
    the default budget and a larger one."""
    budgets = ({}, {"max_complexity": 9, "max_obfuscations": 3})
    return [[fn.__name__ for fn in sample_combination(
                random.Random(seed), all_techniques(), **budget).techniques]
            for seed in range(100) for budget in budgets]


def test_a_draw_without_a_dealer_is_the_same_at_any_weight_and_without_the_slot(
        monkeypatch):
    """It deals nothing, so the heavy encodings stay encode's members: the
    spec's weight, or its absence, moves none of its draws (the budget
    digest pins them too)."""
    drawn = _undealt_draws()
    assert any(name in HEAVY for names in drawn for name in names)
    _at_weight(monkeypatch, 1)
    assert _undealt_draws() == drawn
    spec = {key: value for key, value in load_spec().items()
            if key != "heavy_encoding_weight"}
    monkeypatch.setattr(T, "load_spec", lambda: spec)
    assert _undealt_draws() == drawn


def _eval_draws() -> list[list[str]]:
    from redact.recipe import load_dataset_type

    base = load_dataset_type("eval")["augmentations"]
    return simulate.deal_kind({}, 300, groups=3, seed=43, base=base)[0]


def test_the_eval_type_draws_what_it_drew_without_the_slot(monkeypatch):
    """It draws by count, which deals the heavy encodings as encode's
    members: the slot moves none of its draws (its digest is pinned too)."""
    with_slot = _eval_draws()
    spec = {key: value for key, value in load_spec().items()
            if key != "heavy_encoding_weight"}
    monkeypatch.setattr(T, "load_spec", lambda: spec)
    assert _eval_draws() == with_slot


def test_the_default_attacks_draw_otherwise_without_the_slot(monkeypatch):
    """The other way: a budget kind's draws move with it."""
    with_slot = simulate.deal_kind({}, 300, groups=3)[0]
    spec = {key: value for key, value in load_spec().items()
            if key != "heavy_encoding_weight"}
    monkeypatch.setattr(T, "load_spec", lambda: spec)
    assert simulate.deal_kind({}, 300, groups=3)[0] != with_slot


# --- how often --------------------------------------------------------------------

def _heavy_draws(n: int) -> int:
    drawn, _ = simulate.deal_kind({}, n)
    return sum(name in HEAVY for names in drawn for name in names)


def test_the_weight_sets_how_often_the_heavy_encodings_are_drawn(monkeypatch):
    """About the weight times a family's draws: of 2,000 units in one group,
    17 at 0.2 and 34 at 0.4. The sample is 2,000 (it was 1,000, giving 15 and
    29) since spec 1.5's two exclusive obfuscation families thinned the layer
    and 1,000 units left too few draws at 0.2 to compare. The ratio holds the
    weight, not the count."""
    _at_weight(monkeypatch, 0.2)
    fifth = _heavy_draws(2000)
    _at_weight(monkeypatch, 0.4)
    double = _heavy_draws(2000)
    assert fifth >= 10  # noqa: PLR2004
    assert 1.6 <= double / fifth <= 2.4  # noqa: PLR2004
