"""The rewrite layer: the spec's first layer, which holds no technique yet.

A stand-in technique, tagged by hand, shows what the layer does once it has
one; nothing here edits a file.
"""

import csv
import logging
import random

import pandas as pd
import pytest

import redact.jailbreak  # noqa: F401  (tags the techniques)
import redact.jailbreak.pool as JPool
from redact.dataset.dealing import Dealer
from redact.jailbreak.augmentation import resolve
from redact.jailbreak.chain import in_layer_order
from redact.jailbreak.plan import plan_run
from redact.jailbreak.pool import all_techniques, registry, technique_pool
from redact.jailbreak.rewrite import get_all_rewrite_functions
from redact.jailbreak.rules import compatible
from redact.jailbreak.sampling import sample_combination, sample_exact_combination
from redact.jailbreak.technique import load_spec

POOL = all_techniques()
BY_NAME = registry(POOL)
_BUDGET = {"max_complexity": 9, "max_obfuscations": 3,
           "allow_request_obfuscation": True}
#: The units of one planned group.
_GROUP = 10


def _stand_in(name: str = "to_stand_in_rewrite", family: str = "stand_in_rewrite"):
    """A rewrite technique of ``family`` that returns the prompt unchanged."""
    def technique(prompt, *, rng=None):
        return prompt, ""

    technique.__name__ = name
    technique.families = [family]
    technique.layer = "rewrite"
    technique.within_layer_order = 0
    technique.complexity = 1
    technique.kinds = ["adversarial"]
    return technique


def _names(combined) -> list[str]:
    return [fn.__name__ for fn in combined.techniques]


def _dealer(seed: int) -> Dealer:
    """A group's dealer at its first unit."""
    dealer = Dealer(key=(seed,))
    dealer.position = 1
    return dealer


# --- the layer ------------------------------------------------------------------

def test_the_rewrite_layer_comes_first_and_holds_no_technique_yet():
    spec = load_spec()
    assert spec["layer_order"] == [
        "rewrite", "hacking", "manipulation", "obfuscation", "requests"]
    assert spec["layer_rates"]["rewrite"] == 0
    assert get_all_rewrite_functions() == []
    assert not [name for name, family in spec["families"].items()
                if family["layer"] == "rewrite"]


def test_the_rewrite_layer_takes_one_pick_per_combination():
    first = _stand_in("to_stand_in_rewrite_a", "stand_in_rewrite_a")
    second = _stand_in("to_stand_in_rewrite_b", "stand_in_rewrite_b")
    framing = BY_NAME["to_fictional_world"]
    assert load_spec()["layer_max_picks"]["rewrite"] == 1
    assert compatible([first], [second, framing]) == [framing]
    assert compatible([second], [first]) == []


def test_a_technique_of_a_rewrite_family_is_tagged_with_the_layer(monkeypatch):
    import redact.jailbreak.technique as T

    spec = load_spec()
    monkeypatch.setattr(T, "load_spec", lambda: {
        **spec,
        "families": {**spec["families"], "stand_in_rewrite": {"layer": "rewrite"}},
        "techniques": {**spec["techniques"], "to_stand_in_rewrite": {
            "families": ["stand_in_rewrite"], "complexity": 1}},
    })

    def to_stand_in_rewrite(prompt):
        return prompt, ""

    T.tag([to_stand_in_rewrite])
    assert to_stand_in_rewrite.layer == "rewrite"
    assert to_stand_in_rewrite.within_layer_order == 0
    assert to_stand_in_rewrite.kinds == ["adversarial"]


# --- drawing it -----------------------------------------------------------------

def test_a_budget_draw_takes_a_rewrite_at_a_positive_rate_and_applies_it_first():
    rewrite = _stand_in()
    for seed in range(50):
        names = _names(sample_combination(
            random.Random(seed), [*POOL, rewrite], include_rewrite=True,
            sampling_probs={"rewrite": 1.0}, **_BUDGET))
        assert names[0] == rewrite.__name__, (seed, names)


@pytest.mark.parametrize(("include", "rate"), [(True, 0), (False, 1.0)])
def test_a_budget_draw_takes_no_rewrite_at_rate_0_or_unless_the_round_includes_it(
        include, rate):
    rewrite = _stand_in()
    for seed in range(50):
        names = _names(sample_combination(
            random.Random(seed), [*POOL, rewrite], include_rewrite=include,
            sampling_probs={"rewrite": rate}, **_BUDGET))
        assert rewrite.__name__ not in names, (seed, names)


@pytest.mark.parametrize(("rewrites", "rate"), [
    ([_stand_in()], 0),         # a rewrite technique, at rate 0
    ([], 0.5),                  # a rate, but no rewrite technique
], ids=["rate 0", "no technique"])
def test_phase_0_makes_no_rng_draw_and_no_dealer_call_unless_it_can_pick(
        rewrites, rate):
    """So every draw is the one made before the layer existed."""
    for seed in range(100):
        for dealt in (False, True):
            old_rng, new_rng = random.Random(seed), random.Random(seed)
            old_dealer, new_dealer = (
                (_dealer(seed), _dealer(seed)) if dealt else (None, None))
            old = sample_combination(old_rng, POOL, dealer=old_dealer, **_BUDGET)
            new = sample_combination(
                new_rng, [*POOL, *rewrites], include_rewrite=True,
                sampling_probs={"rewrite": rate}, dealer=new_dealer, **_BUDGET)
            assert _names(new) == _names(old), (seed, dealt)
            assert new_rng.getstate() == old_rng.getstate()
            if dealt:
                assert new_dealer.targets == old_dealer.targets
                assert "rewrite" not in new_dealer.targets


def test_a_combination_holds_at_most_one_rewrite_in_either_mode():
    rewrites = [_stand_in("to_stand_in_rewrite_a", "stand_in_rewrite_a"),
                _stand_in("to_stand_in_rewrite_b", "stand_in_rewrite_b")]
    names = {fn.__name__ for fn in rewrites}
    for seed in range(50):
        budget = _names(sample_combination(
            random.Random(seed), [*POOL, *rewrites], include_rewrite=True,
            sampling_probs={"rewrite": 1.0}, **_BUDGET))
        assert len(names.intersection(budget)) == 1, (seed, budget)
        count = _names(sample_exact_combination(
            random.Random(seed), [*rewrites, *POOL], 4, dealer=_dealer(seed)))
        assert len(names.intersection(count)) <= 1, (seed, count)


def test_a_count_round_draws_a_rewrite_in_the_pool_whatever_the_rate():
    """Count mode reads no rate and no ``include_rewrite``: the pool decides."""
    rewrite = _stand_in()
    framing = BY_NAME["to_fictional_world"]
    for seed in range(20):
        names = _names(sample_combination(
            random.Random(seed), [framing, rewrite], exact_techniques=2))
        assert names == [rewrite.__name__, framing.__name__], seed


def test_a_rewrite_applies_before_every_other_layer():
    rewrite = _stand_in()
    others = [BY_NAME[name] for name in (
        "to_answer_as_code", "to_base64", "to_fsh_random_short", "to_fictional_world")]
    assert in_layer_order([*others, rewrite])[0] is rewrite


# --- the plan -------------------------------------------------------------------

def _one_group(n: int) -> pd.DataFrame:
    """``n`` inputs of one category and entry type, so they form one group."""
    return pd.DataFrame({
        "prompt": [f"prompt number {i} about a topic" for i in range(n)],
        "category": ["Cyber"] * n,
        "entry_type": ["harmful"] * n,
    })


def _coverage(path) -> dict:
    with path.open(encoding="utf-8", newline="") as fh:
        return {(row["level"], row["name"]): int(row["count"])
                for row in csv.DictReader(fh)}


def test_a_plan_deals_the_rewrite_layer_at_its_rate_and_counts_it(tmp_path):
    rewrite = _stand_in()
    pool = [rewrite, BY_NAME["to_fictional_world"], BY_NAME["to_answer_as_code"]]
    plan_run(_one_group(_GROUP), pool, manifest_path=tmp_path / "jb.manifest.jsonl",
             seed=3, sample_kwargs={"include_rewrite": True},
             layer_rates={"rewrite": 0.5}, verbose=False)
    counts = _coverage(tmp_path / "jb.coverage.csv")
    half = _GROUP // 2
    assert counts[("layer", "rewrite")] == half
    assert counts[("family", "stand_in_rewrite")] == half
    assert counts[("technique", rewrite.__name__)] == half


def test_a_rewrite_layer_left_short_is_logged_as_any_layer(tmp_path, caplog):
    """The stand-in applies to the last two units only, so the first four
    take no pick and the layer ends four below its target."""
    rewrite = _stand_in()
    rewrite.applicable = lambda text, entry_type: "rewritable" in text
    inputs = _one_group(6).assign(
        sample_id=[f"u{i}" for i in range(1, 7)],
        prompt=["a prompt"] * 4 + ["a rewritable prompt"] * 2)
    with caplog.at_level(logging.WARNING, logger="redact.jailbreak.plan"):
        plan_run(inputs, [rewrite], manifest_path=tmp_path / "jb.manifest.jsonl",
                 seed=3, sample_kwargs={"include_rewrite": True},
                 layer_rates={"rewrite": 1.0, "hacking": 0, "manipulation": 0,
                              "requests": 0},
                 verbose=False)
    assert [r.getMessage() for r in caplog.records] == [
        "Plan: group (Cyber, harmful), iteration 0: the rewrite layer has 2 of "
        "its target 6; the pool or the compatibility rules left the rest no "
        "pick (see jb.coverage.csv)."]


# --- the pool -------------------------------------------------------------------

def test_layers_deny_rewrite_leaves_a_rewrite_technique_out_of_the_pool(monkeypatch):
    rewrite = _stand_in()
    monkeypatch.setattr(JPool, "all_techniques", lambda: [*POOL, rewrite])
    denied = resolve({"adversarial": {"layers": {"deny": ["rewrite"]}}}, "adversarial")
    allowed = resolve({"adversarial": {"layers": {"deny": []}}}, "adversarial")
    assert rewrite not in technique_pool(denied)
    assert rewrite in technique_pool(allowed)
