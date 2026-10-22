"""The rewrite layer: the spec's first layer, off by default.

``POOL`` is every technique but the layer's own, so a stand-in technique,
tagged by hand, can be the layer's only one and show what the layer does;
nothing here edits a file. The last tests are about the 16 techniques the
layer holds.
"""

import csv
import json
import logging
import random
from collections import Counter

import pandas as pd
import pytest

import redact.jailbreak  # noqa: F401  (tags the techniques)
import redact.jailbreak.augmentation as JAugmentation
import redact.jailbreak.pipeline as JPipe
import redact.jailbreak.pool as JPool
from redact import paths
from redact.dataset import Manifest
from redact.dataset.dealing import Dealer
from redact.dataset.planfile import read_plan
from redact.jailbreak.augmentation import (
    defaults,
    layers_off_by_default,
    resolve,
    rounds_of,
)
from redact.jailbreak.chain import in_layer_order
from redact.jailbreak.plan import default_escalation_schedule, plan_run
from redact.jailbreak.pool import (
    all_techniques,
    registry,
    technique_pool,
    uses_translation,
)
from redact.jailbreak.rewrite import get_all_rewrite_functions
from redact.jailbreak.rules import compatible
from redact.jailbreak.sampling import sample_combination, sample_exact_combination
from redact.jailbreak.technique import KINDS, load_spec, read_spec
from redact.llms.model_config import default_model_for_role
from redact.recipe import augmentations_of, load_dataset_type, load_recipe, resolve_role
from redact.stages import generate_jailbreaks
from tests.jailbreak.test_pipeline_jailbreaks import _inputs

POOL = [fn for fn in all_techniques() if fn.layer != "rewrite"]
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

def test_the_rewrite_layer_comes_first_and_holds_16_techniques_in_three_families():
    spec = load_spec()
    assert spec["layer_order"] == [
        "rewrite", "hacking", "manipulation", "obfuscation", "requests"]
    assert spec["layer_rates"]["rewrite"] == 0
    assert {name for name, family in spec["families"].items()
            if family["layer"] == "rewrite"} == {
        "persuasion", "tense_rewrite", "model_distractors"}
    rewrites = get_all_rewrite_functions()
    assert Counter(fn.families[0] for fn in rewrites) == {
        "persuasion": 12, "tense_rewrite": 2, "model_distractors": 2}
    assert {(fn.layer, fn.requires_llm, tuple(fn.kinds), fn.since, fn.complexity)
            for fn in rewrites} == {("rewrite", True, ("adversarial",), "1.3", 2)}


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


def test_a_plan_deals_the_real_rewrite_families_evenly_at_the_layer_rate(tmp_path):
    """Half of 12 units draw one rewrite technique each: two of each of the
    three families, no technique twice."""
    allowed = resolve({"adversarial": {"layers": {"allow": ["rewrite"]}}},
                      "adversarial")
    plan_run(_one_group(12), technique_pool(allowed),
             manifest_path=tmp_path / "jb.manifest.jsonl", seed=3,
             sample_kwargs={"include_rewrite": True},
             layer_rates={"rewrite": 0.5}, verbose=False)
    counts = _coverage(tmp_path / "jb.coverage.csv")
    assert counts[("layer", "rewrite")] == 6    # noqa: PLR2004
    assert [counts[("family", family)] for family in (
        "persuasion", "tense_rewrite", "model_distractors")] == [2, 2, 2]
    assert max(counts[("technique", fn.__name__)]
               for fn in get_all_rewrite_functions()) == 1


# --- the pool -------------------------------------------------------------------

def _register(monkeypatch, *rewrites) -> None:
    """``rewrites`` registered beside every other layer's technique and in place
    of the layer's own, for the pool and the rounds: none leaves the layer
    empty, as it was before it held any."""
    for module in (JPool, JAugmentation):
        monkeypatch.setattr(module, "all_techniques", lambda: [*POOL, *rewrites])


def test_layers_allow_rewrite_puts_a_rewrite_technique_in_the_pool(monkeypatch):
    rewrite = _stand_in()
    _register(monkeypatch, rewrite)
    allowed = resolve({"adversarial": {"layers": {"allow": ["rewrite"]}}},
                      "adversarial")
    assert rewrite not in technique_pool(resolve(None, "adversarial"))
    assert rewrite in technique_pool(allowed)


# --- the augmentation section ---------------------------------------------------

#: An adversarial budget round as a plan file recorded it before the layer
#: existed: the packaged kind's, an ``include_*`` for each of four layers.
_ROUND_BEFORE = {"max_complexity": 6, "max_obfuscations": 2, "include_hacking": True,
                 "include_manipulation": True, "include_obfuscation": True,
                 "include_requests": True, "allow_request_obfuscation": True}


def test_the_spec_has_the_rewrite_layer_off_by_default():
    assert load_spec()["layers_off_by_default"] == ["rewrite"]
    assert layers_off_by_default() == ["rewrite"]


@pytest.mark.parametrize("off", [["rewrites"], "rewrite"])
def test_the_spec_may_name_only_layers_of_its_layer_order_off_by_default(
        tmp_path, off):
    spec = json.loads(
        (paths.jailbreak_configs_dir() / "combination_spec.json").read_text("utf-8"))
    spec["layers_off_by_default"] = off
    spec["generators"] = {}
    path = tmp_path / "combination_spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError) as raised:
        read_spec(path)
    assert str(raised.value) == (
        "combination_spec.json: layers_off_by_default must be a list of layers of "
        f"layer_order, got {off!r}; write names of ['rewrite', 'hacking', "
        "'manipulation', 'obfuscation', 'requests'].")


@pytest.mark.parametrize("kind", KINDS)
def test_both_packaged_kinds_deny_the_rewrite_layer_by_the_spec(kind):
    """The file names no layer; the spec's list denies it."""
    assert defaults()[kind]["layers"] == {"deny": []}
    assert resolve(None, kind)["layers"] == {"deny": ["rewrite"]}


@pytest.mark.parametrize("dataset_type", ["eval", "training"])
def test_the_dataset_types_resolve_their_layers_as_before(dataset_type):
    """``{"deny": ["rewrite"]}`` and no ``allow`` key, so their resolved
    kinds are what they were (their pools and draws are pinned elsewhere)."""
    base = load_dataset_type(dataset_type)["augmentations"]
    for kind in KINDS:
        resolved = resolve(None, kind, base=base)
        assert resolved is None or resolved["layers"] == {"deny": ["rewrite"]}


@pytest.mark.parametrize(("deny", "denied"), [
    (["hacking"], ["hacking", "rewrite"]),
    (["rewrite", "hacking"], ["rewrite", "hacking"]),
    ([], ["rewrite"]),
])
def test_a_kinds_own_layers_deny_still_denies_the_rewrite_layer(deny, denied):
    """A deny given replaces the default's whole, in a section or a type
    file; the spec's layer joins it whether it names the layer or not."""
    given = {"adversarial": {"layers": {"deny": deny}}}
    for resolved in (resolve(given, "adversarial"),
                     resolve(None, "adversarial", base=given)):
        assert resolved["layers"] == {"deny": denied}


def test_a_kinds_own_layers_deny_keeps_a_registered_rewrite_out(monkeypatch):
    rewrite = _stand_in()
    _register(monkeypatch, rewrite)
    kind = resolve({"adversarial": {"layers": {"deny": ["hacking"]}}}, "adversarial")
    assert rewrite not in technique_pool(kind)
    assert rounds_of(kind) == [{**_ROUND_BEFORE, "include_hacking": False}]


def test_layers_allow_given_over_a_deny_that_names_the_layer_allows_it():
    """An allowed name leaves the default's deny, as for families and
    techniques: a run may allow the layer over a type file, or a
    ``config_dir`` copy, that denies it."""
    base = {"adversarial": {"layers": {"deny": ["rewrite", "hacking"]}}}
    kind = resolve({"adversarial": {"layers": {"allow": ["rewrite"]}}},
                   "adversarial", base=base)
    assert kind["layers"] == {"deny": ["hacking"], "allow": ["rewrite"]}
    assert resolve({"adversarial": kind}, "adversarial") == kind


def test_layers_allow_naming_a_layer_on_by_default_raises():
    with pytest.raises(ValueError) as raised:
        resolve({"adversarial": {"layers": {"allow": ["hacking"]}}}, "adversarial")
    assert str(raised.value) == (
        "augmentations.adversarial.layers.allow names 'hacking', which is on "
        "already: layers.allow turns on only the layers combination_spec.json has "
        "off by default (['rewrite']). Take it out of layers.allow.")


@pytest.mark.parametrize(("given", "key", "name"), [
    ({"techniques": {"allow": ["to_persuade_scarcity"]}}, "techniques.allow",
     "to_persuade_scarcity"),
    ({"families": {"allow": ["persuasion"]}}, "families.allow", "persuasion"),
    ({"family_weights": {"persuasion": 2}}, "family_weights", "persuasion"),
], ids=["technique", "family", "weight"])
def test_naming_a_rewrite_technique_without_layers_allow_raises(given, key, name):
    """The pool would drop it, and a kind that named nothing else would draw
    nothing; with the layer allowed it is drawn."""
    with pytest.raises(ValueError) as raised:
        resolve({"adversarial": given}, "adversarial")
    assert str(raised.value) == (
        f"augmentations.adversarial.{key} names {name!r}, of the rewrite layer, "
        "which combination_spec.json has off by default, so the kind would not "
        'draw it: write "layers": {"allow": ["rewrite"]} in the kind to turn the '
        f"layer on, or take it out of {key}.")
    allowed = resolve({"adversarial": {**given, "layers": {"allow": ["rewrite"]}}},
                      "adversarial")
    assert [fn for fn in technique_pool(allowed) if fn.layer == "rewrite"]


def _config_dir(tmp_path, layers: dict):
    """A config folder whose ``augmentations.json`` is the packaged one with
    ``layers`` in both kinds."""
    section = json.loads(
        (paths.jailbreak_configs_dir() / "augmentations.json").read_text("utf-8"))
    for kind in KINDS:
        section[kind]["layers"] = layers
    (tmp_path / "jailbreak").mkdir()
    (tmp_path / "jailbreak" / "augmentations.json").write_text(
        json.dumps(section), encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("where", ["section", "config_dir"])
def test_a_layer_named_under_both_allow_and_deny_raises(tmp_path, where):
    both = {"allow": ["rewrite"], "deny": ["rewrite"]}
    if where == "section":
        section, config_dir = {"adversarial": {"layers": both}}, None
    else:
        section, config_dir = None, _config_dir(tmp_path, both)
    with pytest.raises(ValueError) as raised:
        resolve(section, "adversarial", config_dir=config_dir)
    assert str(raised.value) == (
        "augmentations.adversarial.layers names ['rewrite'] under both allow and "
        "deny; write each under one of them.")


def test_the_packaged_kinds_rounds_are_those_written_before_the_layer():
    assert rounds_of(resolve(None, "adversarial")) == [_ROUND_BEFORE]
    escalation = resolve({"adversarial": {"rounds": "escalation"}}, "adversarial")
    assert not [r for r in rounds_of(escalation) if "include_rewrite" in r]
    assert rounds_of(resolve(None, "natural")) == [{"exact_techniques": 1}]


def test_a_kind_that_allows_the_empty_layer_keeps_the_rounds_written_before_it(
        monkeypatch):
    """With no rewrite technique registered, allowing the layer changes no draw,
    so its rounds stay as they were."""
    _register(monkeypatch)
    allowed = resolve({"adversarial": {"layers": {"allow": ["rewrite"]}}},
                      "adversarial")
    assert rounds_of(allowed) == [_ROUND_BEFORE]
    hacking_off = resolve(
        {"adversarial": {"layers": {"deny": ["hacking"], "allow": ["rewrite"]}}},
        "adversarial")
    assert rounds_of(hacking_off) == [{**_ROUND_BEFORE, "include_hacking": False}]


def test_a_kind_that_allows_the_layer_says_include_rewrite_for_its_techniques():
    kind = resolve({"adversarial": {"layers": {"allow": ["rewrite"]},
                                    "layer_rates": {"rewrite": 0.25}}}, "adversarial")
    assert kind["layer_rates"] == {"rewrite": 0.25}
    assert rounds_of(kind) == [{**_ROUND_BEFORE, "include_rewrite": True}]
    hacking_off = resolve(
        {"adversarial": {"layers": {"deny": ["hacking"], "allow": ["rewrite"]}}},
        "adversarial")
    assert rounds_of(hacking_off) == [
        {**_ROUND_BEFORE, "include_hacking": False, "include_rewrite": True}]
    assert rounds_of(resolve(None, "adversarial")) == [_ROUND_BEFORE]


def test_a_kind_whose_pool_holds_no_rewrite_keeps_its_rounds_with_the_layer_allowed():
    """The eval type's lists name no rewrite family, so allowing the layer
    there adds nothing to its pool and nothing to its rounds: a kept eval
    plan resumes."""
    base = load_dataset_type("eval")["augmentations"]
    allowed = resolve({"adversarial": {"layers": {"allow": ["rewrite"]}}},
                      "adversarial", base=base)
    assert rounds_of(allowed) == rounds_of(resolve(None, "adversarial", base=base))


def test_the_escalation_carries_the_kinds_rewrite_rate_into_its_budget_rounds():
    kind = resolve({"adversarial": {"layers": {"allow": ["rewrite"]},
                                    "layer_rates": {"rewrite": 0.5},
                                    "rounds": "escalation"}}, "adversarial")
    schedule = default_escalation_schedule()
    rounds = rounds_of(kind)
    assert [r.get("sampling_probs") for r in rounds] == [
        None, None, *({"rewrite": 0.5, **r["sampling_probs"]} for r in schedule[2:])]
    # A kind that does not draw the layer keeps the schedule's rates.
    plain = resolve({"adversarial": {"rounds": "escalation"}}, "adversarial")
    assert [r.get("sampling_probs") for r in rounds_of(plain)] == [
        r.get("sampling_probs") for r in schedule]


_RATE_0 = "draw it at rate 0"


@pytest.mark.parametrize(("given", "flat"), [
    ({}, "[0]"),
    ({"layer_rates": {"rewrite": 0.5}}, None),
    ({"layer_rates": {"rewrite": 0.5}, "rounds": "escalation"}, None),
    ({"rounds": "escalation"}, "[2, 3]"),
    ({"layer_rates": {"rewrite": 0.5},
      "rounds": [{"max_complexity": 6}, {"layer_rates": {"hacking": 0.9}}]}, "[1]"),
], ids=["no rate", "rate", "escalation with a rate", "escalation without",
        "a round's own rates"])
def test_allowing_the_layer_warns_for_each_budget_round_at_rate_0(caplog, given, flat):
    with caplog.at_level("WARNING", logger="redact.jailbreak.augmentation"):
        resolve({"adversarial": {"layers": {"allow": ["rewrite"]}, **given}},
                "adversarial")
    warned = [r.getMessage() for r in caplog.records if _RATE_0 in r.getMessage()]
    if flat is None:
        assert warned == []
    else:
        (message,) = warned
        assert f"its budget round(s) {flat} (from 0) draw it at rate 0" in message


def test_a_rewrite_rate_with_the_layer_off_warns(caplog):
    with caplog.at_level("WARNING", logger="redact.jailbreak.augmentation"):
        resolve({"adversarial": {"layer_rates": {"rewrite": 0.5}}}, "adversarial")
        resolve({"adversarial": {"layer_rates": {"rewrite": 0}}}, "adversarial")
    (record,) = caplog.records
    assert record.getMessage() == (
        "augmentations.adversarial gives the rewrite layer a rate in layer_rates, "
        'but the layer is off, so the kind draws no rewrite: write "layers": '
        '{"allow": ["rewrite"]} to turn it on, or take the rate out.')


@pytest.mark.parametrize("deny", [[], ["rewrite"]], ids=["empty", "names rewrite"])
def test_a_config_dir_copy_made_before_denies_the_layer(tmp_path, monkeypatch, deny):
    """The former releases wrote ``"deny": []`` (before the layer) and
    ``"deny": ["rewrite"]`` (before the spec listed it off by default): both
    deny the layer, a registered rewrite technique included, and their
    rounds say nothing of it."""
    rewrite = _stand_in()
    _register(monkeypatch, rewrite)
    config_dir = _config_dir(tmp_path, {"deny": deny})
    for kind in KINDS:
        assert resolve(None, kind, config_dir=config_dir)["layers"] == {
            "deny": ["rewrite"]}
    kind = resolve(None, "adversarial", config_dir=config_dir)
    assert rewrite not in technique_pool(kind)
    assert rounds_of(kind) == [_ROUND_BEFORE]


def test_the_flat_keywords_keep_the_rewrite_layer_denied():
    with pytest.warns(DeprecationWarning):
        section = augmentations_of({"augmentations": {"include_hacking": False}})
    assert section["adversarial"]["layers"] == {"deny": ["rewrite", "hacking"]}


@pytest.mark.parametrize(("layers", "round_before"), [
    (None, _ROUND_BEFORE),
    ({"deny": ["rewrite"]}, _ROUND_BEFORE),
    ({"deny": ["hacking"]}, {**_ROUND_BEFORE, "include_hacking": False}),
    ({"deny": []}, _ROUND_BEFORE),
    ({"allow": ["rewrite"]}, _ROUND_BEFORE),
], ids=["packaged deny", "former packaged deny", "hacking denied alone", "no deny",
        "layer allowed"])
def test_a_plan_kept_from_before_the_layer_resumes(
        tmp_path, monkeypatch, layers, round_before):
    """Whatever the section's ``layers`` say, the former packaged
    ``{"deny": ["rewrite"]}`` among them, the plan file records the rounds a
    plan made before the layer recorded, and resumes: the layer is denied, so
    the rounds say nothing of it though the layer now holds techniques.
    Where the section allows the layer, its pure ``encode`` pool holds no
    rewrite technique, so the rounds say nothing of it either."""
    out = tmp_path / "jb.csv"
    adversarial = {"pure_only": True, "translation": False,
                   "families": {"allow": ["encode"]}}
    if layers is not None:
        adversarial["layers"] = layers
    keywords = dict(
        inputs=_inputs(), output_path=out, seed=1, auto_generate_benign=False,
        verbose=False, augmentation={"natural": None, "adversarial": adversarial})
    with monkeypatch.context() as patch:
        patch.setattr(JPipe, "drive_units", lambda units, **kwargs: None)
        generate_jailbreaks(**keywords)
    manifest = Manifest.sidecar(out)
    plan = manifest.load()
    assert read_plan(manifest.path)["rounds"] == [round_before]
    rows = generate_jailbreaks(**keywords)
    assert manifest.load() == plan and len(rows) == len(plan)


# --- the models a run plans -----------------------------------------------------

def test_a_rewrite_technique_does_not_plan_the_translation_model(monkeypatch):
    """A rewrite has the generation model write it, as any model-written
    technique: only a translation technique plans the translation model."""
    rewrite = _stand_in()
    rewrite.requires_llm = True
    monkeypatch.setattr(JPool, "all_techniques", lambda: [*POOL, rewrite])
    section = {"adversarial": {"layers": {"allow": ["rewrite"]}, "translation": False},
               "natural": None}
    pool = technique_pool(augmentations_of({"augmentations": section})["adversarial"])
    assert rewrite in pool
    assert not uses_translation(pool)
    assert resolve_role({}, "translation", section) == []


@pytest.mark.parametrize("dataset_type", ["eval", "training"])
def test_the_dataset_types_plan_the_models_they_planned(dataset_type):
    """Both types' attacks translate, so the translation model is planned; the
    layer adds no role."""
    section = load_recipe({"dataset_type": dataset_type})["augmentations"]
    assert resolve_role({}, "translation", section) == [
        default_model_for_role("translation")]
    assert resolve_role({}, "check", section) == resolve_role({}, "gen", section)
