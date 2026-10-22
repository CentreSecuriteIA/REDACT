"""Balanced dealing of a plan, and the coverage table beside it.

The planner deals each layer at its target rate over a ``(category,
entry_type, iteration)`` group, the families of a layer evenly, and the
techniques of a family evenly, in ``sample_id`` order. ``plan.coverage``
records the result beside the manifest.
"""

import csv
from collections import Counter

import pandas as pd
import pytest

import redact.jailbreak  # noqa: F401  (tags the techniques)
from redact.jailbreak.hacking import get_all_hacking_functions
from redact.jailbreak.plan import coverage, plan_run
from redact.jailbreak.technique import applies
from tests.jailbreak import simulate

_GROUP = 20
# Of 200 units, the fewest a request family may get (about 15 each is even).
_LEAST_PER_FAMILY = 10
# Units a simulated plan deals, in three groups.
_SIMULATED = 3000
# The most a technique's share of its family may be, over an even share, by
# budget and in a count round.
_TOP_OVER_EVEN = 1.5
_COUNT_TOP_OVER_EVEN = 1.2


def _one_group(n: int) -> pd.DataFrame:
    """``n`` inputs of one category and entry type, so they form one group."""
    return pd.DataFrame({
        "prompt": [f"prompt number {i} about a topic" for i in range(n)],
        "category": ["Cyber"] * n,
        "entry_type": ["harmful"] * n,
    })


def _read_coverage(path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_a_group_of_twenty_gets_its_hacking_target_dealt_over_families(tmp_path):
    hacking = get_all_hacking_functions()
    manifest = tmp_path / "jb.manifest.jsonl"
    plan_run(_one_group(_GROUP), hacking, manifest_path=manifest, seed=3,
             layer_rates={"hacking": 0.5}, verbose=False)

    cov = {(r["level"], r["name"]): int(r["count"]) for r in _read_coverage(
        tmp_path / "jb.coverage.csv")}

    # Rate 0.5 over 20 units -> exactly 10 hacking picks.
    assert cov[("layer", "hacking")] == _GROUP // 2

    # Those 10 are dealt evenly over the three hacking families (framing,
    # cognitive, persona): each count within one of the others.
    families = sorted(getattr(f, "families", [None])[0] for f in hacking)
    family_counts = [cov.get(("family", fam), 0) for fam in set(families)]
    assert sum(family_counts) == _GROUP // 2
    assert max(family_counts) - min(family_counts) <= 1


def test_a_lower_rate_deals_fewer_picks(tmp_path):
    hacking = get_all_hacking_functions()
    manifest = tmp_path / "jb.manifest.jsonl"
    plan_run(_one_group(_GROUP), hacking, manifest_path=manifest, seed=3,
             layer_rates={"hacking": 0.25}, verbose=False)
    cov = {(r["level"], r["name"]): int(r["count"]) for r in _read_coverage(
        tmp_path / "jb.coverage.csv")}
    assert cov[("layer", "hacking")] == round(0.25 * _GROUP)


def test_groups_of_one_are_dealt_at_the_rate_not_every_time(tmp_path):
    # 200 categories of one input each: each group holds a single unit, so
    # the rate must hold across groups, not round up to a pick in every one.
    hacking = get_all_hacking_functions()
    n = 200
    inputs = pd.DataFrame({
        "prompt": [f"prompt number {i} about a topic" for i in range(n)],
        "category": [f"cat{i}" for i in range(n)],
        "entry_type": ["harmful"] * n,
    })
    plan_run(inputs, hacking, manifest_path=tmp_path / "jb.manifest.jsonl",
             seed=3, layer_rates={"hacking": 0.5}, verbose=False)
    picks = sum(int(r["count"]) for r in _read_coverage(tmp_path / "jb.coverage.csv")
                if (r["level"], r["name"]) == ("layer", "hacking"))
    assert 0.35 * n <= picks <= 0.65 * n


def test_coverage_lists_every_pool_technique_including_zeros(tmp_path):
    hacking = get_all_hacking_functions()
    manifest = tmp_path / "jb.manifest.jsonl"
    plan_run(_one_group(4), hacking, manifest_path=manifest, seed=1,
             layer_rates={"hacking": 0.5}, verbose=False)
    rows = _read_coverage(tmp_path / "jb.coverage.csv")
    technique_names = {r["name"] for r in rows if r["level"] == "technique"}
    assert {f.__name__ for f in hacking} <= technique_names


def test_coverage_returns_none_without_a_manifest(tmp_path):
    assert coverage(tmp_path / "missing.manifest.jsonl", []) is None


def test_a_plans_model_calls_add_up_from_the_spec(tmp_path, caplog):
    """A synthetic plan: a pure unit, typos ([2, 4]), a translation
    ([2, 12]), both, and a persona ([3, 9])."""
    import logging

    from redact.dataset import Manifest
    from redact.jailbreak.plan import log_model_calls
    from redact.jailbreak.pool import all_techniques, registry

    names = ["to_base64", "to_rewrite_with_typos_low", "to_swahili",
             "to_artist_persona"]
    pool = [registry(all_techniques())[name] for name in names]
    manifest = tmp_path / "jb.manifest.jsonl"
    Manifest(manifest).write(
        {"sample_id": str(i), "iteration": 0, "combination": combination}
        for i, combination in enumerate([
            ["to_base64"], ["to_rewrite_with_typos_low", "to_base64"],
            ["to_swahili"], ["to_swahili", "to_rewrite_with_typos_low"],
            ["to_artist_persona"]]))
    with caplog.at_level(logging.INFO, logger="redact.jailbreak.plan"):
        log_model_calls(manifest, pool)
    assert [r.getMessage() for r in caplog.records] == [
        "Model calls: 4 of 5 planned units call a model, 11 to 41 calls "
        "(translation 4 to 24, from the spec's calls)."]


def test_every_technique_that_calls_a_model_has_calls_in_the_spec():
    from redact.jailbreak.pool import all_techniques

    assert [fn.__name__ for fn in all_techniques()
            if fn.requires_llm and fn.calls is None] == []


def test_a_verbose_plan_logs_its_model_calls(tmp_path, caplog):
    import logging

    from redact.jailbreak.obfuscation.typos import to_rewrite_with_typos_low

    with caplog.at_level(logging.INFO, logger="redact.jailbreak.plan"):
        plan_run(_one_group(4), [to_rewrite_with_typos_low],
                 manifest_path=tmp_path / "jb.manifest.jsonl", seed=3,
                 settings_per_iteration=[{"exact_techniques": 1}])
    assert ("Model calls: 4 of 4 planned units call a model, 8 to 16 calls "
            "(translation 0 to 0") in caplog.text


def test_a_layer_left_two_or_more_short_of_its_target_is_logged(tmp_path, caplog):
    import logging

    # No request technique in the pool: no unit can take the requests layer.
    with caplog.at_level(logging.WARNING, logger="redact.jailbreak.plan"):
        plan_run(_one_group(_GROUP), get_all_hacking_functions(),
                 manifest_path=tmp_path / "jb.manifest.jsonl", seed=3,
                 layer_rates={"hacking": 0.5, "manipulation": 0.0, "requests": 0.5},
                 verbose=False)
    assert [r.getMessage() for r in caplog.records] == [
        "Plan: group (Cyber, harmful), iteration 0: the requests layer has 0 of "
        "its target 10; the pool or the compatibility rules left the rest no "
        "pick (see jb.coverage.csv)."]


def test_a_request_obfuscation_left_short_is_logged_as_a_step(tmp_path, caplog):
    import logging

    from redact.jailbreak.requests import get_all_request_functions

    # Every unit takes a request, and no encode or translation technique is
    # in the pool: no unit can take a request obfuscation.
    with caplog.at_level(logging.WARNING, logger="redact.jailbreak.plan"):
        plan_run(_one_group(_GROUP),
                 [*get_all_hacking_functions(), *get_all_request_functions()],
                 manifest_path=tmp_path / "jb.manifest.jsonl", seed=3,
                 sample_kwargs={"allow_request_obfuscation": True},
                 layer_rates={"hacking": 0.5, "manipulation": 0.0, "requests": 1.0,
                              "request_obfuscation": 1.0},
                 verbose=False)
    assert [r.getMessage() for r in caplog.records] == [
        "Plan: group (Cyber, harmful), iteration 0: the request obfuscation step "
        f"has 0 of its target {_GROUP}; the pool or the compatibility rules left "
        "the rest no pick (see jb.coverage.csv)."]


def _plan_kind(tmp_path, section: dict, n: int):
    """Plan ``n`` inputs of one group under the adversarial ``section``; the
    coverage rows."""
    from redact.jailbreak.augmentation import resolve, rounds_of
    from redact.jailbreak.pool import technique_pool

    kind = resolve({"adversarial": section}, "adversarial")
    rounds = rounds_of(kind)
    manifest = tmp_path / "jb.manifest.jsonl"
    plan_run(_one_group(n), technique_pool(kind), manifest_path=manifest, seed=42,
             iterations=len(rounds), settings_per_iteration=rounds,
             layer_rates=kind["layer_rates"] or None, verbose=False)
    return _read_coverage(tmp_path / "jb.coverage.csv")


def test_the_escalation_asks_rates_the_rules_can_meet(tmp_path, caplog):
    """Hacking and manipulation exclude each other, so their rates in a round
    sum to at most 1, and no group falls short of either."""
    import logging

    with caplog.at_level(logging.WARNING, logger="redact.jailbreak.plan"):
        _plan_kind(tmp_path, {"rounds": "escalation"}, 40)
    assert [r.getMessage() for r in caplog.records] == []


def test_the_default_budget_deals_every_request_family(tmp_path):
    """The requests layer is picked before the obfuscations, so the budget
    they spend does not leave only the cheapest request families. Every
    request family the attacks may draw: not the natural-only ones."""
    from redact.jailbreak.technique import family_kinds, load_spec

    families = load_spec()["families"]
    requests = [name for name, family in families.items()
                if family["layer"] == "requests"
                and "adversarial" in family_kinds(name)]
    rows = _plan_kind(tmp_path, {}, 200)
    drawn = {r["name"]: int(r["count"]) for r in rows if r["level"] == "family"}
    short = {name: drawn.get(name, 0) for name in requests
             if drawn.get(name, 0) < _LEAST_PER_FAMILY}
    assert short == {}


# --- the techniques of a family --------------------------------------------------
# Within the family the dealer picks, the unit takes the technique it may take
# that the group has dealt least. The complexity budget still favours a
# family's cheaper techniques where they cost differently (encode,
# sensitive_words): one of complexity 3 fits only a unit whose earlier picks
# leave 3 points, and fsh and dap refuse the heavy encodings, which is
# wanted; each is still drawn.


@pytest.fixture(scope="module")
def default_shares() -> dict:
    return simulate.shares(*simulate.deal_kind({}, _SIMULATED, groups=3))


@pytest.mark.parametrize("family", ["persona", "answer_language"])
def test_the_techniques_of_a_family_the_budget_never_limits_come_out_even(
        default_shares, family):
    assert simulate.top_over_even(default_shares, family) <= _TOP_OVER_EVEN


def test_a_count_round_deals_the_techniques_of_every_family_evenly():
    """No budget limits a count round, so every family comes out even: within
    one of the others per group (drawn alone, answer_language's top member
    had about 1.4 times its share)."""
    table = simulate.shares(*simulate.deal_kind(
        {}, _SIMULATED, kind="natural", groups=3))
    families = {row["family"] for row in table.values()}
    uneven = {family: round(simulate.top_over_even(table, family), 2)
              for family in families
              if simulate.top_over_even(table, family) > _COUNT_TOP_OVER_EVEN}
    assert uneven == {}


@pytest.mark.parametrize("section", [{}, {"budget": "group"}], ids=["unit", "group"])
def test_the_default_attacks_draw_every_technique_of_their_pool(section):
    """No step's picks starve another step's families: the request
    obfuscation once counted its picks as the obfuscation layer's, and 14
    encode and sensitive_words techniques were never drawn. A deprecated
    technique, or one no simulated input may take, is not asked for. The
    nine heavy encodings are the rarest draws (one of complexity 3 fits only
    a unit whose earlier picks leave 3 points, and the other families of
    complexity 3, the word art's among them, take those units too), about 1
    unit in 500 each, and a seed can leave one out of its plan altogether:
    to_braille, which ascii_art refuses, is drawn 0 times at seed 42 and 8
    at seed 43 in 4,000 units. So the plans of two seeds are pooled."""
    units, seeds = 4000, (42, 43)
    plans = [simulate.deal_kind(section, units, groups=3, seed=seed) for seed in seeds]
    pool = plans[0][1]
    table = simulate.shares([names for drawn, _ in plans for names in drawn], pool)
    texts = simulate.inputs(units, groups=3)["prompt"]
    asked = [fn.__name__ for fn in pool if not getattr(fn, "deprecated", None)
             and any(applies(fn, text, "harmful") for text in texts)]
    assert [name for name in asked if table[name]["count"] == 0] == []


def test_a_kept_plan_replays_each_row_as_it_was_recorded(tmp_path):
    """Dealing changes what a fresh plan draws; a plan already written is
    replayed by name, in its recorded order, whatever the dealer does now."""
    from redact.dataset import Manifest
    from redact.jailbreak.augmentation import resolve, rounds_of
    from redact.jailbreak.plan import combination_of
    from redact.jailbreak.pool import all_techniques, registry, technique_pool

    kind = resolve({"adversarial": {"rounds": "escalation"}}, "adversarial")
    rounds = rounds_of(kind)
    manifest = tmp_path / "jb.manifest.jsonl"
    plan_run(_one_group(40), technique_pool(kind), manifest_path=manifest, seed=42,
             iterations=len(rounds), settings_per_iteration=rounds, verbose=False)
    # A row as an earlier planner wrote it: the request obfuscation last.
    recorded = [row["combination"] for row in Manifest(manifest).load()]
    recorded.append(["to_persona_roleplay", "to_answer_as_tutorial", "to_rot13"])
    every = registry(all_techniques())
    replayed = [[fn.__name__ for fn in combination_of(names, every).techniques]
                for names in recorded]
    assert replayed == recorded


# --- the request obfuscation -----------------------------------------------------
# With allow_request_obfuscation on, as in the packaged attacks, the request
# obfuscation is dealt at its rate like a layer, to units that drew a request,
# and holds one obfuscation slot. Off (the eval type, or a round that says
# false), the draws are those the planner made before it was counted.

_BUDGETS = [
    {},
    {"max_complexity": 9, "max_obfuscations": 3, "allow_request_obfuscation": False,
     "sampling_probs": {"hacking": 0.9, "manipulation": 0.1, "requests": 0.9,
                        "request_obfuscation": 0.7}},
]
# The obfuscation renumbering (whole-text rewrites before word-level steps)
# moved this digest: the same techniques are drawn, some in another order.
# The script rule moved it again: six draws paired a Latin-only encoding
# with a translation into another script, and a few more drew from a list
# the rule narrowed. The language table moved it once more: the answer
# languages take its order, Swahili before Thai, so the draws that named one
# of the two name the other, and nothing else changed. The fifteen spec 1.2
# encodings moved it again: a wider encode family changes which member an
# encode draw lands on for some seeds. The spec 1.2 noise family moved it once
# more: one more obfuscation family to draw changes which family some draws
# land on. The applicability of to_stripped_accents and to_abbreviations
# moved it again: the simulated inputs hold no accented letter, so
# to_stripped_accents, which 7 of the 240 dealt units drew, is never drawn
# for them, and the casual writing picks are dealt over the other members;
# since a unit's rng and its group's counts follow the picks before, 65 units
# draw differently, 40 of them outside casual writing. The draws without a
# dealer filter nothing and do not move. The spec 1.2 neutral_container family
# moved it again: one more obfuscation family to draw changes which family
# some draws land on. The six spec 1.2 answer languages moved it once more: a
# wider answer_language family changes the member some draws land on, and
# through the unit's rng the rest of its draw (6 of the 240 dealt units, 4 of
# the 200 undealt draws). The three spec 1.2 natural-only request families
# (small talk before and after, benign roles) moved it again: all_techniques()
# holds them, whatever their kinds, and three more request families to deal
# change most draws (224 of the 240 dealt units, 49 of the 200 undealt ones).
# The simulated inputs are harmful, so no dealt unit draws a benign role; the
# undealt draws, which read no applicability, do. The four spec 1.2 many-shot
# fsh and dap variants moved it once more: a wider fsh and dap change the
# member a manipulation draw lands on, and through the unit's rng and its
# group's counts the rest of its draw (128 of the 240 dealt units; 6 of the
# 200 undealt draws, each now drawing a many-shot variant). Scoring
# to_html_entities as heavy, complexity 3 (it was light, 2) moved it last: the
# budget it takes and the members it leaves out of fsh and dap units change the
# picks after it (25 of the 240 dealt units, 3 of the 200 undealt draws).
# Scoring the two selected_many variants 3 (they were 4: the scale stays 0 to
# 3) moved it once more: they take a point less of a unit's budget (53 of the
# 240 dealt units, 3 of the 200 undealt draws). The spec 1.3 persona_template
# family moved it again: one more hacking family to deal, eleven members wide,
# changes the hacking pick of many units and, through the unit's rng and its
# group's counts, the rest of their draw (223 of the 240 dealt units, 125 of
# the 200 undealt draws). The three spec 1.3 answer formats moved it once
# more: a wider answer_format family changes the member some request draws
# land on (5 of the 240 dealt units, 11 of the 200 undealt draws). The spec
# 1.3 fake_turns family moved it again: one more manipulation family to deal,
# needing no benign data, changes the manipulation pick of many dealt units
# and, through their rng and their group's counts, the rest of their draw
# (116 of the 240 dealt units, 9 of the 200 undealt draws). The seven spec 1.3
# members of the thin request families (impersonation, temporal, the two
# distractors, asking) moved it once more: wider families change the member
# some request draws land on and, through the unit's rng and its group's
# counts, the rest of their draw (99 of the 240 dealt units, 26 of the 200
# undealt draws). The two spec 1.3 payload splits moved it again: one more
# obfuscation family to deal, which refuses encode, changes the family some
# obfuscation draws land on and, through the unit's rng and its group's
# counts, the rest of their draw (221 of the 240 dealt units, 73 of the 200
# undealt draws). The spec 1.3 word art moved it once more: one more
# obfuscation family to deal, complexity 3 and refusing the other word-level
# steps, changes the family some obfuscation draws land on and, through the
# unit's rng and its group's counts, the rest of their draw (204 of the 240
# dealt units, 27 of the 200 undealt draws). The three spec 1.3 persona
# archetypes moved it once more: a wider persona family changes the member
# some hacking draws land on and, through the unit's rng and its group's
# counts, the rest of their draw (149 of the 240 dealt units, 42 of the 200
# undealt draws). The four spec 1.3 translations (Hmong, Chinese, Hindi,
# Amharic) moved it once more: a wider translation family changes the member
# some translation draws land on and, through the unit's rng and its group's
# counts, the rest of their draw (38 of the 240 dealt units, 7 of the 200
# undealt draws). The rule keeping a translation from the families that have a
# model mark words (sensitive_words, tokenbreak, word_art) moved it again: a
# draw that held both now takes another technique, and through the unit's rng
# and its group's counts the rest of its draw (185 of the 240 dealt units, 68 of
# the 200 undealt draws). The spec 1.3 wording family (a code name with a key,
# a word game) moved it once more: one more obfuscation family to deal, which
# refuses the other word-level steps, a translation and ascii_art, changes the
# family some obfuscation draws land on and, through the unit's rng and its
# group's counts, the rest of their draw (221 of the 240 dealt units, 123 of the
# 200 undealt draws). The rule keeping a translation from the word-level and
# character-level changes (typos, casual_writing, noise, payload_split) moved
# it again: a draw that held both now takes another technique, and through the
# unit's rng and its group's counts the rest of its draw (179 of the 240 dealt
# units, 75 of the 200 undealt draws). The dealer's jitter per unit (in place
# of one offset per layer and group, which locked the layers of one rate in or
# out of phase for a whole group) moved it last: each layer falls on other
# units of its group, at the same rate, and through the units' rng and counts
# the rest of their draws follow (235 of the 240 dealt units; the 200 undealt
# draws, which ask no dealer, do not move).
_PINNED_DRAWS = "b7a07d7e3e1eb68c10737e1b57a49ef2d8e5b672247d18a914b66cb69f5c4a1c"


def test_draws_without_the_request_obfuscation_are_those_made_before_it_was_counted():
    import hashlib
    import json
    import random

    from redact.jailbreak.pool import all_techniques
    from redact.jailbreak.sampling import sample_combination

    pool = all_techniques()
    dealt = [simulate.deal(pool, 60, groups=2, seed=seed, rounds=_BUDGETS)
             for seed in (1, 2)]
    alone = [[t.__name__ for t in sample_combination(
                 random.Random(seed), pool, **budget).techniques]
             for seed in range(100) for budget in _BUDGETS]
    digest = hashlib.sha256(json.dumps([dealt, alone]).encode()).hexdigest()
    assert digest == _PINNED_DRAWS


def test_the_plan_deals_the_request_obfuscation_at_its_rate():
    """Half of a group of 40 takes one, exactly: dealt, not drawn per unit."""
    from redact.dataset.dealing import Dealer
    from redact.jailbreak.plan import assign_combination
    from redact.jailbreak.pool import all_techniques

    pool = all_techniques()
    dealer = Dealer(key=(1, "Cyber", "harmful", 0))
    for position in range(1, 2 * _GROUP + 1):
        dealer.position = position
        assign_combination(
            f"unit{position}", pool, seed=1, dealer=dealer, max_complexity=9,
            max_obfuscations=2, allow_request_obfuscation=True,
            sampling_probs={"request_obfuscation": 0.5})
    assert dealer.included["request_obfuscation"] == _GROUP
    assert dealer.targets["request_obfuscation"] == _GROUP


def _request_obfuscations(monkeypatch, section: dict, n: int, **kwargs) -> list:
    """Each unit's request obfuscation (``None`` without one) in a one-round
    plan of the adversarial ``section``, read off each draw: the manifest
    names a combination's techniques, not which of them is the request
    obfuscation. One round, so no unit is drawn again."""
    import redact.jailbreak.plan as JP

    sample = JP.sample_combination
    drawn = []

    def recording(*args, **sample_kwargs):
        combined = sample(*args, **sample_kwargs)
        drawn.append(getattr(combined, "request_obfuscation", None))
        return combined

    monkeypatch.setattr(JP, "sample_combination", recording)
    simulate.deal_kind(section, n, **kwargs)
    return drawn


# Units of the packaged attacks a test of the request obfuscation plans, in
# three groups; the spec's request_obfuscation rate is 0.5.
_ATTACKS = 600


def test_the_packaged_attacks_deal_a_request_obfuscation_to_half_their_units(
        monkeypatch):
    """Dealt among the units that hold a request (0.7 of them), half of all
    the units take one; a group whose last units hold no request may fall
    one short of its half."""
    drawn = _request_obfuscations(monkeypatch, {}, _ATTACKS, groups=3)
    assert _ATTACKS // 2 - 3 <= sum(fn is not None for fn in drawn) <= _ATTACKS // 2


@pytest.mark.parametrize("dealt", [False, True], ids=["drawn", "dealt"])
def test_a_unit_without_a_request_takes_no_request_obfuscation(dealt):
    """It obfuscates the finished request, so a unit that drew no request is
    not asked, whatever the rate: it never lands after the outermost
    wrapper. Half the units take a request here, and each of those one."""
    import random

    import redact.jailbreak.plan as JP
    from redact.dataset.dealing import Dealer
    from redact.jailbreak.pool import all_techniques
    from redact.jailbreak.sampling import sample_combination

    dealer = Dealer(key=(1, "Cyber", "harmful", 0)) if dealt else None
    held = []
    for position in range(1, 2 * _GROUP + 1):
        if dealer is not None:
            dealer.position = position
        combined = sample_combination(
            random.Random(position), all_techniques(), dealer=dealer,
            allow_request_obfuscation=True,
            sampling_probs={"requests": 0.5, "request_obfuscation": 1.0})
        if dealer is not None:
            JP._observe(dealer, combined)
        held.append((any(fn.layer == "requests" for fn in combined.techniques),
                     combined.request_obfuscation is not None))
    assert [request for request, obfuscated in held if obfuscated] == [True] * sum(
        obfuscated for _, obfuscated in held)
    assert sum(obfuscated for _, obfuscated in held) >= _GROUP // 2


def test_the_request_obfuscation_is_a_light_encoding_or_a_translation(monkeypatch):
    """Light ``encode`` techniques, the light ``to_sensitive_words_encode_*``
    (first family ``sensitive_words``) and the translations: all three are
    drawn by the packaged attacks."""
    drawn = [fn for fn in _request_obfuscations(monkeypatch, {}, _ATTACKS, groups=3)
             if fn is not None]
    assert {fn.families[0] for fn in drawn} == {
        "encode", "sensitive_words", "translation"}
    assert all(getattr(fn, "encode_weight", None) == "light"
               or "translation" in fn.families for fn in drawn)


def test_the_request_obfuscation_takes_the_family_weights(monkeypatch):
    """Its families are counted apart from the obfuscation layer's and keep
    their weights: translation at 3 takes three picks to each of the other
    two families' one. A budget of 12 leaves every family room, and the
    manipulation layer, whose families refuse a translation, is denied."""
    section = {"max_complexity": 12, "family_weights": {"translation": 3},
               "layers": {"deny": ["manipulation"]}}
    drawn = _request_obfuscations(monkeypatch, section, _ATTACKS, groups=3)
    families = Counter(fn.families[0] for fn in drawn if fn is not None)
    assert families == {"translation": 180, "encode": 60, "sensitive_words": 60}


# The eval type's plan of 300 inputs in three groups at its seed, 43, pinned
# before the packaged attacks turned the request obfuscation on. The eval
# pool and its draws are held (fixtures/eval_pool.json); they move only with
# the owner's eval review.
_PINNED_EVAL_DRAWS = "95083fbb5221e4b1bd39125bfa297bc06bfc51ddbdadde2ad1eb45a6d59b7afb"


def _eval_draws(section: dict) -> list[list[str]]:
    from redact.recipe import load_dataset_type

    base = load_dataset_type("eval")["augmentations"]
    dealt, _ = simulate.deal_kind(section, 300, groups=3, seed=43, base=base)
    return dealt


def test_the_eval_types_draws_are_held():
    import hashlib
    import json

    digest = hashlib.sha256(json.dumps(_eval_draws({})).encode()).hexdigest()
    assert digest == _PINNED_EVAL_DRAWS


def test_a_count_round_ignores_the_request_obfuscation():
    """The eval type's rounds draw an exact count, which never reads the
    switch: on, they draw what they draw off."""
    assert _eval_draws({"allow_request_obfuscation": True}) == _eval_draws({})
