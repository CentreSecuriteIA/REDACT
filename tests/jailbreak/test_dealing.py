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
# budget and in a count round. The count-round bound rose from 1.2 to 1.25
# when the spec 1.4 language-mix families joined the natural pool: with more
# families each gets fewer of the count round's picks, so a family's members
# complete fewer full even cycles and its top member sits a little higher
# (casual_writing's reaches 1.215 at seeds 42 and 44). The spec 1.4 retune of
# heavy_encoding_weight cannot bring it back to 1.2: a count round deals the
# heavy encodings as encode's members and never reads the weight, and the
# natural kind draws none of them (casual_writing's top member is 1.215 at
# 0.2 and at 0.25 alike).
_TOP_OVER_EVEN = 1.5
_COUNT_TOP_OVER_EVEN = 1.25


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

    # Those 10 are dealt evenly over the four hacking families (framing,
    # cognitive, persona, persona_template): each count within one of the
    # others.
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


@pytest.mark.parametrize(("size", "logged"), [(4, True), (2, False)],
                         ids=["two short", "one short"])
def test_a_layer_is_logged_from_two_picks_short_of_its_target(
        tmp_path, caplog, size, logged):
    """At rate 0.5 a group of ``size`` is due exactly ``size / 2`` request
    picks, and no unit can take one: two short is logged, one is not."""
    import logging

    with caplog.at_level(logging.WARNING, logger="redact.jailbreak.plan"):
        plan_run(_one_group(size), get_all_hacking_functions(),
                 manifest_path=tmp_path / "jb.manifest.jsonl", seed=3,
                 layer_rates={"hacking": 0.5, "manipulation": 0.0, "requests": 0.5},
                 verbose=False)
    assert bool(caplog.records) is logged


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
    technique, or one no simulated input may take, is not asked for. When
    the layers of one rate were locked in or out of phase (commit 590babe),
    some techniques were drawn 2 times at seed 42 and 14 at 43; one plan now
    holds them all. The first eval release's rules left the ten heavy full-text
    encodings 5 draws in all under the unit budget and 10 under the group
    budget, five and two of them never, until they got a dealing slot of
    their own (``sampling.HEAVY_ENCODINGS``)."""
    units = 4000
    drawn, pool = simulate.deal_kind(section, units, groups=3, seed=42)
    table = simulate.shares(drawn, pool)
    texts = simulate.inputs(units, groups=3)["prompt"]
    asked = [fn.__name__ for fn in pool if not getattr(fn, "deprecated", None)
             and any(applies(fn, text, "harmful") for text in texts)]
    assert [name for name in asked if table[name]["count"] == 0] == []


# The ten heavy full-text encodings, dealt in a slot of their own beside encode
# in a budget draw (sampling.HEAVY_ENCODINGS, at the spec's
# heavy_encoding_weight, 0.25).
_HEAVY = ("to_base64", "to_unicode_escape", "to_ascii_ordinal", "to_morse",
          "to_braille", "to_hex", "to_binary", "to_html_entities", "to_base32",
          "to_base85")


def test_the_default_attacks_draw_each_heavy_encoding_a_few_times(default_shares):
    """3,000 units in three groups at seed 42: 39 draws of the ten, 3 to 5
    each. The slot fixes the total, not a count for each encoding; the total
    is 37 or 38 in one group, 38 or 39 in three and 40 in ten (seeds 42 to
    46); the least drawn of the ten is drawn 3 times in one and three groups,
    1 or 2 times in ten. The four spec 1.4 language-mix families (dialect,
    code-switching, transliteration, romanization) joined the obfuscation
    layer, so the layer's family-first picks are shared among more families
    and at the weight of 0.2 the slot's count fell from 45 to 30 (the bounds
    were 28 to 42 then); spec 1.4 retuned the weight to 0.25 for about 40,
    and the bounds are those of spec 1.3 again. Dealt as encode's members the
    ten got 12, two of them never; dealt as a family of their own (weight 1),
    ~200."""
    counts = [default_shares[name]["count"] for name in _HEAVY]
    assert 36 <= sum(counts) <= 60  # noqa: PLR2004
    assert min(counts) >= 3  # noqa: PLR2004


@pytest.mark.parametrize("seed", [1, 2, 3, 7, 42, 43, 44, 45, 46, 123])
def test_no_group_of_the_default_attacks_goes_without_a_heavy_encoding_or_ascii_art(
        seed):
    """Every group of 100 units, in one, three and ten groups, draws a heavy
    full-text encoding (one of the ten, not each of them) and ASCII art: at
    least 2 and 4 times over seeds 1 to 3, 7, 42 to 46 and 123. Dealt as
    encode's members, the heavy encodings were missing from 1 to 9 groups in
    29 of those 30 plans. ASCII art refuses many families (the hacking and
    manipulation techniques, a translation, the wrappers, wording, word art,
    the payload split, the heavy encodings), so it is drawn on the units left
    with none of them."""
    per_group = 100
    for groups in (1, 3, 10):
        rows, _ = simulate.plan_kind(
            {}, simulate.inputs(per_group * groups, groups), seed=seed)
        by_group: dict[str, Counter] = {}
        for row in rows:
            by_group.setdefault(row["category"], Counter()).update(row["combination"])
        assert len(by_group) == groups
        without = {category: [what for what, drawn in (
            ("heavy", sum(counts[name] for name in _HEAVY)),
            ("ascii_art", counts["to_ascii_art"])) if not drawn]
            for category, counts in by_group.items()}
        assert {category: what for category, what in without.items() if what} == {}


def test_no_family_of_the_default_attacks_thins_out_in_a_group():
    """Every family of the pool gets, in every group of 200 units, at least
    half its mean count per group, over four seeds in one group and in
    three. With the layers of one rate locked in or out of phase for a whole
    group, ascii_art, which refuses both a hacking technique and a
    translation, got 1 draw in a group where its mean was 7; now every
    family gets 0.9 of its mean or more."""
    per_group = 200
    counts: list[Counter] = []
    for seed in (1, 7, 42, 43):
        for groups in (1, 3):
            rows, pool = simulate.plan_kind(
                {}, simulate.inputs(per_group * groups, groups), seed=seed)
            family = {fn.__name__: fn.families[0] for fn in pool}
            by_group: dict[str, Counter] = {}
            for row in rows:
                by_group.setdefault(row["category"], Counter()).update(
                    family[name] for name in row["combination"])
            counts.extend(by_group.values())
    families = {name for group in counts for name in group}
    mean = {name: sum(group[name] for group in counts) / len(counts)
            for name in families}
    thin = {name: min(group[name] for group in counts) for name in families
            if min(group[name] for group in counts) < mean[name] / 2}
    assert thin == {}


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
    # Rows as earlier planners wrote them: the request obfuscation last, and
    # the manipulation layer before the obfuscations and the request.
    recorded = [row["combination"] for row in Manifest(manifest).load()]
    recorded.append(["to_persona_roleplay", "to_answer_as_tutorial", "to_rot13"])
    recorded.append(["to_fsh_random_short", "to_rot13", "to_use_past_tense"])
    every = registry(all_techniques())
    replayed = [[fn.__name__ for fn in combination_of(names, every).techniques]
                for names in recorded]
    assert replayed == recorded


# --- the request obfuscation -----------------------------------------------------
# With allow_request_obfuscation on, as in the packaged attacks, the request
# obfuscation is dealt at its rate like a layer, to units that drew a request,
# and holds one obfuscation slot. Off (the eval type, or a round that says
# false), the sampler never decides it, so the draws are those of a sampler
# without the step.

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
# out of phase for a whole group) moved it once more: each layer falls on other
# units of its group, at the same rate, and through the units' rng and counts
# the rest of their draws follow (235 of the 240 dealt units; the 200 undealt
# draws, which ask no dealer, do not move). The rule keeping a translation from
# every encode technique moved it in the first eval release: a draw that held both
# now takes another technique, and through the unit's rng and its group's
# counts the rest of its draw (128 of the 240 dealt units, 19 of the 200
# undealt draws). The script rule it made dead, a Latin-only technique beside a
# translation into another script, is gone. The rule keeping ASCII art from
# the nine heavy full-text encodings that did not refuse it (braille did) moved
# it once more: a draw that held both takes another technique (4 of the 240
# dealt units, 1 of the 200 undealt draws). The rule keeping sensitive_words and
# tokenbreak apart, one model marking step per text, moved it again: a draw that
# held both takes another technique, and through the unit's rng and its group's
# counts the rest of its draw (159 of the 240 dealt units, 30 of the 200
# undealt draws). The manipulation layer applying last, around the finished
# question, moved it once more, by order alone: the same techniques are drawn,
# and a draw that holds a manipulation technique beside a later step names it
# last (48 of the 240 dealt units, 19 of the 200 undealt draws). The heavy
# full-text encodings' dealing slot moved it again: a budget draw's
# obfuscations deal them under a name of their own beside encode, so the
# group's family counts and the names the dealer breaks a tie among change
# from the first obfuscation pick on (219 of the 240 dealt units; the 200
# undealt draws ask no dealer and do not move). Spec 1.4's twelve persona
# templates moved it once more: the family has 23 members instead of 11, so a
# draw that holds one takes another, and through the unit's rng and its
# group's counts the rest of its draw (65 of the 240 dealt units, 41 of the
# 200 undealt draws). The eval type's pin below does not move: no eval pool
# draws the family. The rest of spec 1.4 moved it again: the Yoruba
# translation widens the translation family and the 24 published suffixes the
# suffixes family (4 members to 28), so a draw that holds one of the two
# families may take a new member, and through the unit's rng and its group's
# counts the rest of its draw (57 of the 240 dealt units, 46 of them holding
# one of the 25 new techniques; 68 of the 200 undealt draws, 62 holding one).
# The eval type's pin does not move: no eval pool draws Yoruba or a suffix.
# The published suffixes refusing ascii_art moved it once more, by one draw:
# an undealt draw that took a published suffix first no longer has ascii_art
# among its candidates, so the rng's next picks land elsewhere (1 of the 200
# undealt draws; none of the 240 dealt units, and no draw held both before).
# Spec 1.4's to_letter_spacing (a light encode member, so encode's members
# are dealt among more) and guardrail_address (a new request-layer family of
# three attack-only techniques, so the request layer's families are dealt
# among one more) moved it again: the request each unit takes and, through
# its rng and its group's counts, the rest of its draw change (189 of the 240
# dealt units, 19 of them holding one of the four new techniques; 46 of the
# 200 undealt draws, 17 holding one). Spec 1.4's four language-mix
# obfuscation families (within_layer_order 11 to 14, complexity 2: four pure
# transliterations, ten code-switching techniques, ten romanizations and six
# dialects, on the translation model under the switch, which the default
# attacks have on) moved it again: they join the obfuscation family-first
# picks, so the group's family counts and the tie-break names change from the
# first obfuscation pick on, and the undealt draws gain the new families (233
# of the 240 dealt units, 96 of them holding one of the 30; 176 of the 200
# undealt draws, 54 holding one). Retuning heavy_encoding_weight from 0.2 to
# 0.25 did not move it (0 of the 240 dealt units, 0 of the 200 undealt
# draws): in groups of 30 units the slot takes its first pick early and its
# next only after more units than the group has, so any weight from 0.2 to
# 0.3 deals it on the same units (a weight of 1 moves 120 of the 240).
# Renaming the de-CH dialect from to_dialect_swiss_german to
# to_dialect_german_switzerland (standard German as written in Switzerland,
# not the Swiss German dialects) moved it by the name alone: the draws are
# the same, with the new name in the 5 of the 240 dealt units and the 2 of
# the 200 undealt draws that held the technique (its row keeps its place).
_PINNED_DRAWS = "ca39fc158e15fa9ab4ec9b1afa893de5e5c58693f878c64562291ef39ed379a2"


def test_the_budget_draws_are_the_pinned_ones():
    """A change detector, not a property: two dealt plans and 200 undealt
    draws of every registered technique, under the default budget and a
    larger one, the request obfuscation off. Any change in what the planner
    or the sampler draws moves the digest; re-pin it on purpose, and say in
    the commit and the comment above what moved and why."""
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


@pytest.mark.parametrize("dealt", [False, True], ids=["drawn", "dealt"])
def test_a_draw_with_the_request_obfuscation_off_never_decides_it(
        monkeypatch, dealt):
    """Off, the sampler neither asks the dealer for the step nor draws its
    candidates, at any rate: the draws are those a sampler without the step
    makes."""
    import random

    import redact.jailbreak.sampling as JS
    from redact.dataset.dealing import Dealer
    from redact.jailbreak.plan import assign_combination
    from redact.jailbreak.pool import all_techniques

    def never(pool):
        raise AssertionError("the request obfuscation's candidates were drawn")

    monkeypatch.setattr(JS, "request_obfuscation_candidates", never)
    dealer = Dealer(key=(1, "Cyber", "harmful", 0)) if dealt else None
    for position in range(1, 2 * _GROUP + 1):
        settings = {"allow_request_obfuscation": False,
                    "sampling_probs": {"requests": 1.0, "request_obfuscation": 1.0}}
        if dealer is None:
            JS.sample_combination(random.Random(position), all_techniques(), **settings)
            continue
        dealer.position = position
        assign_combination(f"unit{position}", all_techniques(), seed=1,
                           dealer=dealer, **settings)
    assert dealer is None or "request_obfuscation" not in dealer.targets


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
            JP._observe(dealer, combined, heavy_apart=True)
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


def _translates_after_a_request(names: list[str], by: dict) -> bool:
    """Whether a translation applies after the request in ``names``: it is
    then the request obfuscation (a phase-4 translation runs before the
    requests), wherever it stands."""
    layers = [by[name].layer for name in names]
    return "requests" in layers and any(
        "translation" in by[name].families
        for name in names[layers.index("requests") + 1:])


def test_a_planned_request_obfuscation_that_translates_holds_nothing_to_translate():
    """A translation applied after a request can only be the request
    obfuscation. Of the 3,000 units about 500 take one, and none holds what
    it would translate: a suffix, a wrapper's keys or a container's lines."""
    rows, pool = simulate.plan_kind({}, simulate.inputs(_SIMULATED, groups=3))
    by = {fn.__name__: fn for fn in pool}
    translating = [row["combination"] for row in rows
                   if _translates_after_a_request(row["combination"], by)]
    assert len(translating) > _SIMULATED // 8
    kept_out = {"suffixes", "structural", "neutral_container"}
    assert [names for names in translating
            if kept_out & {family for name in names for family in by[name].families}
            ] == []


# The eval type's plan of 300 inputs in three groups at its seed, 43, pinned
# before the packaged attacks turned the request obfuscation on. The eval
# pool and its draws are pinned (fixtures/eval_pool.json); they move only in
# an eval review. The first eval release moved it, first by the rule keeping a
# translation from every encode technique: the pool lost 40 of its 335
# pairs (each of its eight translations beside each of its five encodings),
# and 283 of the 300 round-2 units are dealt another pair, the group counts
# following the lost pairs (round 1 draws one technique and does not move).
# In a plan of 3,000 units, 304 round-2 units held such a pair. Then ASCII art
# refusing the heavy full-text encodings took 4 more pairs (base64, unicode
# escapes, ASCII ordinals and morse beside it; 291 of 630 remain; braille had
# the rule), 485 of the 3,000 round-2 units of that plan holding one after
# the first change, and 279 of the 300 round-2 units are dealt another pair.
# The rule keeping sensitive_words and tokenbreak apart did not move it: the
# pool holds neither. The manipulation layer applying last moved it again, by
# order alone: the same pairs go to the same units, and the 86 of the 300
# round-2 units that hold fsh or dap beside to_json, to_xml, to_markdown,
# to_refusal_then_comply or to_use_past_tense (40 of the 291 pairs) name it
# last; a plan of 3,000 inputs holds 858 such units of its 3,000 round-2 ones.
# Eval pool v2 (the third eval release, an eval review) moved it on purpose:
# eval.json admits eleven pure families (persona_template, indirect, the two
# distractors, impersonation, asking, guardrail_address, neutral_container,
# transliteration, fake_turns, and suffixes less its four noise suffixes),
# to_letter_spacing and the eight answer languages of its translations, so
# the pool goes from 36 techniques in 10 families to 126 in 22 and its pairs
# from 291 of 630 to 5,613 of 7,875 (no rule changed: the 291 are among
# them). Every group deals among 22 families instead of 10, so round 1 deals
# 281 of its 300 units another technique (19 draw what they drew) and round 2
# all 300 units another pair.
_PINNED_EVAL_DRAWS = "f075952d32211b4dc4a831be81e1c312c9145ac020634086eeaa195b45d14e1c"


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


def test_every_eval_technique_is_drawn_in_3000_units_and_its_families_evenly():
    """Eval pool v2 deals 22 families: in 1,500 inputs (3,000 units of its
    two rounds) in three groups at its seed every one of its 126 techniques
    is drawn (5 times at the least), each family but translation, weighted
    3, holds an even share of the rows, and translation three times one."""
    from redact.recipe import load_dataset_type

    base = load_dataset_type("eval")["augmentations"]
    dealt, pool = simulate.deal_kind({}, 1500, groups=3, seed=43, base=base)
    drawn = Counter(name for names in dealt for name in names)
    assert sorted(drawn) == sorted(fn.__name__ for fn in pool)
    assert len(drawn) == 126 and min(drawn.values()) >= 5   # noqa: PLR2004
    family_of = {fn.__name__: fn.families[0] for fn in pool}
    rows = Counter(family for names in dealt
                   for family in {family_of[name] for name in names})
    even = len(dealt) * 3 / 2 / 24      # 1.5 picks a unit over 24 weights
    assert {family for family, count in rows.items()
            if abs(count / (3 if family == "translation" else 1) - even) > 0.05 * even
            } == set()
