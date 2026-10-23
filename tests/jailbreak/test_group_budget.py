"""The group budget: an adversarial kind's opt-in ``budget: "group"``.

A unit draws at most ``max_complexity`` by default. With ``"group"`` it may
spend what the units dealt before it in its group left unspent of it, up to
``max_unit_complexity``, so the group's mean stays at or below
``max_complexity`` and no unit spends more than the ceiling. Left out or
``"unit"``, nothing changes: every draw is the one the planner made before the
option existed. The Dealer's bank is tested with the Dealer
(``tests/dataset/test_dealing.py``), the keys' validation with the section
(``test_augmentation.py``).
"""

import hashlib
import json

import pytest

import redact.jailbreak  # noqa: F401  (tags the techniques)
from redact.dataset import Manifest
from redact.dataset.dealing import Dealer
from redact.dataset.planfile import read_plan
from redact.jailbreak.augmentation import resolve, rounds_of
from redact.jailbreak.plan import assign_combination, plan_run
from redact.jailbreak.pool import all_techniques, technique_pool
from tests.jailbreak import simulate
from tests.jailbreak.test_dealing import _BUDGETS, _PINNED_EVAL_DRAWS, _eval_draws
from tests.jailbreak.test_plan_tolerance import _planned, _resume, _warnings

# Units a simulated plan deals, in three groups; the kind's max_complexity.
_UNITS = 3000
_GROUPS = 3
_MEAN = 6
_GROUP = {"budget": "group"}
_BUDGET_KEYS = ("budget", "max_unit_complexity")


def _plan(section: dict, frame) -> dict[str, list[str]]:
    """The combination each input of ``frame`` is planned, under the default
    attacks with ``section`` over them."""
    rows, _ = simulate.plan_kind(section, frame)
    return {row["sample_id"]: row["combination"] for row in rows}


# --- off: the draws of before ------------------------------------------------


@pytest.mark.parametrize("section", [{"budget": "unit"}, {"budget": None}],
                         ids=["unit", "null"])
def test_a_kind_that_says_unit_or_null_draws_what_a_kind_without_the_key_draws(section):
    assert (simulate.deal_kind(section, 600, groups=3)
            == simulate.deal_kind({}, 600, groups=3))


def test_a_round_that_says_unit_deals_the_draws_a_round_without_it_deals():
    """The draws ``test_dealing.py`` pins, round by round."""
    pool = all_techniques()
    unit = [{**budget, "budget": "unit"} for budget in _BUDGETS]
    for seed in (1, 2):
        assert (simulate.deal(pool, 60, groups=2, seed=seed, rounds=unit)
                == simulate.deal(pool, 60, groups=2, seed=seed, rounds=_BUDGETS))


def test_without_a_dealer_the_group_budget_changes_nothing():
    pool = all_techniques()
    drawn = [assign_combination(f"unit{i}", pool, seed=1, budget=budget)
             for i in range(30) for budget in (None, "group")]
    assert drawn[0::2] == drawn[1::2]


def test_a_count_round_ignores_the_group_budget():
    """The eval type's rounds draw an exact count, which has no budget to
    bank: its draws are the pinned ones."""
    held = _eval_draws({"budget": "group"})
    assert held == _eval_draws({})
    assert hashlib.sha256(json.dumps(held).encode()).hexdigest() == _PINNED_EVAL_DRAWS


# --- on: what a unit and a group spend ------------------------------------------


@pytest.fixture(scope="module", params=[None, 7], ids=["default ceiling", "ceiling 7"])
def spent(request):
    """The combinations of ``_UNITS`` units in three groups under the group
    budget, by category, with the pool, and the ceiling the kind says."""
    ceiling = request.param
    section = {**_GROUP, **({"max_unit_complexity": ceiling} if ceiling else {})}
    rows, pool = simulate.plan_kind(section, simulate.inputs(_UNITS, groups=_GROUPS))
    by_category: dict[str, list[list[str]]] = {}
    for row in rows:
        by_category.setdefault(row["category"], []).append(row["combination"])
    return by_category, pool, ceiling or _MEAN + 3


def test_each_groups_mean_complexity_is_at_or_below_max_complexity(spent):
    by_category, pool, _ = spent
    assert len(by_category) == _GROUPS
    for category, combinations in by_category.items():
        assert simulate.spend(combinations, pool)["mean"] <= _MEAN, category


def test_no_unit_spends_more_than_the_ceiling(spent):
    by_category, pool, ceiling = spent
    for combinations in by_category.values():
        assert simulate.spend(combinations, pool)["highest"] <= ceiling


def test_units_spend_more_than_max_complexity_with_what_the_others_left(spent):
    by_category, pool, _ = spent
    for combinations in by_category.values():
        assert simulate.spend(combinations, pool)["above"] > 0


def test_a_units_draw_adds_its_complexity_to_what_the_group_spent():
    """The request obfuscation's complexity is counted too."""
    pool = all_techniques()
    complexity = {fn.__name__: fn.complexity for fn in pool}
    dealer = Dealer(key=(1, "Cyber", "harmful", 0))
    for position in range(1, 21):
        dealer.position = position
        before = dealer.spent
        names = assign_combination(
            f"unit{position}", pool, seed=1, dealer=dealer, max_complexity=9,
            max_obfuscations=2, allow_request_obfuscation=True,
            sampling_probs={"request_obfuscation": 1})
        assert dealer.spent - before == sum(complexity[name] for name in names)
    assert dealer.spent > 0


# --- groups and order ------------------------------------------------------------


def test_a_groups_plan_does_not_depend_on_another_group():
    both = simulate.inputs(400, groups=2)
    alone = both[both["category"] == "category 0"]
    together, apart = _plan(_GROUP, both), _plan(_GROUP, alone)
    assert apart and all(together[sample] == names for sample, names in apart.items())


def test_the_plan_does_not_depend_on_the_order_of_the_inputs():
    frame = simulate.inputs(400, groups=2)
    shuffled = frame.sample(frac=1, random_state=7)
    assert _plan(_GROUP, shuffled) == _plan(_GROUP, frame)


# --- the plan file and a kept plan ---------------------------------------------------


@pytest.mark.parametrize(("section", "carried"), [
    ({}, {}),
    ({"budget": "unit"}, {}),
    ({"budget": "group"}, {"budget": "group"}),
    ({"budget": "group", "max_unit_complexity": 8},
     {"budget": "group", "max_unit_complexity": 8}),
], ids=["left out", "unit", "group", "group and ceiling"])
def test_the_plan_file_and_the_rows_carry_the_keys_only_when_set(
        tmp_path, section, carried):
    kind = resolve({"adversarial": section}, "adversarial")
    manifest = tmp_path / "jb.manifest.jsonl"
    plan_run(simulate.inputs(6), technique_pool(kind), manifest_path=manifest,
             settings_per_iteration=rounds_of(kind), verbose=False)
    (planned,) = read_plan(manifest)["rounds"]
    (settings, *_) = (row["settings"] for row in Manifest(manifest).load())
    for written in (planned, settings):
        assert {k: v for k, v in written.items() if k in _BUDGET_KEYS} == carried


@pytest.mark.parametrize("kind", [{}, {"budget": "unit"}, {"budget": None}],
                         ids=["left out", "unit", "null"])
def test_a_kept_plan_made_without_the_group_budget_resumes_quietly(
        tmp_path, monkeypatch, caplog, kind):
    out = tmp_path / "jb.csv"
    plan = _planned(out, monkeypatch)
    with caplog.at_level("WARNING"):
        rows = _resume(out, **kind)
    assert _warnings(caplog) == [] and len(rows) == len(plan)


def test_a_kept_plan_made_without_the_group_budget_raises_when_a_section_turns_it_on(
        tmp_path, monkeypatch):
    out = tmp_path / "jb.csv"
    _planned(out, monkeypatch)
    with pytest.raises(ValueError, match=r"this run's \['rounds'\].*resume=False"):
        _resume(out, **_GROUP)


def test_a_kept_group_plan_resumes_quietly_and_raises_when_the_section_drops_it(
        tmp_path, monkeypatch, caplog):
    out = tmp_path / "jb.csv"
    plan = _planned(out, monkeypatch, **_GROUP)
    with caplog.at_level("WARNING"):
        rows = _resume(out, **_GROUP)
    assert _warnings(caplog) == [] and len(rows) == len(plan)
    with pytest.raises(ValueError, match=r"this run's \['rounds'\].*resume=False"):
        _resume(out)
