"""A library that adds and retires techniques: what a kind's pool draws, and
what a kept plan does on resume.

The spec dates a technique with ``since``, the version that added it, and
``deprecated``, the one that retired it; the plan file records the spec
version its pool was drawn at. A kept plan whose pool differs only by what
the library added or retired after that version resumes with a WARNING;
any other difference raises. The plans these tests make are drawn at the
library's own version, so they add a dummy technique, or retire a real one,
by hand at a version after it. The last tests are about the techniques the
library itself retired.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

import redact.jailbreak.pipeline as JPipe
import redact.jailbreak.pool as JPool
import redact.jailbreak.technique as T
from redact.dataset import Manifest
from redact.dataset.planfile import read_plan, write_plan
from redact.jailbreak.augmentation import resolve
from redact.jailbreak.plan import combination_of
from redact.jailbreak.pool import all_techniques, registry, technique_pool
from redact.stages import generate_jailbreaks
from tests.conftest import MockBackend
from tests.jailbreak.test_pipeline_jailbreaks import (
    GEN,
    _inputs,
    _llm_run,
    _pure,
    upper,
)

#: The spec version the plans these tests make are drawn at, and one after
#: it, for a technique a test adds or retires after the plan.
NOW = T.load_spec()["version"]
LATER = f"{NOW}.1"


def _retire(monkeypatch, name: str = "to_rot13", version: str = LATER):
    """The technique ``name``, retired by spec version ``version``."""
    fn = registry(all_techniques())[name]
    monkeypatch.setattr(fn, "deprecated", version)
    return fn


def _add(monkeypatch, since: str):
    """``to_reversed``, a pure encoding whose spec entry says the library
    added it in version ``since``, tagged and registered after every other."""
    def to_reversed(prompt):
        return prompt[::-1], ""

    spec = T.load_spec()
    entry = {"families": ["encode"], "complexity": 1, "encode_weight": "light",
             "since": since}
    with monkeypatch.context() as patch:
        patch.setattr(T, "load_spec", lambda: {
            **spec, "techniques": {**spec["techniques"], "to_reversed": entry}})
        T.tag([to_reversed])
    registered = all_techniques()
    monkeypatch.setattr(JPool, "all_techniques", lambda: [*registered, to_reversed])
    return to_reversed


def _planned(out, monkeypatch, **kind) -> list[dict]:
    """The plan of a run of the pure obfuscations into ``out``, stopped before
    its first unit, as its manifest holds it."""
    with monkeypatch.context() as patch:
        patch.setattr(JPipe, "drive_units", lambda units, **kwargs: None)
        generate_jailbreaks(inputs=_inputs(), output_path=out, seed=1, **_pure(**kind))
    return Manifest.sidecar(out).load()


def _resume(out, seed=1, **kind):
    return generate_jailbreaks(inputs=_inputs(), output_path=out, seed=seed,
                               **_pure(**kind))


def _warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records]


# --- the pool ----------------------------------------------------------------


def test_a_deprecated_technique_leaves_the_pool(monkeypatch):
    retired = _retire(monkeypatch)
    pool = technique_pool(resolve(None, "adversarial"))
    assert pool and retired not in pool


def test_allowing_its_family_does_not_keep_a_deprecated_technique(monkeypatch):
    retired = _retire(monkeypatch)
    pool = technique_pool(resolve(
        {"adversarial": {"families": {"allow": ["encode"]}, "translation": False}},
        "adversarial"))
    assert retired not in pool
    assert [fn for fn in pool if "encode" in fn.families]


def test_naming_a_deprecated_technique_in_techniques_allow_draws_it_with_a_warning(
        monkeypatch):
    retired = _retire(monkeypatch)
    with pytest.warns(DeprecationWarning,
                      match=f"'to_rot13', which spec version {LATER} deprecated"):
        pool = technique_pool(resolve(
            {"adversarial": {"techniques": {"allow": ["to_rot13"]},
                             "translation": False}}, "adversarial"))
    assert pool == [retired]


def test_naming_a_deprecated_technique_in_techniques_allow_is_logged(
        monkeypatch, caplog):
    """The DeprecationWarning shows only for code run as __main__, so a run
    also logs it."""
    _retire(monkeypatch)
    with caplog.at_level("WARNING", logger="redact.jailbreak.augmentation"):
        resolve({"adversarial": {"techniques": {"allow": ["to_rot13"]},
                                 "translation": False}}, "adversarial")
    (record,) = caplog.records
    assert record.getMessage().startswith(
        f"augmentations.adversarial.techniques.allow names to_rot13 ({LATER}), "
        "which the spec retired")


# --- a kept plan on resume -----------------------------------------------------


def test_a_technique_added_after_the_plan_warns_once_and_keeps_the_plan(
        tmp_path, monkeypatch, caplog):
    out = tmp_path / "jb.csv"
    plan = _planned(out, monkeypatch)
    _add(monkeypatch, since=LATER)
    with caplog.at_level("WARNING"):
        rows = _resume(out)
    (warning,) = _warnings(caplog)
    assert "added ['to_reversed'] and retired []" in warning
    assert "does not draw the added techniques" in warning
    assert "resume=False" in warning
    # Every planned unit ran as planned; none drew the new technique.
    assert Manifest.sidecar(out).load() == plan
    assert len(rows) == len(plan)
    assert not any("to_reversed" in t.split("+") for t in rows["technique"])


def test_a_technique_added_without_a_later_since_raises(tmp_path, monkeypatch):
    """A name the plan's version already had, as one a section newly allows."""
    out = tmp_path / "jb.csv"
    _planned(out, monkeypatch)
    _add(monkeypatch, since=NOW)
    with pytest.raises(ValueError, match=r"\['pool'\].*resume=False"):
        _resume(out)


def test_a_technique_no_longer_registered_raises(tmp_path, monkeypatch):
    out = tmp_path / "jb.csv"
    _planned(out, monkeypatch)
    registered = [fn for fn in all_techniques() if fn.__name__ != "to_rot13"]
    monkeypatch.setattr(JPool, "all_techniques", lambda: registered)
    with pytest.raises(ValueError, match=r"\['pool'\].*resume=False"):
        _resume(out)


def test_a_technique_retired_after_the_plan_warns_and_still_replays(
        tmp_path, monkeypatch, caplog):
    out = tmp_path / "jb.csv"
    plan = _planned(out, monkeypatch)
    unit = next(row for row in plan if row["combination"])
    retired = _retire(monkeypatch, unit["combination"][0])
    with caplog.at_level("WARNING"):
        rows = _resume(out)
    (warning,) = _warnings(caplog)
    assert f"added [] and retired ['{retired.__name__}']" in warning
    assert "resume=False" in warning
    replayed = rows[(rows["input_id"] == unit["sample_id"])
                    & (rows["iteration"] == unit["iteration"])]
    assert list(replayed["technique"]) == ["+".join(unit["combination"])]


def test_a_technique_retired_by_the_plan_s_own_version_raises(tmp_path, monkeypatch):
    """Retired no later than the plan was made, it left the pool by the
    section's doing, not the library's."""
    out = tmp_path / "jb.csv"
    _planned(out, monkeypatch)
    _retire(monkeypatch, "to_rot13", version=NOW)
    with pytest.raises(ValueError, match=r"\['pool'\].*resume=False"):
        _resume(out)


def _rewrite_plan_file(out, **made_with) -> None:
    """Set the keys ``made_with`` in the plan file beside ``out``; a key
    given as None is taken out, as a plan file written before it held it."""
    manifest = Manifest.sidecar(out).path
    record = read_plan(manifest)
    for key, value in made_with.items():
        if value is None:
            del record[key]
        else:
            record[key] = value
    write_plan(manifest, record)


def _without_spec_version(out) -> None:
    """Make the plan file beside ``out`` one written before it recorded
    ``spec_version``."""
    _rewrite_plan_file(out, spec_version=None)


def test_a_plan_file_without_a_spec_version_warns_and_its_rows_record_1_0(
        tmp_path, monkeypatch, caplog):
    """It was drawn at the first version, which is not today's."""
    out = tmp_path / "jb.csv"
    plan = _planned(out, monkeypatch)
    _without_spec_version(out)
    with caplog.at_level("WARNING"):
        rows = _resume(out)
    (warning,) = _warnings(caplog)
    assert f"(spec version 1.0, now {NOW})" in warning
    assert f"0 of its {len(plan)} pending units" in warning
    assert set(rows["combination_spec_version"].astype(str)) == {"1.0"}


def test_a_plan_file_without_rules_warns_that_it_does_not_record_them(
        tmp_path, monkeypatch, caplog):
    out = tmp_path / "jb.csv"
    plan = _planned(out, monkeypatch)
    _rewrite_plan_file(out, rules=None)
    with caplog.at_level("WARNING"):
        rows = _resume(out)
    (warning,) = _warnings(caplog)
    assert "the plan file does not record the rules it was drawn by" in warning
    assert f"0 of its {len(plan)} pending units" in warning
    assert len(rows) == len(plan)


def test_the_rules_id_moves_with_a_rule_not_with_the_pool_order_or_a_comment(
        monkeypatch):
    import redact.jailbreak.plan as JP

    every = registry(all_techniques())
    pool = [every["to_base64"], every["to_rot13"]]
    before = JP.rules_id(pool)
    assert JP.rules_id(pool[::-1]) == before
    spec = T.load_spec()
    with monkeypatch.context() as patch:
        patch.setattr(JP, "load_spec", lambda: {**spec, "layer_rates": {
            **spec["layer_rates"], "_comment": "reworded"}})
        assert JP.rules_id(pool) == before
    monkeypatch.setattr(pool[0], "complexity", pool[0].complexity + 1)
    assert JP.rules_id(pool) != before


def test_the_rules_id_moves_when_a_translation_stops_refusing_encode(monkeypatch):
    """The eval release made the family rule: a plan kept from before it has
    another rules id, so its resume warns."""
    import copy

    import redact.jailbreak.plan as JP

    every = registry(all_techniques())
    pool = [every["to_base64"], every["to_french"]]
    before = JP.rules_id(pool)
    spec = copy.deepcopy(T.load_spec())
    refused = spec["families"]["translation"]["cross_incompatible_with"]
    assert "encode" in refused
    refused.remove("encode")
    monkeypatch.setattr(JP, "load_spec", lambda: spec)
    assert JP.rules_id(pool) != before


def test_the_rules_id_moves_when_a_technique_stops_refusing_ascii_art(monkeypatch):
    """The eval release gave the heavy full-text encodings a rule of their
    own against ``ascii_art``: a technique's own list is in the id."""
    import redact.jailbreak.plan as JP

    base64 = registry(all_techniques())["to_base64"]
    assert base64.cross_incompatible_with == ["ascii_art"]
    before = JP.rules_id([base64])
    monkeypatch.setattr(base64, "cross_incompatible_with", [])
    assert JP.rules_id([base64]) != before


def test_the_rules_id_moves_with_what_a_request_obfuscation_keeps_out(monkeypatch):
    """A draw with the request obfuscation on reads the spec's
    ``request_obfuscation_refuses``, so its content is in the id, its comment
    is not."""
    import copy

    import redact.jailbreak.plan as JP

    every = registry(all_techniques())
    pool = [every["to_french"], every["to_json"]]
    before = JP.rules_id(pool)
    spec = copy.deepcopy(T.load_spec())
    spec["request_obfuscation_refuses"]["_comment"] = "reworded"
    monkeypatch.setattr(JP, "load_spec", lambda: spec)
    assert JP.rules_id(pool) == before
    spec["request_obfuscation_refuses"]["translation"].remove("structural")
    assert JP.rules_id(pool) != before


def test_a_resume_under_the_plan_s_version_and_rules_is_quiet(
        tmp_path, monkeypatch, caplog):
    out = tmp_path / "jb.csv"
    _planned(out, monkeypatch)
    with caplog.at_level("WARNING"):
        _resume(out)
    assert _warnings(caplog) == []


def test_a_resume_across_a_rule_change_warns_counts_and_stamps_the_plan_s_version(
        tmp_path, monkeypatch, caplog):
    """A plan of two obfuscations each, drawn at 1.2; today one of the first
    unit's techniques may no longer join the other's family. The units that
    pair the two still run as planned, and every row records 1.2."""
    out, pair = tmp_path / "jb.csv", [{"exact_techniques": 2}]
    plan = _planned(out, monkeypatch, rounds=pair)
    _rewrite_plan_file(out, spec_version="1.2")
    every = registry(all_techniques())
    first, second = (every[name] for name in plan[0]["combination"])
    monkeypatch.setattr(first, "cross_incompatible_with", [second.families[0]])
    refused = sum(
        first.__name__ in row["combination"]
        and any(second.families[0] in every[name].families
                for name in row["combination"])
        for row in plan)
    with caplog.at_level("WARNING"):
        rows = _resume(out, rounds=pair)
    (warning,) = _warnings(caplog)
    assert f"(spec version 1.2, now {NOW})" in warning
    assert "changed the rules a combination is drawn by" in warning
    assert f"{refused} of its {len(plan)} pending units" in warning and refused
    assert "+".join(plan[0]["combination"]) in set(rows["technique"])
    assert set(rows["combination_spec_version"].astype(str)) == {"1.2"}


def test_each_kind_s_rows_record_the_version_its_plan_was_drawn_at(
        tmp_path, monkeypatch):
    """One run, two plans: the attacks kept from 1.2, the variations planned
    afresh. Each file's rows record the version their plan was drawn at."""
    out = tmp_path / "jb.csv"
    _planned(out, monkeypatch)
    _rewrite_plan_file(out, spec_version="1.2")
    generate_jailbreaks(
        inputs=_inputs(), output_path=out, seed=1, auto_generate_benign=False,
        verbose=False, augmentation={
            "adversarial": _pure()["augmentation"]["adversarial"],
            "natural": {"families": {"allow": ["casual_writing"]}}})
    stamps = {name: set(pd.read_csv(tmp_path / name, dtype=str)
                        ["combination_spec_version"])
              for name in ("jb.csv", "variations.csv")}
    assert stamps == {"jb.csv": {"1.2"}, "variations.csv": {NOW}}


def test_a_plan_file_without_a_spec_version_was_made_at_the_first_version(
        tmp_path, monkeypatch, caplog):
    out = tmp_path / "jb.csv"
    _planned(out, monkeypatch)
    _without_spec_version(out)
    _add(monkeypatch, since=LATER)
    with caplog.at_level("WARNING"):
        _resume(out)
    (warning,) = _warnings(caplog)
    assert "spec version 1.0" in warning and "added ['to_reversed']" in warning


@pytest.mark.parametrize(("seed", "kind", "named"), [
    (2, {}, "seed"),
    (1, {"rounds": [{"max_complexity": 3}]}, "rounds"),
    (1, {"family_weights": {"encode": 2}}, "family_weights"),
    (1, {"layer_rates": {"requests": 0.1}}, "layer_rates"),
], ids=["seed", "rounds", "family_weights", "layer_rates"])
def test_another_seed_or_section_raises_though_the_library_grew(
        tmp_path, monkeypatch, seed, kind, named):
    out = tmp_path / "jb.csv"
    _planned(out, monkeypatch)
    _add(monkeypatch, since=LATER)
    with pytest.raises(ValueError, match=rf"this run's \[{named!r}\].*resume=False"):
        _resume(out, seed=seed, **kind)


#: The pools of the kinds at the released spec 1.2, as a plan made then
#: recorded them (``fixtures/spec_1_2_pools.json``).
_SPEC_1_2 = json.loads((Path(__file__).parents[1] / "fixtures" / "spec_1_2_pools.json")
                       .read_text(encoding="utf-8"))
_FOUR_TRANSLATIONS = ["to_amharic", "to_chinese", "to_hindi", "to_hmong"]


def _growth_since_1_2(key: str, pool: list) -> tuple[list[str], list[str]] | None:
    """What :func:`~redact.jailbreak.pipeline._library_change` finds in
    ``pool`` against the pool a spec 1.2 plan recorded for ``key``."""
    made_with = {"spec_version": "1.2", "pool": _SPEC_1_2[key]}
    return JPipe._library_change(made_with, pool, registry(all_techniques()))


def test_an_attack_plan_of_spec_1_2_finds_spec_1_3_s_36_techniques_as_growth():
    """The attack pool of a plan made at 1.2 differs from today's by 36 names,
    all dated 1.3 (the four translations switched on among them), none
    retired: the library added them after the plan, so the plan is kept and
    the run warns."""
    added, retired = _growth_since_1_2(
        "adversarial", technique_pool(resolve(None, "adversarial")))
    assert (len(added), retired) == (36, [])     # noqa: PLR2004
    assert set(_FOUR_TRANSLATIONS) <= set(added)
    assert {registry(all_techniques())[name].since for name in added} == {"1.3"}


def test_a_natural_plan_of_spec_1_2_finds_the_urgent_ask_as_the_library_s_growth():
    """Spec 1.3 added ``to_ask_with_urgency`` to the natural pool, the one
    technique it added there."""
    pool = technique_pool(resolve(None, "natural"))
    assert _growth_since_1_2("natural", pool) == (["to_ask_with_urgency"], [])


def test_an_eval_plan_of_spec_1_2_finds_no_growth():
    """The eval type's attack pool is pinned: the 36 techniques of 1.2 are the
    36 of today's, so a 1.2 eval plan resumes without a pool warning."""
    from redact.recipe import load_dataset_type

    pool = technique_pool(resolve(
        load_dataset_type("eval")["augmentations"], "adversarial"))
    assert _growth_since_1_2("eval:adversarial", pool) == ([], [])
    eval_pool = json.loads((Path(__file__).parents[1] / "fixtures" / "eval_pool.json")
                           .read_text(encoding="utf-8"))
    assert _SPEC_1_2["eval:adversarial"] == sorted(eval_pool)


# --- the techniques the library retired ----------------------------------------

#: Retired by spec version 1.1: three duplicates of
#: ``to_sensitive_words_encode_separator``, and four near-duplicate personas.
RETIRED_IN_1_1 = [
    "to_deformed_professor_persona", "to_mentally_ill_persona", "to_politician_persona",
    "to_religious_figure_persona", "to_sensitive_words_hyphen",
    "to_sensitive_words_star", "to_sensitive_words_underscore",
]


def test_spec_1_1_retires_three_word_separators_and_four_personas():
    retired = sorted(fn.__name__ for fn in all_techniques() if fn.deprecated)
    assert retired == RETIRED_IN_1_1
    assert {registry(all_techniques())[name].deprecated for name in retired} == {"1.1"}


def test_the_separator_encoding_draws_the_retired_separators_but_not_a_space():
    """So ``to_sensitive_words_split``, which splits with a space, stays."""
    from redact.jailbreak.obfuscation.encoding import SEPARATORS

    assert {"*", "-", "_"} <= set(SEPARATORS) and " " not in SEPARATORS


@pytest.mark.parametrize("kind", ["adversarial", "natural"])
def test_no_default_pool_draws_a_retired_technique(kind):
    pool = {fn.__name__ for fn in technique_pool(resolve(None, kind))}
    assert pool and not pool & set(RETIRED_IN_1_1)


@pytest.mark.parametrize("name", RETIRED_IN_1_1)
def test_a_kept_plan_still_finds_each_retired_technique(name):
    combined = combination_of([name], registry(all_techniques()))
    assert [fn.__name__ for fn in combined.techniques] == [name]


def test_a_kept_plan_runs_the_retired_word_separators_as_planned(tmp_path, monkeypatch):
    """A plan made before 1.1 names them; its units still run them."""
    retired = registry(all_techniques())
    plan = [("a word here", retired["to_sensitive_words_star"]),
            ("one word more", retired["to_sensitive_words_hyphen"]),
            ("that word too", retired["to_sensitive_words_underscore"])]
    out = _llm_run(tmp_path, monkeypatch, plan, MockBackend("word", model=GEN),
                   techniques=[upper, *(fn for _, fn in plan)])
    rows = pd.read_csv(out)
    assert dict(zip(rows["prompt"], rows["jailbreak"])) == {
        "a word here": "a w*o*r*d here", "one word more": "one w-o-r-d more",
        "that word too": "that w_o_r_d too"}
    assert rows["accepted"].all()
