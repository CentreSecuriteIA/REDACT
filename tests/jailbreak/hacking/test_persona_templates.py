"""The persona-template techniques: a fixed frame per persona, a mechanic
drawn into it.

The packaged frames and mechanics are adversarial on purpose; these tests
assert properties of what the techniques write (a slot filled, a name in
the info), never print it, and show the composition on stand-in texts.
"""

import random
import re

import pandas as pd
import pytest

import redact.jailbreak.hacking.persona_templates as PT
from redact.jailbreak.augmentation import resolve
from redact.jailbreak.directives import PoolRef, draw_variant, read_pools
from redact.jailbreak.pool import registry, technique_pool, variant_pools
from redact.jailbreak.requests.answer import to_answer_as_code
from redact.jailbreak.variants import variants_of

_PROBE = "a neutral probe about the weather"
_FNS = PT.get_persona_template_functions()
_MECHANICS = PT._POOLS[PT.MECHANICS]
#: A slot of a frame or a mechanic left in a text.
_SLOT = re.compile(r"\{(prompt|mechanic|persona)\}")


def test_there_is_one_technique_per_frame_of_the_file():
    names = [fn.__name__ for fn in _FNS]
    assert names == [f"to_{key}_persona_template" for key in PT.PERSONAS]
    assert len(names) == 11     # noqa: PLR2004
    assert set(read_pools("hacking/persona_templates.json")) == {
        *PT.PERSONAS, PT.MECHANICS}


@pytest.mark.parametrize("template", [
    "{prompt} and no mechanic", "{prompt} {mechanic} {mechanic}"],
    ids=["none", "twice"])
def test_a_frame_must_name_the_mechanic_exactly_once(monkeypatch, template):
    """A frame without it would still report ``variant=<mechanic>`` for a
    mechanic that was never written; with two, it would be written twice."""
    pools = {**PT._POOLS, "dan": [{"name": "dan", "template": template}]}
    monkeypatch.setattr(PT, "_POOLS", pools)
    with pytest.raises(ValueError, match=r"directive 'dan': the frame needs the "
                       r"\{mechanic\} slot exactly once"):
        PT._check_mechanic_slots()


def test_the_packaged_frames_name_the_mechanic_exactly_once():
    PT._check_mechanic_slots()


def test_every_frame_has_a_display_name():
    assert all(PT._ENTRIES[key]["display_name"].strip() for key in PT.PERSONAS)


@pytest.mark.parametrize("fn", _FNS, ids=lambda fn: fn.__name__)
def test_the_spec_tags_a_pure_adversarial_hacking_technique_of_spec_1_3(fn):
    assert fn.families == ["persona_template"]
    assert (fn.layer, fn.complexity, fn.requires_llm, fn.requires_benign) == (
        "hacking", 2, False, False)
    assert fn.kinds == ["adversarial"]
    assert fn.since == "1.3"


@pytest.mark.parametrize("fn", _FNS, ids=lambda fn: fn.__name__)
def test_a_technique_fills_every_slot_and_names_the_mechanic_drawn(fn):
    """Compared as flags: no text is shown."""
    key = fn.__name__.removeprefix("to_").removesuffix("_persona_template")
    for seed in range(12):
        text, info = fn(_PROBE, rng=random.Random(seed))
        drawn = draw_variant(_MECHANICS, random.Random(seed))["name"]
        assert info == f"category={key}_persona_template;variant={drawn}"
        found = (_PROBE in text, PT.display_name(key) in text,
                 _SLOT.search(text) is None)
        assert found == (True, True, True), (fn.__name__, seed)


def test_the_draws_reach_every_mechanic():
    infos = {_FNS[0](_PROBE, rng=random.Random(seed))[1] for seed in range(60)}
    drawn = {info.rsplit("variant=", 1)[1] for info in infos}
    assert drawn == {variant["name"] for variant in _MECHANICS}
    assert len(drawn) == 6      # noqa: PLR2004


def test_the_mechanic_takes_the_display_name_then_goes_into_the_frame(monkeypatch):
    """On stand-in texts: a literal brace written ``{{`` stays one brace, and
    neither the filled mechanic nor the prompt is read for slots again."""
    monkeypatch.setattr(PT, "_POOLS", {
        "pilot": [{"name": "pilot", "template": "[{mechanic}] then: {prompt}"}],
        PT.MECHANICS: [{"name": "greet", "template": "be {persona} {{calm}}"}],
    })
    monkeypatch.setattr(PT, "_ENTRIES", {"pilot": {"display_name": "The Pilot"}})
    fn = PT._make_persona_template_fn("pilot")
    assert fn("say {mechanic} hi") == (
        "[be The Pilot {calm}] then: say {mechanic} hi",
        "category=pilot_persona_template;variant=greet")


def test_a_frame_without_a_display_name_is_called_by_its_key(monkeypatch):
    monkeypatch.setattr(PT, "_ENTRIES", {"rogue_ai": {}})
    assert PT.display_name("rogue_ai") == "Rogue Ai"


def test_the_mechanics_are_the_one_pool_drawn_and_hold_no_request_slot():
    ref = PoolRef("hacking/persona_templates.json", "mechanics", ("persona",),
                  "variant", request_slot=False)
    assert all(fn.variant_pool == ref for fn in _FNS)
    counted = {name: pool for name, pool in variant_pools().items()
               if pool["family"] == "persona_template"}
    assert counted == {ref.name: {"family": "persona_template",
                                  "techniques": [fn.__name__ for fn in _FNS],
                                  "variants": 6}}


def test_the_variants_table_pairs_the_mechanic_with_its_technique():
    every = registry([*_FNS, to_answer_as_code])
    assert variants_of(
        "to_dan_persona_template+to_answer_as_code",
        "category=dan_persona_template;variant=dual_answer;"
        "directive=answer_as_code;variant=v2", every) == [
        ("to_dan_persona_template", "dual_answer"), ("to_answer_as_code", "v2")]


def test_a_kind_counts_the_mechanic_each_row_drew(tmp_path):
    from redact.stages import generate_jailbreaks

    inputs = pd.DataFrame({"prompt": [f"prompt number {i} about a topic"
                                      for i in range(6)],
                           "category": ["Cyber"] * 6, "entry_type": ["harmful"] * 6})
    generate_jailbreaks(
        inputs=inputs, output_path=tmp_path / "jb.csv", seed=3,
        augmentation={"adversarial": {"families": {"allow": ["persona_template"]},
                                      "translation": False,
                                      "rounds": [{"exact_techniques": 1}]},
                      "natural": None},
        auto_generate_benign=False, verbose=False)

    table = pd.read_csv(tmp_path / "jb.variants.csv")
    rows = pd.read_csv(tmp_path / "jb.csv")
    assert table["rows"].sum() == rows["accepted"].sum() == 6     # noqa: PLR2004
    assert len(table) == 11 * 6     # noqa: PLR2004
    assert set(table["variant"]) == {variant["name"] for variant in _MECHANICS}


def test_neither_the_natural_kind_nor_the_eval_type_draws_them():
    from redact.recipe import load_dataset_type

    eval_section = load_dataset_type("eval")["augmentations"]
    for pool in (technique_pool(resolve(None, "natural")),
                 technique_pool(resolve(eval_section, "adversarial"))):
        assert not [fn for fn in pool if "persona_template" in fn.families]
    assert set(_FNS) <= set(technique_pool(resolve(None, "adversarial")))
