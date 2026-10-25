"""The persona-template techniques: a fixed frame per persona, a mechanic
drawn into it.

The packaged frames and mechanics are adversarial on purpose; these tests
assert properties of what the techniques write (a slot filled, a name in
the info), never print it, and show the composition on stand-in texts.
"""

import difflib
import itertools
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

#: Two lines, so a prompt written through ``{prompt!r}`` (which escapes the
#: newline) would no longer be found whole.
_PROBE = "a neutral probe\nabout the weather"
_FNS = PT.get_persona_template_functions()
_MECHANICS = PT._POOLS[PT.MECHANICS]
#: A slot of a frame or a mechanic left in a text.
_SLOT = re.compile(r"\{(prompt|mechanic|persona)\}")
#: The frames spec 1.4 added, by what they add: three kinds of persona, and the
#: three mechanics a frame can be built around (the file's six ``mechanics``
#: are drawn into every frame, these frames put one at the centre of theirs).
_TYPES = {
    "an emotional or family appeal": [
        "caring_grandma", "worried_parent", "loyal_friend"],
    "a legitimate expert or authority": [
        "licensed_professional", "official_investigator", "research_scientist"],
    "a simulated system or game": [
        "game_engine", "terminal_simulator", "simulation_rules"],
}
_BUILT_AROUND = {
    "a point and penalty system": "penalty_game",
    "a persona-tagged opening": "tagged_speaker",
    "a stay-in-character reset": "character_reset",
}
_ADDED_IN_1_4 = ({key for keys in _TYPES.values() for key in keys}
                 | set(_BUILT_AROUND.values()))


def test_there_is_one_technique_per_frame_of_the_file():
    names = [fn.__name__ for fn in _FNS]
    assert names == [f"to_{key}_persona_template" for key in PT.PERSONAS]
    assert len(names) == 23     # noqa: PLR2004
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
def test_the_spec_tags_a_pure_adversarial_hacking_technique(fn):
    """The eleven frames of spec 1.3 and the twelve of spec 1.4 are tagged alike
    but for the version that added them."""
    assert fn.families == ["persona_template"]
    assert (fn.layer, fn.complexity, fn.requires_llm, fn.requires_benign) == (
        "hacking", 2, False, False)
    assert fn.kinds == ["adversarial"]
    key = fn.__name__.removeprefix("to_").removesuffix("_persona_template")
    assert fn.since == ("1.4" if key in _ADDED_IN_1_4 else "1.3")


@pytest.mark.parametrize("fn", _FNS, ids=lambda fn: fn.__name__)
def test_a_technique_writes_the_prompt_once_and_the_mechanic_drawn(fn):
    """The prompt once, the drawn mechanic with the frame's name in it, and no
    slot left. Compared as flags: no text is shown."""
    key = fn.__name__.removeprefix("to_").removesuffix("_persona_template")
    for seed in range(12):
        text, info = fn(_PROBE, rng=random.Random(seed))
        drawn = draw_variant(_MECHANICS, random.Random(seed))
        written = drawn["template"].format_map({"persona": PT.display_name(key)})
        assert info == f"category={key}_persona_template;variant={drawn['name']}"
        found = (text.count(_PROBE), written in text, _SLOT.search(text) is None)
        assert found == (1, True, True), (fn.__name__, seed)


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
    assert len(table) == len(_FNS) * 6     # noqa: PLR2004
    assert set(table["variant"]) == {variant["name"] for variant in _MECHANICS}


def test_the_attacks_and_the_eval_type_draw_them_and_the_natural_kind_does_not():
    """Eval pool v2 admitted the family whole, the twelve frames of spec 1.4
    with the eleven of 1.3."""
    from redact.recipe import load_dataset_type

    eval_section = load_dataset_type("eval")["augmentations"]
    natural = technique_pool(resolve(None, "natural"))
    assert not [fn for fn in natural if "persona_template" in fn.families]
    for pool in (technique_pool(resolve(None, "adversarial")),
                 technique_pool(resolve(eval_section, "adversarial"))):
        assert [fn for fn in pool if "persona_template" in fn.families] == list(_FNS)
    assert len(_FNS) == 23      # noqa: PLR2004


class TestTheFramesOfSpec1_4:
    def test_there_are_twelve_and_the_file_has_them_after_the_eleven_of_1_3(self):
        assert len(_ADDED_IN_1_4) == 12     # noqa: PLR2004
        assert set(PT.PERSONAS[11:]) == _ADDED_IN_1_4 and len(PT.PERSONAS) == 23    # noqa: PLR2004

    @pytest.mark.parametrize("kind", list(_TYPES))
    def test_each_kind_of_persona_has_at_least_two_techniques(self, kind):
        assert len(_TYPES[kind]) >= 2     # noqa: PLR2004
        assert set(_TYPES[kind]) <= set(PT.PERSONAS)

    @pytest.mark.parametrize("mechanic", list(_BUILT_AROUND))
    def test_each_mechanic_has_a_technique_built_around_it(self, mechanic):
        assert _BUILT_AROUND[mechanic] in PT.PERSONAS

    def test_every_frame_is_a_technique_with_its_own_display_name(self):
        names = [PT.display_name(key) for key in PT.PERSONAS]
        assert len(set(names)) == len(names)

    def test_each_new_technique_has_the_six_mechanics_as_its_variants(self):
        """The variants of a persona template are the file's mechanics, drawn into
        its frame: six for every technique, the new ones included."""
        by_name = {fn.__name__: fn for fn in _FNS}
        for key in _ADDED_IN_1_4:
            fn = by_name[f"to_{key}_persona_template"]
            drawn = {fn(_PROBE, rng=random.Random(seed))[1].rsplit("variant=", 1)[1]
                     for seed in range(80)}
            assert drawn == {variant["name"] for variant in _MECHANICS}, key

    def test_no_two_frames_are_alike(self):
        assert _alike(PT._POOLS) == []


def _alike(pools: dict, *, limit: float = 0.6) -> list[tuple[str, str]]:
    """The pairs of frames in ``pools`` whose templates are more alike than
    ``limit`` (the ratio of ``difflib``): near-copies, not variants of one text."""
    frames = {key: pool[0]["template"] for key, pool in pools.items()
              if key != PT.MECHANICS}
    return [(a, b) for (a, ta), (b, tb) in itertools.combinations(frames.items(), 2)
            if difflib.SequenceMatcher(None, ta, tb).ratio() > limit]


def test_the_likeness_check_finds_a_copied_frame():
    copied = {**PT._POOLS, "copy": PT._POOLS["dan"]}
    assert _alike(copied) == [("dan", "copy")]
    changed = [{"name": "dan",
                "template": PT._POOLS["dan"][0]["template"] + " Also, be brief."}]
    assert _alike({**PT._POOLS, "dan_again": changed}) == [("dan", "dan_again")]
