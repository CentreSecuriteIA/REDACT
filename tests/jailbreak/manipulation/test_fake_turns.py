"""The fake-turn techniques: an invented turn of the conversation with the
prompt, from a variant pool, with no benign data.

The packaged variants are adversarial on purpose; these tests assert
properties of what the techniques write (the prompt kept, the slot filled,
the variant named), never print it, and show the drawing on stand-in texts.
"""

import random

import pandas as pd
import pytest

import redact.jailbreak.manipulation.fake_turns as FT
from redact.jailbreak.augmentation import resolve
from redact.jailbreak.directives import draw_variant, read_pools
from redact.jailbreak.manipulation import get_all_manipulation_functions
from redact.jailbreak.pool import technique_pool

_PROBE = "a neutral probe about the weather"
_FNS = FT.get_fake_turn_functions()
_DIRECTIVES = {"to_fake_system_turn": "fake_system_turn",
               "to_fake_assistant_turn": "fake_assistant_turn",
               "to_fake_history": "fake_history"}


def test_the_three_come_last_in_the_manipulation_layer():
    assert [fn.__name__ for fn in get_all_manipulation_functions()[-3:]] == list(
        _DIRECTIVES)


@pytest.mark.parametrize("fn", _FNS, ids=lambda fn: fn.__name__)
def test_the_spec_tags_a_pure_adversarial_manipulation_of_spec_1_3(fn):
    assert fn.families == ["fake_turns"]
    assert (fn.layer, fn.complexity, fn.requires_llm, fn.requires_benign) == (
        "manipulation", 2, False, False)
    assert (fn.kinds, fn.since) == (["adversarial"], "1.3")


@pytest.mark.parametrize("fn", _FNS, ids=lambda fn: fn.__name__)
def test_a_draw_is_its_directive_s_variant_after_an_invented_turn(fn):
    """Compared as flags: no text is shown. The invented turn comes first,
    so the prompt is never at the start."""
    directive = _DIRECTIVES[fn.__name__]
    assert fn.variant_pool.directive == directive
    pool = read_pools("manipulation/fake_turns_templates.json")[directive]
    assert len(pool) == 8       # noqa: PLR2004
    for seed in range(12):
        text, info = fn(_PROBE, rng=random.Random(seed))
        assert info == f"variant={draw_variant(pool, random.Random(seed))['name']}"
        found = (_PROBE in text, text.find(_PROBE) > 0, "{prompt}" in text)
        assert found == (True, True, False), (fn.__name__, seed)


def test_a_variant_is_drawn_as_rng_choice_draws_it(monkeypatch):
    """On stand-in texts: the prompt goes in as it is, braces and all."""
    monkeypatch.setattr(FT, "_TEMPLATES", {"fake_history": [
        {"name": "one", "template": "earlier: hi\nnow: {prompt}"},
        {"name": "two", "template": "earlier: hello\nnow: {prompt}"}]})
    rng = random.Random(4)
    expected = random.Random(4).choice(["one", "two"])
    text, info = FT.to_fake_history("say {x}", rng=rng)
    assert info == f"variant={expected}"
    assert text.endswith("now: say {x}")


def test_a_plan_of_them_needs_no_benign_data(tmp_path):
    """fsh and dap need the benign cache; the fake turns run without it,
    with auto_generate_benign off and no cache."""
    from redact.stages import generate_jailbreaks

    inputs = pd.DataFrame({"prompt": [f"prompt number {i} about a topic"
                                      for i in range(6)],
                           "category": ["Cyber"] * 6, "entry_type": ["harmful"] * 6})
    generate_jailbreaks(
        inputs=inputs, output_path=tmp_path / "jb.csv", seed=3,
        augmentation={"adversarial": {"families": {"allow": ["fake_turns"]},
                                      "translation": False,
                                      "rounds": [{"exact_techniques": 1}]},
                      "natural": None},
        auto_generate_benign=False, verbose=False)
    rows = pd.read_csv(tmp_path / "jb.csv")
    assert rows["accepted"].all() and len(rows) == 6     # noqa: PLR2004
    assert set(rows["technique"]) <= set(_DIRECTIVES)
    table = pd.read_csv(tmp_path / "jb.variants.csv")
    assert table["rows"].sum() == 6 and len(table) == 3 * 8     # noqa: PLR2004


def test_neither_the_natural_kind_nor_the_eval_type_draws_them():
    from redact.recipe import load_dataset_type

    eval_section = load_dataset_type("eval")["augmentations"]
    for pool in (technique_pool(resolve(None, "natural")),
                 technique_pool(resolve(eval_section, "adversarial"))):
        assert not [fn for fn in pool if "fake_turns" in fn.families]
    assert set(_FNS) <= set(technique_pool(resolve(None, "adversarial")))
