"""Residency per stage: lifetimes, resident sets and the one share per engine."""

import random
from unittest.mock import patch

import pytest

from redact.llms.model_config import MODEL_REGISTRY, VLLMConfig, register_model
from redact.llms.resources import estimate
from redact.residency import plan_phases
from tests.test_residency import assert_no_card_is_overfull, fit_exists, packed_sizes


@pytest.fixture
def local_model():
    """Register vLLM models, of 5 GiB unless a size is given; planning reads
    no HF config."""
    made = []

    def _make(name, gb=5.0, **setup):
        register_model(name, backend_type="vllm", vllm=VLLMConfig(
            hf_model_id=f"org/{name}", vram_gb=gb, **setup))
        made.append(name)
        return name

    with patch("redact.llms.resources.estimate._load_config", return_value=None):
        yield _make
    for name in made:
        MODEL_REGISTRY.pop(name, None)


def _plan(stage_models, gib=24.0):
    return plan_phases(stage_models, gpus=1, gib_per_gpu=gib)


def _share(phase, name):
    return next(fp.planned_utilization for fp in phase.plan.footprints
                if fp.model == name)


def test_a_model_is_resident_from_its_first_stage_to_its_last(local_model):
    x, y = local_model("_ph_x"), local_model("_ph_y")
    phases = _plan({"a": [x], "b": [y], "c": [x], "d": []})
    assert [p.stage for p in phases] == ["a", "b", "c", "d"]
    # x stays through b, where only y is called, because c calls it again.
    assert [p.resident for p in phases] == [[x], [x, y], [x], []]


def test_stages_with_the_same_resident_models_share_one_plan(local_model):
    x = local_model("_ph_same")
    first, second = _plan({"a": [x], "b": [x]})
    assert first.plan is second.plan


def test_an_engine_gets_its_smallest_share_in_every_stage(local_model):
    x, y = local_model("_ph_alone"), local_model("_ph_guest")
    alone, shared = _plan({"a": [x], "b": [x, y]})
    # Alone, x would get 0.9 of the card; beside y it gets 0.45, and it loads
    # only once, so 0.45 is its share in both stages.
    assert _share(alone, x) == _share(shared, x) == 0.45
    assert _share(shared, y) == 0.45
    assert alone.plan.fits and shared.plan.fits


def test_two_models_on_one_engine_have_one_lifetime(local_model):
    x = local_model("_ph_engine")
    twin = "_ph_engine_twin"
    register_model(twin, backend_type="vllm",
                   vllm=VLLMConfig(hf_model_id=f"org/{x}", vram_gb=5.0))
    try:
        phases = _plan({"a": [x], "b": [], "c": [twin]})
    finally:
        MODEL_REGISTRY.pop(twin, None)
    # Stage b calls neither, but the engine is needed again in c.
    assert [p.resident for p in phases] == [[x, twin], [x, twin], [x, twin]]


def test_api_and_unregistered_models_are_ignored(local_model):
    x = local_model("_ph_local")
    phases = _plan({"a": ["claude-opus-4-6", "_ph_not_registered"], "b": [x]})
    assert [p.resident for p in phases] == [[], [x]]


def test_loadable_is_what_fits_together(local_model):
    x, y = local_model("_ph_fit_x"), local_model("_ph_fit_y")
    (phase,) = _plan({"a": [x, y]}, gib=8.0)      # 5 + 5 GiB is over 0.9 x 8
    assert not phase.plan.fits
    assert phase.loadable == [x]


def test_a_share_lowered_for_a_crowded_stage_is_logged(local_model, caplog):
    import logging

    x, y = local_model("_ph_log_x"), local_model("_ph_log_y")
    with caplog.at_level(logging.INFO, logger="redact.residency.phases"):
        _plan({"a": [x], "b": [x, y]})
    assert f"{x} loads once, with gpu_memory_utilization=0.45" in caplog.text
    assert y not in caplog.text


def _cards(phase, name):
    """``(devices, copies)`` a stage plans for a model."""
    return next((fp.devices, fp.copies) for fp in phase.plan.footprints
                if fp.model == name)


def test_an_engine_gets_its_smallest_copies_in_every_stage(local_model, caplog):
    import logging

    x, y = local_model("_ph_cp_x"), local_model("_ph_cp_y", 15.0)
    with caplog.at_level(logging.INFO, logger="redact.residency.phases"):
        alone, shared = plan_phases({"a": [x], "b": [x, y]}, gpus=2, gib_per_gpu=24.0)
    # Alone, x would have a copy on card 1. Stage b plans y there, with no
    # room left beside it, and x loads only once, so it has one copy in
    # both stages.
    assert _cards(alone, x) == _cards(shared, x) == ((0,), 1)
    assert _cards(shared, y) == ((1,), 1)
    assert (alone.plan.gpus_required, shared.plan.gpus_required) == (1, 2)
    assert _share(alone, x) == _share(shared, x) == 0.9
    assert [r.getMessage() for r in caplog.records] == [
        f"[residency] {x} loads once, with 1 copy: on the cards it holds in "
        "every stage."]
    assert "card 1 is idle" in alone.plan.explain()


def test_an_engine_keeps_a_copy_beside_a_later_stages_model(local_model, caplog):
    """Stage b plans y on card 1, where x's copy fits beside it. x keeps both
    copies, with the share card 1 grants it."""
    import logging

    x, y = local_model("_ph_beside_x"), local_model("_ph_beside_y")
    with caplog.at_level(logging.INFO, logger="redact.residency.phases"):
        alone, shared = plan_phases({"a": [x], "b": [x, y]}, gpus=2, gib_per_gpu=24.0)
    assert _cards(alone, x) == _cards(shared, x) == ((0, 1), 2)
    assert _cards(shared, y) == ((1,), 1)
    assert [_share(alone, x), _share(shared, x), _share(shared, y)] == [0.45] * 3
    assert [r.getMessage() for r in caplog.records] == [
        f"[residency] {x} loads once, with gpu_memory_utilization=0.45: its share "
        "in the stage where its card is most crowded."]


def test_an_engine_keeps_the_copies_every_stage_has_a_card_for(local_model):
    x, y = local_model("_ph_keep_x"), local_model("_ph_keep_y", 15.0)
    alone, shared = plan_phases({"a": [x], "b": [x, y]}, gpus=3, gib_per_gpu=24.0)
    # Stage b plans y on card 1, with no room left for a copy of x beside
    # it, so x keeps its copy on card 2 only.
    assert _cards(alone, x) == _cards(shared, x) == ((0, 2), 2)
    assert _cards(shared, y) == ((1,), 1)
    assert alone.plan.gpus_required == shared.plan.gpus_required == 3


def test_a_given_count_reaches_every_stage(local_model, caplog):
    import logging

    x, y = local_model("_ph_given_x"), local_model("_ph_given_y")
    with caplog.at_level(logging.INFO, logger="redact.residency.phases"):
        alone, shared = plan_phases(
            {"a": [x], "b": [x, y]}, gpus=3, gib_per_gpu=24.0, copies={x: 1})
    assert _cards(alone, x) == _cards(shared, x) == ((0,), 1)
    assert caplog.records == []


def test_an_engine_keeps_the_card_it_loaded_on(local_model, caplog):
    """y loads beside x, on card 1, where x has a copy. When x is released
    y stays there and z takes card 0. Placed afresh, y would be on card 0
    and z on top of where it really is. y has one copy in both stages: the
    first has none for it."""
    import logging

    x, y, z = (local_model(f"_ph_stay_{n}") for n in "xyz")
    with caplog.at_level(logging.INFO, logger="redact.residency"):
        first, second = plan_phases({"a": [x, y], "b": [y, z]}, gpus=2, gib_per_gpu=24.0)
    assert (_cards(first, x), _cards(first, y)) == (((0, 1), 2), ((1,), 1))
    assert (_cards(second, y), _cards(second, z)) == (((1,), 1), ((0,), 1))
    assert first.plan.fits and second.plan.fits
    assert [_share(first, y), _share(second, y), _share(second, z)] == [
        0.45, 0.45, 0.9]
    assert [r.getMessage() for r in caplog.records] == [
        f"[residency] {y} loads once, with gpu_memory_utilization=0.45: its share "
        "in the stage where its card is most crowded.",
        f"[residency] {y} loads once, with 1 copy: on the cards it holds in every "
        "stage."]


def test_a_kept_engine_gets_no_copy_it_did_not_load_with(local_model, caplog):
    """Three cards. In stage a x has copies on cards 1 and 2, and y is on
    card 1. In stage b y stays on card 1 and z takes card 0. Card 2 is idle
    then, but y loaded with one copy."""
    import logging

    x, y, z = (local_model(f"_ph_three_{n}") for n in "xyz")
    with caplog.at_level(logging.INFO, logger="redact.residency"):
        first, second = plan_phases({"a": [x, y], "b": [y, z]}, gpus=3, gib_per_gpu=24.0)
    assert (_cards(first, x), _cards(first, y)) == (((0, 1, 2), 3), ((1,), 1))
    assert (_cards(second, y), _cards(second, z)) == (((1,), 1), ((0,), 1))
    assert (first.plan.gpus_required, second.plan.gpus_required) == (3, 2)
    assert [(r.levelname, r.getMessage()) for r in caplog.records] == [
        ("INFO", f"[residency] {y} loads once, with gpu_memory_utilization=0.45: "
                 "its share in the stage where its card is most crowded."),
        ("INFO", f"[residency] {y} loads once, with 1 copy: on the cards it holds "
                 "in every stage.")]
    assert f"  note: card 2 is idle: {y} is limited to 1 copy" in second.plan.explain()


def test_a_kept_engine_with_a_given_count_keeps_all_its_cards(local_model):
    """Four cards: y has two copies, on cards 1 and 2, in both stages. x
    has a copy on every card."""
    x, y, z = (local_model(f"_ph_count_{n}") for n in "xyz")
    first, second = plan_phases(
        {"a": [x, y], "b": [y, z]}, gpus=4, gib_per_gpu=24.0, copies={y: 2})
    assert (_cards(first, x), _cards(first, y)) == (((0, 1, 2, 3), 4), ((1, 2), 2))
    assert (_cards(second, y), _cards(second, z)) == (((1, 2), 2), ((0,), 1))
    assert first.plan.fits and second.plan.fits
    assert f"  note: card 3 is idle: {y} is limited to 2 copies" in second.plan.explain()


def test_an_engine_needed_again_stays_on_its_card_in_between(local_model):
    """Stage b does not call x, but stage c does, so x is not released: it
    stays on card 1 while w leaves and y comes. No lifetime has a gap, so no
    engine is released and placed again. w has a copy beside x."""
    w, x, y = (local_model(f"_ph_again_{n}") for n in "wxy")
    a, b, c = plan_phases({"a": [w, x], "b": [y], "c": [x]}, gpus=2, gib_per_gpu=24.0)
    assert [p.resident for p in (a, b, c)] == [[w, x], [x, y], [x]]
    assert _cards(a, x) == _cards(b, x) == _cards(c, x) == ((1,), 1)
    assert (_cards(a, w), _cards(b, y)) == (((0, 1), 2), ((0,), 1))
    assert a.plan.fits and b.plan.fits and c.plan.fits


def test_a_newcomer_that_does_not_fit_beside_kept_engines_opens_a_group(local_model):
    """An 8 GiB card holds one 5 GiB engine. y stays on card 1, the first
    newcomer takes card 0 and the second has no card left."""
    w, y, n1, n2 = (local_model(f"_ph_full_{n}") for n in ("w", "y", "n1", "n2"))
    first, second = plan_phases({"a": [w, y], "b": [y, n1, n2]}, gpus=2, gib_per_gpu=8.0)
    assert _cards(first, y) == _cards(second, y) == ((1,), 1)
    assert second.plan.as_dict()["groups"] == [[y, n1], [n2]]
    assert (_cards(second, n1), _cards(second, n2)) == (((0,), 1), ((0,), 1))
    assert first.plan.fits and not second.plan.fits
    assert second.loadable == [y, n1]
    assert second.plan.problems[0].startswith(
        "these groups cannot be resident together, so a later group would load "
        "on top of the earlier ones.")


def test_a_twin_is_kept_through_its_engine(local_model):
    """x and its twin are two names on one checkpoint. Stage b calls only the
    twin, and the engine stays on card 1 when w leaves."""
    w, x, z = (local_model(f"_ph_twin_{n}") for n in "wxz")
    twin = "_ph_twin_x_twin"
    register_model(twin, backend_type="vllm",
                   vllm=VLLMConfig(hf_model_id=f"org/{x}", vram_gb=5.0))
    try:
        first, second = plan_phases(
            {"a": [w, x], "b": [twin, z]}, gpus=2, gib_per_gpu=24.0)
    finally:
        MODEL_REGISTRY.pop(twin, None)
    assert [p.resident for p in (first, second)] == [[w, x, twin], [x, twin, z]]
    assert _cards(first, w) == ((0, 1), 2)
    for phase in (first, second):
        assert _cards(phase, x) == _cards(phase, twin) == ((1,), 1)
    assert _cards(second, z) == ((0,), 1)


def test_a_kept_engine_on_whole_cards_does_not_join_a_shared_card(local_model):
    """Stage a does not fit: big, tensor parallel, opens a group of its own
    on cards 0 and 1, and gen is on card 0 in the first. Kept there in stage
    b, the two cannot be resident together either: a card held whole is no
    card to share, and gen keeps its full share."""
    from redact.llms.model_config import IntrospectConfig

    gen = local_model("_ph_whole_gen")
    big, spanning = "_ph_whole_big", "_ph_whole_tf"
    register_model(big, backend_type="vllm",
                   vllm=VLLMConfig(hf_model_id=f"org/{big}", vram_gb=20.0, min_gpus=2))
    register_model(spanning, backend_type="introspect", introspect=IntrospectConfig(
        hf_model_id=f"org/{spanning}", log_dir="x", vram_gb=18.0))
    try:
        first, second = plan_phases(
            {"a": [gen, big, spanning], "b": [gen, big]}, gpus=2, gib_per_gpu=24.0)
    finally:
        MODEL_REGISTRY.pop(big, None)
        MODEL_REGISTRY.pop(spanning, None)
    assert first.plan.as_dict()["groups"] == [[gen], [big], [spanning]]
    assert second.plan.as_dict()["groups"] == [[gen], [big]]
    assert not first.plan.fits and not second.plan.fits
    assert (_cards(second, gen), _cards(second, big)) == (((0,), 1), ((0, 1), 1))
    assert _share(first, gen) == _share(second, gen) == 0.9
    assert _share(second, big) == 0.9


def test_a_transformers_model_guessed_elsewhere_later_is_warned_about(
        local_model, caplog):
    """The plan does not place a transformers model. Beside w, 18 GiB of
    weights (20.5 with overhead) spill onto card 1. With w released the plan
    counts all of it on card 0 and hands card 1 to z."""
    import logging

    from redact.llms.model_config import IntrospectConfig

    w, z = local_model("_ph_tf_w"), local_model("_ph_tf_z")
    t = "_ph_tf_spanning"
    register_model(t, backend_type="introspect", introspect=IntrospectConfig(
        hf_model_id=f"org/{t}", log_dir="x", vram_gb=18.0))
    try:
        with caplog.at_level(logging.INFO, logger="redact.residency"):
            first, second = plan_phases(
                {"a": [w, t], "b": [t, z]}, gpus=2, gib_per_gpu=24.0)
    finally:
        MODEL_REGISTRY.pop(t, None)
    assert (_cards(first, t), _cards(second, t)) == (((0, 1), 1), ((0,), 1))
    assert _cards(second, z) == ((1,), 1)
    assert [(r.levelname, r.getMessage()) for r in caplog.records] == [(
        "WARNING",
        f"[residency] {t} is planned on other cards in a later stage. A "
        "transformers model stays where its device_map put it when it loaded, "
        "so that stage's plan may hand out memory it holds.")]


def test_an_early_stage_leaves_a_later_one_its_card(local_model, caplog):
    """A 24 GiB card grants 21.6 GiB; x and y need 6.1 each and z 17.1,
    which fits beside neither. Placed for stage a alone, x and y take a
    card each and stage b has none for z. On one card they fit both."""
    import logging

    x, y = local_model("_ph_se_x"), local_model("_ph_se_y")
    z = local_model("_ph_se_z", 15.0)
    with caplog.at_level(logging.INFO, logger="redact.residency"):
        first, second = plan_phases(
            {"a": [x, y], "b": [x, y, z]}, gpus=2, gib_per_gpu=24.0)
    assert first.plan.fits and second.plan.fits
    assert len(first.plan.groups) == len(second.plan.groups) == 1
    for phase in (first, second):
        assert _cards(phase, x) == _cards(phase, y) == ((0,), 1)
        assert _share(phase, x) == _share(phase, y) == 0.45
    assert (_cards(second, z), _share(second, z)) == (((1,), 1), 0.9)
    assert [r.getMessage() for r in caplog.records] == [
        "[residency] placed in another arrangement than the order given: "
        f"{x} -> card 0, {y} -> card 0, {z} -> card 1",
        f"[residency] {x} loads once, with 1 copy: on the cards it holds in every "
        "stage."]
    assert f"  note: card 1 is idle: {x} is limited to 1 copy" in first.plan.explain()


def test_cards_that_fit_one_stage_but_not_the_next_are_not_taken(local_model, caplog):
    """Stage a fits with x and y on one card, which leaves the other to z.
    Stage b fits with them apart, each beside an engine of 15.45 GiB. No
    cards fit both, so both are planned in the order given."""
    import logging

    x, y = local_model("_ph_no_x"), local_model("_ph_no_y")
    z = local_model("_ph_no_z", 15.0)
    u, w = local_model("_ph_no_u", 13.5), local_model("_ph_no_w", 13.5)
    with caplog.at_level(logging.INFO, logger="redact.residency.placement"):
        first, second = plan_phases(
            {"a": [x, y, z], "b": [x, y, u, w]}, gpus=2, gib_per_gpu=24.0)
    assert first.plan.as_dict()["groups"] == [[x, y], [z]]
    assert not first.plan.fits and second.plan.fits
    assert [_cards(first, m) for m in (x, y, z)] == [((0,), 1), ((1,), 1), ((0,), 1)]
    assert [_cards(second, m) for m in (x, y, u, w)] == [
        ((0,), 1), ((1,), 1), ((0,), 1), ((1,), 1)]
    assert caplog.records == []


def test_a_rearranged_engine_gets_no_copy_on_a_card_a_later_stage_fills(
        local_model, caplog):
    """Alone in stage a, x would have a copy on card 1. Stage b fits only
    with y beside x and z on card 1, and x loads once."""
    import logging

    x, y = local_model("_ph_cs_x"), local_model("_ph_cs_y")
    z = local_model("_ph_cs_z", 15.0)
    with caplog.at_level(logging.INFO, logger="redact.residency"):
        alone, shared = plan_phases(
            {"a": [x], "b": [x, y, z]}, gpus=2, gib_per_gpu=24.0)
    assert alone.plan.fits and shared.plan.fits
    assert _cards(alone, x) == _cards(shared, x) == _cards(shared, y) == ((0,), 1)
    assert _cards(shared, z) == ((1,), 1)
    assert _share(alone, x) == _share(shared, x) == 0.45
    assert [r.getMessage() for r in caplog.records] == [
        "[residency] placed in another arrangement than the order given: "
        f"{x} -> card 0, {y} -> card 0, {z} -> card 1",
        f"[residency] {x} loads once, with gpu_memory_utilization=0.45: its "
        "share in the stage where its card is most crowded.",
        f"[residency] {x} loads once, with 1 copy: on the cards it holds in "
        "every stage."]


def test_an_engine_leaves_the_card_of_a_later_transformers_model(local_model):
    """Alone in stage a, x would take card 0. Stage b brings a transformers
    model fixed on card 0, with 12 GiB that do not fit beside x's 17.1. So
    the two cards are not alike in stage a either, and x takes card 1."""
    from redact.llms.model_config import IntrospectConfig

    x = local_model("_ph_lt_x", 15.0)
    t = "_ph_lt_fixed"
    register_model(t, backend_type="introspect", introspect=IntrospectConfig(
        hf_model_id=f"org/{t}", log_dir="x", vram_gb=10.0, device_map="cuda:0"))
    try:
        first, second = plan_phases({"a": [x], "b": [x, t]}, gpus=2, gib_per_gpu=24.0)
    finally:
        MODEL_REGISTRY.pop(t, None)
    assert first.plan.fits and second.plan.fits
    assert _cards(first, x) == _cards(second, x) == ((1,), 1)
    assert _cards(second, t) == ((0,), 1)


def test_an_engine_on_whole_cards_keeps_them_in_a_rearranged_run(local_model):
    """Stage a fits four cards only with x and y on one, z on the next and
    the tensor-parallel engine on cards 2 and 3. It keeps those in stage b,
    and the engine that comes there is placed around it."""
    x, y = local_model("_ph_wk_x"), local_model("_ph_wk_y")
    z, new = local_model("_ph_wk_z", 15.0), local_model("_ph_wk_new", 8.0)
    split = local_model("_ph_wk_split", 30.0, min_gpus=2)
    first, second = plan_phases(
        {"a": [x, y, z, split], "b": [split, new]}, gpus=4, gib_per_gpu=24.0)
    assert first.plan.fits and second.plan.fits
    assert [_cards(first, m) for m in (x, y, z, split)] == [
        ((0,), 1), ((0,), 1), ((1,), 1), ((2, 3), 1)]
    # Cards 0 and 1 are idle in stage b: the new engine gets a copy on each.
    assert [_cards(second, m) for m in (split, new)] == [((2, 3), 1), ((0, 1), 2)]


def test_twins_of_different_sizes_are_not_crowded_into_one_group(local_model, caplog):
    """small and big are two names on one checkpoint, and the engine is
    placed by small's 6.1 GiB. Beside a (14.9 GiB), with b alone, stage y
    would be one group, but big needs 17.1 GiB and would be granted 6.4. So
    the order given stands, and stage x keeps the engine's whole share."""
    import logging

    small = local_model("_ph_tw_small")
    a, b = local_model("_ph_tw_a", 13.0), local_model("_ph_tw_b", 15.0)
    big = "_ph_tw_big"
    register_model(big, backend_type="vllm",
                   vllm=VLLMConfig(hf_model_id=f"org/{small}", vram_gb=15.0))
    try:
        with caplog.at_level(logging.INFO, logger="redact.residency.placement"):
            first, second = plan_phases(
                {"x": [small, big], "y": [small, big, a, b]}, gpus=2, gib_per_gpu=24.0)
    finally:
        MODEL_REGISTRY.pop(big, None)
    assert first.plan.fits and not second.plan.fits
    assert _share(first, small) == _share(first, big) == 0.9
    assert second.plan.as_dict()["groups"] == [[small, big, a], [b]]
    assert second.plan.oversized == []
    assert [_cards(second, m) for m in (small, big, a, b)] == [
        ((0,), 1), ((0,), 1), ((1,), 1), ((0,), 1)]
    assert caplog.records == []


def loads_of(called):
    """Per stage, its models as loads, should every engine be unloaded
    wherever no stage calls it: each a model and the stage it has been
    called since without a stage in between that does not."""
    def since(model, k):
        while k and model in called[k - 1]:
            k -= 1
        return k

    return [[(m, since(m, k)) for m in stage] for k, stage in enumerate(called)]


def test_a_run_fits_exactly_when_one_set_of_cards_fits_every_stage(local_model, caplog):
    """300 seeded runs of 2 to 4 stages over 3 to 5 engines, on 2 to 4
    cards: every stage fits exactly when trying every assignment finds one
    set of cards that fits them all, with every engine loaded once or with
    engines unloaded where no stage calls them."""
    import logging

    rng = random.Random(20261012)
    card = 24.0
    pool = estimate.DEFAULT_GPU_MEMORY_UTILIZATION * card
    unfit = 0
    with caplog.at_level(logging.INFO, logger="redact.residency.placement"):
        for i in range(300):
            cards = rng.randint(2, 4)
            names = [local_model(f"_ph_or{i}_{j}", gb) for j, gb in enumerate(
                packed_sizes(rng, cards, pool, range(3, 6)))]
            called = [rng.sample(names, rng.randint(1, len(names)))
                      for _ in range(rng.randint(2, 4))]
            phases = plan_phases(
                {f"s{k}": stage for k, stage in enumerate(called)},
                gpus=cards, gib_per_gpu=card)
            # Loaded once, an engine is resident from the first stage that
            # calls it to the last.
            resident = [
                [m for m in names if any(m in stage for stage in called[:k + 1])
                 and any(m in stage for stage in called[k:])]
                for k in range(len(called))]
            needs = {fp.model: fp.padded_gb
                     for phase in phases for fp in phase.plan.footprints}
            loads = loads_of(called)
            exists = fit_exists(needs, resident, cards, pool) or fit_exists(
                {load: needs[load[0]] for stage in loads for load in stage},
                loads, cards, pool)
            assert all(phase.plan.fits for phase in phases) is exists, (
                i, cards, called)
            # One set of cards for as long as an engine stays loaded.
            held, before = {}, []
            for phase in phases:
                if exists:
                    assert_no_card_is_overfull(phase.plan, pool)
                for fp in phase.plan.footprints:
                    if fp.model not in before:
                        held[fp.model] = fp.devices
                    assert held[fp.model] == fp.devices
                before = phase.resident
            unfit += not exists
    # Of the 300, the runs that fit only on other cards than the order
    # given takes, and those that fit on none. Two more fit with an engine
    # unloaded, one of them on such other cards.
    assert (len(caplog.records), unfit) == (69, 51)


def _unloaded(model, during, again, so_that=None):
    """The line that says a model is unloaded between two stages that call it."""
    return (f"[residency] {model} is unloaded for stage {during} and loaded again "
            f"for stage {again}, so that {so_that or during} fits. This costs one "
            "more load.")


def _messages(caplog):
    return [r.getMessage() for r in caplog.records]


def test_an_engine_is_unloaded_for_a_stage_that_does_not_fit_beside_it(
        local_model, caplog):
    """An 8 GiB card holds one 5 GiB engine. Stage b does not call x and
    stage c does, so x is unloaded for b and loaded again for c."""
    import logging

    x, y = local_model("_ph_gap_x"), local_model("_ph_gap_y")
    with caplog.at_level(logging.INFO, logger="redact.residency"):
        a, b, c = _plan({"a": [x], "b": [y], "c": [x]}, gib=8.0)
    assert [p.resident for p in (a, b, c)] == [[x], [y], [x]]
    assert a.plan.fits and b.plan.fits and c.plan.fits
    # Two loads of x are two plans, though the same model is resident.
    assert a.plan is not c.plan
    assert _share(a, x) == _share(b, y) == _share(c, x) == 0.9
    assert _messages(caplog) == [_unloaded(x, "b", "c")]


def test_no_engine_is_unloaded_where_another_card_holds_it(local_model, caplog):
    """The same run on two such cards: y takes the other card, and x stays
    loaded through stage b."""
    import logging

    x, y = local_model("_ph_two_x"), local_model("_ph_two_y")
    with caplog.at_level(logging.INFO, logger="redact.residency"):
        a, b, c = plan_phases(
            {"a": [x], "b": [y], "c": [x]}, gpus=2, gib_per_gpu=8.0)
    assert [p.resident for p in (a, b, c)] == [[x], [x, y], [x]]
    assert a.plan.fits and b.plan.fits and a.plan is c.plan
    assert (_cards(a, x), _cards(b, x), _cards(b, y)) == (
        ((0,), 1), ((0,), 1), ((1,), 1))
    assert _messages(caplog) == [
        f"[residency] {x} loads once, with 1 copy: on the cards it holds in "
        "every stage."]


def test_a_run_that_fits_with_every_engine_loaded_once_is_planned_as_before(
        local_model, caplog):
    """Runs in which a stage holds an engine it does not call, and fits:
    resident models, the stages that share a plan, cards, copies, shares,
    text and log lines as the planner made them before it unloaded any."""
    import logging

    from redact.residency import placement
    from tests.test_residency import _kv

    w, x, y = (local_model(f"_ph_as_{n}") for n in "wxy")
    z = local_model("_ph_as_z", 15.0)
    runs = [({"a": [x], "b": [y], "c": [x], "d": []}, 1),
            ({"a": [w, x], "b": [y], "c": [x]}, 2),
            ({"a": [x, y], "b": [z], "c": [x, y]}, 2),
            ({"a": [x], "b": [y], "c": [x]}, 3)]
    made = []
    with _kv(0.0), caplog.at_level(logging.INFO, logger="redact.residency"), \
         patch.object(placement._Stage, "only", side_effect=AssertionError):
        for stage_models, cards in runs:
            caplog.clear()
            phases = plan_phases(stage_models, gpus=cards, gib_per_gpu=24.0)
            plans = list({id(p.plan): p.plan for p in phases}.values())
            made.append((
                [p.resident for p in phases],
                [next(n for n, plan in enumerate(plans) if plan is p.plan)
                 for p in phases],
                [[(fp.model, fp.devices, fp.copies, fp.planned_utilization)
                  for fp in plan.footprints] for plan in plans],
                [plan.explain() for plan in plans],
                _messages(caplog)))
    assert made == [(resident, shared, placed, ["\n".join(text) for text in texts], log)
                    for resident, shared, placed, texts, log in _AS_BEFORE]


def _line(model, where, grant, size="6.5", weights="5.0"):
    share = {"21.6GiB": "0.90", "10.8GiB": "0.45", "21.6GiB each": "0.90",
             "10.8GiB each": "0.45"}[grant]
    return (f"    _ph_as_{model}[vllm]  ~{size}GiB  (declared: {weights} weights + "
            f"0.00 KV, x1.1 +0.6)  -> {where}, vLLM gets {grant} "
            f"(gpu_memory_utilization={share})")


_ONE, _TWO, _THREE = (
    f"plan needs {n} GPU(s); planning for {n} GPU(s), 24.0GiB each" for n in (1, 2, 3))
_COPIES = "      -> x{} copies (data_parallel_size={}), one per card, ~6.5GiB each"
#: Recorded at commit 125a000, per run: the resident models of each stage,
#: the plan each stage has, and per plan ``(model, cards, copies, share)``
#: of each model and the lines of ``explain()``; then the log lines. Since
#: copies share cards, the first engine of a stage has a copy beside the
#: others where they leave room, which the last three runs show.
_AS_BEFORE = [
    ([["_ph_as_x"], ["_ph_as_x", "_ph_as_y"], ["_ph_as_x"], []],
     [0, 1, 0, 2],
     [[("_ph_as_x", (0,), 1, 0.45)],
      [("_ph_as_x", (0,), 1, 0.45), ("_ph_as_y", (0,), 1, 0.45)],
      []],
     [(_ONE, "  group 1:", _line("x", "card 0", "10.8GiB")),
      (_ONE, "  group 1:", _line("x", "card 0", "10.8GiB"),
       _line("y", "card 0", "10.8GiB")),
      ("plan needs 0 GPU(s); planning for 1 GPU(s), 24.0GiB each",)],
     ["[residency] _ph_as_x loads once, with gpu_memory_utilization=0.45: its "
      "share in the stage where its card is most crowded."]),
    ([["_ph_as_w", "_ph_as_x"], ["_ph_as_x", "_ph_as_y"], ["_ph_as_x"]],
     [0, 1, 2],
     [[("_ph_as_w", (0, 1), 2, 0.45), ("_ph_as_x", (1,), 1, 0.45)],
      [("_ph_as_x", (1,), 1, 0.45), ("_ph_as_y", (0,), 1, 0.9)],
      [("_ph_as_x", (1,), 1, 0.45)]],
     [(_TWO, "  group 1:", _line("w", "cards 0-1", "10.8GiB each"),
       _COPIES.format(2, 2), _line("x", "card 1", "10.8GiB")),
      (_TWO, "  group 1:", _line("x", "card 1", "10.8GiB"),
       _line("y", "card 0", "21.6GiB")),
      (_TWO, "  group 1:", _line("x", "card 1", "10.8GiB"),
       "  note: card 0 is idle: _ph_as_x is limited to 1 copy")],
     ["[residency] _ph_as_x loads once, with gpu_memory_utilization=0.45: its "
      "share in the stage where its card is most crowded.",
      "[residency] _ph_as_x loads once, with 1 copy: on the cards it holds in "
      "every stage."]),
    ([["_ph_as_x", "_ph_as_y"], ["_ph_as_x", "_ph_as_y", "_ph_as_z"],
      ["_ph_as_x", "_ph_as_y"]],
     [0, 1, 0],
     [[("_ph_as_x", (0,), 1, 0.45), ("_ph_as_y", (0,), 1, 0.45)],
      [("_ph_as_x", (0,), 1, 0.45), ("_ph_as_y", (0,), 1, 0.45),
       ("_ph_as_z", (1,), 1, 0.9)]],
     [("plan needs 1 GPU(s); planning for 2 GPU(s), 24.0GiB each", "  group 1:",
       _line("x", "card 0", "10.8GiB"), _line("y", "card 0", "10.8GiB"),
       "  note: card 1 is idle: _ph_as_x is limited to 1 copy"),
      (_TWO, "  group 1:", _line("x", "card 0", "10.8GiB"),
       _line("y", "card 0", "10.8GiB"),
       _line("z", "card 1", "21.6GiB", size="17.5", weights="15.0"))],
     ["[residency] placed in another arrangement than the order given: "
      "_ph_as_x -> card 0, _ph_as_y -> card 0, _ph_as_z -> card 1",
      "[residency] _ph_as_x loads once, with 1 copy: on the cards it holds in "
      "every stage."]),
    ([["_ph_as_x"], ["_ph_as_x", "_ph_as_y"], ["_ph_as_x"]],
     [0, 1, 0],
     [[("_ph_as_x", (0, 1, 2), 3, 0.45)],
      [("_ph_as_x", (0, 1, 2), 3, 0.45), ("_ph_as_y", (1,), 1, 0.45)]],
     [(_THREE, "  group 1:", _line("x", "cards 0-2", "10.8GiB each"),
       _COPIES.format(3, 3)),
      (_THREE, "  group 1:", _line("x", "cards 0-2", "10.8GiB each"),
       _COPIES.format(3, 3), _line("y", "card 1", "10.8GiB"))],
     ["[residency] _ph_as_x loads once, with gpu_memory_utilization=0.45: its "
      "share in the stage where its card is most crowded."]),
]


def test_the_release_a_run_fits_without_is_given_back(local_model, caplog):
    """A 24 GiB card grants 21.6 GiB. Stage b's engine needs 9.4 and does
    not fit beside small (6.1) and large (11.6) together, but beside
    either. large is kept: its load is the longer one to repeat."""
    import logging

    small, large = local_model("_ph_back_small"), local_model("_ph_back_large", 10.0)
    mid = local_model("_ph_back_mid", 8.0)
    with caplog.at_level(logging.INFO, logger="redact.residency"):
        a, b, c = _plan({"a": [small, large], "b": [mid], "c": [small, large]})
    assert [p.resident for p in (a, b, c)] == [
        [small, large], [large, mid], [small, large]]
    assert a.plan.fits and b.plan.fits and c.plan.fits
    # large has one share from stage a to stage c: stage b's.
    assert [_share(p, large) for p in (a, b, c)] == [0.4958] * 3
    assert (_share(a, small), _share(b, mid), _share(c, small)) == (
        0.3354, 0.4041, 0.3354)
    assert _messages(caplog) == [
        _unloaded(small, "b", "c"),
        f"[residency] {large} loads once, with gpu_memory_utilization=0.50: its "
        "share in the stage where its card is most crowded."]


def test_engines_unloaded_for_different_stages_are_said_in_stage_order(
        local_model, caplog):
    """One 24 GiB card holds x, y (6.1 each) and big (14.9) two at a time.
    Each is unloaded for the stage that calls the other two, and has one
    share for every time it is loaded."""
    import logging

    x, y = local_model("_ph_turn_x"), local_model("_ph_turn_y")
    big = local_model("_ph_turn_big", 13.0)
    with caplog.at_level(logging.INFO, logger="redact.residency"):
        phases = _plan({"a": [x], "b": [x, y], "c": [x, big], "d": [x, y],
                        "e": [y, big], "f": [x]})
    assert [p.resident for p in phases] == [
        [x], [x, y], [x, big], [x, y], [y, big], [x]]
    assert all(p.plan.fits for p in phases)
    assert [_share(p, x) for p in (*phases[:4], phases[5])] == [0.2666] * 4 + [0.9]
    assert [_share(phases[k], y) for k in (1, 3, 4)] == [0.45, 0.2666, 0.2666]
    assert _messages(caplog) == [
        _unloaded(y, "c", "d"), _unloaded(big, "d", "e"), _unloaded(x, "e", "f"),
        f"[residency] {x} loads for stage a, with gpu_memory_utilization=0.27: its "
        "share in the stage where its card is most crowded.",
        f"[residency] {y} loads for stage d, with gpu_memory_utilization=0.27: its "
        "share in the stage where its card is most crowded."]


def test_every_engine_a_stage_has_no_room_for_is_unloaded(local_model, caplog):
    """Stage b's engine needs 17.1 of the 21.6 GiB, beside which neither
    small nor large fits."""
    import logging

    small, large = local_model("_ph_all_small"), local_model("_ph_all_large", 10.0)
    big = local_model("_ph_all_big", 15.0)
    with caplog.at_level(logging.INFO, logger="redact.residency"):
        a, b, c = _plan({"a": [small, large], "b": [big], "c": [small, large]})
    assert [p.resident for p in (a, b, c)] == [[small, large], [big], [small, large]]
    assert a.plan.fits and b.plan.fits and c.plan.fits
    assert _messages(caplog) == [_unloaded(small, "b", "c"), _unloaded(large, "b", "c")]


def test_no_engine_is_unloaded_for_a_stage_that_does_not_fit_without_it_either(
        local_model, caplog):
    """Stage b's own two engines do not fit one 8 GiB card, with x or
    without it: x stays loaded, and stages a and c share their plan."""
    import logging

    x, y, z = (local_model(f"_ph_stuck_{n}") for n in "xyz")
    with caplog.at_level(logging.INFO, logger="redact.residency"):
        a, b, c = _plan({"a": [x], "b": [y, z], "c": [x]}, gib=8.0)
    assert [p.resident for p in (a, b, c)] == [[x], [x, y, z], [x]]
    assert b.plan.as_dict()["groups"] == [[x], [y], [z]]
    assert a.plan.fits and not b.plan.fits and a.plan is c.plan
    assert caplog.records == []


def test_no_engine_is_unloaded_unless_every_stage_then_fits(local_model, caplog):
    """Unloading x for stage b and y for stage c would make both fit, but
    stage d calls y and z, which no 8 GiB card holds together."""
    import logging

    x, y, z = (local_model(f"_ph_half_{n}") for n in "xyz")
    with caplog.at_level(logging.INFO, logger="redact.residency"):
        phases = _plan({"a": [x], "b": [y], "c": [x], "d": [y, z]}, gib=8.0)
    assert [p.resident for p in phases] == [[x], [x, y], [x, y], [y, z]]
    assert [p.plan.fits for p in phases] == [True, False, False, False]
    assert caplog.records == []


def test_no_engine_is_unloaded_for_a_stage_with_a_problem_it_does_not_cause(
        local_model, caplog):
    """Without x, stage b's models are one group on the 8 GiB card, but
    one of them has no known size, and such a plan does not fit."""
    import logging

    x, y = local_model("_ph_prob_x"), local_model("_ph_prob_y")
    unsized = local_model("_ph_prob_unsized", None)
    with caplog.at_level(logging.INFO, logger="redact.residency.phases"):
        a, b, c = _plan({"a": [x], "b": [unsized, y], "c": [x]}, gib=8.0)
    assert [p.resident for p in (a, b, c)] == [[x], [x, unsized, y], [x]]
    assert a.plan.fits and not b.plan.fits and a.plan is c.plan
    assert _messages(caplog) == [
        f"[residency] {x} loads once, with gpu_memory_utilization=0.83: its "
        "share in the stage where its card is most crowded."]


def test_an_engine_loaded_twice_counts_once_against_the_search_limit(
        local_model, caplog):
    """Five engines, each kept through stages that do not call it. With
    every one unloaded where a stage does not fit, ten lifetimes are placed:
    more than the search tries every card for when lifetimes count, so that
    attempt failed and nothing was unloaded. Unloading e0 for s3 fits."""
    import logging

    from tests.test_residency import _kv

    e = {n: local_model(f"_ph_lim_e{n}", gb)
         for n, gb in enumerate((6.3, 5.3, 8.3, 9.3, 11.2))}
    stages = {"s0": [2, 3], "s1": [2, 3, 4], "s2": [0], "s3": [1, 4], "s4": [0],
              "s5": [2], "s6": [3]}
    with _kv(0.0), caplog.at_level(logging.INFO, logger="redact.residency.phases"):
        phases = plan_phases({s: [e[n] for n in ns] for s, ns in stages.items()},
                             gpus=2, gib_per_gpu=24.0,
                             copies=dict.fromkeys(e.values(), 1))
    assert all(phase.plan.fits for phase in phases)
    assert [r.getMessage() for r in caplog.records if "unloaded" in r.getMessage()] == [
        f"[residency] {e[0]} is unloaded for stage s3 and loaded again for stage "
        "s4, so that s3 fits. This costs one more load."]


def lifetime_fit_exists(needs, calls, released, stages, cards, pool):
    """Whether one card per lifetime keeps every card within ``pool`` in
    every stage, with the engines unloaded over the ``released`` gaps,
    found by trying cards one lifetime at a time.

    Args:
        needs: Each engine's need with overhead.
        calls: The stages that call each engine, in order.
        released: Gaps, each as an engine, the call before and the call after.
    """
    from redact.residency.phases import _lifetimes

    resident = [[] for _ in range(stages)]
    lifetimes = []
    for engine, at in calls.items():
        gaps = [(b, a) for e, b, a in released if e == engine]
        for first, last in _lifetimes(at, gaps):
            for i in range(first, last + 1):
                resident[i].append(len(lifetimes))
            lifetimes.append(needs[engine])
    load = [[0.0] * cards for _ in range(stages)]
    spans = [[i for i in range(stages) if n in resident[i]]
             for n in range(len(lifetimes))]

    def place(n):
        if n == len(lifetimes):
            return True
        # Cards no lifetime is on yet are alike: try the first of them only.
        used = {d for i in range(stages) for d in range(cards) if load[i][d] > 0}
        tried = [d for d in range(cards) if d in used] + [
            d for d in range(cards) if d not in used][:1]
        for d in tried:
            if all(load[i][d] + lifetimes[n] <= pool + 1e-6 for i in spans[n]):
                for i in spans[n]:
                    load[i][d] += lifetimes[n]
                if place(n + 1):
                    return True
                for i in spans[n]:
                    load[i][d] -= lifetimes[n]
        return False

    return place(0)


def test_a_run_one_set_of_unloads_would_fit_is_planned_to_fit(local_model, caplog):
    """300 seeded runs of 5 to 8 engines over 5 to 8 stages on 2 to 4 cards.
    Where unloading the engines a stage that does not fit holds without
    calling them gives a fit, found by trying every card for every
    lifetime, the plan fits: unloading more never makes a fit harder, so
    the planner's attempt with all of them unloaded finds one too. With
    more engines than _SEARCH_LIMIT it may not: they are placed largest
    first."""
    import logging
    import random

    from tests.test_residency import _kv

    rng = random.Random(20261016)
    checked = released = 0
    caplog.set_level(logging.INFO, logger="redact.residency.phases")
    with _kv(0.0):
        for run in range(300):
            cards = rng.randint(2, 4)
            names = [local_model(f"_ph_rel{run}_{n}", round(rng.uniform(4.0, 16.0), 1))
                     for n in range(rng.randint(5, 8))]
            count = rng.randint(5, 8)
            stage_models = {f"s{i}": rng.sample(names, rng.randint(1, 1 + cards))
                            for i in range(count)}
            caplog.clear()
            phases = plan_phases(stage_models, gpus=cards, gib_per_gpu=24.0,
                                 copies=dict.fromkeys(names, 1))
            released += any("unloaded" in r.getMessage() for r in caplog.records)
            unfit = [i for i, phase in enumerate(phases) if not phase.plan.fits]
            if not unfit:
                continue
            needs = {fp.model: fp.padded_gb for phase in phases
                     for fp in phase.plan.footprints}
            calls = {name: [i for i, models in enumerate(stage_models.values())
                            if name in models] for name in names}
            calls = {name: at for name, at in calls.items() if at}
            gaps = [(name, before, after) for name, at in calls.items()
                    for before, after in zip(at, at[1:])
                    if any(before < i < after for i in unfit)]
            if not gaps:
                continue
            checked += 1
            assert not lifetime_fit_exists(
                needs, calls, gaps, count, cards,
                estimate.DEFAULT_GPU_MEMORY_UTILIZATION * 24.0), (run, stage_models)
    # The runs checked, and those that fit by unloading.
    assert (checked, released) == (72, 80)


def test_no_engine_is_unloaded_only_to_move_it(local_model, caplog):
    """Two 24 GiB cards grant 21.6 GiB each. Stage a fits only with x and y
    on one card (18.8), beside p (20.4), and stage c only with them apart,
    each beside an engine of 11.6. x could change cards while stage b does
    not call it, but stage b fits with it, and stage c calls every engine
    it holds."""
    import logging

    p, x, y = (local_model(f"_ph_move_{n}", gb)
               for n, gb in (("p", 18.0), ("x", 8.0), ("y", 8.0)))
    r, s = local_model("_ph_move_r", 10.0), local_model("_ph_move_s", 10.0)
    with caplog.at_level(logging.INFO, logger="redact.residency"):
        phases = plan_phases(
            {"a": [p, x, y], "b": [y], "c": [r, s, x, y]}, gpus=2, gib_per_gpu=24.0)
    assert [ph.resident for ph in phases] == [[p, x, y], [x, y], [x, y, r, s]]
    assert [ph.plan.fits for ph in phases] == [True, True, False]
    # Alone on two cards in stage b, x would have a copy on card 0.
    assert [r.getMessage() for r in caplog.records] == [
        f"[residency] {x} loads once, with 1 copy: on the cards it holds in every "
        "stage."]


def test_an_engine_is_unloaded_for_every_stage_between_two_calls(local_model, caplog):
    """Stage b calls no local model and would fit with x, stage c does not.
    x is unloaded when stage a is over, and stage b holds nothing."""
    import logging

    x, y = local_model("_ph_long_x"), local_model("_ph_long_y")
    with caplog.at_level(logging.INFO, logger="redact.residency"):
        a, b, c, d = _plan({"a": [x], "b": [], "c": [y], "d": [x]}, gib=8.0)
    assert [p.resident for p in (a, b, c, d)] == [[x], [], [y], [x]]
    assert all(p.plan.fits for p in (a, b, c, d))
    assert (b.plan.footprints, b.plan.gpus_required) == ([], 0)
    assert _messages(caplog) == [_unloaded(x, "b", "d", so_that="c")]


def test_two_models_on_one_engine_are_unloaded_together(local_model, caplog):
    import logging

    x, y = local_model("_ph_pair_x"), local_model("_ph_pair_y")
    twin = "_ph_pair_x_twin"
    register_model(twin, backend_type="vllm",
                   vllm=VLLMConfig(hf_model_id=f"org/{x}", vram_gb=5.0))
    try:
        with caplog.at_level(logging.INFO, logger="redact.residency"):
            a, b, c = _plan({"a": [x], "b": [y], "c": [twin]}, gib=8.0)
    finally:
        MODEL_REGISTRY.pop(twin, None)
    assert [p.resident for p in (a, b, c)] == [[x, twin], [y], [x, twin]]
    assert a.plan.fits and b.plan.fits and c.plan.fits
    assert _messages(caplog) == [_unloaded(x, "b", "c")]


def test_an_engine_loaded_again_has_cards_copies_and_a_share_of_its_own(
        local_model, caplog):
    """Two 24 GiB cards. Alone in stage a, x has a copy on each. Stage b's
    engines need a card each (17.1 GiB), so x is unloaded for it. Loaded
    again for stage c, x finds g on card 0 and takes card 1. It stays there
    in stage d, where placed afresh it would take card 0, and shares it
    with q: one copy at half the card, in both stages."""
    import logging

    x, q = local_model("_ph_life_x"), local_model("_ph_life_q")
    g, c, big = (local_model(f"_ph_life_{n}", 15.0) for n in ("g", "c", "big"))
    with caplog.at_level(logging.INFO, logger="redact.residency"):
        first, between, third, fourth = plan_phases(
            {"a": [x], "b": [g, c], "c": [x, g], "d": [x, big, q]},
            gpus=2, gib_per_gpu=24.0)
    assert [ph.resident for ph in (first, between, third, fourth)] == [
        [x], [g, c], [x, g], [x, big, q]]
    assert all(ph.plan.fits for ph in (first, between, third, fourth))
    assert (_cards(first, x), _share(first, x)) == (((0, 1), 2), 0.9)
    for phase in (third, fourth):
        assert (_cards(phase, x), _share(phase, x)) == (((1,), 1), 0.45)
    assert (_cards(third, g), _cards(fourth, big)) == (((0,), 1), ((0,), 1))
    assert (_cards(fourth, q), _share(fourth, q)) == (((1,), 1), 0.45)
    assert _messages(caplog) == [
        _unloaded(x, "b", "c"),
        f"[residency] {x} loads for stage c, with gpu_memory_utilization=0.45: "
        "its share in the stage where its card is most crowded."]


def test_an_engine_loaded_again_keeps_the_copies_its_later_stages_have_a_card_for(
        local_model, caplog):
    """Two 8 GiB cards, one 5 GiB engine each. x has a copy on both in
    stage a, is unloaded for stage b, and alone in stage c would have both
    again. Stage d plans p on card 1, so its second load is one copy."""
    import logging

    x, y, z, p = (local_model(f"_ph_cut_{n}") for n in "xyzp")
    with caplog.at_level(logging.INFO, logger="redact.residency"):
        a, b, c, d = plan_phases(
            {"a": [x], "b": [y, z], "c": [x], "d": [x, p]}, gpus=2, gib_per_gpu=8.0)
    assert [ph.resident for ph in (a, b, c, d)] == [[x], [y, z], [x], [x, p]]
    assert _cards(a, x) == ((0, 1), 2)
    assert _cards(c, x) == _cards(d, x) == ((0,), 1)
    assert _messages(caplog) == [
        _unloaded(x, "b", "c"),
        f"[residency] {x} loads for stage c, with 1 copy: on the cards it holds "
        "in every stage."]


def assert_shares_stay_under_the_ceiling(plan):
    """The shares on a card sum to 0.9 at most, up to the step each is
    rounded by, and each covers its engine's need."""
    for card in range(plan.gpus_available):
        on = [fp for fp in plan.footprints if card in fp.devices]
        assert sum(fp.planned_utilization for fp in on) <= (
            estimate.DEFAULT_GPU_MEMORY_UTILIZATION + 1e-4 * len(on) + 1e-9)
        for fp in on:
            assert fp.planned_utilization * plan.per_gpu_gb >= fp.padded_gb - 1e-6


def test_engines_are_unloaded_only_where_a_run_does_not_fit_with_them(
        local_model, caplog):
    """600 seeded runs of 3 to 6 stages, each calling one or two of 3 to 5
    engines, on 1 to 3 cards. Trying every assignment says what must hold:

    - A run with cards for every engine loaded once is planned so.
    - Any other fits exactly when it has cards with engines unloaded where
      no stage calls them, and then with no engine unloaded that could
      have stayed.
    - An engine is resident in every stage that calls it and unloaded only
      between two of them. For as long as it stays loaded it has one set
      of cards and one share, and no card of a stage that fits is over its
      size.
    """
    import logging

    rng = random.Random(20261013)
    card = 24.0
    pool = estimate.DEFAULT_GPU_MEMORY_UTILIZATION * card
    once = again = never = reloaded = 0
    for i in range(600):
        cards = rng.randint(1, 3)
        names = [local_model(f"_ph_un{i}_{j}", gb) for j, gb in enumerate(
            packed_sizes(rng, cards, pool, range(3, 6)))]
        called = [rng.sample(names, rng.randint(1, 2))
                  for _ in range(rng.randint(3, 6))]
        names = [m for m in names if any(m in stage for stage in called)]
        with caplog.at_level(logging.INFO, logger="redact.residency.phases"):
            caplog.clear()
            phases = plan_phases(
                {f"s{k}": stage for k, stage in enumerate(called)},
                gpus=cards, gib_per_gpu=card)
        unloaded = [m for m in _messages(caplog) if "is unloaded for stage" in m]
        fits = all(phase.plan.fits for phase in phases)
        needs = {fp.model: fp.padded_gb
                 for phase in phases for fp in phase.plan.footprints}
        kept = [[m for m in names if any(m in stage for stage in called[:k + 1])
                 and any(m in stage for stage in called[k:])]
                for k in range(len(called))]
        resident = [set(phase.resident) for phase in phases]

        def has_cards(stages):
            """Whether cards exist for the models of each stage, a model
            being one load for as long as it stays."""
            loads = loads_of([sorted(stage) for stage in stages])
            return fit_exists({load: needs[load[0]] for stage in loads
                               for load in stage}, loads, cards, pool)

        if fit_exists(needs, kept, cards, pool):
            once += 1
            assert fits and not unloaded and resident == [set(k) for k in kept], i
        else:
            assert fits is has_cards(called), (i, cards, called)
            again += fits
            never += not fits
        reloads = 0
        for model in names:
            at = [k for k, stage in enumerate(resident) if model in stage]
            for before, after in zip(at, at[1:]):
                if after - before > 1:
                    reloads += 1
                    stays = [stage | {model} if before < k < after else stage
                             for k, stage in enumerate(resident)]
                    assert not has_cards(stays), (i, model, before, after)
        assert len(unloaded) == reloads and (fits or not reloads), i
        reloaded += reloads
        held, earlier = {}, set()
        for k, phase in enumerate(phases):
            assert set(called[k]) <= resident[k] <= set(kept[k]), (i, k)
            if fits:
                assert_no_card_is_overfull(phase.plan, pool)
                assert_shares_stay_under_the_ceiling(phase.plan)
            for fp in phase.plan.footprints:
                now = (fp.devices, fp.copies, fp.planned_utilization)
                if fp.model not in earlier:
                    held[fp.model] = now
                assert held[fp.model] == now, (i, k, fp.model)
            earlier = resident[k]
    # Of the 600: the runs that fit with every engine loaded once, those
    # that fit only with engines unloaded, those that fit neither way, and
    # the times an engine is loaded again.
    assert (once, again, never, reloaded) == (532, 50, 18, 70)


def test_preload_skips_a_model_no_longer_wanted(local_model):
    from redact import residency

    a, b = local_model("_ph_pre_a"), local_model("_ph_pre_b")
    wanted, created = {a, b}, []

    def create(name, backend_type=None):
        created.append(name)
        wanted.discard(b)           # dropped while the first model loads

    with patch("redact.llms.client.ModelClient.create", side_effect=create):
        residency.preload([a, b], verbose=False, wanted=wanted).join()
    assert created == [a]
