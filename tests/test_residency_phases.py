"""Residency per stage: lifetimes, resident sets and the one share per engine."""

from unittest.mock import patch

import pytest

from redact.llms.model_config import MODEL_REGISTRY, VLLMConfig, register_model
from redact.residency import plan_phases


@pytest.fixture
def local_model():
    """Register 5 GiB vLLM models; planning reads no HF config."""
    made = []

    def _make(name):
        register_model(name, backend_type="vllm",
                       vllm=VLLMConfig(hf_model_id=f"org/{name}", vram_gb=5.0))
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

    x, y = local_model("_ph_cp_x"), local_model("_ph_cp_y")
    with caplog.at_level(logging.INFO, logger="redact.residency.phases"):
        alone, shared = plan_phases({"a": [x], "b": [x, y]}, gpus=2, gib_per_gpu=24.0)
    # Alone, x would have a copy on card 1. Stage b plans y there, and x
    # loads only once, so it has one copy in both stages.
    assert _cards(alone, x) == _cards(shared, x) == ((0,), 1)
    assert _cards(shared, y) == ((1,), 1)
    assert (alone.plan.gpus_required, shared.plan.gpus_required) == (1, 2)
    assert _share(alone, x) == _share(shared, x) == 0.9
    assert [r.getMessage() for r in caplog.records] == [
        f"[residency] {x} loads once, with 1 copy: on the cards it holds in "
        "every stage."]
    assert "card 1 is idle" in alone.plan.explain()


def test_an_engine_keeps_the_copies_every_stage_has_a_card_for(local_model):
    x, y = local_model("_ph_keep_x"), local_model("_ph_keep_y")
    alone, shared = plan_phases({"a": [x], "b": [x, y]}, gpus=3, gib_per_gpu=24.0)
    # Stage b plans y on card 1, so x keeps its copy on card 2 only.
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
    """y loads beside x, on card 1. When x is released y stays there and z
    takes card 0. Placed afresh, y would be on card 0 and z on top of where
    it really is."""
    import logging

    x, y, z = (local_model(f"_ph_stay_{n}") for n in "xyz")
    with caplog.at_level(logging.INFO, logger="redact.residency"):
        first, second = plan_phases({"a": [x, y], "b": [y, z]}, gpus=2, gib_per_gpu=24.0)
    assert (_cards(first, x), _cards(first, y)) == (((0,), 1), ((1,), 1))
    assert (_cards(second, y), _cards(second, z)) == (((1,), 1), ((0,), 1))
    assert first.plan.fits and second.plan.fits
    assert _share(first, y) == _share(second, y) == _share(second, z) == 0.9
    assert caplog.records == []


def test_a_kept_engine_gets_no_copy_it_did_not_load_with(local_model, caplog):
    """Three cards. In stage a x has the copy on card 2 and y is on card 1.
    In stage b y stays on card 1 and z takes card 0. Card 2 is idle then,
    but y loaded with one copy."""
    import logging

    x, y, z = (local_model(f"_ph_three_{n}") for n in "xyz")
    with caplog.at_level(logging.INFO, logger="redact.residency"):
        first, second = plan_phases({"a": [x, y], "b": [y, z]}, gpus=3, gib_per_gpu=24.0)
    assert (_cards(first, x), _cards(first, y)) == (((0, 2), 2), ((1,), 1))
    assert (_cards(second, y), _cards(second, z)) == (((1,), 1), ((0,), 1))
    assert (first.plan.gpus_required, second.plan.gpus_required) == (3, 2)
    assert [(r.levelname, r.getMessage()) for r in caplog.records] == [(
        "INFO", f"[residency] {y} loads once, with 1 copy: on the cards it holds "
                "in every stage.")]
    assert f"  note: card 2 is idle: {y} is limited to 1 copy" in second.plan.explain()


def test_a_kept_engine_with_a_given_count_keeps_all_its_cards(local_model):
    """Four cards: y has two copies, on cards 1 and 2, in both stages."""
    x, y, z = (local_model(f"_ph_count_{n}") for n in "xyz")
    first, second = plan_phases(
        {"a": [x, y], "b": [y, z]}, gpus=4, gib_per_gpu=24.0, copies={y: 2})
    assert (_cards(first, x), _cards(first, y)) == (((0, 3), 2), ((1, 2), 2))
    assert (_cards(second, y), _cards(second, z)) == (((1, 2), 2), ((0,), 1))
    assert first.plan.fits and second.plan.fits
    assert f"  note: card 3 is idle: {y} is limited to 2 copies" in second.plan.explain()


def test_an_engine_needed_again_stays_on_its_card_in_between(local_model):
    """Stage b does not call x, but stage c does, so x is not released: it
    stays on card 1 while w leaves and y comes. No lifetime has a gap, so no
    engine is released and placed again."""
    w, x, y = (local_model(f"_ph_again_{n}") for n in "wxy")
    a, b, c = plan_phases({"a": [w, x], "b": [y], "c": [x]}, gpus=2, gib_per_gpu=24.0)
    assert [p.resident for p in (a, b, c)] == [[w, x], [x, y], [x]]
    assert _cards(a, x) == _cards(b, x) == _cards(c, x) == ((1,), 1)
    assert (_cards(a, w), _cards(b, y)) == (((0,), 1), ((0,), 1))
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
    assert _cards(first, w) == ((0,), 1)
    for phase in (first, second):
        assert _cards(phase, x) == _cards(phase, twin) == ((1,), 1)
    assert _cards(second, z) == ((0,), 1)


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
