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


def test_an_engine_planned_on_different_cards_is_warned_about(local_model, caplog):
    """y loads beside x on card 1. Stage b plans it afresh, on card 0, with z
    on the card y is really on."""
    import logging

    x, y, z = (local_model(f"_ph_move_{n}") for n in "xyz")
    with caplog.at_level(logging.INFO, logger="redact.residency.phases"):
        first, second = plan_phases({"a": [x, y], "b": [y, z]}, gpus=2, gib_per_gpu=24.0)
    assert (_cards(first, y), _cards(second, y)) == (((1,), 1), ((0,), 1))
    assert _cards(second, z) == ((1,), 1)
    assert [(r.levelname, r.getMessage()) for r in caplog.records] == [(
        "WARNING",
        f"[residency] {y} is planned on different cards in different stages. "
        "It loads once and stays on the cards it loaded on, so a later "
        "stage's plan may put another model there.")]


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
