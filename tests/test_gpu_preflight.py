"""The run's check of the cards before it loads a local model."""

import logging
import os
from unittest.mock import patch

import pandas as pd
import pytest

from redact import telemetry
from redact.driver import run_pipeline
from redact.llms.resources import measure
from redact.llms.resources.measure import CardUse

LOCAL = "llama-3.2-3b-debug"


def _card(index, used_gib=0.0, processes=()):
    return CardUse(index=index, uuid=f"GPU-{index}", used_gib=used_gib,
                   total_gib=48.0, processes=tuple(processes))


@pytest.fixture()
def run(tmp_path, monkeypatch):
    """Run an inputs stage on one copy of the local model, planned on card 0
    of two (the default plans a copy per card), with ``nvidia-smi`` answering
    ``cards``. ``_run.ran`` is what the last run ran: the stage, and each
    preload."""
    import redact

    def _run(cards, models=None):
        ran = _run.ran = []
        monkeypatch.setattr(redact, "generate_inputs",
                            lambda **k: ran.append("inputs") or pd.DataFrame())
        models = models or {"gen": {"name": LOCAL, "copies": 1}}
        recipe = {"dataset_type": "eval", "data_dir": str(tmp_path),
                  "stages": ["inputs"], "models": models}
        preloaded = lambda names, **kw: ran.append(("preload", names))  # noqa: E731
        try:
            with patch.object(measure, "card_use", return_value=cards), \
                 patch.object(measure, "detect_gpus", return_value=("FakeGPU", 2)), \
                 patch.object(measure, "card_sizes_gib", return_value=[48.0]), \
                 patch("redact.llms.resources.estimate._load_config",
                       return_value=None), \
                 patch("redact.residency.preload", side_effect=preloaded), \
                 patch("redact.llms.ModelClient.create"):
                run_pipeline(recipe, params={"inputs": {}}, verbose=False)
        finally:
            telemetry.uninstall()
        return ran

    return _run


def test_another_process_on_a_planned_card_stops_the_run_before_any_stage(run):
    holders = [(21131, "VLLM::EngineCore", 28.4), (21140, "python", None)]
    cards = [_card(0, 28.4, holders), _card(1)]
    with pytest.raises(RuntimeError) as raised:
        run(cards)
    message = str(raised.value)
    assert "GPU 0" in message and "GPU 1" not in message
    assert "21131 (VLLM::EngineCore, 28.4 GiB)" in message
    assert "21140 (python)" in message
    assert "kill -9 21131 21140" in message
    assert run.ran == []                # no stage ran and no model loaded


def test_a_process_on_a_card_the_plan_leaves_free_is_a_warning(run, caplog):
    cards = [_card(0), _card(1, 12.0, [(4242, "python", 12.0)])]
    with caplog.at_level(logging.WARNING, logger="redact"):
        ran = run(cards)
    assert "inputs" in ran
    (warning,) = [r.getMessage() for r in caplog.records if "4242" in r.getMessage()]
    assert "GPU 1" in warning


def test_memory_in_use_on_a_planned_card_with_no_process_listed_is_a_warning(
        run, caplog):
    """Another container's process holds memory nvidia-smi lists no process for."""
    with caplog.at_level(logging.WARNING, logger="redact"):
        ran = run([_card(0, 6.0), _card(1, 0.3)])
    assert "inputs" in ran
    warnings = [r.getMessage() for r in caplog.records if "in use" in r.getMessage()]
    assert len(warnings) == 1
    assert "GPU 0" in warnings[0] and "6.0 of 48.0 GiB" in warnings[0]


def test_this_process_on_a_planned_card_does_not_stop_the_run(run, caplog):
    """Its engines are the plan's to warn about."""
    with caplog.at_level(logging.WARNING, logger="redact.driver"):
        ran = run([_card(0, 4.0, [(os.getpid(), "python", 4.0)]), _card(1)])
    assert "inputs" in ran
    assert caplog.records == []


def test_the_plan_counts_its_cards_over_the_visible_ones(run, monkeypatch):
    """With CUDA_VISIBLE_DEVICES=1 the plan's card 0 is nvidia-smi's GPU 1."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with pytest.raises(RuntimeError, match="GPU 1"):
        run([_card(0), _card(1, 10.0, [(4242, "python", 10.0)])])


def test_without_nvidia_smi_or_a_local_model_there_is_nothing_to_check(run):
    assert "inputs" in run(None)
    held = [_card(0, 28.4, [(21131, "VLLM::EngineCore", 28.4)])]
    assert "inputs" in run(held, models={"gen": "venice-uncensored"})
