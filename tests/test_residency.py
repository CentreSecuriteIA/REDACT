"""VRAM footprints, residency planning, the load locks, and background preload.

None of this needs a GPU: capacity detection and the loaders are stubbed, and
the measurement path is exercised against a fake ``mem_get_info``. Real
load-time measurement against a live engine is a GPU-session job.
"""

import json
import threading
import time
from unittest.mock import patch

import pytest

from redact import telemetry
from redact.llms.model_config import MODEL_REGISTRY, VLLMConfig, register_model
from redact.llms.resources import measure, residency

GIB = 1024 ** 3
#: Fixture card size and the TP factor the suggestions land on.
CARD_GIB = 48.0
EXPECTED_TP = 2


@pytest.fixture(autouse=True)
def _clean():
    yield
    telemetry.uninstall()


@pytest.fixture()
def registered():
    """Register throwaway local models; clean up after."""
    made = []

    def _make(name, **vllm_kwargs):
        register_model(name, backend_type="vllm", vllm=VLLMConfig(**vllm_kwargs))
        made.append(name)
        return name

    yield _make
    for n in made:
        MODEL_REGISTRY.pop(n, None)


class TestReporting:
    """explain() is the pre-flight message — it has to be actionable, not just
    true. These cover the two things it grew: the vLLM grant check, and a
    concrete suggested configuration instead of 'go configure something'."""

    def test_grant_warning_fires_when_utilization_is_too_low(self, registered):
        """A model can fit the card and still OOM because gpu_memory_utilization
        caps what vLLM may reserve. That is the constraint which actually
        produces the failure, and nothing checked it before."""
        m = registered("_res_grant", hf_model_id="org/g", vram_gb=20.0,
                       vllm_kwargs={"gpu_memory_utilization": 0.1})
        caps = _capacity(48.0, 1)
        with caps[0], caps[1]:
            plan = residency.plan_residency([m])
        out = plan.explain()
        assert "gpu_memory_utilization=0.1 grants 4.8GiB" in out
        assert "Raise it to >=" in out

    def test_grant_check_applies_to_a_declared_footprint(self, registered):
        """The util is an engine fact, not an estimate detail — pinning
        vram_gb must not silently skip the check."""
        m = registered("_res_grant_decl", hf_model_id="org/gd", vram_gb=40.0,
                       vllm_kwargs={"gpu_memory_utilization": 0.5})
        caps = _capacity(48.0, 1)
        with caps[0], caps[1]:
            plan = residency.plan_residency([m])
        assert "WARNING" in plan.explain()

    def test_oversized_model_gets_a_concrete_configuration(self, registered):
        m = registered("_res_suggest", hf_model_id="org/s", vram_gb=52.0)
        caps = _capacity(48.0, 2)
        with caps[0], caps[1]:
            plan = residency.plan_residency([m])
        out = plan.explain()
        assert "suggested: min_gpus=2 (tensor_parallel_size=2)" in out
        assert "~26.0GiB per GPU" in out

    def test_tensor_parallel_split_is_shown(self, registered):
        m = registered("_res_tp_report", hf_model_id="org/t", vram_gb=60.0,
                       min_gpus=2)
        caps = _capacity(48.0, 2)
        with caps[0], caps[1]:
            plan = residency.plan_residency([m])
        assert "-> 2 GPUs (tensor_parallel_size=2), ~30.0GiB each" in plan.explain()


class TestMeasurement:
    """Post-hoc diagnostic only — nothing here feeds planning any more."""

    def test_no_ops_without_cuda(self):
        with patch("redact.llms.resources.measure._torch", return_value=None):
            with measure.Measurement() as m:
                pass
        assert m.claimed_gib is None and m.weights_gib is None

    def test_reports_claimed_and_weights_in_gib(self):
        fake = _FakeTorch(free_before=20 * GIB, free_after=8 * GIB,
                          peak=4 * GIB, total=24 * GIB)
        with patch("redact.llms.resources.measure._torch", return_value=fake):
            with measure.Measurement() as m:
                pass
        assert m.claimed_gib == pytest.approx(12.0)   # 20 -> 8 free
        assert m.weights_gib == pytest.approx(4.0)    # torch allocator peak
        assert m.total_gib == pytest.approx(24.0)

    def test_weights_exclude_what_was_already_resident(self):
        """reset_peak_memory_stats() resets the peak to *currently allocated*,
        not to zero — so with another checkpoint on the card the raw peak is
        (resident + mine). preload() loads sequentially in one process, so this
        is the normal case, not an edge one."""
        fake = _FakeTorch(free_before=20 * GIB, free_after=16 * GIB,
                          peak=10 * GIB, total=24 * GIB, already_allocated=6 * GIB)
        with patch("redact.llms.resources.measure._torch", return_value=fake):
            with measure.Measurement() as m:
                pass
        assert m.weights_gib == pytest.approx(4.0)   # 10 peak - 6 resident

    def test_load_failure_is_not_swallowed(self):
        fake = _FakeTorch(free_before=20 * GIB, free_after=20 * GIB,
                          peak=0, total=24 * GIB)
        with patch("redact.llms.resources.measure._torch", return_value=fake):
            with pytest.raises(RuntimeError), measure.Measurement():
                raise RuntimeError("load blew up")


class _FakeTorch:
    """Minimal stand-in for the torch surface Measurement touches."""

    def __init__(self, free_before, free_after, peak, total, already_allocated=0):
        self._free = [free_before, free_after]
        self._peak = peak
        self._total = total
        self._allocated = already_allocated
        self.cuda = self

    def memory_allocated(self):
        return self._allocated

    def is_available(self):
        return True

    def mem_get_info(self):
        return (self._free.pop(0) if len(self._free) > 1 else self._free[0]), self._total

    def reset_peak_memory_stats(self):
        pass

    def max_memory_allocated(self):
        return self._peak


def _capacity(total_gb, n_gpus, name="FakeGPU"):
    """Pin every capacity source, so the real machine never leaks into a test."""
    return (
        patch("redact.llms.resources.measure.free_total_gib",
              return_value=(total_gb, total_gb)),
        patch("redact.telemetry.detect_gpus", return_value=(name, n_gpus)),
    )


class TestFootprintResolution:
    def test_declared_value_is_used_when_nothing_was_measured(self, registered):
        m = registered("_res_declared", hf_model_id="org/a", vram_gb=20.0)
        fp = residency.footprint(m)
        assert (fp.gb, fp.source) == (20.0, "declared")

    def test_declared_beats_estimated(self, registered):
        """vram_gb is now the *override*, not the fallback — the place someone
        pins a number they know better than the shape math."""
        m = registered("_res_override", hf_model_id="org/b", vram_gb=20.0)
        with patch("redact.llms.resources.estimate.estimate_weights_gib",
                   return_value=99.0):
            fp = residency.footprint(m)
        assert (fp.gb, fp.source) == (20.0, "declared")

    def test_estimated_when_nothing_is_declared(self, registered):
        """A newly registered model must plan correctly with nothing
        hand-derived — the whole point of dropping the measurement loop."""
        m = registered("_res_estimated", hf_model_id="org/c")
        with patch("redact.llms.resources.estimate.estimate_weights_gib",
                   return_value=12.0),              patch("redact.llms.resources.estimate.estimate_kv_gib",
                   return_value=1.0):
            fp = residency.footprint(m)
        assert fp.source == "estimated"
        assert fp.gb > 13.0  # noqa: PLR2004 — weights 12 + KV 1, before headroom
        assert "12.0 weights" in fp.breakdown()

    def test_estimate_failure_is_unknown_not_zero(self, registered):
        m = registered("_res_noconfig", hf_model_id="org/unreadable")
        with patch("redact.llms.resources.estimate.estimate_weights_gib",
                   return_value=None):
            fp = residency.footprint(m)
        assert fp.gb is None and fp.source == "unknown"

    def test_api_only_model_has_no_footprint(self):
        assert residency.footprint("claude-opus-4-6") is None

    def test_unknown_when_nothing_declared_or_measured(self, registered):
        m = registered("_res_unknown", hf_model_id="org/d")
        fp = residency.footprint(m)
        assert fp.gb is None and fp.source == "unknown" and not fp.known


class TestPlanResidency:
    def test_two_small_models_share_one_gpu(self, registered):
        a = registered("_res_small_a", hf_model_id="org/sa", vram_gb=3.0)
        b = registered("_res_small_b", hf_model_id="org/sb", vram_gb=3.0)
        cap, gpus = _capacity(8.0, 1)
        with cap, gpus:
            plan = residency.plan_residency([a, b])
        assert len(plan.groups) == 1
        assert plan.gpus_required == 1 and plan.fits

    def test_one_large_model_takes_the_card_alone(self, registered):
        a = registered("_res_big_a", hf_model_id="org/ba", vram_gb=20.0)
        b = registered("_res_big_b", hf_model_id="org/bb", vram_gb=20.0)
        cap, gpus = _capacity(24.0, 1)
        with cap, gpus:
            plan = residency.plan_residency([a, b])
        assert len(plan.groups) == 2 and plan.sequential
        assert "unload between" in plan.explain()

    def test_models_sharing_a_checkpoint_are_one_load(self, registered):
        # venice-uncensored[vllm] and venice-paraphraser do exactly this. Two
        # rows, one engine — counting it twice would split a plan that fits.
        a = registered("_res_shared_a", hf_model_id="org/same", vram_gb=20.0)
        b = registered("_res_shared_b", hf_model_id="org/same", vram_gb=20.0)
        cap, gpus = _capacity(24.0, 1)
        with cap, gpus:
            plan = residency.plan_residency([a, b])
        assert len(plan.groups) == 1
        assert plan.gpus_required == 1 and plan.fits

    def test_multi_gpu_model_claims_whole_devices(self, registered):
        # The locally-served-translator case: too big for one card, so it takes
        # two outright rather than sharing the leftover GB on card 0.
        big = registered("_res_tp2", hf_model_id="org/huge", vram_gb=70.0, min_gpus=2)
        cap, gpus = _capacity(80.0, 1)
        with cap, gpus:
            plan = residency.plan_residency([big])
        assert plan.gpus_required == 2
        assert not plan.fits
        assert "needs 2 GPU(s)" in plan.explain()

    def test_unknown_capacity_does_not_invent_a_sequential_plan(self, registered):
        a = registered("_res_nocap_a", hf_model_id="org/na", vram_gb=20.0)
        b = registered("_res_nocap_b", hf_model_id="org/nb", vram_gb=20.0)
        # Both capacity sources blind: no torch, and no nvidia-smi either.
        with patch("redact.llms.resources.measure.free_total_gib", return_value=None):
            with patch("redact.telemetry.detect_gpus", return_value=(None, 0)):
                with patch("redact.telemetry.detect_gpu_memory_gib", return_value=None):
                    plan = residency.plan_residency([a, b])
        assert len(plan.groups) == 1          # grouped, not falsely split
        assert not plan.sequential

    def test_unknown_footprints_are_flagged_in_the_explanation(self, registered):
        a = registered("_res_noft", hf_model_id="org/nf")
        cap, gpus = _capacity(24.0, 1)
        with cap, gpus:
            plan = residency.plan_residency([a])
        assert "no footprint for" in plan.explain()


class TestLoadLocks:
    """Preload runs off-thread, so the caches must not be check-then-act.

    Without the double-checked lock, two threads missing together both build an
    engine — two copies of the same checkpoint on one card, OOM, and a
    traceback that points nowhere useful.
    """

    def test_concurrent_engine_misses_load_exactly_once(self):
        import redact.llms.backends.vllm as vllm_module

        vllm_module._engines.clear()
        vllm_module._engine_loaded_at.clear()
        loads = []

        class _SlowLLM:
            def __init__(self, **kw):
                # Wide enough that a second thread is certainly inside the
                # window a naive check-then-act would leave open.
                time.sleep(0.15)
                loads.append(kw["model"])

        fake_vllm = type("m", (), {"LLM": _SlowLLM})
        with patch.dict("sys.modules", {"vllm": fake_vllm}):
            with patch.object(vllm_module, "_prepare_environment"):
                results = []
                threads = [
                    threading.Thread(
                        target=lambda: results.append(
                            vllm_module._engine("org/same", None, {})
                        )
                    )
                    for _ in range(4)
                ]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

        assert loads == ["org/same"]                 # one load, not four
        assert len({id(r) for r in results}) == 1    # everyone got the same engine
        vllm_module._engines.clear()
        vllm_module._engine_loaded_at.clear()

    def test_concurrent_transformers_misses_load_exactly_once(self):
        import redact.llms.backends.introspection as intro

        intro._models.clear()
        loads = []

        def _slow_load(key, hf_model_id, device_map, torch_dtype, hf_kwargs):
            time.sleep(0.15)
            loads.append(hf_model_id)
            intro._models[key] = ("tok", "model")
            return intro._models[key]

        with patch.object(intro, "_load_locked", _slow_load):
            threads = [
                threading.Thread(target=lambda: intro._load("org/x", "auto", "auto", {}))
                for _ in range(4)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert loads == ["org/x"]
        intro._models.clear()


class TestPreload:
    def test_returns_none_when_nothing_is_local(self):
        assert residency.preload(["claude-opus-4-6"]) is None

    def test_warms_local_models_in_the_background(self, registered):
        m = registered("_pre_ok", hf_model_id="org/warm", vram_gb=1.0)
        seen = []
        with patch("redact.llms.ModelClient.create", side_effect=lambda n, **kw: seen.append(n)):
            thread = residency.preload([m])
            thread.join(timeout=5)
        assert seen == [m]

    def test_a_failed_preload_never_aborts_the_run(self, registered, tmp_path, caplog):
        # The whole point: constitution is ledger-backed and its Opus output is
        # saved as it goes, so a GPU that won't load must not kill it. The stage
        # that needs the model fails later, at the point of use.
        import logging

        m = registered("_pre_boom", hf_model_id="org/boom", vram_gb=1.0)
        telemetry.install(data_dir=tmp_path)
        with patch("redact.llms.ModelClient.create", side_effect=RuntimeError("no CUDA")):
            with caplog.at_level(logging.ERROR, logger="redact.residency"):
                thread = residency.preload([m])
                thread.join(timeout=5)          # returns normally, does not raise

        assert "no CUDA" in caplog.text
        assert "run continues" in caplog.text
        events = json.loads(telemetry.collector().trace_path.read_text().splitlines()[0])
        assert events["ev"] == "preload_failed" and events["model"] == m


class TestOversizedModels:
    """One model bigger than one card is a config problem, not a packing result."""

    def test_model_larger_than_a_device_is_reported_as_unplaceable(self, registered):
        # ceil(50/8) = 7 would imply splitting one model across 7 cards. Only
        # tensor parallelism spans devices; packing never does.
        big = registered("_res_oversized", hf_model_id="org/huge", vram_gb=50.0)
        cap, gpus = _capacity(8.0, 1)
        with cap, gpus:
            plan = residency.plan_residency([big])
        assert not plan.fits
        assert [fp.model for fp in plan.oversized] == [big]
        assert "packing never splits one model across cards" in plan.explain()
        assert "min_gpus" in plan.explain()

    def test_declaring_min_gpus_clears_the_problem(self, registered):
        big = registered("_res_tp_ok", hf_model_id="org/huge2", vram_gb=50.0, min_gpus=8)
        cap, gpus = _capacity(8.0, 8)
        with cap, gpus:
            plan = residency.plan_residency([big])
        assert plan.oversized == []
        assert plan.gpus_required == 8 and plan.fits

    def test_a_model_that_fits_is_not_flagged(self, registered):
        ok = registered("_res_fits", hf_model_id="org/small", vram_gb=7.0)
        cap, gpus = _capacity(8.0, 1)
        with cap, gpus:
            plan = residency.plan_residency([ok])
        assert plan.oversized == [] and plan.fits


class TestCapacityDetection:
    def test_nvidia_smi_supplies_capacity_when_torch_is_unavailable(self, registered):
        # A machine with a card but no torch/CUDA stack still gets a real plan
        # instead of the optimistic "capacity unknown" fallback.
        m = registered("_res_smi", hf_model_id="org/smi", vram_gb=50.0)
        with patch("redact.llms.resources.measure.free_total_gib", return_value=None):
            with patch("redact.telemetry.detect_gpus", return_value=("FakeGPU", 1)):
                with patch("redact.telemetry.detect_gpu_memory_gib", return_value=8.0):
                    plan = residency.plan_residency([m])
        assert plan.per_gpu_gb == 8.0
        assert not plan.fits          # 50GB on an 8GB card: correctly refused

    def test_torch_wins_over_nvidia_smi_when_both_are_available(self, registered):
        m = registered("_res_both", hf_model_id="org/both", vram_gb=1.0)
        with patch("redact.llms.resources.measure.free_total_gib", return_value=(20.0, 24.0)):
            with patch("redact.telemetry.detect_gpus", return_value=("FakeGPU", 1)):
                with patch("redact.telemetry.detect_gpu_memory_gib", return_value=8.0):
                    plan = residency.plan_residency([m])
        assert plan.per_gpu_gb == 24.0     # the live figure, not the static one


class TestRegistryEstimates:
    """The shipped entries declare a starting footprint, so planning works on
    a fresh install before anything has ever been measured."""

    @pytest.mark.parametrize("model,backend_type", [
        ("venice-uncensored", "vllm"),
        ("venice-paraphraser", None),
        ("llama-3.2-3b-debug", None),
    ])
    def test_every_local_model_declares_an_estimate(self, model, backend_type):
        fp = residency.footprint(model, backend_type)
        assert fp is not None and fp.known, f"{model} has no vram_gb estimate"
        assert fp.source == "declared"

    def test_the_shared_checkpoint_declares_the_same_footprint(self):
        a = residency.footprint("venice-uncensored", "vllm")
        b = residency.footprint("venice-paraphraser")
        assert a.hf_model_id == b.hf_model_id
        assert a.gb == b.gb        # one engine, one footprint


def test_estimated_tensor_parallel_footprint_is_a_total_like_a_declared_one():
    name = "_res_tp_estimated"
    register_model(name, backend_type="vllm", vllm=VLLMConfig(
        hf_model_id="org/tp", min_gpus=2, vllm_kwargs={"gpu_memory_utilization": 0.5}))
    try:
        with patch("redact.llms.resources.estimate.estimate_weights_gib",
                   return_value=30.0), \
             patch("redact.llms.resources.estimate.estimate_kv_gib", return_value=0.0), \
             patch("redact.llms.resources.measure.free_total_gib",
                   return_value=(48.0, 48.0)), \
             patch("redact.telemetry.detect_gpus", return_value=("FakeGPU", 2)):
            fp = residency.footprint(name)
            out = residency.plan_residency([name]).explain()
    finally:
        MODEL_REGISTRY.pop(name, None)
    assert fp.gb == 68.0            # (30 + 0) * 1.1 + 0.6 -> 34.0 per GPU, x2
    assert "~34.0GiB each" in out
    assert "needs ~34.0GiB per GPU but gpu_memory_utilization=0.5 grants 24.0GiB" in out


def test_same_checkpoint_with_different_engine_settings_is_two_loads():
    names = ("_res_same_a", "_res_same_b")
    for name, length in zip(names, (4096, 8192)):
        register_model(name, backend_type="vllm", vllm=VLLMConfig(
            hf_model_id="org/same", vram_gb=20.0, vllm_kwargs={"max_model_len": length}))
    try:
        with patch("redact.llms.resources.measure.free_total_gib",
                   return_value=(24.0, 24.0)), \
             patch("redact.telemetry.detect_gpus", return_value=("FakeGPU", 1)):
            plan = residency.plan_residency(list(names))
    finally:
        for name in names:
            MODEL_REGISTRY.pop(name, None)
    assert len(plan.groups) == 2    # 2 x 20GiB on one 24GiB card: one after the other


def test_planner_engine_id_matches_the_vllm_cache_key():
    import redact.llms.backends.vllm as vllm_module

    cfg = VLLMConfig(hf_model_id="org/m", quantization="awq", min_gpus=2,
                     vllm_kwargs={"max_model_len": 4096})
    assert residency._engine_id(cfg, "vllm")[1:] == vllm_module._engine_key(
        cfg.hf_model_id, cfg.quantization, cfg.engine_kwargs)


def test_unload_local_keeps_rate_limit_windows():
    from redact.llms import wrappers
    from redact.llms.client import clear_client_cache

    window = wrappers.shared_limiter("_test_unload_key")
    try:
        residency.unload_local()
        assert wrappers.shared_limiter("_test_unload_key") is window
    finally:
        clear_client_cache()
