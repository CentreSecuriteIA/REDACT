"""VRAM footprints, residency planning, the load locks, and background preload.

None of this needs a GPU: capacity detection and the loaders are stubbed, and
the measurement path is exercised against a fake ``mem_get_info``. Real
load-time measurement against a live engine is a GPU-session job.
"""

import json
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from redact import residency, telemetry
from redact.llms.backends import vram
from redact.llms.model_config import MODEL_REGISTRY, VLLMConfig, register_model


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


class TestPlanningGb:
    """The KV-cache trap: what vLLM *reserved* is not what the model *needs*."""

    def test_vllm_claimed_gb_is_never_used_for_planning(self):
        # A 3B at gpu_memory_utilization=0.86 reserves ~86% of the card. Planning
        # from that would say the model needs 6.9GB and make co-residency
        # impossible forever.
        entry = {"backend": "vllm", "claimed_gb": 6.9,
                 "weights_gb": 2.1, "kv_gb_est": 0.4}
        assert vram.planning_gb(entry) == pytest.approx(2.5)

    def test_vllm_without_a_weights_floor_declines_to_guess(self):
        assert vram.planning_gb({"backend": "vllm", "claimed_gb": 6.9}) is None

    def test_introspect_delta_really_is_the_need(self):
        # transformers preallocates no KV pool, so there the delta is honest.
        assert vram.planning_gb({"backend": "introspect", "claimed_gb": 6.2}) == 6.2


class TestEstimateKvGb:
    """The variable half of a footprint, derived from shapes not from a delta."""

    @staticmethod
    def _fake_transformers(cfg):
        """A stand-in ``transformers`` exposing only AutoConfig.from_pretrained."""
        mod = MagicMock()
        mod.AutoConfig.from_pretrained.return_value = cfg
        return patch.dict(sys.modules, {"transformers": mod})

    def test_computes_two_x_layers_x_kv_heads_x_head_dim_x_tokens(self):
        # Llama-3-8B's shapes, head_dim = 4096 // 32 = 128:
        #   2 * 32 * 8 * 128 * 8192 * 2 bytes = 1.0737e9 -> 1.07 GB.
        # Matches the ~1GB-per-sequence-at-8k figure quoted for that model,
        # which is what makes this a check on the formula and not just on
        # arithmetic I wrote twice.
        cfg = SimpleNamespace(
            num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=8,
            hidden_size=4096, max_position_embeddings=8192,
        )
        with self._fake_transformers(cfg):
            got = vram.estimate_kv_gb("fake/model", max_model_len=8192)
        # 2 * 32 * 8 * 128 * 8192 * 2 = 1.073...e9 bytes
        assert got == pytest.approx(1.07, abs=0.01)

    def test_gqa_caches_kv_heads_not_attention_heads(self):
        """The 4x difference between 8 KV heads and 32 attention heads is real."""
        mha = SimpleNamespace(
            num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=32,
            hidden_size=4096, max_position_embeddings=8192,
        )
        gqa = SimpleNamespace(
            num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=8,
            hidden_size=4096, max_position_embeddings=8192,
        )
        with self._fake_transformers(mha):
            big = vram.estimate_kv_gb("fake/model", max_model_len=8192)
        with self._fake_transformers(gqa):
            small = vram.estimate_kv_gb("fake/model", max_model_len=8192)
        assert big == pytest.approx(small * 4, rel=0.01)

    def test_max_model_len_falls_back_to_the_models_own_context(self):
        cfg = SimpleNamespace(
            num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=8,
            hidden_size=4096, max_position_embeddings=8192,
        )
        with self._fake_transformers(cfg):
            explicit = vram.estimate_kv_gb("fake/model", max_model_len=8192)
            implied = vram.estimate_kv_gb("fake/model")
        assert explicit == implied

    def test_shorter_context_needs_proportionally_less(self):
        cfg = SimpleNamespace(
            num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=8,
            hidden_size=4096, max_position_embeddings=8192,
        )
        with self._fake_transformers(cfg):
            full = vram.estimate_kv_gb("fake/model", max_model_len=8192)
            half = vram.estimate_kv_gb("fake/model", max_model_len=4096)
        assert half == pytest.approx(full / 2, rel=0.01)

    def test_tensor_parallel_shards_the_per_gpu_need(self):
        cfg = SimpleNamespace(
            num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=8,
            hidden_size=4096, max_position_embeddings=8192,
        )
        with self._fake_transformers(cfg):
            one = vram.estimate_kv_gb("fake/model", max_model_len=8192)
            four = vram.estimate_kv_gb(
                "fake/model", max_model_len=8192, tensor_parallel_size=4
            )
        assert four == pytest.approx(one / 4, rel=0.01)

    def test_fp32_kv_is_twice_bf16(self):
        cfg = SimpleNamespace(
            num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=8,
            hidden_size=4096, max_position_embeddings=8192,
        )
        with self._fake_transformers(cfg):
            bf16 = vram.estimate_kv_gb("fake/model", max_model_len=8192, dtype="bfloat16")
            fp32 = vram.estimate_kv_gb("fake/model", max_model_len=8192, dtype="float32")
        assert fp32 == pytest.approx(bf16 * 2, rel=0.01)

    def test_nested_text_config_is_unwrapped(self):
        """A multimodal entry keeps the language model's shapes one level down."""
        inner = SimpleNamespace(
            num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=8,
            hidden_size=4096, max_position_embeddings=8192,
        )
        with self._fake_transformers(SimpleNamespace(text_config=inner)):
            assert vram.estimate_kv_gb("fake/model", max_model_len=8192) == pytest.approx(
                1.07, abs=0.01
            )

    def test_incomplete_config_declines_to_guess(self):
        with self._fake_transformers(SimpleNamespace(num_hidden_layers=32)):
            assert vram.estimate_kv_gb("fake/model", max_model_len=8192) is None

    def test_unreadable_config_never_raises(self):
        """A planning hint must not take down a load that would have worked."""
        mod = MagicMock()
        mod.AutoConfig.from_pretrained.side_effect = OSError("no network")
        with patch.dict(sys.modules, {"transformers": mod}):
            assert vram.estimate_kv_gb("fake/model") is None

    def test_planning_uses_the_estimate_once_it_is_recorded(self, tmp_path):
        """End to end: a recorded estimate is what lifts planning off the floor."""
        path = tmp_path / "vram.json"
        m = vram.Measurement()
        m.claimed_gb, m.weights_gb, m.total_gb = 6.9, 2.1, 8.0
        m.record(path, "vllm:fake/model", {}, backend="vllm", kv_gb_est=0.4)
        entry = vram.load_cache(path)["vllm:fake/model"]
        assert entry["kv_gb_est"] == 0.4
        assert vram.planning_gb(entry) == pytest.approx(2.5)


class TestMeasurement:
    def test_no_ops_without_cuda(self, tmp_path):
        with patch("redact.llms.backends.vram._torch", return_value=None):
            with vram.Measurement() as m:
                pass
            m.record(tmp_path / "vram.json", "vllm:org/x", {}, backend="vllm")
        assert m.claimed_gb is None
        assert not (tmp_path / "vram.json").exists()   # nothing to record

    def test_records_all_three_numbers_with_settings(self, tmp_path):
        fake = _FakeTorch(free_before=20e9, free_after=8e9, peak=4e9, total=24e9)
        with patch("redact.llms.backends.vram._torch", return_value=fake):
            with vram.Measurement() as m:
                pass
            m.record(tmp_path / "vram.json", "vllm:org/x",
                     {"max_model_len": 4096, "gpu_memory_utilization": 0.9},
                     backend="vllm", kv_gb_est=1.5)
        entry = json.loads((tmp_path / "vram.json").read_text())["vllm:org/x"]
        assert entry["claimed_gb"] == pytest.approx(12.0)   # what it reserved
        assert entry["weights_gb"] == pytest.approx(4.0)    # the real floor
        assert entry["settings"]["max_model_len"] == 4096
        assert vram.planning_gb(entry) == pytest.approx(5.5)


class _FakeTorch:
    """Minimal stand-in for the torch surface Measurement touches."""

    def __init__(self, free_before, free_after, peak, total):
        self._free = [free_before, free_after]
        self._peak = peak
        self._total = total
        self.cuda = self

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
        patch("redact.llms.backends.vram.free_total_gb",
              return_value=(total_gb, total_gb)),
        patch("redact.telemetry.detect_gpus", return_value=(name, n_gpus)),
    )


class TestFootprintResolution:
    def test_declared_value_is_used_when_nothing_was_measured(self, registered):
        m = registered("_res_declared", hf_model_id="org/a", vram_gb=20.0)
        fp = residency.footprint(m)
        assert (fp.gb, fp.source) == (20.0, "declared")

    def test_measured_beats_declared_when_settings_match(self, registered, tmp_path):
        m = registered("_res_measured", hf_model_id="org/b", vram_gb=20.0)
        cache = {"vllm:org/b": {"backend": "vllm", "claimed_gb": 22.0,
                                "weights_gb": 12.0, "kv_gb_est": 1.0,
                                "settings": {"gpu_memory_utilization": None,
                                             "max_model_len": None,
                                             "tensor_parallel_size": 1,
                                             "quantization": None, "dtype": None}}}
        (tmp_path / "vram.json").write_text(json.dumps(cache))
        with patch("redact.paths.vram_cache_json", return_value=tmp_path / "vram.json"):
            fp = residency.footprint(m)
        assert (fp.gb, fp.source) == (13.0, "measured")

    def test_measurement_is_not_reused_when_settings_differ(self, registered, tmp_path):
        # Measured at max_model_len=384 says nothing about the same checkpoint
        # at 32k, so it must fall back to the declared estimate.
        m = registered("_res_stale", hf_model_id="org/c", vram_gb=20.0,
                       vllm_kwargs={"max_model_len": 32768})
        cache = {"vllm:org/c": {"backend": "vllm", "weights_gb": 12.0,
                                "settings": {"max_model_len": 384}}}
        (tmp_path / "vram.json").write_text(json.dumps(cache))
        with patch("redact.paths.vram_cache_json", return_value=tmp_path / "vram.json"):
            fp = residency.footprint(m)
        assert (fp.gb, fp.source) == (20.0, "declared")

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
        with patch("redact.llms.backends.vram.free_total_gb", return_value=None):
            with patch("redact.telemetry.detect_gpus", return_value=(None, 0)):
                with patch("redact.telemetry.detect_gpu_memory_gb", return_value=None):
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
        assert "cannot be split across" in plan.explain()
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
        with patch("redact.llms.backends.vram.free_total_gb", return_value=None):
            with patch("redact.telemetry.detect_gpus", return_value=("FakeGPU", 1)):
                with patch("redact.telemetry.detect_gpu_memory_gb", return_value=8.0):
                    plan = residency.plan_residency([m])
        assert plan.per_gpu_gb == 8.0
        assert not plan.fits          # 50GB on an 8GB card: correctly refused

    def test_torch_wins_over_nvidia_smi_when_both_are_available(self, registered):
        m = registered("_res_both", hf_model_id="org/both", vram_gb=1.0)
        with patch("redact.llms.backends.vram.free_total_gb", return_value=(20.0, 24.0)):
            with patch("redact.telemetry.detect_gpus", return_value=("FakeGPU", 1)):
                with patch("redact.telemetry.detect_gpu_memory_gb", return_value=8.0):
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
