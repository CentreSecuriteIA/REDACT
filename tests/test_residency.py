"""VRAM footprints, residency planning, the load locks, and background preload.

None of this needs a GPU or the network: capacity detection, HF config reads
and the loaders are stubbed, and the measurement path is exercised against a
fake ``mem_get_info``. Real load-time measurement against a live engine is a
GPU-session job.
"""

import json
import logging
import threading
import time
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from redact import telemetry
from redact.llms import observe
from redact.llms.model_config import (
    MODEL_REGISTRY,
    IntrospectConfig,
    VLLMConfig,
    register_model,
)
from redact import residency
from redact.llms.resources import estimate, measure
from redact.residency.footprint import _engine_id
from redact.residency.placement import _Cards, _exceeds_its_cards

GIB = 1024 ** 3
#: Fixture card size and the TP factor the suggestions land on.
CARD_GIB = 48.0
EXPECTED_TP = 2

# Real shapes: Mistral-Small-24B (32 heads) and Qwen2.5-7B (28 heads).
MISTRAL_24B = SimpleNamespace(
    num_hidden_layers=40, hidden_size=5120, intermediate_size=32768,
    num_attention_heads=32, num_key_value_heads=8, head_dim=128,
    vocab_size=131072, max_position_embeddings=32768,
    tie_word_embeddings=False, torch_dtype="bfloat16", model_type="mistral",
)
QWEN_7B = SimpleNamespace(
    num_hidden_layers=28, hidden_size=3584, intermediate_size=18944,
    num_attention_heads=28, num_key_value_heads=4, vocab_size=152064,
    max_position_embeddings=32768, tie_word_embeddings=False,
    torch_dtype="bfloat16", model_type="qwen2",
)


@pytest.fixture(autouse=True)
def _clean():
    """No HF config is readable unless a test supplies one via ``_config``."""
    with patch.object(estimate, "_load_config", return_value=None):
        yield
    telemetry.uninstall()


def _config(cfg):
    """Make every checkpoint's HF config read as ``cfg``."""
    return patch.object(estimate, "_load_config", return_value=cfg)


@pytest.fixture()
def registered():
    """Register throwaway local models; clean up after.

    ``setup="introspect"`` registers a transformers model, anything else a
    vLLM one.
    """
    made = []

    def _make(name, setup="vllm", **kwargs):
        if setup == "introspect":
            register_model(name, backend_type="introspect",
                           introspect=IntrospectConfig(log_dir="x", **kwargs))
        else:
            register_model(name, backend_type="vllm", vllm=VLLMConfig(**kwargs))
        made.append(name)
        return name

    yield _make
    for n in made:
        MODEL_REGISTRY.pop(n, None)


def _capacity(total_gb, n_gpus, name="FakeGPU"):
    """Pin every capacity source, so the real machine never leaks into a test."""
    # nvidia-smi is asked first, so pinning it keeps torch out as well.
    return (
        patch("redact.llms.resources.measure.detect_gpu_memory_gib", return_value=total_gb),
        patch("redact.llms.resources.measure.detect_gpus", return_value=(name, n_gpus)),
    )


def _kv(gib):
    """Make every KV estimate read as ``gib`` per GPU."""
    return patch.object(estimate, "estimate_kv_gib", return_value=gib)


def _plan(models, total_gb, n_gpus, **kwargs):
    cap, gpus = _capacity(total_gb, n_gpus)
    with cap, gpus:
        return residency.plan_residency(models, **kwargs)


class TestReporting:
    """explain() is the pre-flight message — it has to be actionable, not just
    true."""

    def test_grant_check_applies_to_declared_weights(self, registered):
        m = registered("_res_grant_decl", hf_model_id="org/gd", vram_gb=40.0)
        with _kv(2.0):
            plan = _plan([m], 48.0, 1)
        # 40 declared + 2 KV, padded to 46.8, against the 43.2 GiB grant.
        assert "needs 46.8GiB with overhead (42.0 weights + KV, x1.1 +0.6) but " \
               "gpu_memory_utilization=0.90 grants 43.2GiB." in plan.problems[0]

    def test_grant_check_uses_the_padded_need(self, registered):
        """The grant has to cover (weights + KV) x 1.1 + 0.6, unrounded."""
        m = registered("_res_grant_est", hf_model_id="org/ge")
        with patch.object(estimate, "estimate_weights_gib", return_value=38.0), \
             patch.object(estimate, "estimate_kv_gib", return_value=0.5):
            fits_grant = _plan([m], 48.0, 1)          # 38.5 x 1.1 + 0.6 = 42.95 <= 43.2
        with patch.object(estimate, "estimate_weights_gib", return_value=38.0), \
             patch.object(estimate, "estimate_kv_gib", return_value=1.0):
            over_grant = _plan([m], 48.0, 1)          # 39 x 1.1 + 0.6 = 43.5 > 43.2
        fp = fits_grant.groups[0][0]
        assert (fp.need_gb, fp.gb) == (38.5, 43.0)    # gb is 42.95 rounded up
        assert fp.padded_gb == pytest.approx(42.95)
        assert fits_grant.fits and fits_grant.problems == []
        assert not over_grant.fits                    # the bare 39.0 would fit
        assert "needs 43.5GiB with overhead (39.0 weights + KV, x1.1 +0.6) but " \
               "gpu_memory_utilization=0.90 grants 43.2GiB." in over_grant.problems[0]

    def test_a_padded_need_equal_to_the_grant_fits(self, registered):
        """22 x 1.1 + 0.6 = 24.8 = 0.9 x 27.5556."""
        planned = registered("_res_edge_p", hf_model_id="org/ep", vram_gb=22.0)
        with _kv(0.0):
            at_ceiling = _plan([planned], 24.8 / 0.9, 1)
            over_ceiling = _plan([planned], 24.7 / 0.9, 1)
        assert at_ceiling.fits and not over_ceiling.fits
        assert at_ceiling.footprints[0].planned_utilization == 0.9

    def test_a_lone_engine_is_checked_at_the_card_ceiling(self, registered):
        m = registered("_res_grant_default", hf_model_id="org/gdf")
        with patch.object(estimate, "estimate_weights_gib", return_value=81.5), \
             patch.object(estimate, "estimate_kv_gib", return_value=0.0):
            plan = _plan([m], 100.0, 1)               # 81.5 x 1.1 + 0.6 = 90.25
        out = plan.explain()
        assert not plan.fits
        assert "gpu_memory_utilization=0.90 grants 90.0GiB" in out

    @pytest.mark.parametrize("card,grant,fits", [
        (8.0, 7.2, False), (24.0, 21.6, True), (80.0, 72.0, True)])
    def test_shipped_debug_model_on_three_cards(self, card, grant, fits):
        """6.0 GiB of weights + 0.04 GiB of KV at max_model_len=384 is 6.04;
        with overhead 6.04 x 1.1 + 0.6 = 7.244, over the 7.2 an 8 GiB card
        grants."""
        with _kv(0.04):
            plan = _plan(["llama-3.2-3b-debug"], card, 1)
        fp = plan.footprints[0]
        assert fp.need_gb == pytest.approx(6.04)
        assert fp.padded_gb == pytest.approx(7.244)
        assert fp.reserved_gb == pytest.approx(grant)
        assert fp.planned_utilization == 0.9
        assert plan.fits is fits and plan.warnings == []
        assert f"vLLM gets {grant}GiB (gpu_memory_utilization=0.90)" in plan.explain()
        assert "(declared: 6.0 weights + 0.04 KV, x1.1 +0.6)" in plan.explain()
        if not fits:
            assert "llama-3.2-3b-debug[vllm] needs 7.2GiB with overhead (6.0 " \
                   "weights + KV, x1.1 +0.6) but gpu_memory_utilization=0.90 " \
                   "grants 7.2GiB." in plan.problems[0]

    def test_an_unsized_kv_cache_is_reported(self):
        """No HF config: the need counts weights only, and the plan says so."""
        plan = _plan(["llama-3.2-3b-debug"], 8.0, 1)
        assert plan.footprints[0].need_gb == 6.0
        assert "KV cache of llama-3.2-3b-debug could not be estimated" in plan.warnings[0]
        assert "counts the weights only" in plan.explain()

    def test_a_kv_cache_larger_than_the_weights_is_pointed_out(self, registered):
        m = registered("_res_kv_big", hf_model_id="org/kb", vram_gb=6.0)
        with _kv(14.0):
            out = _plan([m], 48.0, 1).explain()
        assert "the KV cache of _res_kv_big (14.0GiB at the default context " \
               "of up to 10000 tokens) is larger than its weights (6.0GiB); " \
               "lower max_model_len to shrink it" in out

    def test_a_kv_cache_that_breaks_the_fit_is_named(self, registered):
        m = registered("_res_kv_over", hf_model_id="org/ko", vram_gb=6.0)
        with _kv(14.0):
            plan = _plan([m], 16.0, 1)                # 20.0 against 14.4
        assert not plan.fits
        assert "the KV cache (14.0GiB) is what does not fit: lower " \
               "max_model_len" in plan.problems[0]

    def test_oversized_model_gets_a_concrete_configuration(self, registered):
        m = registered("_res_suggest", hf_model_id="org/s", vram_gb=52.0)
        out = _plan([m], 48.0, 2).explain()
        assert "suggested: min_gpus=2 (tensor_parallel_size=2)" in out
        # Weights and KV halve: 26 x 1.1 + 0.6.
        assert "-> 29.2GiB per GPU with overhead" in out
        assert "~3x smaller at awq/gptq" in out

    def test_the_suggested_size_divides_the_need(self, registered):
        m = registered("_res_suggest_est", hf_model_id="org/se")
        with _config(MISTRAL_24B):
            plan = _plan([m], 48.0, 2)
            out = plan.explain()
        assert plan.groups[0][0].gb == 51.0          # KV at the default 10000 tokens
        # (43.91 + 1.53) / 2 x 1.1 + 0.6, against a 43.2 GiB grant per card.
        assert "suggested: min_gpus=2 (tensor_parallel_size=2) -> 25.6GiB per GPU" in out

    def test_suggested_size_divides_the_attention_heads(self, registered):
        """Qwen2.5-7B has 28 heads: 8 is not a valid tensor-parallel size."""
        m = registered("_res_suggest_heads", hf_model_id="org/sh")
        with _config(QWEN_7B):
            two_cards = _plan([m], 8.0, 4).explain()      # 2 too big, 4 fits
            tiny_cards = _plan([m], 4.0, 8).explain()     # only 8 would fit
        assert "suggested: min_gpus=4 (tensor_parallel_size=4)" in two_cards
        assert "suggested" not in tiny_cards
        assert "no tensor-parallel size up to 8 fits a 4.0GiB device" in tiny_cards

    def test_no_size_is_invented_when_none_fits(self, registered):
        m = registered("_res_huge", hf_model_id="org/h", vram_gb=500.0)
        out = _plan([m], 48.0, 2).explain()
        assert "no tensor-parallel size up to 8 fits a 48.0GiB device" in out
        assert "tensor_parallel_size=11" not in out

    def test_suggestion_skips_a_size_that_does_not_fit(self, registered):
        m = registered("_res_suggest_skip", hf_model_id="org/ss", vram_gb=98.0)
        out = _plan([m], 48.0, 2).explain()
        # 49.0 GiB per card at 2 pads to 54.5, over the 43.2 GiB grant; at 4
        # it is 24.5, padded to 27.55.
        assert "suggested: min_gpus=4 (tensor_parallel_size=4) -> 27.6GiB " \
               "per GPU with overhead  (more GPUs than detected)" in out



    def test_tensor_parallel_split_is_shown(self, registered):
        m = registered("_res_tp_report", hf_model_id="org/t", vram_gb=60.0,
                       min_gpus=2)
        plan = _plan([m], 48.0, 2)
        out = plan.explain()
        # 30 GiB of weights per card: 30 x 1.1 + 0.6 = 33.6, up to 34.0.
        assert "-> 2 GPUs (tensor_parallel_size=2), ~34.0GiB each" in out
        assert "-> cards 0-1, vLLM gets 43.2GiB each " \
               "(gpu_memory_utilization=0.90)" in out
        assert plan.fits and plan.footprints[0].planned_utilization == 0.9

    def test_report_logs_by_severity(self, registered, caplog):
        fine = registered("_res_log_ok", hf_model_id="org/lo", vram_gb=10.0)
        big = registered("_res_log_err", hf_model_id="org/le", vram_gb=60.0)
        # An unsized KV cache is the warning case.
        for model, kv, level in ((fine, 1.0, "INFO"), (fine, None, "WARNING"),
                                 (big, 1.0, "ERROR")):
            caplog.clear()
            with caplog.at_level(logging.INFO, logger=residency.logger.name), _kv(kv):
                residency.report(_plan([model], 48.0, 1))
            assert [r.levelname for r in caplog.records] == [level]


class TestMeasurement:
    """Post-hoc diagnostic only — nothing here feeds planning."""

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


class TestFootprintResolution:
    def test_declared_value_is_the_weights(self, registered):
        m = registered("_res_declared", hf_model_id="org/a", vram_gb=20.0)
        fp = residency.footprint(m)
        # 20 x 1.1 + 0.6 = 22.6, up to 23.0.
        assert (fp.gb, fp.source, fp.need_gb) == (23.0, "declared", 20.0)
        assert fp.breakdown() == "declared: 20.0 weights, x1.1 +0.6"

    def test_declared_weights_get_an_estimated_kv_cache(self, registered):
        """vram_gb overrides the weights only; the KV term is still estimated."""
        m = registered("_res_override", hf_model_id="org/b", vram_gb=20.0,
                       vllm_kwargs={"max_model_len": 8192})
        with _config(MISTRAL_24B), \
             patch.object(estimate, "estimate_weights_gib", return_value=99.0):
            fp = residency.footprint(m)
        assert (fp.source, fp.parts["weights_gib"], fp.parts["kv_gib"]) == (
            "declared", 20.0, 1.25)
        assert fp.need_gb == 21.25
        assert fp.gb == 24.0          # 21.25 x 1.1 + 0.6 = 23.98
        assert fp.breakdown() == "declared: 20.0 weights + 1.25 KV, x1.1 +0.6"

    def test_declared_weights_are_a_total_across_tensor_parallel_cards(self, registered):
        m = registered("_res_decl_tp", hf_model_id="org/dt", vram_gb=40.0, min_gpus=2)
        with _config(MISTRAL_24B):
            fp = residency.footprint(m)
        assert (fp.parts["weights_gib"], fp.parts["kv_gib"]) == (20.0, 0.76)
        assert fp.gb == 47.0          # (20.76 x 1.1 + 0.6 = 23.44 -> 23.5) x 2

    def test_a_pinned_transformers_model_is_not_split(self, registered):
        """device_map decides the span: 60 GiB pinned to one 48 GiB card."""
        m = registered("_res_intro_span", setup="introspect", hf_model_id="org/is",
                       vram_gb=60.0, device_map="cuda:0")
        plan = _plan([m], 48.0, 2)
        fp = plan.footprints[0]
        assert (fp.min_gpus, fp.per_gpu_gb, fp.devices) == (1, 67.0, (0,))
        assert not plan.fits
        out = plan.explain()
        assert "needs ~67.0GiB but its card has 48.0GiB." in out
        assert "max_model_len" not in out and ".api" not in out
        assert "never splits" not in out

    def test_estimated_when_nothing_is_declared(self, registered):
        """A newly registered model must plan correctly with nothing
        hand-derived."""
        m = registered("_res_estimated", hf_model_id="org/c")
        with patch("redact.llms.resources.estimate.estimate_weights_gib",
                   return_value=12.0), \
             patch("redact.llms.resources.estimate.estimate_kv_gib",
                   return_value=1.0):
            fp = residency.footprint(m)
        assert fp.source == "estimated"
        assert fp.gb == 15.0          # (12 + 1) x 1.1 + 0.6 = 14.9, up to 15.0
        assert fp.need_gb == 13.0
        assert fp.breakdown() == "estimated: 12.0 weights + 1.00 KV, x1.1 +0.6"

    def test_estimate_names_what_drives_it(self, registered):
        m = registered("_res_drivers", hf_model_id="org/dr", quantization="awq",
                       vllm_kwargs={"max_model_len": 8192})
        with _config(MISTRAL_24B):
            fp = residency.footprint(m)
        assert fp.breakdown() == (
            "estimated: 14.2 weights + 1.25 KV, x1.1 +0.6 [bfloat16, awq 4-bit]")
        assert fp.parts["attention_heads"] == 32

    def test_introspect_estimate_has_no_kv_term(self, registered):
        m = registered("_res_intro", setup="introspect", hf_model_id="org/in")
        with _config(MISTRAL_24B):
            fp = residency.footprint(m)
        assert fp.parts["kv_gib"] is None
        assert fp.gb == 49.0          # 43.91 x 1.1 + 0.6 = 48.9

    def test_estimate_failure_is_unknown_not_zero(self, registered):
        m = registered("_res_noconfig", hf_model_id="org/unreadable")
        with patch("redact.llms.resources.estimate.estimate_weights_gib",
                   return_value=None):
            fp = residency.footprint(m)
        assert fp.gb is None and fp.source == "unknown"

    def test_api_only_model_has_no_footprint(self):
        assert residency.footprint("claude-opus-4-6") is None

    def test_unknown_when_neither_declared_nor_estimable(self, registered):
        m = registered("_res_unknown", hf_model_id="org/d")
        fp = residency.footprint(m)
        assert fp.gb is None and fp.source == "unknown" and not fp.known
        assert fp.breakdown() == "unknown"

    def test_none_tensor_parallel_size_does_not_raise(self, registered):
        m = registered("_res_tp_none", hf_model_id="org/tn",
                       vllm_kwargs={"tensor_parallel_size": None})
        with _config(MISTRAL_24B):
            assert residency.footprint(m).gb == 51.0


class TestPlanResidency:
    def test_two_engines_share_a_card_when_their_grants_fit(self, registered):
        a = registered("_res_small_a", hf_model_id="org/sa", vram_gb=2.5)
        b = registered("_res_small_b", hf_model_id="org/sb", vram_gb=2.5)
        plan = _plan([a, b], 8.0, 1)
        assert len(plan.groups) == 1
        assert plan.gpus_required == 1 and plan.fits
        assert [fp.reserved_gb for fp in plan.footprints] == [pytest.approx(3.6)] * 2
        assert "vLLM gets 3.6GiB (gpu_memory_utilization=0.45)" in plan.explain()

    def test_a_lone_engine_gets_the_card_ceiling(self, registered):
        a = registered("_res_one", hf_model_id="org/one", vram_gb=20.0)
        with _kv(2.0):
            fp = _plan([a], 48.0, 1).footprints[0]
        assert fp.reserved_gb == pytest.approx(43.2)
        assert fp.planned_utilization == 0.9

    def test_two_engines_split_the_spare_equally(self, registered):
        """Pool 43.2; padded needs 24.8 + 12.7 leave 5.7, so each gets 2.85
        on top."""
        a = registered("_res_dflt_a", hf_model_id="org/da", vram_gb=20.0)
        b = registered("_res_dflt_b", hf_model_id="org/db", vram_gb=10.0)
        with patch.object(estimate, "estimate_kv_gib", side_effect=[2.0, 1.0]):
            plan = _plan([a, b], 48.0, 1)
        assert len(plan.groups) == 1 and plan.fits
        assert [fp.padded_gb for fp in plan.footprints] == [
            pytest.approx(24.8), pytest.approx(12.7)]
        assert [fp.reserved_gb for fp in plan.footprints] == [
            pytest.approx(27.65), pytest.approx(15.55)]
        assert [fp.planned_utilization for fp in plan.footprints] == [0.576, 0.3239]
        assert "(gpu_memory_utilization=0.58)" in plan.explain()

    def test_three_engines_split_the_spare_equally(self, registered):
        """Pool 72; padded needs 13.8 + 11.6 + 6.1 leave 40.5, so each gets
        13.5 on top."""
        names = [registered(f"_res_three_{i}", hf_model_id=f"org/th{i}", vram_gb=gb)
                 for i, gb in enumerate((12.0, 10.0, 5.0))]
        with _kv(0.0):
            plan = _plan(names, 80.0, 1)
        assert plan.fits
        assert [fp.reserved_gb for fp in plan.footprints] == [
            pytest.approx(27.3), pytest.approx(25.1), pytest.approx(19.6)]
        assert [fp.planned_utilization for fp in plan.footprints] == [
            0.3412, 0.3137, 0.245]

    def test_every_vllm_engine_gets_a_computed_fraction(self, registered):
        """A transformers model holds 12 of the card; pool 31.2, padded needs
        22.6 + 6.1 leave 2.5, so each engine gets 1.25 on top."""
        fixed = registered("_res_mix_x", setup="introspect", hf_model_id="org/mx",
                           vram_gb=10.0)
        a = registered("_res_mix_a", hf_model_id="org/ma", vram_gb=20.0)
        b = registered("_res_mix_b", hf_model_id="org/mb", vram_gb=5.0)
        with _kv(0.0):
            plan = _plan([fixed, a, b], 48.0, 1)
        assert plan.fits and len(plan.groups) == 1
        assert [fp.reserved_gb for fp in plan.footprints] == [
            pytest.approx(12.0), pytest.approx(23.85), pytest.approx(7.35)]
        assert [fp.planned_utilization for fp in plan.footprints] == [
            None, 0.4968, 0.1531]
        assert [m["gpu_memory_utilization"] for m in plan.as_dict()["models"]] == [
            None, 0.4968, 0.1531]

    def test_engines_whose_needs_exceed_the_ceiling_do_not_share(self, registered):
        """22 + 22 is over 0.9 x 48, though under the card."""
        a = registered("_res_over_a", hf_model_id="org/oa", vram_gb=22.0)
        b = registered("_res_over_b", hf_model_id="org/ob", vram_gb=22.0)
        plan = _plan([a, b], 48.0, 1)
        assert len(plan.groups) == 2 and not plan.fits
        assert "needs with overhead sum to at most 0.9 of it" in plan.explain()

    def test_models_that_cannot_be_resident_together_do_not_fit(self, registered):
        """Nothing is unloaded during a run, so a second group is not a
        schedule: it would load on top of the first."""
        a = registered("_res_big_a", hf_model_id="org/ba", vram_gb=20.0)
        b = registered("_res_big_b", hf_model_id="org/bb", vram_gb=20.0)
        plan = _plan([a, b], 24.0, 1)
        out = plan.explain()
        assert len(plan.groups) == 2
        assert plan.sequential and not plan.co_resident and not plan.fits
        assert "group 2:  -> cannot be resident with the above" in out
        assert "unload between" not in out
        assert "separate run_pipeline invocations" in out
        assert "process exit frees the card" in out

    def test_single_gpu_vllm_engines_never_spread_across_cards(self, registered):
        """Nothing assigns a device at load time, so a second card adds no
        room: three 15 GiB engines on 2 x 24 GiB do not fit."""
        names = [registered(f"_res_c0_{i}", hf_model_id=f"org/c0{i}", vram_gb=15.0)
                 for i in range(3)]
        plan = _plan(names, 24.0, 2)
        assert len(plan.groups) == 3 and not plan.fits
        assert [fp.devices for fp in plan.footprints] == [(0,)] * 3
        assert plan.gpus_required == 1
        assert "single-GPU vLLM engines all load on card 0" in plan.explain()

    def test_a_single_card_plan_does_not_mention_other_cards(self, registered):
        m = registered("_res_one_card", hf_model_id="org/oc", vram_gb=10.0)
        assert "card 0, since" not in _plan([m], 24.0, 1).explain()

    def test_transformers_models_are_packed_per_card(self, registered):
        """device_map='auto' may split a model, so three 15 GiB models fit
        2 x 24 GiB and a fourth does not."""
        # 13 GiB of weights: 13 x 1.1 + 0.6 = 14.9, up to 15.0.
        names = [registered(f"_res_tf_{i}", setup="introspect",
                            hf_model_id=f"org/tf{i}", vram_gb=13.0)
                 for i in range(4)]
        three = _plan(names[:3], 24.0, 2)
        assert len(three.groups) == 1 and three.fits and three.gpus_required == 2
        assert [fp.devices for fp in three.footprints] == [(0,), (0, 1), (1,)]
        four = _plan(names, 24.0, 2)
        assert len(four.groups) == 2 and not four.fits

    def test_a_pinned_transformers_model_stays_on_its_card(self, registered):
        a = registered("_res_pin_a", setup="introspect", hf_model_id="org/pa",
                       vram_gb=15.0, device_map="cuda:1")
        b = registered("_res_pin_b", setup="introspect", hf_model_id="org/pb",
                       vram_gb=15.0, device_map="cuda:1")
        plan = _plan([a, b], 24.0, 2)
        assert plan.footprints[0].devices == (1,)
        assert len(plan.groups) == 2          # 30 GiB on one 24 GiB card

    def test_models_sharing_a_checkpoint_are_one_load(self, registered):
        # venice-uncensored[vllm] and venice-paraphraser do exactly this. Two
        # rows, one engine — counting it twice would split a plan that fits.
        a = registered("_res_shared_a", hf_model_id="org/same", vram_gb=18.0)
        b = registered("_res_shared_b", hf_model_id="org/same", vram_gb=18.0)
        plan = _plan([a, b], 24.0, 1)
        assert len(plan.groups) == 1
        assert plan.gpus_required == 1 and plan.fits
        assert plan.footprints[1].devices == (0,)

    def test_multi_gpu_model_claims_whole_devices(self, registered):
        # The locally-served-translator case: too big for one card, so it takes
        # two outright rather than sharing the leftover GB on card 0.
        big = registered("_res_tp2", hf_model_id="org/huge", vram_gb=70.0, min_gpus=2)
        plan = _plan([big], 80.0, 1)
        assert plan.gpus_required == 2
        assert not plan.fits
        assert "needs 2 GPU(s)" in plan.explain()

    def test_a_single_gpu_engine_cannot_join_a_tensor_parallel_one(self, registered):
        """The tensor-parallel engine reserves card 0 too."""
        big = registered("_res_tp_first", hf_model_id="org/tf", vram_gb=70.0, min_gpus=2)
        small = registered("_res_tp_then", hf_model_id="org/tt", vram_gb=5.0)
        plan = _plan([big, small], 80.0, 2)
        assert len(plan.groups) == 2 and not plan.fits

    def test_unknown_capacity_does_not_invent_a_sequential_plan(self, registered):
        a = registered("_res_nocap_a", hf_model_id="org/na", vram_gb=20.0)
        b = registered("_res_nocap_b", hf_model_id="org/nb", vram_gb=20.0)
        # Both capacity sources blind: no torch, and no nvidia-smi either.
        with patch("redact.llms.resources.measure.free_total_gib", return_value=None):
            with patch("redact.llms.resources.measure.detect_gpus", return_value=(None, 0)):
                with patch("redact.llms.resources.measure.detect_gpu_memory_gib", return_value=None):
                    plan = residency.plan_residency([a, b])
        assert len(plan.groups) == 1          # grouped, not falsely split
        assert not plan.sequential

    def test_unknown_footprints_are_flagged_in_the_explanation(self, registered):
        a = registered("_res_noft", hf_model_id="org/nf")
        assert "no footprint for" in _plan([a], 24.0, 1).explain()

    def test_unknown_vllm_models_share_the_card_and_are_flagged(self, registered):
        a = registered("_res_unk_a", hf_model_id="org/ua")
        b = registered("_res_unk_b", hf_model_id="org/ub")
        plan = _plan([a, b], 24.0, 1)
        assert len(plan.groups) == 1
        assert [fp.planned_utilization for fp in plan.footprints] == [0.45, 0.45]
        assert "no footprint for _res_unk_a, _res_unk_b" in plan.explain()

    def test_a_new_group_starts_with_empty_cards(self, registered):
        a = registered("_res_grp_a", hf_model_id="org/ga", vram_gb=20.0)
        b = registered("_res_grp_b", hf_model_id="org/gb", vram_gb=3.0)
        c = registered("_res_grp_c", hf_model_id="org/gc", vram_gb=3.0)
        plan = _plan([a, b, c], 24.0, 1)
        assert [[fp.model for fp in g] for g in plan.groups] == [[a], [b, c]]

    def test_too_few_gpus_is_a_named_problem(self, registered):
        big = registered("_res_tp_short", hf_model_id="org/ts", vram_gb=70.0, min_gpus=2)
        plan = _plan([big], 80.0, 1)
        assert not plan.fits
        assert any("needs 2 GPUs but 1 are available" in p for p in plan.problems)
        assert "PROBLEM: the plan needs 2 GPUs" in plan.explain()

    def test_no_gpu_is_a_named_problem(self, registered):
        a = registered("_res_nogpu", hf_model_id="org/ng", vram_gb=5.0)
        plan = residency.plan_residency([a], gpus=0, gib_per_gpu=24.0)
        assert not plan.fits
        assert any("no GPU is available" in p for p in plan.problems)

    def test_unknown_card_size_is_flagged(self, registered):
        a = registered("_res_nosize", hf_model_id="org/ns", vram_gb=40.0)
        with patch("redact.llms.resources.measure.free_total_gib", return_value=None):
            with patch("redact.llms.resources.measure.detect_gpus", return_value=("FakeGPU", 2)):
                with patch("redact.llms.resources.measure.detect_gpu_memory_gib", return_value=None):
                    plan = residency.plan_residency([a])
        assert any("card size is unknown" in w for w in plan.warnings)
        assert "card size is unknown" in plan.explain()


class TestPlanForAGivenMachine:
    def test_plans_for_the_machine_given_without_detecting(self, registered):
        m = registered("_res_given", hf_model_id="org/gv", vram_gb=50.0)
        with patch("redact.llms.resources.measure.detect_gpus", side_effect=AssertionError), \
             patch("redact.llms.resources.measure.free_total_gib",
                   side_effect=AssertionError):
            small = residency.plan_residency([m], gpus=1, gib_per_gpu=24.0)
            pod = residency.plan_residency([m], gpus=1, gib_per_gpu=80.0)
        assert not small.fits and pod.fits
        assert pod.machine_given and pod.gpu_name is None
        assert "planning for 1 GPU(s), 80.0GiB each" in pod.explain()

    def test_card_size_alone_keeps_the_detected_count(self, registered):
        m = registered("_res_given_size", hf_model_id="org/gs", vram_gb=50.0)
        with patch("redact.llms.resources.measure.detect_gpus", return_value=("FakeGPU", 2)), \
             patch("redact.llms.resources.measure.free_total_gib",
                   side_effect=AssertionError):
            plan = residency.plan_residency([m], gib_per_gpu=80.0)
        assert (plan.gpus_available, plan.per_gpu_gb) == (2, 80.0)


class TestAsDict:
    def test_carries_what_the_explanation_does(self, registered):
        ok = registered("_res_dict_ok", hf_model_id="org/do", vram_gb=10.0)
        big = registered("_res_dict_big", hf_model_id="org/dbg", vram_gb=60.0)
        with _kv(1.0):
            plan = _plan([ok, big], 48.0, 1)
        data = json.loads(json.dumps(plan.as_dict()))     # plain data
        assert data["fits"] is False and data["co_resident"] is False
        assert data["groups"] == [[ok], [big]]
        assert data["oversized"] == [big]
        assert (data["gpus_available"], data["per_gpu_gib"]) == (1, 48.0)
        assert data["machine"] == "detected"
        assert len(data["problems"]) == 2 and data["warnings"] == []
        first, second = data["models"]
        assert first == {
            "model": ok, "setup": "vllm", "hf_model_id": "org/do", "gib": 13.0,
            "source": "declared",
            "derivation": "declared: 10.0 weights + 1.00 KV, x1.1 +0.6",
            "min_gpus": 1, "devices": [0],
            "reserved_gib_per_device": pytest.approx(43.2),
            "need_gib_per_device": 11.0,
            "padded_gib_per_device": pytest.approx(12.7),
            "gpu_memory_utilization": 0.9,
            "group": 0,
        }
        assert second["group"] == 1
        assert second["gpu_memory_utilization"] == 0.9

    def test_the_residency_event_carries_the_plan(self, registered, tmp_path):
        m = registered("_res_event", hf_model_id="org/ev", vram_gb=20.0)
        telemetry.install(data_dir=tmp_path)
        residency.report(_plan([m], 48.0, 1), verbose=False)
        event = json.loads(telemetry.collector().trace_path.read_text().splitlines()[0])
        assert event["ev"] == "residency" and event["fits"] is True
        assert event["groups"] == [[m]]
        assert event["models"][0]["devices"] == [0]


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

    def test_loads_in_the_order_given_and_skips_api_models(self, registered):
        a = registered("_pre_first", hf_model_id="org/p1")
        b = registered("_pre_second", hf_model_id="org/p2")
        seen = []
        with patch("redact.llms.ModelClient.create", side_effect=lambda n, **kw: seen.append(n)):
            residency.preload([b, "claude-opus-4-6", a, b]).join(timeout=5)
        assert seen == [b, a]

    def test_locality_is_decided_without_estimating(self, registered):
        """Asking "is this local" must not read an HF config."""
        m = registered("_pre_no_estimate", hf_model_id="org/pne")
        with patch.object(estimate, "estimate", side_effect=AssertionError), \
             patch("redact.llms.ModelClient.create"):
            residency.preload([m]).join(timeout=5)

    def test_a_failed_preload_never_aborts_the_run(self, registered, tmp_path, caplog):
        # The whole point: constitution is ledger-backed and its Opus output is
        # saved as it goes, so a GPU that won't load must not kill it. The stage
        # that needs the model fails later, at the point of use.
        m = registered("_pre_boom", hf_model_id="org/boom", vram_gb=1.0)
        telemetry.install(data_dir=tmp_path)
        with patch("redact.llms.ModelClient.create", side_effect=RuntimeError("no CUDA")):
            with caplog.at_level(logging.ERROR, logger=residency.logger.name):
                thread = residency.preload([m])
                thread.join(timeout=5)          # returns normally, does not raise

        assert "no CUDA" in caplog.text
        assert "run continues" in caplog.text
        events = json.loads(telemetry.collector().trace_path.read_text().splitlines()[0])
        assert events["ev"] == "preload_failed" and events["model"] == m


class TestOversizedModels:
    """One model bigger than its cards is a config problem, not a packing result."""

    def test_model_larger_than_a_device_is_reported_as_unplaceable(self, registered):
        big = registered("_res_oversized", hf_model_id="org/huge", vram_gb=50.0)
        plan = _plan([big], 8.0, 1)
        assert not plan.fits
        assert [fp.model for fp in plan.oversized] == [big]
        assert "needs 55.6GiB with overhead (50.0 weights + KV, x1.1 +0.6) but " \
               "gpu_memory_utilization=0.90 grants 7.2GiB." in plan.explain()
        # 6.25 per card at 8 pads to 7.475, over the 7.2 a card grants.
        assert "no tensor-parallel size up to 8 fits a 8.0GiB device" in plan.explain()

    def test_an_oversized_model_is_one_problem(self, registered):
        big = registered("_res_oversized_once", hf_model_id="org/h1", vram_gb=50.0)
        with _kv(1.0):
            plan = _plan([big], 48.0, 1)
        assert len(plan.problems) == 1 and plan.warnings == []

    def test_declaring_min_gpus_clears_the_problem(self, registered):
        big = registered("_res_tp_ok", hf_model_id="org/huge2", vram_gb=50.0, min_gpus=8)
        plan = _plan([big], 10.0, 8)           # 6.25 pads to 7.475 of the 9.0
        assert plan.oversized == []
        assert plan.gpus_required == 8 and plan.fits

    def test_tensor_parallel_model_larger_than_its_cards_is_flagged(self, registered):
        """60 GiB on each of two 48 GiB cards does not fit either."""
        big = registered("_res_tp_over", hf_model_id="org/tpo", vram_gb=120.0, min_gpus=2)
        plan = _plan([big], 48.0, 2)
        out = plan.explain()
        assert not plan.fits and [fp.model for fp in plan.oversized] == [big]
        assert "needs 66.6GiB per GPU with overhead (60.0 weights + KV, x1.1 " \
               "+0.6) but gpu_memory_utilization=0.90 grants 43.2GiB." in out
        assert "suggested: min_gpus=4 (tensor_parallel_size=4) -> 33.6GiB per GPU" in out

    def test_a_spanning_transformers_model_may_exceed_one_card(self, registered):
        m = registered("_res_span", setup="introspect", hf_model_id="org/sp", vram_gb=40.0)
        assert _plan([m], 24.0, 2).fits
        assert not _plan([m], 24.0, 1).fits

    def test_a_model_that_fits_is_not_flagged(self, registered):
        ok = registered("_res_fits", hf_model_id="org/small", vram_gb=5.0)
        plan = _plan([ok], 8.0, 1)
        assert plan.oversized == [] and plan.fits


class TestCapacityDetection:
    def test_nvidia_smi_supplies_capacity_without_touching_torch(self, registered):
        """Asking torch would start CUDA in this process before vLLM loads."""
        m = registered("_res_smi", hf_model_id="org/smi", vram_gb=50.0)
        with patch("redact.llms.resources.measure.free_total_gib",
                   side_effect=AssertionError("torch probed")):
            with patch("redact.llms.resources.measure.detect_gpus", return_value=("FakeGPU", 1)):
                with patch("redact.llms.resources.measure.detect_gpu_memory_gib", return_value=8.0):
                    plan = residency.plan_residency([m])
        assert plan.per_gpu_gb == 8.0
        assert not plan.fits          # 50GB on an 8GB card: correctly refused

    def test_torch_supplies_capacity_when_nvidia_smi_cannot(self, registered):
        m = registered("_res_both", hf_model_id="org/both", vram_gb=1.0)
        with patch("redact.llms.resources.measure.free_total_gib", return_value=(20.0, 24.0)):
            with patch("redact.llms.resources.measure.detect_gpus", return_value=("FakeGPU", 1)):
                with patch("redact.llms.resources.measure.detect_gpu_memory_gib", return_value=None):
                    plan = residency.plan_residency([m])
        assert plan.per_gpu_gb == 24.0

    def test_an_api_only_plan_never_probes_capacity(self):
        """The torch probe imports torch; an API-only run must not pay for it."""
        with patch("redact.llms.resources.measure.free_total_gib",
                   side_effect=AssertionError("probed")), \
             patch("redact.llms.resources.measure.detect_gpu_memory_gib",
                   side_effect=AssertionError("probed")), \
             patch("redact.llms.resources.measure.detect_gpus", return_value=("FakeGPU", 1)):
            plan = residency.plan_residency(["claude-opus-4-6", "deepseek-v3.2"])
        assert plan.groups == [] and plan.per_gpu_gb is None
        assert plan.gpus_available == 1


class TestRegistryFootprints:
    """The shipped local entries declare a footprint, so planning works
    without reading any HF config."""

    @pytest.mark.parametrize("model,backend_type", [
        ("venice-uncensored", "vllm"),
        ("venice-paraphraser", None),
        ("llama-3.2-3b-debug", None),
    ])
    def test_every_local_model_declares_a_footprint(self, model, backend_type):
        fp = residency.footprint(model, backend_type)
        assert fp is not None and fp.known, f"{model} has no vram_gb"
        assert fp.source == "declared"

    def test_the_shared_checkpoint_declares_the_same_footprint(self):
        a = residency.footprint("venice-uncensored", "vllm")
        b = residency.footprint("venice-paraphraser")
        assert a.hf_model_id == b.hf_model_id
        assert a.gb == b.gb        # one engine, one footprint


def test_estimated_tensor_parallel_footprint_is_a_total_like_a_declared_one():
    name = "_res_tp_estimated"
    register_model(name, backend_type="vllm", vllm=VLLMConfig(
        hf_model_id="org/tp", min_gpus=2))
    try:
        with patch("redact.llms.resources.estimate.estimate_weights_gib",
                   return_value=30.0), \
             patch("redact.llms.resources.estimate.estimate_kv_gib", return_value=0.0), \
             patch("redact.llms.resources.measure.detect_gpu_memory_gib", return_value=36.0), \
             patch("redact.llms.resources.measure.detect_gpus", return_value=("FakeGPU", 2)):
            fp = residency.footprint(name)
            out = residency.plan_residency([name]).explain()
    finally:
        MODEL_REGISTRY.pop(name, None)
    assert fp.gb == 68.0            # (30 + 0) * 1.1 + 0.6 -> 34.0 per GPU, x2
    assert "~34.0GiB each" in out
    # The grant has to cover the 30 GiB of weights with overhead: 33.6.
    assert "needs 33.6GiB per GPU with overhead (30.0 weights + KV, x1.1 +0.6) " \
           "but gpu_memory_utilization=0.90 grants 32.4GiB" in out


def test_same_checkpoint_with_different_engine_settings_is_two_loads():
    names = ("_res_same_a", "_res_same_b")
    for name, length in zip(names, (4096, 8192)):
        register_model(name, backend_type="vllm", vllm=VLLMConfig(
            hf_model_id="org/same", vram_gb=20.0, vllm_kwargs={"max_model_len": length}))
    try:
        with patch("redact.llms.resources.measure.detect_gpu_memory_gib", return_value=24.0), \
             patch("redact.llms.resources.measure.detect_gpus", return_value=("FakeGPU", 1)):
            plan = residency.plan_residency(list(names))
    finally:
        for name in names:
            MODEL_REGISTRY.pop(name, None)
    assert len(plan.groups) == 2    # 2 x 20GiB on one 24GiB card: one after the other


def test_planner_engine_id_matches_the_vllm_cache_key():
    import redact.llms.backends.vllm as vllm_module

    cfg = VLLMConfig(hf_model_id="org/m", quantization="awq", min_gpus=2,
                     vllm_kwargs={"max_model_len": 4096})
    assert _engine_id(cfg, "vllm")[1:] == vllm_module._engine_key(
        cfg.hf_model_id, cfg.quantization, cfg.engine_kwargs)


class TestUnloadLocal:
    @pytest.fixture()
    def events(self):
        seen = []
        observe.set_emitter(seen.append)
        yield seen
        observe.set_emitter(None)

    def test_kept_vllm_engine_stays_cached_and_timed(self, registered, events):
        import redact.llms.backends.vllm as vllm_module

        kept = registered("_unload_keep", hf_model_id="org/keep")
        dropped = registered("_unload_drop", hf_model_id="org/drop")
        key = {n: vllm_module._engine_key(f"org/{n.split('_')[-1]}", None, {})
               for n in (kept, dropped)}
        vllm_module._engines.clear()
        vllm_module._engine_loaded_at.clear()
        try:
            for k in key.values():
                vllm_module._engines[k] = object()
                vllm_module._engine_loaded_at[k] = time.perf_counter()
            engine = vllm_module._engines[key[kept]]

            residency.unload_local(keep=[kept])

            assert list(vllm_module._engines) == [key[kept]]
            assert vllm_module._engines[key[kept]] is engine
            assert list(vllm_module._engine_loaded_at) == [key[kept]]   # meter runs on
            released = [e["hf_model_id"] for e in events if e.get("phase") == "release"]
            assert released == ["org/drop"]

            residency.unload_local()
            assert vllm_module._engines == {} and vllm_module._engine_loaded_at == {}
            released = [e["hf_model_id"] for e in events if e.get("phase") == "release"]
            assert released == ["org/drop", "org/keep"]
        finally:
            vllm_module._engines.clear()
            vllm_module._engine_loaded_at.clear()

    def test_kept_transformers_model_stays_cached_and_timed(self, registered, events):
        import redact.llms.backends.introspection as intro

        kept = registered("_unload_keep_tf", setup="introspect", hf_model_id="org/ktf")
        keep_key = ("org/ktf", "auto", "auto", "[]")
        drop_key = ("org/dtf", "auto", "auto", "[]")
        intro._models.clear()
        intro._models_loaded_at.clear()
        try:
            for k in (keep_key, drop_key):
                intro._models[k] = ("tok", "model")
                intro._models_loaded_at[k] = time.perf_counter()

            residency.unload_local(keep=[kept, "claude-opus-4-6", "not-registered"])

            assert list(intro._models) == [keep_key]
            assert list(intro._models_loaded_at) == [keep_key]
            released = [e["hf_model_id"] for e in events if e.get("phase") == "release"]
            assert released == ["org/dtf"]
        finally:
            intro._models.clear()
            intro._models_loaded_at.clear()

    def test_a_model_kept_under_a_non_default_setup_keeps_its_engine(self, events):
        """venice-uncensored defaults to its API setup but also runs on vLLM."""
        import redact.llms.backends.vllm as vllm_module

        key = vllm_module._engine_key("dphn/Dolphin-Mistral-24B-Venice-Edition", None, {})
        vllm_module._engines.clear()
        try:
            vllm_module._engines[key] = object()
            residency.unload_local(keep=["venice-uncensored"])
            assert list(vllm_module._engines) == [key]
        finally:
            vllm_module._engines.clear()
            vllm_module._engine_loaded_at.clear()

    def test_clear_cache_waits_for_a_load_in_progress(self):
        import redact.llms.backends.introspection as intro
        import redact.llms.backends.vllm as vllm_module

        for lock, backend in ((vllm_module._engines_lock, vllm_module.VLLMBackend),
                              (intro._models_lock,
                               intro.TransformersIntrospectionBackend)):
            done = threading.Event()
            with lock:
                thread = threading.Thread(
                    target=lambda: (backend.clear_cache(), done.set()))
                thread.start()
                assert not done.wait(0.1)
            thread.join(timeout=5)
            assert done.is_set()

    def test_api_connection_pools_are_left_alone(self):
        from redact.llms.backends.openai import OpenAIBackend

        with patch.object(OpenAIBackend, "clear_cache",
                          side_effect=AssertionError("cleared")):
            residency.unload_local()


def test_unload_local_keeps_rate_limit_windows():
    from redact.llms import wrappers
    from redact.llms.client import clear_client_cache

    window = wrappers.shared_limiter("_test_unload_key")
    try:
        residency.unload_local()
        assert wrappers.shared_limiter("_test_unload_key") is window
    finally:
        clear_client_cache()


_HI = [[{"role": "user", "content": "hi"}]]


class TestRelease:
    """A released local model is shut down, its memory is freed and nothing
    refers to it any longer. A kept one is untouched."""

    @pytest.fixture()
    def runtime(self):
        """A fake ``vllm`` and ``torch``; yields the modules and what they saw."""
        import redact.llms.backends.introspection as intro
        import redact.llms.backends.vllm as vllm_module

        log = SimpleNamespace(loads=[], shutdowns=[], emptied=[], events=[])

        class _LLM:
            hook = True     # False: a vLLM version with no shutdown hook

            def __init__(self, **kw):
                model = kw["model"]
                log.loads.append(model)
                if self.hook:
                    # vLLM's V1 layout: the hook is on the engine core.
                    self.llm_engine = SimpleNamespace(engine_core=SimpleNamespace(
                        shutdown=lambda: log.shutdowns.append(model)))

            def chat(self, chat_inputs, sampling_params):
                return [SimpleNamespace(prompt_token_ids=[1], outputs=[
                    SimpleNamespace(text="ok", token_ids=[1])]) for _ in chat_inputs]

        torch = SimpleNamespace(cuda=SimpleNamespace(
            is_initialized=lambda: True, empty_cache=lambda: log.emptied.append(1)))
        fake_vllm = SimpleNamespace(LLM=_LLM, SamplingParams=lambda **kw: kw)
        vllm_module.VLLMBackend.clear_cache()
        intro.TransformersIntrospectionBackend.clear_cache()
        observe.set_emitter(log.events.append)
        with patch.dict("sys.modules", {"vllm": fake_vllm, "torch": torch}), \
             patch.object(vllm_module, "_prepare_environment"):
            yield SimpleNamespace(vllm=vllm_module, intro=intro, LLM=_LLM, log=log)
            vllm_module.VLLMBackend.clear_cache()
            intro.TransformersIntrospectionBackend.clear_cache()
        observe.set_emitter(None)

    @staticmethod
    def _backend(model):
        from redact.llms.backends import backend_for
        from redact.llms.model_config import get_model_config

        return backend_for(get_model_config(model))

    @staticmethod
    def _released(log):
        return [e["hf_model_id"] for e in log.events if e.get("phase") == "release"]

    def test_a_released_engine_is_shut_down_and_unreferenced(self, runtime, registered):
        backend = self._backend(registered("_rel_gone", hf_model_id="org/gone"))
        engine = weakref.ref(backend._llm)

        residency.unload_local()

        # The backend is still alive here, and must not keep the engine alive.
        assert engine() is None
        assert runtime.log.shutdowns == ["org/gone"]
        assert runtime.log.emptied == [1]
        assert runtime.vllm._engines == {}

    def test_an_engine_without_a_shutdown_hook_is_still_dropped(
            self, runtime, registered, caplog):
        from redact.llms.backends import clear_transport_caches

        runtime.LLM.hook = False
        backend = self._backend(registered("_rel_nohook", hf_model_id="org/nohook"))
        engine = weakref.ref(backend._llm)
        with caplog.at_level(logging.DEBUG, logger=runtime.vllm.logger.name):
            clear_transport_caches()
        assert engine() is None
        assert runtime.log.shutdowns == [] and runtime.log.emptied == [1]
        assert "_LLM has no shutdown hook" in caplog.text

    def test_shutdown_calls_the_outermost_hook_and_falls_back_when_it_fails(
            self, runtime):
        called = []

        def hook(name, fail=False):
            def _hook():
                called.append(name)
                if fail:
                    raise RuntimeError("already stopped")
            return _hook

        engine = SimpleNamespace(
            shutdown=hook("engine", fail=True),
            engine_core=SimpleNamespace(shutdown=hook("core")),
            model_executor=SimpleNamespace(shutdown=hook("executor")),
        )
        runtime.vllm._shut_down(SimpleNamespace(llm_engine=engine))
        assert called == ["engine", "core"]
        called.clear()
        runtime.vllm._shut_down(
            SimpleNamespace(shutdown=hook("llm"), llm_engine=engine))
        assert called == ["llm"]
        runtime.vllm._shut_down(object())       # no hook: nothing raised

    def test_a_kept_engine_is_untouched(self, runtime, registered):
        kept = self._backend(registered("_rel_keep", hf_model_id="org/keep"))
        self._backend(registered("_rel_drop", hf_model_id="org/drop"))
        engine = kept._llm

        residency.unload_local(keep=["_rel_keep"])

        assert runtime.log.shutdowns == ["org/drop"]
        assert kept._llm is engine
        assert kept.generate(_HI) == ["ok"]
        # The event, the meter and the cache name the same engines.
        assert self._released(runtime.log) == ["org/drop"]
        assert list(runtime.vllm.held_seconds()) == ["org/keep"]
        assert [key[0] for key in runtime.vllm._engines] == ["org/keep"]

    def test_a_second_release_does_nothing(self, runtime, registered):
        self._backend(registered("_rel_once", hf_model_id="org/once"))
        residency.unload_local()
        residency.unload_local()
        assert runtime.log.shutdowns == ["org/once"]
        assert runtime.log.emptied == [1]
        assert self._released(runtime.log) == ["org/once"]

    def test_a_backend_on_a_released_engine_raises_and_does_not_reload(
            self, runtime, registered):
        name = registered("_rel_stale", hf_model_id="org/stale")
        backend = self._backend(name)
        residency.unload_local()

        with pytest.raises(RuntimeError, match="was released"):
            backend.generate(_HI)
        assert runtime.log.loads == ["org/stale"]
        assert [e for e in runtime.log.events if e["ev"] == "call"] == []

        self._backend(name)                     # a new client loads it again
        assert backend.generate(_HI) == ["ok"]
        assert runtime.log.loads == ["org/stale", "org/stale"]

    def test_a_release_waits_for_a_load_in_progress(self, runtime):
        loading, proceed = threading.Event(), threading.Event()
        load = runtime.LLM.__init__

        def slow_load(self, **kw):
            loading.set()
            assert proceed.wait(5)
            load(self, **kw)

        runtime.LLM.__init__ = slow_load
        released = threading.Event()
        loader = threading.Thread(
            target=lambda: runtime.vllm._engine("org/slow", None, {}))
        releaser = threading.Thread(
            target=lambda: (runtime.vllm.VLLMBackend.clear_cache(), released.set()))
        loader.start()
        assert loading.wait(5)
        releaser.start()
        assert not released.wait(0.1)           # blocked behind the load
        proceed.set()
        loader.join(5)
        releaser.join(5)

        assert released.is_set()                # no deadlock
        assert runtime.log.loads == ["org/slow"]        # loaded once ...
        assert runtime.log.shutdowns == ["org/slow"]    # ... and then released
        assert runtime.vllm._engines == {}

    def test_a_released_transformers_model_is_freed(self, runtime):
        intro = runtime.intro

        class _Part:
            pass

        key = intro._model_key("org/tf", "auto", "auto", {})
        kept_key = intro._model_key("org/tf-kept", "auto", "auto", {})
        tokenizer, model = _Part(), _Part()
        refs = [weakref.ref(tokenizer), weakref.ref(model)]
        intro._models[key] = (tokenizer, model)
        intro._models[kept_key] = kept = (_Part(), _Part())
        intro._models_loaded_at[key] = intro._models_loaded_at[kept_key] = 0.0
        backend = object.__new__(intro.TransformersIntrospectionBackend)
        backend._key, backend.model, backend.hf_model_id = key, "m", "org/tf"
        assert backend._loaded()[1] is model
        del tokenizer, model

        intro.TransformersIntrospectionBackend.clear_cache(keep=[kept_key])

        assert [ref() for ref in refs] == [None, None]
        assert runtime.log.emptied == [1]
        assert intro._models == {kept_key: kept}
        assert self._released(runtime.log) == ["org/tf"]
        assert list(intro.held_seconds()) == ["org/tf-kept"]
        with pytest.raises(RuntimeError, match="was released"):
            backend._loaded()

    def test_keep_is_a_collection_of_keys_not_one_key(self, runtime):
        for module, backend, key in (
            (runtime.vllm, runtime.vllm.VLLMBackend, ("org/x", None, "[]")),
            (runtime.intro, runtime.intro.TransformersIntrospectionBackend,
             ("org/x", "auto", "auto", "[]")),
        ):
            cache = module._engines if module is runtime.vllm else module._models
            cache[key] = loaded = (object(), object())
            with pytest.raises(TypeError, match="collection of"):
                backend.clear_cache(keep=key)
            backend.clear_cache(keep=[key])
            assert cache == {key: loaded}       # neither call released it

    def test_the_summary_counts_a_released_engine_once_and_a_kept_one(
            self, runtime, registered, tmp_path):
        for name, hf_id in (("_rel_sum_keep", "org/sk"), ("_rel_sum_drop", "org/sd")):
            registered(name, hf_model_id=hf_id)
        telemetry.install(data_dir=tmp_path)
        now = [100.0]
        with patch("time.perf_counter", lambda: now[0]), \
             patch("redact.llms.resources.measure.detect_gpus", return_value=(None, 0)):
            self._backend("_rel_sum_keep")
            self._backend("_rel_sum_drop")
            now[0] = 160.0
            residency.unload_local(keep=["_rel_sum_keep"])
            now[0] = 200.0
            for _ in range(2):                  # reading it twice adds nothing
                assert telemetry.summary()["local_s"] == {
                    "org/sd": 60.0, "org/sk": 100.0}
            assert runtime.vllm.held_seconds() == {"org/sk": 100.0}
            residency.unload_local()
            assert telemetry.summary()["local_s"] == {"org/sd": 60.0, "org/sk": 100.0}
            assert runtime.vllm.held_seconds() == {}


class TestPlannedUtilization:
    """The plan's computed gpu_memory_utilization reaches ``vllm.LLM``."""

    @pytest.fixture()
    def engine(self):
        """Load through a fake ``vllm.LLM``; yields ``(module, constructor calls)``."""
        import redact.llms.backends.vllm as vllm_module

        calls = []

        class _LLM:
            def __init__(self, **kw):
                time.sleep(0.05)
                calls.append(kw)

        vllm_module.VLLMBackend.clear_cache()
        with patch.dict("sys.modules", {"vllm": type("m", (), {"LLM": _LLM})}), \
             patch.object(vllm_module, "_prepare_environment"):
            yield vllm_module, calls
        vllm_module.VLLMBackend.clear_cache()
        vllm_module.set_planned_utilization({})     # a release keeps the plan

    def test_the_planned_value_reaches_the_engine_constructor(self, registered, engine):
        vllm_module, calls = engine
        a = registered("_pu_a", hf_model_id="org/pua", vram_gb=20.0)
        b = registered("_pu_b", hf_model_id="org/pub", vram_gb=10.0)
        with _kv(0.0):
            residency.apply_plan(_plan([a, b], 48.0, 1))
        vllm_module._engine("org/pua", None, {})
        vllm_module._engine("org/pub", None, {})
        # Pool 43.2, padded needs 22.6 + 11.6, spare 9.0: 27.1 and 16.1 of the 48.
        assert [c["gpu_memory_utilization"] for c in calls] == [0.5645, 0.3354]

    def test_no_plan_leaves_the_vllm_default(self, engine):
        vllm_module, calls = engine
        vllm_module._engine("org/noplan", None, {"max_model_len": 512})
        assert calls == [{"model": "org/noplan", "quantization": None,
                          "max_model_len": 512}]

    def test_a_plan_is_swapped_in_whole(self, engine):
        """A load must never read a cleared, half-filled plan."""
        vllm_module, _ = engine
        vllm_module.set_planned_utilization({("a",): 0.5})
        before = vllm_module._planned_utilization
        vllm_module.set_planned_utilization({("b",): 0.4})
        assert before == {("a",): 0.5}                      # not cleared in place
        assert vllm_module._planned_utilization == {("b",): 0.4}
        vllm_module.set_planned_utilization({("b",): 0.1, ("c",): 0.2}, replace=False)
        assert vllm_module._planned_utilization == {("b",): 0.4, ("c",): 0.2}

    def test_a_registered_model_loads_with_the_planned_value(self, registered, engine):
        from redact.llms.model_config import get_model_config

        vllm_module, calls = engine
        m = registered("_pu_reg", hf_model_id="org/pur", vram_gb=5.0,
                       vllm_kwargs={"max_model_len": 512})
        residency.apply_plan(_plan([m], 48.0, 1))
        vllm_module.VLLMBackend.from_config(get_model_config(m))
        assert calls == [{"model": "org/pur", "quantization": None,
                          "max_model_len": 512, "gpu_memory_utilization": 0.9}]

    def test_a_directly_built_engine_keeps_its_own_value(self, engine):
        """Outside the registry: no plan is keyed on these kwargs."""
        vllm_module, calls = engine
        kw = {"gpu_memory_utilization": 0.5, "max_model_len": 512}
        vllm_module.set_planned_utilization(
            {vllm_module._engine_key("org/pue", None, kw): 0.9})
        vllm_module._engine("org/pue", None, kw)
        assert calls[0]["gpu_memory_utilization"] == 0.5

    def test_the_cache_key_is_unchanged_and_a_shared_checkpoint_loads_once(
            self, registered, engine):
        vllm_module, calls = engine
        a = registered("_pu_share_a", hf_model_id="org/pus", vram_gb=20.0)
        b = registered("_pu_share_b", hf_model_id="org/pus", vram_gb=20.0)
        plan = _plan([a, b], 48.0, 1)
        residency.apply_plan(plan)
        assert [fp.planned_utilization for fp in plan.footprints] == [0.9, 0.9]
        first = vllm_module._engine("org/pus", None, {})
        assert vllm_module._engine("org/pus", None, {}) is first
        assert list(vllm_module._engines) == [vllm_module._engine_key("org/pus", None, {})]
        assert len(calls) == 1 and calls[0]["gpu_memory_utilization"] == 0.9

    def test_preload_and_first_use_load_with_the_same_value(self, registered, engine):
        vllm_module, calls = engine
        a = registered("_pu_thr_a", hf_model_id="org/pta", vram_gb=10.0)
        b = registered("_pu_thr_b", hf_model_id="org/ptb", vram_gb=10.0)
        with _kv(0.0):
            residency.apply_plan(_plan([a, b], 40.0, 1))
        threads = [threading.Thread(target=vllm_module._engine,
                                    args=("org/pta", None, {})) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert [c["gpu_memory_utilization"] for c in calls] == [0.45]

    def test_a_second_plan_does_not_reuse_the_first(self, registered, engine):
        vllm_module, calls = engine
        a = registered("_pu_two_a", hf_model_id="org/p2a", vram_gb=10.0)
        b = registered("_pu_two_b", hf_model_id="org/p2b", vram_gb=10.0)
        key_a, key_b = (vllm_module._engine_key(f"org/p2{x}", None, {}) for x in "ab")
        with _kv(0.0):
            residency.apply_plan(_plan([a, b], 40.0, 1))
            assert vllm_module._planned_utilization == {key_a: 0.45, key_b: 0.45}
            residency.apply_plan(_plan([a], 40.0, 1))
        assert vllm_module._planned_utilization == {key_a: 0.9}
        vllm_module._engine("org/p2b", None, {})
        assert "gpu_memory_utilization" not in calls[0]

    def test_replace_false_keeps_the_values_already_set(self, registered, engine):
        vllm_module, _ = engine
        a = registered("_pu_keep_a", hf_model_id="org/pka", vram_gb=10.0)
        b = registered("_pu_keep_b", hf_model_id="org/pkb", vram_gb=10.0)
        key_a, key_b = (vllm_module._engine_key(f"org/pk{x}", None, {}) for x in "ab")
        with _kv(0.0):
            residency.apply_plan(_plan([a, b], 40.0, 1))
            residency.apply_plan(_plan([a], 40.0, 1), replace=False)
        assert vllm_module._planned_utilization == {key_a: 0.45, key_b: 0.45}

    def test_a_released_engine_reloads_with_its_planned_share(self, registered, engine):
        """A share belongs to the plan, not to the load: a release keeps it."""
        vllm_module, calls = engine
        a = registered("_pu_clr_a", hf_model_id="org/pca", vram_gb=10.0)
        b = registered("_pu_clr_b", hf_model_id="org/pcb", vram_gb=10.0)
        key_a, key_b = (vllm_module._engine_key(f"org/pc{x}", None, {}) for x in "ab")
        with _kv(0.0):
            plan = _plan([a, b], 40.0, 1)
        residency.apply_plan(plan)
        vllm_module._engine("org/pca", None, {})
        residency.unload_local()
        assert vllm_module._planned_utilization == {key_a: 0.45, key_b: 0.45}
        vllm_module._engine("org/pca", None, {})            # loaded again
        vllm_module._engine("org/pcb", None, {})            # loaded for the first time
        assert [(c["model"], c["gpu_memory_utilization"]) for c in calls] == [
            ("org/pca", 0.45), ("org/pca", 0.45), ("org/pcb", 0.45)]

    def test_an_unknown_card_size_plans_no_value(self, registered, engine):
        vllm_module, _ = engine
        m = registered("_pu_nosize", hf_model_id="org/pns", vram_gb=10.0)
        with patch("redact.llms.resources.measure.free_total_gib", return_value=None), \
             patch("redact.llms.resources.measure.detect_gpus", return_value=("FakeGPU", 1)), \
             patch("redact.llms.resources.measure.detect_gpu_memory_gib", return_value=None):
            residency.apply_plan(residency.plan_residency([m]))
        assert vllm_module._planned_utilization == {}


class TestPaddedGrant:
    """A vLLM engine claims, is granted and is checked against its need with
    overhead: (weights + KV) x 1.1 + 0.6, unrounded."""

    def test_one_engine(self, registered):
        a = registered("_pg_one", hf_model_id="org/pg1", vram_gb=20.0)
        with _kv(2.0):
            plan = _plan([a], 48.0, 1)
        fp = plan.footprints[0]
        assert (fp.need_gb, fp.gb) == (22.0, 25.0)
        assert fp.padded_gb == pytest.approx(24.8)         # 22 x 1.1 + 0.6
        assert fp.reserved_gb == pytest.approx(43.2) and fp.planned_utilization == 0.9
        assert plan.fits

    def test_two_engines_claim_their_padded_needs(self, registered):
        """Bare needs 20 + 18 are under the 43.2 pool; padded 22.6 + 20.4
        are 43.0, and 20 + 18.2 pad to 43.22, which is over it."""
        a = registered("_pg_two_a", hf_model_id="org/pg2a", vram_gb=20.0)
        b = registered("_pg_two_b", hf_model_id="org/pg2b", vram_gb=18.0)
        c = registered("_pg_two_c", hf_model_id="org/pg2c", vram_gb=18.2)
        with _kv(0.0):
            fits = _plan([a, b], 48.0, 1)
            over = _plan([a, c], 48.0, 1)
        assert fits.fits and len(fits.groups) == 1
        assert [fp.reserved_gb for fp in fits.footprints] == [
            pytest.approx(22.7), pytest.approx(20.5)]     # 0.2 spare, split
        assert [fp.planned_utilization for fp in fits.footprints] == [0.4729, 0.427]
        assert len(over.groups) == 2 and not over.fits

    @pytest.mark.parametrize("weights", [
        (26.7384, 11.4434), (20.0, 18.181818), (7.77, 3.33, 9.1), (25.4321,)])
    def test_the_fraction_covers_the_padded_need_after_rounding(self, registered, weights):
        names = [registered(f"_pg_round_{i}", hf_model_id=f"org/pgr{i}", vram_gb=w)
                 for i, w in enumerate(weights)]
        with _kv(0.0):
            plan = _plan(names, 48.0, 1)
        assert plan.fits
        for fp in plan.footprints:
            assert fp.planned_utilization * 48.0 >= fp.padded_gb - 1e-6
            assert fp.planned_utilization > 0

    def test_rounding_up_applies_only_where_rounding_down_would_undercut(self, registered):
        """Padded needs 33.6123 + 9.5877 fill the 43.2 pool exactly: 0.70025
        and 0.19974 of the card floor to 0.7002 and 0.1997, under the needs."""
        a = registered("_pg_tight_a", hf_model_id="org/pta", vram_gb=30.0)
        b = registered("_pg_tight_b", hf_model_id="org/ptb", vram_gb=8.0)
        with patch.object(estimate, "estimate_kv_gib",
                          side_effect=[0.0111818, 0.1706364]):
            plan = _plan([a, b], 48.0, 1)
        assert plan.fits
        assert [fp.planned_utilization for fp in plan.footprints] == [0.7003, 0.1998]


class TestResidentEngines:
    @pytest.fixture()
    def loaded(self):
        import redact.llms.backends.introspection as intro
        import redact.llms.backends.vllm as vllm_module

        vllm_module._engines.clear()
        intro._models.clear()
        yield vllm_module._engines, intro._models
        vllm_module._engines.clear()
        intro._models.clear()

    def test_a_plan_warns_once_about_engines_it_does_not_cover(
            self, registered, loaded, caplog):
        engines, models = loaded
        m = registered("_re_new", hf_model_id="org/new", vram_gb=5.0)
        engines[("org/new", None, "[]")] = object()            # in the plan
        engines[("org/old", None, "[]")] = object()
        models[("org/old-tf", "auto", "auto", "[]")] = ("tok", "model")
        with caplog.at_level(logging.WARNING, logger=residency.logger.name), _kv(0.0):
            plan = _plan([m], 48.0, 1)
        assert plan.fits                                       # the plan is unchanged
        assert [r.getMessage() for r in caplog.records] == [
            "[residency] already loaded and not in this plan, so the plan hands "
            "their memory out again: org/old-tf[introspect], org/old[vllm]"]

    def test_no_warning_when_every_loaded_engine_is_planned(
            self, registered, loaded, caplog):
        engines, _ = loaded
        m = registered("_re_same", hf_model_id="org/same-re", vram_gb=5.0)
        engines[("org/same-re", None, "[]")] = object()
        with caplog.at_level(logging.WARNING, logger=residency.logger.name), _kv(0.0):
            _plan([m], 48.0, 1)
            _plan(["claude-opus-4-6"], 48.0, 1)                # nothing local
        assert caplog.records == []

    def test_unload_keeps_the_engines_it_is_handed(self, registered, loaded):
        engines, models = loaded
        pool = registered("_re_pool", hf_model_id="org/pool")
        other, mine = ("org/other", None, "[]"), ("org/pool", None, "[]")
        engines[other] = engines[mine] = object()
        models[("org/tf", "auto", "auto", "[]")] = ("tok", "model")
        resident = residency.loaded_engines(exclude=[pool])
        assert resident == {"vllm": {other},
                            "introspect": {("org/tf", "auto", "auto", "[]")}}
        residency.unload_local(engines=resident)
        assert list(engines) == [other] and len(models) == 1


class TestDefaultContextLength:
    """An unset max_model_len is 10000 tokens, in the plan and at load."""

    SMALL = SimpleNamespace(**{**vars(MISTRAL_24B), "max_position_embeddings": 4096})

    @pytest.fixture()
    def engine(self):
        import redact.llms.backends.vllm as vllm_module

        calls = []
        vllm_module.VLLMBackend.clear_cache()
        fake = type("m", (), {"LLM": lambda **kw: calls.append(kw) or object()})
        with patch.dict("sys.modules", {"vllm": fake}), \
             patch.object(vllm_module, "_prepare_environment"):
            yield vllm_module, calls
        vllm_module.VLLMBackend.clear_cache()
        vllm_module.set_planned_utilization({})     # a release keeps the plan

    def test_the_plan_sizes_the_kv_cache_at_the_default(self, registered):
        unset = registered("_dc_unset", hf_model_id="org/dcu", vram_gb=20.0)
        explicit = registered("_dc_explicit", hf_model_id="org/dce", vram_gb=20.0,
                              vllm_kwargs={"max_model_len": 10_000})
        full = registered("_dc_full", hf_model_id="org/dcf", vram_gb=20.0,
                          vllm_kwargs={"max_model_len": 32768})
        with _config(MISTRAL_24B):
            kv = [residency.footprint(m).parts["kv_gib"] for m in (unset, explicit, full)]
        assert kv == [1.53, 1.53, 5.0]

    def test_the_plan_caps_the_default_at_the_checkpoint_maximum(self, registered):
        m = registered("_dc_small", hf_model_id="org/dcs", vram_gb=20.0)
        with _config(self.SMALL):
            assert residency.footprint(m).parts["kv_gib"] == 0.62   # 4096 tokens

    def test_the_default_reaches_the_engine_without_a_plan(self, engine):
        vllm_module, calls = engine
        with _config(MISTRAL_24B):
            vllm_module._engine("org/dc-noplan", None, {})
        assert calls == [{"model": "org/dc-noplan", "quantization": None,
                          "max_model_len": 10_000}]

    def test_the_default_reaches_the_engine_with_a_plan(self, registered, engine):
        vllm_module, calls = engine
        m = registered("_dc_plan", hf_model_id="org/dcp", vram_gb=20.0)
        with _config(MISTRAL_24B):
            plan = _plan([m], 48.0, 1)
            residency.apply_plan(plan)
            vllm_module._engine("org/dcp", None, {})
        assert plan.footprints[0].parts["kv_gib"] == 1.53
        assert calls[0]["max_model_len"] == 10_000
        assert calls[0]["gpu_memory_utilization"] == 0.9

    def test_a_small_checkpoint_is_not_given_more_than_its_maximum(self, engine):
        vllm_module, calls = engine
        with _config(self.SMALL):
            vllm_module._engine("org/dc-small", None, {})
        assert calls[0]["max_model_len"] == 4096

    def test_an_unreadable_config_gets_the_default(self, engine):
        vllm_module, calls = engine
        vllm_module._engine("org/dc-unread", None, {})
        assert calls[0]["max_model_len"] == 10_000

    def test_an_explicit_value_wins_and_reads_no_config(self, engine):
        vllm_module, calls = engine
        with patch.object(estimate, "_load_config", side_effect=AssertionError):
            vllm_module._engine("org/dc-explicit", None, {"max_model_len": 32768})
        assert calls[0]["max_model_len"] == 32768

    def test_the_cache_key_and_engine_identity_are_unchanged(self, registered, engine):
        vllm_module, calls = engine
        a = registered("_dc_share_a", hf_model_id="org/dcsh", vram_gb=20.0)
        b = registered("_dc_share_b", hf_model_id="org/dcsh", vram_gb=20.0)
        key = vllm_module._engine_key("org/dcsh", None, {})
        first = vllm_module._engine("org/dcsh", None, {})
        assert vllm_module._engine("org/dcsh", None, {}) is first and len(calls) == 1
        assert list(vllm_module._engines) == [key]
        assert residency.footprint(a).engine == residency.footprint(b).engine
        assert residency.footprint(a).engine[1:] == key
        residency.unload_local(keep=[b])                     # keep= still matches
        assert list(vllm_module._engines) == [key]


class TestEngineSettingsSizeTheModel:
    """A vLLM setup's dtype and quantization reach the estimate, for estimated
    and for declared weights."""

    FLOAT32 = SimpleNamespace(**{**vars(MISTRAL_24B), "torch_dtype": "float32"})

    def test_a_float32_checkpoint_is_sized_as_vllm_loads_it(self, registered):
        """vLLM's default dtype loads float32 as 16-bit; an explicit
        dtype=float32 keeps 32-bit and needs twice the card."""
        default = registered("_es_f32", hf_model_id="org/esf")
        explicit = registered("_es_f32_explicit", hf_model_id="org/esf",
                              vllm_kwargs={"dtype": "float32"})
        with _config(self.FLOAT32):
            half, full = _plan([default], 80.0, 1), _plan([explicit], 80.0, 1)
        sized = [(plan.footprints[0].parts["weights_gib"],
                  plan.footprints[0].parts["kv_gib"]) for plan in (half, full)]
        assert sized == [(43.91, 1.53), (87.81, 3.05)]
        assert "[float16, cast from float32]" in half.footprints[0].breakdown()
        assert "[float32]" in full.footprints[0].breakdown()
        assert half.fits and not full.fits        # 50.6 and 100.5 against 72.0

    def test_declared_weights_get_a_kv_cache_at_the_engine_dtype(self, registered):
        default = registered("_es_decl", hf_model_id="org/esd", vram_gb=20.0)
        explicit = registered("_es_decl_explicit", hf_model_id="org/esd",
                              vram_gb=20.0, vllm_kwargs={"dtype": "float32"})
        with _config(self.FLOAT32):
            kv = [residency.footprint(m).parts["kv_gib"] for m in (default, explicit)]
        assert kv == [1.53, 3.05]

    def test_quantization_decides_whether_the_model_fits(self, registered):
        plain = registered("_es_plain", hf_model_id="org/esq")
        awq = registered("_es_awq", hf_model_id="org/esq", quantization="awq")
        with _config(MISTRAL_24B):
            unquantized, quantized = _plan([plain], 48.0, 1), _plan([awq], 48.0, 1)
        # 43.91 or 14.15 GiB of weights plus 1.53 of KV, against a 43.2 grant.
        assert quantized.footprints[0].parts["weights_gib"] == 14.15
        assert quantized.fits and not unquantized.fits


class TestSharedEngineAcrossGroups:
    """A model joins the group of the engine it shares, wherever that is."""

    def test_a_later_twin_joins_its_engines_group(self, registered):
        a = registered("_sg_a", hf_model_id="org/sga", vram_gb=30.0)
        big = registered("_sg_big", hf_model_id="org/sgb", vram_gb=30.0)
        twin = registered("_sg_twin", hf_model_id="org/sga", vram_gb=30.0)
        with _kv(0.0):
            plan = _plan([a, big, twin], 48.0, 1)
        # 30 + 30 GiB do not share a 48 GiB card: two engines, two groups.
        assert [[fp.model for fp in g] for g in plan.groups] == [[a, twin], [big]]
        first, shared, second = plan.footprints
        assert (shared.devices, shared.reserved_gb, shared.planned_utilization) == (
            first.devices, first.reserved_gb, first.planned_utilization)
        assert (first.planned_utilization, second.planned_utilization) == (0.9, 0.9)
        assert plan.as_dict()["groups"] == [[a, twin], [big]]
        assert [m["group"] for m in plan.as_dict()["models"]] == [0, 0, 1]
        # Two groups for two engines, and none for a model beside its own engine.
        assert plan.explain().count("cannot be resident with the above") == 1

    def test_the_shares_handed_to_the_loader_are_one_per_engine(self, registered):
        import redact.llms.backends.vllm as vllm_module

        a = registered("_sg_key_a", hf_model_id="org/sgka", vram_gb=30.0)
        big = registered("_sg_key_big", hf_model_id="org/sgkb", vram_gb=30.0)
        twin = registered("_sg_key_twin", hf_model_id="org/sgka", vram_gb=30.0)
        with _kv(0.0):
            plan = _plan([a, big, twin], 48.0, 1)
        try:
            residency.apply_plan(plan)
            assert vllm_module._planned_utilization == {
                vllm_module._engine_key("org/sgka", None, {}): 0.9,
                vllm_module._engine_key("org/sgkb", None, {}): 0.9,
            }
        finally:
            vllm_module.set_planned_utilization({})


class TestPlannerHoles:
    def test_an_unknown_engine_does_not_join_a_tensor_parallel_card(self, registered):
        big = registered("_ph_tp", hf_model_id="org/phtp", vram_gb=40.0, min_gpus=2)
        unknown = registered("_ph_unknown", hf_model_id="org/phu")
        plan = _plan([big, unknown], 48.0, 2)
        assert [[fp.model for fp in g] for g in plan.groups] == [[big], [unknown]]
        assert not plan.fits
        assert plan.footprints[0].planned_utilization == 0.9

    def test_no_fraction_at_or_below_zero_is_planned(self, registered):
        """Fixed claims that leave a card nothing: the engine gets no value,
        and the plan says so instead of publishing 0."""
        import redact.llms.backends.vllm as vllm_module

        m = registered("_ph_zero", hf_model_id="org/phz", vram_gb=5.0)
        unknown = registered("_ph_zero_u", hf_model_id="org/phzu")
        for name in (m, unknown):
            fp = residency.footprint(name)
            cards = _Cards(1, 48.0)
            cards.free[0] = 2.0                     # under the 4.8 kept back
            cards.shared[0].append(fp)
            fp.devices = (0,)
            cards.settle()
            assert fp.planned_utilization is None
            assert fp.reserved_gb == 0.0
            assert _exceeds_its_cards(fp, 48.0, 1)
            plan = residency.ResidencyPlan(groups=[[fp]], gpus_required=1,
                                           gpus_available=1, per_gpu_gb=48.0,
                                           oversized=[fp])
            assert "is left no memory on its card" in plan.explain()
            assert plan.as_dict()["models"][0]["gpu_memory_utilization"] is None
            residency.apply_plan(plan)
            assert vllm_module._planned_utilization == {}

    def test_a_tensor_parallel_size_must_divide_the_heads(self, registered):
        """Qwen2.5-7B has 28 heads: 4 divides them, 8 does not."""
        bad = registered("_ph_heads_bad", hf_model_id="org/phb", min_gpus=8)
        good = registered("_ph_heads_ok", hf_model_id="org/pho", min_gpus=4)
        with _config(QWEN_7B):
            bad_plan, good_plan = _plan([bad], 48.0, 8), _plan([good], 48.0, 8)
        assert good_plan.fits and good_plan.problems == []
        assert not bad_plan.fits and bad_plan.oversized == []
        assert bad_plan.problems == [
            "_ph_heads_bad[vllm] has 28 attention heads, which "
            "tensor_parallel_size=8 does not divide, so vLLM cannot load it."]
        assert "PROBLEM: _ph_heads_bad[vllm] has 28 attention heads" in bad_plan.explain()
        assert bad_plan.as_dict()["fits"] is False

    def test_unknown_card_size_with_two_engines_does_not_fit(self, registered):
        a = registered("_ph_ns_a", hf_model_id="org/phna", vram_gb=5.0)
        b = registered("_ph_ns_b", hf_model_id="org/phnb", vram_gb=5.0)
        twin = registered("_ph_ns_twin", hf_model_id="org/phna", vram_gb=5.0)
        with patch("redact.llms.resources.measure.free_total_gib", return_value=None):
            two = _plan([a, b], None, 1)
            one = _plan([a, twin], None, 1)            # one shared engine
        assert not two.fits and len(two.groups) == 1
        assert "the card size is unknown, so no gpu_memory_utilization is planned " \
               "for the 2 vLLM engines" in two.problems[0]
        assert [fp.planned_utilization for fp in two.footprints] == [None, None]
        assert one.fits
        assert any("card size is unknown" in w for w in one.warnings)

    def test_zero_gpus_given_invents_no_card(self, registered):
        m = registered("_ph_nocard", hf_model_id="org/phnc", vram_gb=5.0)
        with _kv(0.0):
            plan = residency.plan_residency([m], gpus=0, gib_per_gpu=24.0)
        fp = plan.footprints[0]
        assert not plan.fits and plan.gpus_available == 0
        assert (fp.reserved_gb, fp.planned_utilization) == (None, None)
        assert "vLLM gets" not in plan.explain()
        assert any("no GPU is available" in p for p in plan.problems)

    def test_a_size_without_a_count_is_still_one_card(self, registered):
        """torch reports a size where nvidia-smi is missing."""
        m = registered("_ph_torch", hf_model_id="org/pht", vram_gb=5.0)
        with patch("redact.llms.resources.measure.free_total_gib",
                   return_value=(20.0, 24.0)), _kv(0.0):
            with patch("redact.llms.resources.measure.detect_gpus", return_value=(None, 0)), \
                 patch("redact.llms.resources.measure.detect_gpu_memory_gib", return_value=None):
                plan = residency.plan_residency([m])
        assert plan.footprints[0].planned_utilization == 0.9


class TestWhatThePlanCannotTell:
    def test_an_unsized_engine_beside_another_does_not_fit(self, registered):
        known = registered("_ct_known", hf_model_id="org/ctk", vram_gb=10.0)
        unsized = registered("_ct_unsized", hf_model_id="org/ctu")
        with _kv(0.0):
            plan = _plan([known, unsized], 24.0, 1)
        assert not plan.fits
        assert any("has no known size" in p and unsized in p for p in plan.problems)

    def test_a_lone_unsized_engine_is_flagged_not_refused(self, registered):
        unsized = registered("_ct_alone", hf_model_id="org/cta")
        plan = _plan([unsized], 24.0, 1)
        assert plan.fits
        assert any("has no known size" in w for w in plan.warnings)

    def test_torch_alone_reports_the_card_it_plans(self, registered):
        m = registered("_ct_torch", hf_model_id="org/ctt", vram_gb=5.0)
        with _kv(0.0), \
             patch("redact.llms.resources.measure.detect_gpus", return_value=(None, 0)), \
             patch("redact.llms.resources.measure.detect_gpu_memory_gib", return_value=None), \
             patch("redact.llms.resources.measure.free_total_gib",
                   return_value=(24.0, 24.0)):
            plan = residency.plan_residency([m])
        assert plan.gpus_available == 1 and plan.fits

    def test_a_cpu_model_is_never_too_large_for_a_card(self, registered):
        m = registered("_ct_cpu", setup="introspect", hf_model_id="org/ctc",
                       vram_gb=40.0, device_map="cpu")
        plan = _plan([m], 24.0, 1)
        assert plan.fits and plan.oversized == []
