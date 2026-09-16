"""Static VRAM estimation from HF config shapes.

The estimator replaced a measure-and-cache loop, so these tests are the only
thing standing between a shape-math slip and a planner that silently plans
from a wrong number. Both fixtures are the real configs of the two local
checkpoints this library ships, and the assertions are anchored on their
independently-known parameter counts.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from redact.llms.resources import estimate as E

# Real shapes: meta-llama/Llama-3.2-3B-Instruct (the debug model's base).
LLAMA_3B = SimpleNamespace(
    num_hidden_layers=28, hidden_size=3072, intermediate_size=8192,
    num_attention_heads=24, num_key_value_heads=8, head_dim=128,
    vocab_size=128256, max_position_embeddings=131072,
    hidden_act="silu", tie_word_embeddings=True,
)

# Real shapes: Mistral-Small-24B, the base of Dolphin-Mistral-24B-Venice.
MISTRAL_24B = SimpleNamespace(
    num_hidden_layers=40, hidden_size=5120, intermediate_size=32768,
    num_attention_heads=32, num_key_value_heads=8, head_dim=128,
    vocab_size=131072, max_position_embeddings=32768,
    hidden_act="silu", tie_word_embeddings=False,
)


#: The hand-derived vram_gb the registry declares for Dolphin-Mistral-24B.
DECLARED_MISTRAL_GB = 50.0
#: How far the estimate may sit from that hand-derived figure.
DECLARED_TOLERANCE_GB = 5.0
#: weights + kv for the planning-headroom check below.
FLOOR_GIB = 12.0


def _with(cfg):
    return patch.object(E, "_load_config", return_value=cfg)


class TestWeights:
    @pytest.mark.parametrize(
        "cfg,expected_b_params",
        [(LLAMA_3B, 3.2), (MISTRAL_24B, 23.6)],
        ids=["llama-3.2-3b", "mistral-24b"],
    )
    def test_parameter_count_matches_the_model_name(self, cfg, expected_b_params):
        """The shape math must reproduce the count the checkpoint is named for.

        This is the real check: 'Llama-3.2-**3B**' and the 23.6B figure the
        registry's notes cite were derived independently of this code.
        """
        with _with(cfg):
            gib = E.estimate_weights_gib("x")
        billions = gib * (1024 ** 3) / 2 / 1e9  # bf16 -> 2 bytes/param
        assert billions == pytest.approx(expected_b_params, abs=0.15)

    def test_lands_near_the_hand_declared_vram_gb(self):
        """Both shipped entries declare a hand-derived vram_gb; the estimate
        plus headroom must agree, or one of the two is wrong."""
        with _with(MISTRAL_24B):
            plan = E.planning_gib(E.estimate_weights_gib("x"),
                                  E.estimate_kv_gib("x", max_model_len=8192))
        assert abs(plan - DECLARED_MISTRAL_GB) <= DECLARED_TOLERANCE_GB

    def test_tied_embeddings_are_not_double_counted(self):
        untied = SimpleNamespace(**{**vars(LLAMA_3B), "tie_word_embeddings": False})
        with _with(LLAMA_3B):
            tied_gib = E.estimate_weights_gib("x")
        with _with(untied):
            untied_gib = E.estimate_weights_gib("x")
        assert untied_gib > tied_gib

    def test_quantization_beats_dtype(self):
        with _with(MISTRAL_24B):
            bf16 = E.estimate_weights_gib("x", dtype="bfloat16")
            awq = E.estimate_weights_gib("x", dtype="bfloat16", quantization="awq")
        assert awq == pytest.approx(bf16 / 4, rel=0.01)

    def test_sharded_across_tensor_parallel_ranks(self):
        with _with(MISTRAL_24B):
            one = E.estimate_weights_gib("x")
            four = E.estimate_weights_gib("x", tensor_parallel_size=4)
        assert four == pytest.approx(one / 4, rel=0.01)

    def test_gated_mlp_detection_changes_the_answer(self):
        """n_mats is 3 for a gated MLP and 2 otherwise — a third of the MLP,
        which dominates the parameter count, so it cannot be guessed."""
        ungated = SimpleNamespace(**{**vars(MISTRAL_24B), "hidden_act": "gelu"})
        with _with(MISTRAL_24B):
            gated = E.estimate_weights_gib("x")
        with _with(ungated):
            plain = E.estimate_weights_gib("x")
        assert gated > plain

    def test_unreadable_config_is_unknown_not_zero(self):
        with _with(None):
            assert E.estimate_weights_gib("x") is None

    def test_missing_shape_fields_is_unknown(self):
        with _with(SimpleNamespace(num_hidden_layers=28)):
            assert E.estimate_weights_gib("x") is None


class TestKV:
    def test_scales_linearly_with_sequence_and_batch(self):
        # Large enough that the 2-decimal rounding isn't a visible share of
        # the value — at seq_len=1024 the figure is ~0.16 GiB and the rounding
        # alone breaks a 1% comparison.
        with _with(MISTRAL_24B):
            one = E.estimate_kv_gib("x", seq_len=16384)
            longer = E.estimate_kv_gib("x", seq_len=32768)
            batched = E.estimate_kv_gib("x", seq_len=16384, batch=4)
        assert longer == pytest.approx(one * 2, rel=0.01)
        assert batched == pytest.approx(one * 4, rel=0.01)

    def test_seq_len_is_capped_at_max_model_len(self):
        """Asking for more context than the engine allows describes a request
        the engine would reject, so the estimate clamps rather than inflating."""
        with _with(MISTRAL_24B):
            asked = E.estimate_kv_gib("x", seq_len=99999, max_model_len=4096)
            capped = E.estimate_kv_gib("x", seq_len=4096, max_model_len=4096)
        assert asked == capped

    def test_falls_back_to_max_model_len_then_to_config(self):
        with _with(MISTRAL_24B):
            from_arg = E.estimate_kv_gib("x", max_model_len=4096)
            from_cfg = E.estimate_kv_gib("x")
        assert from_arg < from_cfg  # 4096 < max_position_embeddings=32768

    def test_gqa_is_modelled(self):
        """kv_heads (8) not attention heads (32) — a 4x difference here."""
        mha = SimpleNamespace(**{**vars(MISTRAL_24B), "num_key_value_heads": 32})
        with _with(MISTRAL_24B):
            gqa = E.estimate_kv_gib("x", seq_len=4096)
        with _with(mha):
            full = E.estimate_kv_gib("x", seq_len=4096)
        assert full == pytest.approx(gqa * 4, rel=0.01)


class TestPlanning:
    def test_adds_headroom_over_the_floor(self):
        """weights + kv is a bound no model runs in; the result must exceed it."""
        assert E.planning_gib(10.0, 2.0) > FLOOR_GIB

    def test_rounds_up_to_half_a_gib(self):
        got = E.planning_gib(10.0, 2.0)
        assert got * 2 == int(got * 2)

    def test_unknown_weights_stay_unknown(self):
        """Headroom over an unknown base would be a fabricated number."""
        assert E.planning_gib(None, 2.0) is None

    def test_missing_kv_still_yields_a_lower_bound(self):
        assert E.planning_gib(10.0, None) is not None

    def test_headroom_terms_are_overridable(self):
        default = E.planning_gib(10.0, 2.0)
        generous = E.planning_gib(10.0, 2.0, fragmentation=1.5, cuda_context_gib=2.0)
        assert generous > default


def test_gib_not_decimal_gb():
    """The unit bug this rework fixed: GiB throughout, matching nvidia-smi and
    every GPU spec sheet. A decimal-GB divisor would read ~7% higher."""
    cfg = SimpleNamespace(
        num_hidden_layers=1, hidden_size=1024, intermediate_size=1024,
        num_attention_heads=8, num_key_value_heads=8, head_dim=128,
        vocab_size=1000, max_position_embeddings=128,
        hidden_act="gelu", tie_word_embeddings=True,
    )
    with _with(cfg):
        gib = E.estimate_weights_gib("x")
    # 1 layer: attn 4*1024*1024, mlp 2*1024*1024, embed 1000*1024 -> x2 bytes
    expected_bytes = (1 * (4 * 1024 * 1024 + 2 * 1024 * 1024) + 1000 * 1024) * 2
    assert gib == pytest.approx(expected_bytes / 1024 ** 3, abs=0.01)
