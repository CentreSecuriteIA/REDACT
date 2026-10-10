"""Static VRAM estimation from HF config shapes.

These tests are the only thing standing between a shape-math slip and a
planner that silently plans from a wrong number, so the assertions are
anchored on real config shapes and their independently-known parameter
counts. Nothing here touches the network: configs are plain namespaces and
``transformers.AutoConfig`` is faked.
"""

import builtins
import logging
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from redact.llms.resources import estimate as E

GIB = 1024 ** 3

# Real shapes: meta-llama/Llama-3.2-3B-Instruct (the debug model's base).
LLAMA_3B = SimpleNamespace(
    num_hidden_layers=28, hidden_size=3072, intermediate_size=8192,
    num_attention_heads=24, num_key_value_heads=8, head_dim=128,
    vocab_size=128256, max_position_embeddings=131072,
    hidden_act="silu", tie_word_embeddings=True, torch_dtype="bfloat16",
    model_type="llama",
)

# Real shapes: Mistral-Small-24B, the base of Dolphin-Mistral-24B-Venice.
MISTRAL_24B = SimpleNamespace(
    num_hidden_layers=40, hidden_size=5120, intermediate_size=32768,
    num_attention_heads=32, num_key_value_heads=8, head_dim=128,
    vocab_size=131072, max_position_embeddings=32768,
    hidden_act="silu", tie_word_embeddings=False, torch_dtype="bfloat16",
    model_type="mistral",
)
#: Attention + MLP parameters of MISTRAL_24B, and its embedding + lm_head.
MISTRAL_LINEAR = 40 * (2 * 5120 * 4096 + 2 * 5120 * 1024 + 3 * 5120 * 32768)
MISTRAL_EMBED = 2 * 131072 * 5120

LLAMA_8B = SimpleNamespace(
    num_hidden_layers=32, hidden_size=4096, intermediate_size=14336,
    num_attention_heads=32, num_key_value_heads=8, vocab_size=128256,
    max_position_embeddings=131072, hidden_act="silu",
    tie_word_embeddings=False, torch_dtype="bfloat16", model_type="llama",
)
QWEN_7B = SimpleNamespace(
    num_hidden_layers=28, hidden_size=3584, intermediate_size=18944,
    num_attention_heads=28, num_key_value_heads=4, vocab_size=152064,
    max_position_embeddings=32768, hidden_act="silu",
    tie_word_embeddings=False, torch_dtype="bfloat16", model_type="qwen2",
)
# A gated MLP behind a GELU activation: the activation name must not decide.
GEMMA_9B = SimpleNamespace(
    num_hidden_layers=42, hidden_size=3584, intermediate_size=14336,
    num_attention_heads=16, num_key_value_heads=8, head_dim=256,
    vocab_size=256000, max_position_embeddings=8192,
    hidden_act="gelu_pytorch_tanh", hidden_activation="gelu_pytorch_tanh",
    tie_word_embeddings=True, torch_dtype="float32", model_type="gemma2",
)
MIXTRAL = SimpleNamespace(
    num_hidden_layers=32, hidden_size=4096, intermediate_size=14336,
    num_attention_heads=32, num_key_value_heads=8, vocab_size=32000,
    max_position_embeddings=32768, hidden_act="silu",
    tie_word_embeddings=False, torch_dtype="bfloat16", model_type="mixtral",
    num_local_experts=8,
)
# Qwen2-57B-A14B: 64 routed experts and one wider shared expert per layer.
QWEN2_MOE_57B = SimpleNamespace(
    num_hidden_layers=28, hidden_size=3584, intermediate_size=18944,
    num_attention_heads=28, num_key_value_heads=4, vocab_size=151936,
    max_position_embeddings=32768, hidden_act="silu",
    tie_word_embeddings=False, torch_dtype="bfloat16", model_type="qwen2_moe",
    num_experts=64, moe_intermediate_size=2560,
    shared_expert_intermediate_size=20480,
)

#: The hand-derived vram_gb the registry declares for Dolphin-Mistral-24B.
DECLARED_MISTRAL_GB = 50.0
#: How far the estimate may sit from that hand-derived figure.
DECLARED_TOLERANCE_GB = 5.0
#: weights + kv for the planning-headroom check below.
FLOOR_GIB = 12.0


@pytest.fixture(autouse=True)
def _fresh_module_state():
    """The log-once set is module state."""
    E._reported.clear()
    yield
    E._reported.clear()


def _with(cfg):
    return patch.object(E, "_load_config", return_value=cfg)


def _variant(cfg, drop=(), **changes):
    fields = {**vars(cfg), **changes}
    return SimpleNamespace(**{k: v for k, v in fields.items() if k not in drop})


def _params(cfg, **kwargs):
    """The parameter count the bf16 weight estimate implies."""
    with _with(cfg):
        return E.estimate_weights_gib("x", dtype="bfloat16", **kwargs) * GIB / 2


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
        billions = gib * GIB / 2 / 1e9  # bf16 -> 2 bytes/param
        assert billions == pytest.approx(expected_b_params, abs=0.15)

    @pytest.mark.parametrize("cfg,true_params", [
        (LLAMA_3B, 3_212_749_824),
        (LLAMA_8B, 8_030_261_248),
        (QWEN_7B, 7_615_616_512),
        (MISTRAL_24B, 23_572_403_200),
    ], ids=["llama-3.2-3b", "llama-3.1-8b", "qwen2.5-7b", "mistral-24b"])
    def test_dense_architectures_stay_within_a_tenth_of_a_percent(
            self, cfg, true_params):
        assert _params(cfg) == pytest.approx(true_params, rel=0.001)

    def test_lands_near_the_hand_declared_vram_gb(self):
        """Both shipped entries declare a hand-derived vram_gb; the estimate
        plus headroom must agree, or one of the two is wrong."""
        with _with(MISTRAL_24B):
            plan = E.planning_gib(E.estimate_weights_gib("x"),
                                  E.estimate_kv_gib("x", max_model_len=8192))
        assert abs(plan - DECLARED_MISTRAL_GB) <= DECLARED_TOLERANCE_GB

    def test_tied_embeddings_are_not_double_counted(self):
        untied = _variant(LLAMA_3B, tie_word_embeddings=False)
        with _with(LLAMA_3B):
            tied_gib = E.estimate_weights_gib("x")
        with _with(untied):
            untied_gib = E.estimate_weights_gib("x")
        assert untied_gib - tied_gib == pytest.approx(128256 * 3072 * 2 / GIB, abs=0.01)

    def test_awq_quantizes_the_layers_and_keeps_the_embeddings(self):
        """4.5 bits per attention/MLP weight; embeddings and lm_head stay bf16."""
        with _with(MISTRAL_24B):
            bf16 = E.estimate_weights_gib("x")
            awq = E.estimate_weights_gib("x", quantization="awq")
        assert bf16 == pytest.approx(43.91, abs=0.01)
        assert awq == pytest.approx(
            (MISTRAL_LINEAR * 4.5 / 8 + MISTRAL_EMBED * 2) / GIB, abs=0.01)
        assert awq == pytest.approx(14.15, abs=0.01)

    def test_quantization_bit_width_is_read_from_the_config(self):
        as_dict = _variant(MISTRAL_24B, quantization_config={"bits": 8})
        as_object = _variant(MISTRAL_24B, quantization_config=SimpleNamespace(bits=8))
        expected = (MISTRAL_LINEAR * 8.5 / 8 + MISTRAL_EMBED * 2) / GIB
        for cfg in (as_dict, as_object):
            with _with(cfg):
                got = E.estimate_weights_gib("x", quantization="gptq")
                assert got == pytest.approx(expected, abs=0.01)

    def test_config_bits_alone_do_not_quantize(self):
        """Only a quantization the caller names is applied."""
        with _with(_variant(MISTRAL_24B, quantization_config={"bits": 4})):
            assert E.estimate_weights_gib("x") == pytest.approx(43.91, abs=0.01)

    @pytest.mark.parametrize("method", ["bitsandbytes", "gguf", "awq_marlin", "int8"])
    def test_unknown_quantization_is_sized_unquantized(self, method):
        """An unmodelled method must not shrink the plan."""
        with _with(MISTRAL_24B):
            assert E.estimate_weights_gib("x", quantization=method) == pytest.approx(
                43.91, abs=0.01)

    def test_quantized_rate_never_exceeds_the_dtype(self):
        with _with(_variant(MISTRAL_24B, quantization_config={"bits": 16})):
            assert E.estimate_weights_gib("x", quantization="awq") == pytest.approx(
                43.91, abs=0.01)

    def test_sharded_across_tensor_parallel_ranks(self):
        with _with(MISTRAL_24B):
            one = E.estimate_weights_gib("x")
            four = E.estimate_weights_gib("x", tensor_parallel_size=4)
        assert four == pytest.approx(one / 4, rel=0.01)

    def test_none_tensor_parallel_size_counts_as_one(self):
        with _with(MISTRAL_24B):
            assert E.estimate_weights_gib("x", tensor_parallel_size=None) == \
                pytest.approx(43.91, abs=0.01)

    def test_gemma_is_sized_as_a_gated_mlp(self):
        """Three matrices despite the GELU activation; two would read 7.08B."""
        assert _params(GEMMA_9B) == pytest.approx(9_241_705_984, rel=0.002)

    def test_unknown_architecture_is_sized_as_a_gated_mlp(self):
        """Without a known model_type the larger figure is taken."""
        untyped = _variant(MISTRAL_24B, drop=("model_type",), hidden_act="gelu")
        assert _params(untyped) == pytest.approx(23_572_403_200, rel=0.001)

    def test_known_ungated_architecture_uses_two_matrices(self):
        # phi-2 shapes: attention 4 x 2560^2, MLP 2 x 2560 x 10240, untied head.
        phi2 = SimpleNamespace(
            num_hidden_layers=32, hidden_size=2560, intermediate_size=10240,
            num_attention_heads=32, vocab_size=51200, hidden_act="gelu_new",
            tie_word_embeddings=False, model_type="phi",
        )
        expected = 32 * (4 * 2560 * 2560 + 2 * 2560 * 10240) + 2 * 51200 * 2560
        assert _params(phi2) == pytest.approx(expected, rel=0.002)

    def test_moe_counts_every_expert(self):
        """One expert alone would read 7.2B for a 46.7B checkpoint."""
        assert _params(MIXTRAL) == pytest.approx(46_702_792_704, rel=0.001)

    def test_moe_expert_width_and_shared_experts_are_read(self):
        moe = SimpleNamespace(
            num_hidden_layers=64, hidden_size=1024, intermediate_size=4096,
            moe_intermediate_size=512, num_experts=4, n_shared_experts=1,
            num_attention_heads=8, vocab_size=1000, tie_word_embeddings=True,
        )
        per_layer = 4 * 1024 * 1024 + 5 * 3 * 1024 * 512 + 1024 * 5
        assert _params(moe) == pytest.approx(64 * per_layer + 1000 * 1024, rel=0.005)

    def test_moe_shared_expert_of_its_own_width_is_counted(self):
        """Without it the 57.4B checkpoint reads 51.2B: 95.45 GiB for 106.9."""
        assert _params(QWEN2_MOE_57B) == pytest.approx(57_408_225_280, rel=0.001)
        with _with(QWEN2_MOE_57B):
            assert E.estimate_weights_gib("x") == pytest.approx(106.93, abs=0.01)
        routed_only = _variant(QWEN2_MOE_57B, drop=("shared_expert_intermediate_size",))
        with _with(routed_only):
            assert E.estimate_weights_gib("x") == pytest.approx(95.45, abs=0.01)

    def test_a_shared_expert_width_is_read_for_an_moe_config_only(self):
        dense = _variant(MISTRAL_24B, shared_expert_intermediate_size=20480)
        assert _params(dense) == pytest.approx(23_572_403_200, rel=0.001)

    def test_fp32_is_four_bytes_per_parameter(self):
        with _with(MISTRAL_24B):
            assert E.estimate_weights_gib("x", dtype="float32") == pytest.approx(
                87.81, abs=0.02)

    def test_unreadable_config_is_unknown_not_zero(self):
        with _with(None):
            assert E.estimate_weights_gib("x") is None

    def test_missing_shape_fields_is_unknown(self):
        with _with(SimpleNamespace(num_hidden_layers=28)):
            assert E.estimate_weights_gib("x") is None


class TestKV:
    def test_magnitude_is_pinned(self):
        """2 x 40 layers x 8 KV heads x 128 x 32768 tokens x 2 bytes = 5 GiB."""
        with _with(MISTRAL_24B):
            assert E.estimate_kv_gib("x", max_model_len=32768) == 5.0
            assert E.estimate_kv_gib("x", max_model_len=8192) == 1.25
            assert E.estimate_kv_gib(
                "x", max_model_len=32768, tensor_parallel_size=2) == 2.5
            assert E.estimate_kv_gib("x", max_model_len=32768, dtype="float32") == 10.0
        with _with(LLAMA_3B):
            assert E.estimate_kv_gib("x", max_model_len=131072) == 14.0

    def test_default_context_is_10000_tokens(self):
        """2 x 40 x 8 x 128 x 10000 x 2 bytes = 1.53 GiB, not the 5.0 GiB of
        the checkpoint's full 32768."""
        assert E.DEFAULT_MAX_MODEL_LEN == 10_000
        with _with(MISTRAL_24B):
            assert E.estimate_kv_gib("x") == 1.53
            assert E.estimate_kv_gib("x") == E.estimate_kv_gib("x", max_model_len=10_000)
            assert E.estimate("x").kv_gib == 1.53

    def test_default_context_is_capped_by_the_checkpoint_maximum(self):
        small = _variant(MISTRAL_24B, max_position_embeddings=4096)
        with _with(small):
            assert E.effective_max_model_len("x") == 4096
            assert E.estimate_kv_gib("x") == E.estimate_kv_gib("x", max_model_len=4096)
            # An explicit value is taken as given.
            assert E.effective_max_model_len("x", 8192) == 8192
        with _with(MISTRAL_24B):
            assert E.effective_max_model_len("x") == 10_000
        with _with(None):                       # unreadable: the default stands
            assert E.effective_max_model_len("x") == 10_000

    def test_dtype_follows_the_checkpoint_then_defaults_to_float32(self):
        with _with(_variant(MISTRAL_24B, drop=("torch_dtype",))):
            assert E.estimate_kv_gib("x", max_model_len=32768) == 10.0
            assert E.estimate_kv_gib("x", max_model_len=32768, dtype="auto") == 10.0
            assert E.estimate_kv_gib("x", max_model_len=32768, dtype="bfloat16") == 5.0

    def test_none_tensor_parallel_size_counts_as_one(self):
        with _with(MISTRAL_24B):
            assert E.estimate_kv_gib(
                "x", max_model_len=32768, tensor_parallel_size=None) == 5.0

    def test_missing_context_length_uses_the_default(self):
        cfg = SimpleNamespace(num_hidden_layers=40, hidden_size=5120,
                              num_attention_heads=32, torch_dtype="bfloat16")
        with _with(cfg):
            # No num_key_value_heads: all 32 heads are assumed cached.
            assert E.estimate_kv_gib("x") == 7.63          # at 10000 tokens
            assert E.estimate_kv_gib("x", max_model_len=32768) == 25.0

    def test_scales_linearly_with_the_context_length(self):
        with _with(MISTRAL_24B):
            one = E.estimate_kv_gib("x", max_model_len=16384)
            longer = E.estimate_kv_gib("x", max_model_len=32768)
        assert longer == pytest.approx(one * 2, rel=0.01)

    def test_falls_back_to_the_default_context(self):
        with _with(MISTRAL_24B):
            from_arg = E.estimate_kv_gib("x", max_model_len=4096)
            from_default = E.estimate_kv_gib("x")
        assert from_arg < from_default  # 4096 < 10000

    def test_gqa_is_modelled(self):
        """kv_heads (8) not attention heads (32) — a 4x difference here."""
        mha = _variant(MISTRAL_24B, num_key_value_heads=32)
        with _with(MISTRAL_24B):
            gqa = E.estimate_kv_gib("x", max_model_len=4096)
        with _with(mha):
            full = E.estimate_kv_gib("x", max_model_len=4096)
        assert full == pytest.approx(gqa * 4, rel=0.01)


class TestDtype:
    """An explicit dtype, then the checkpoint's own, then float32."""

    @pytest.mark.parametrize("dtype,expected", [
        ("float64", 8), ("float32", 4), ("torch.float16", 2), ("bfloat16", 2),
        ("half", 2), ("not-a-dtype", 4),
    ])
    def test_explicit_dtype_wins(self, dtype, expected):
        assert E._resolve_dtype(
            dtype, SimpleNamespace(torch_dtype="float64"))[1] == expected

    @pytest.mark.parametrize("dtype", [None, "auto"])
    def test_auto_and_none_use_the_checkpoint_dtype(self, dtype):
        assert E._resolve_dtype(dtype, SimpleNamespace(torch_dtype="bfloat16"))[1] == 2
        assert E._resolve_dtype(dtype, SimpleNamespace(torch_dtype="float32"))[1] == 4
        assert E._resolve_dtype(dtype, SimpleNamespace(dtype="float16"))[1] == 2

    @pytest.mark.parametrize("dtype", [None, "auto"])
    def test_undeclared_dtype_is_float32(self, dtype):
        assert E._resolve_dtype(dtype)[1] == 4
        assert E._resolve_dtype(dtype, SimpleNamespace())[1] == 4
        assert E._resolve_dtype(dtype, SimpleNamespace(torch_dtype="auto"))[1] == 4

    def test_nested_text_config_dtype_is_found(self):
        cfg = SimpleNamespace(text_config=SimpleNamespace(torch_dtype="bfloat16"))
        assert E._resolve_dtype(None, cfg)[1] == 2

    def test_weights_follow_the_resolution(self):
        bf16, fp32 = pytest.approx(5.98, abs=0.01), pytest.approx(11.97, abs=0.01)
        with _with(LLAMA_3B):
            assert E.estimate_weights_gib("x") == bf16
            assert E.estimate_weights_gib("x", dtype="auto") == bf16
        with _with(_variant(LLAMA_3B, torch_dtype="float32")):
            assert E.estimate_weights_gib("x", dtype="auto") == fp32
            assert E.estimate_weights_gib("x", dtype="bfloat16") == bf16
        with _with(_variant(LLAMA_3B, drop=("torch_dtype",))):
            assert E.estimate_weights_gib("x") == fp32


class _FakeAutoConfig:
    """Stands in for ``transformers.AutoConfig``."""

    calls: list = []
    configs: dict = {}

    @classmethod
    def from_pretrained(cls, hf_model_id, **kwargs):
        cls.calls.append((hf_model_id, kwargs))
        if hf_model_id not in cls.configs:
            raise OSError("401 gated repo")
        return cls.configs[hf_model_id]


@pytest.fixture()
def fake_transformers():
    _FakeAutoConfig.calls = []
    _FakeAutoConfig.configs = {"org/llama": LLAMA_3B}
    module = SimpleNamespace(AutoConfig=_FakeAutoConfig)
    with patch.dict(sys.modules, {"transformers": module}):
        yield _FakeAutoConfig


class TestLoadConfig:
    def test_reads_through_autoconfig_without_remote_code(self, fake_transformers):
        assert E._load_config("org/llama") is LLAMA_3B
        assert fake_transformers.calls == [("org/llama", {"trust_remote_code": False})]

    def test_one_estimate_reads_the_config_once(self, fake_transformers):
        est = E.estimate("org/llama", max_model_len=384)
        assert est.weights_gib == pytest.approx(5.98, abs=0.01) and est.kv_gib == 0.04
        assert len(fake_transformers.calls) == 1

    def test_nothing_is_kept_between_estimates(self, fake_transformers):
        E.estimate("org/llama")
        E.estimate("org/llama")
        assert len(fake_transformers.calls) == 2

    def test_a_failed_read_is_not_repeated_within_one_estimate(self, fake_transformers):
        assert E.estimate("org/gated") is None
        assert len(fake_transformers.calls) == 1

    def test_unreadable_config_is_none_and_says_why_once(
            self, fake_transformers, caplog):
        with caplog.at_level(logging.INFO, logger=E.logger.name):
            assert E.estimate_weights_gib("org/gated") is None
            assert E.estimate_kv_gib("org/gated") is None
        messages = [r.getMessage() for r in caplog.records]
        assert len(messages) == 1
        assert "org/gated" in messages[0] and "401 gated repo" in messages[0]

    def test_missing_transformers_is_none_and_says_why(self, caplog):
        real_import = builtins.__import__

        def no_transformers(name, *args, **kwargs):
            if name == "transformers":
                raise ImportError("No module named 'transformers'")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", no_transformers), \
             caplog.at_level(logging.INFO, logger=E.logger.name):
            assert E._load_config("org/any") is None
        assert "transformers is not installed" in caplog.text

    def test_missing_shape_fields_say_why(self, caplog):
        with _with(SimpleNamespace(num_hidden_layers=28)), \
             caplog.at_level(logging.INFO, logger=E.logger.name):
            assert E.estimate_weights_gib("org/odd") is None
        assert "lacks the shape fields" in caplog.text

    def test_text_config_supplies_the_shapes(self, fake_transformers):
        """Only the language model is sized; its dtype may sit on the outer config."""
        fake_transformers.configs["org/vlm"] = SimpleNamespace(
            text_config=_variant(LLAMA_3B, drop=("torch_dtype",)),
            vision_config=SimpleNamespace(hidden_size=1280),
            torch_dtype="bfloat16",
        )
        assert E.estimate_weights_gib("org/vlm") == pytest.approx(5.98, abs=0.01)
        assert E.estimate_kv_gib("org/vlm", max_model_len=131072) == 14.0


class TestPlanning:
    def test_adds_headroom_over_the_floor(self):
        """weights + kv is a bound no model runs in; the result must exceed it."""
        assert E.planning_gib(10.0, 2.0) > FLOOR_GIB

    def test_exact_value(self):
        assert E.planning_gib(10.0, 2.0) == 14.0     # 12 x 1.1 + 0.6 = 13.8
        assert E.planning_gib(43.91, 5.0) == 54.5    # 48.91 x 1.1 + 0.6 = 54.40
        assert E.planning_gib(10.0, None) == 12.0    # 10 x 1.1 + 0.6 = 11.6

    def test_rounds_up_to_half_a_gib(self):
        got = E.planning_gib(10.0, 2.0)
        assert got * 2 == int(got * 2)

    def test_unknown_weights_stay_unknown(self):
        """Headroom over an unknown base would be a fabricated number."""
        assert E.planning_gib(None, 2.0) is None

    def test_missing_kv_still_yields_a_lower_bound(self):
        assert E.planning_gib(10.0, None) is not None

    def test_padded_is_the_unrounded_figure(self):
        assert E.padded_gib(12.0) == pytest.approx(13.8)     # 12 x 1.1 + 0.6
        assert E.planning_gib(10.0, 2.0) == 14.0             # rounded up from it


def test_gib_not_decimal_gb():
    """GiB throughout, matching nvidia-smi and every GPU spec sheet. A
    decimal-GB divisor would read ~7% higher."""
    cfg = SimpleNamespace(
        num_hidden_layers=64, hidden_size=1024, intermediate_size=1024,
        num_attention_heads=8, num_key_value_heads=8, head_dim=128,
        vocab_size=1000, max_position_embeddings=128,
        hidden_act="gelu", tie_word_embeddings=True, torch_dtype="float16",
        model_type="gpt_neox",
    )
    with _with(cfg):
        gib = E.estimate_weights_gib("x")
    # Per layer: attention 4 x 1024^2, a two-matrix MLP 2 x 1024^2. x2 bytes.
    expected_bytes = (64 * 6 * 1024 * 1024 + 1000 * 1024) * 2
    assert gib == pytest.approx(expected_bytes / GIB, abs=0.006)
    assert gib != pytest.approx(expected_bytes / 1e9, abs=0.02)


class TestEstimate:
    """The one call a planner makes: every figure, from one config read."""

    def test_returns_every_figure(self):
        with _with(MISTRAL_24B):
            est = E.estimate("x", max_model_len=32768)
        assert est.weights_gib == pytest.approx(43.91, abs=0.01)
        assert est.kv_gib == 5.0
        assert est.per_gpu_gib == 54.5 and est.total_gib == 54.5
        assert (est.tensor_parallel_size, est.attention_heads) == (1, 32)
        assert est.description == "43.9 weights + 5.00 KV, x1.1 +0.6 [bfloat16]"

    def test_tensor_parallel_figures_are_per_gpu_and_total(self):
        with _with(MISTRAL_24B):
            est = E.estimate("x", max_model_len=32768, tensor_parallel_size=2)
        # (21.95 + 2.5) x 1.1 + 0.6 = 27.5 on each card: more than 54.5 / 2,
        # since every card carries its own context.
        assert est.per_gpu_gib == 27.5 and est.total_gib == 55.0
        assert est.description == (
            "21.9 weights + 2.50 KV, x1.1 +0.6, per GPU of 2 [bfloat16]")

    def test_max_model_len_sizes_the_kv_cache(self):
        with _with(MISTRAL_24B):
            est = E.estimate("x", max_model_len=8192)
        assert est.kv_gib == 1.25 and est.per_gpu_gib == 50.5

    def test_kv_can_be_left_out(self):
        with _with(MISTRAL_24B):
            est = E.estimate("x", kv=False)
        assert est.kv_gib is None
        assert est.per_gpu_gib == 49.0         # 43.91 x 1.1 + 0.6 = 48.9
        assert est.description == "43.9 weights, x1.1 +0.6 [bfloat16]"

    def test_description_names_what_drives_the_number(self):
        with _with(MISTRAL_24B):
            awq = E.estimate("x", quantization="awq").description
            unmodelled = E.estimate("x", quantization="gguf").description
            explicit = E.estimate("x", dtype="float32").description
        with _with(MIXTRAL):
            moe = E.estimate("x").description
        with _with(_variant(MISTRAL_24B, drop=("torch_dtype",))):
            assumed = E.estimate("x").description
        assert awq.endswith("[bfloat16, awq 4-bit]")
        assert unmodelled.endswith("[bfloat16, gguf sized unquantized]")
        assert explicit.endswith("[float32]")
        assert moe.endswith("[bfloat16, 8 experts]")
        assert assumed.endswith("[float32 assumed]")

    def test_unreadable_config_is_none(self):
        with _with(None):
            assert E.estimate("x") is None

    def test_none_tensor_parallel_size_counts_as_one(self):
        with _with(MISTRAL_24B):
            assert E.estimate(
                "x", max_model_len=32768, tensor_parallel_size=None).total_gib == 54.5

    def test_one_config_read_serves_the_whole_estimate(self, fake_transformers):
        est = E.estimate("org/llama", max_model_len=384)
        assert est.per_gpu_gib == 7.5          # (5.98 + 0.04) x 1.1 + 0.6 = 7.22
        assert len(fake_transformers.calls) == 1


class TestFloat32AsHalf:
    """vLLM's default dtype loads a float32 checkpoint as 16-bit."""

    def test_a_float32_checkpoint_is_sized_at_16_bit(self):
        with _with(GEMMA_9B):
            full = E.estimate_weights_gib("x")
            half = E.estimate_weights_gib("x", float32_as_half=True)
        assert half == pytest.approx(full / 2, abs=0.01)

    def test_an_undeclared_dtype_is_sized_at_16_bit(self):
        cfg = SimpleNamespace(**{**vars(LLAMA_3B), "torch_dtype": None})
        with _with(cfg):
            assert E.estimate_weights_gib("x", float32_as_half=True) == pytest.approx(
                5.98, abs=0.01)

    def test_an_explicit_float32_is_kept(self):
        with _with(GEMMA_9B):
            full = E.estimate_weights_gib("x")
            asked = E.estimate_weights_gib("x", dtype="float32", float32_as_half=True)
        assert asked == full

    def test_the_description_says_it_was_cast(self):
        with _with(GEMMA_9B):
            est = E.estimate("x", float32_as_half=True)
        assert "float16, cast from float32" in est.description

    def test_an_unrecognised_dtype_is_not_halved(self):
        with _with(LLAMA_3B):
            fp32 = E.estimate_weights_gib("x", dtype="float32")
            odd = E.estimate_weights_gib("x", dtype="int8", float32_as_half=True)
        assert odd == fp32
