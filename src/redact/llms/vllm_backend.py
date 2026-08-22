"""Local vLLM backend for self-hosted inference.

Heavy init — loads model weights into GPU memory once. After that,
generate() and batch_generate() are fast. batch_generate() is
especially efficient because vLLM processes all prompts in a single
engine pass.

Reference: output dataset paraphraser.py (TheBloke/dolphin-2.2-70B-GPTQ).
"""

import contextlib
import multiprocessing
import os

from .base import LLMBackend
from .model_config import get_model_config

# Mistral-3.x instruct format, used only when use_mistral_format=True (see
# _build_mistral_prompt below) — CAUTION, see that function's docstring.
_MISTRAL_SYSTEM_TEMPLATE = "[SYSTEM_PROMPT]{system}[/SYSTEM_PROMPT]"
_MISTRAL_INST_TEMPLATE = "[INST]{user}[/INST]"


def _build_mistral_prompt(messages: list[dict]) -> str:
    """Render OpenAI-style chat messages as a raw Mistral-3.x instruct prompt.

    CAUTION: this is a best-effort reconstruction, written without access to
    a live vLLM + Mistral-3.x checkpoint to validate against — see the
    ``use_mistral_format`` note in ``VLLMBackend.__init__`` before relying on
    this in production.

    Targets the Mistral-3.x format: any system message(s) become one leading
    ``[SYSTEM_PROMPT]...[/SYSTEM_PROMPT]`` block, then each user turn becomes
    ``[INST]...[/INST]`` with assistant turns appended as plain text between
    them (matching multi-turn Mistral instruct conversations). System
    messages after the first user turn are folded into that leading block too
    (Mistral's format has no mid-conversation system slot).
    """
    system_parts = [m["content"] for m in messages if m.get("role") == "system"]
    parts = []
    if system_parts:
        parts.append(_MISTRAL_SYSTEM_TEMPLATE.format(system="\n".join(system_parts)))
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue
        elif role == "user":
            parts.append(_MISTRAL_INST_TEMPLATE.format(user=m["content"]))
        elif role == "assistant":
            parts.append(m["content"])
    return "".join(parts)


class VLLMBackend(LLMBackend):
    """Backend for local vLLM inference."""

    def __init__(
        self,
        model: str,
        quantization: str | None = None,
        **vllm_kwargs,
    ):
        """Load a model with vLLM.

        Args:
            model: HuggingFace model ID or local path.
            quantization: Quantization method (e.g. "gptq", "awq").
            **vllm_kwargs: Passed to vllm.LLM() (e.g. revision,
                           trust_remote_code, gpu_memory_utilization), with
                           one exception: ``use_mistral_format`` (bool) is
                           popped out here rather than forwarded — it is not
                           a real vllm.LLM() kwarg, it's this backend's own
                           flag (see the note below).
        """
        # vLLM v1 spawns engine core subprocesses. On Linux the default
        # multiprocessing start method is 'fork', which causes CUDA to fail
        # if it was already initialised in the parent (e.g. in a notebook).
        with contextlib.suppress(RuntimeError):
            multiprocessing.set_start_method("spawn", force=True)

        # WSL2-specific workaround, verified by actually hitting this on a
        # real WSL2 + RTX 2080 Super setup: vLLM's V2 GPU model runner
        # unconditionally allocates a UVA (Unified Virtual Addressing)
        # buffer, but is_pin_memory_available() defaults to False on WSL2,
        # so UVA allocation raises `RuntimeError: UVA is not available` on
        # every load. This is an open, unmerged upstream bug as of vLLM
        # 0.27.1 (vllm-project/vllm#47387, fix proposed in #47579) — forcing
        # the older V1 runner sidesteps it entirely (no UVA dependency).
        # setdefault so an explicit VLLM_USE_V2_MODEL_RUNNER from the caller
        # still wins; remove this block once the upstream fix ships.
        with contextlib.suppress(OSError), open("/proc/version") as f:
            if "microsoft" in f.read().lower():
                os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")

        from vllm import LLM  # Lazy import — vllm is heavy and optional

        self._model_name = model
        # CAUTION — best-effort implementation, not yet validated against a
        # live vLLM engine + real Mistral-3.x weights (no GPU available where
        # this was written): when True, generate()/batch_generate() build a
        # raw [INST]/[SYSTEM_PROMPT] prompt string (_build_mistral_prompt)
        # and call self._llm.generate() instead of self._llm.chat() — for
        # checkpoints whose tokenizer_config.json lacks a working
        # chat_template, so vLLM's own template application can't be trusted.
        # Required for Mistral-3.x family models per the project docs. Before
        # depending on this in production, verify the constructed prompt
        # against the target checkpoint's own tokenizer (e.g. compare with
        # AutoTokenizer.apply_chat_template, or against passing
        # tokenizer_mode="mistral" — a real vllm.LLM() kwarg that lets vLLM's
        # own mistral-common integration format the prompt instead of this
        # hand-rolled version, if available for the installed vllm version).
        self._use_mistral_format = bool(vllm_kwargs.pop("use_mistral_format", False))
        if "download_dir" not in vllm_kwargs and os.environ.get("HF_HOME"):
            vllm_kwargs["download_dir"] = os.environ["HF_HOME"]
        self._llm = LLM(
            model=model,
            quantization=quantization,
            **vllm_kwargs,
        )

    def _clean_output(self, text: str) -> str:
        """Strip known tokenizer artifacts from raw vLLM output.

        Belt-and-suspenders fallback. With llm.chat() the model's own
        tokenizer handles stop tokens, so this should rarely fire.
        """
        text = text.strip()
        if text.startswith("|>"):
            text = text[2:].lstrip()
        return text

    def generate(
        self,
        messages: list[dict],
        model: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        **kwargs,
    ) -> str:
        """Generate a single response via local vLLM.

        Args:
            messages: Chat messages in OpenAI format.
            model: Model identifier — validated against loaded model but
                   not used to switch models (vLLM loads one model at init).
            max_tokens: Max tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
        """
        from vllm import SamplingParams

        config = get_model_config(model)
        resolved_max_tokens = max_tokens if max_tokens is not None else config.default_max_tokens
        resolved_temperature = temperature if temperature is not None else (config.default_temperature or 0.7)
        resolved_top_p = top_p if top_p is not None else 0.85

        sampling_params = SamplingParams(
            max_tokens=resolved_max_tokens,
            temperature=resolved_temperature,
            top_p=resolved_top_p,
        )

        if self._use_mistral_format:
            outputs = self._llm.generate([_build_mistral_prompt(messages)], sampling_params)
        else:
            outputs = self._llm.chat([messages], sampling_params)
        return self._clean_output(outputs[0].outputs[0].text)

    def batch_generate(
        self,
        messages_list: list[list[dict]],
        model: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        **kwargs,
    ) -> list[str]:
        """Generate responses for multiple message lists in a single vLLM pass.

        Much faster than looping over generate() because vLLM batches
        all prompts through the engine at once.
        """
        from vllm import SamplingParams

        config = get_model_config(model)
        resolved_max_tokens = max_tokens if max_tokens is not None else config.default_max_tokens
        resolved_temperature = temperature if temperature is not None else (config.default_temperature or 0.7)
        resolved_top_p = top_p if top_p is not None else 0.85

        sampling_params = SamplingParams(
            max_tokens=resolved_max_tokens,
            temperature=resolved_temperature,
            top_p=resolved_top_p,
        )

        if not messages_list:
            return []
        if self._use_mistral_format:
            prompts = [_build_mistral_prompt(m) for m in messages_list]
            outputs = self._llm.generate(prompts, sampling_params)
        else:
            outputs = self._llm.chat(messages_list, sampling_params)
        return [self._clean_output(out.outputs[0].text) for out in outputs]

    @property
    def backend_name(self) -> str:
        return "vllm"

    @property
    def supports_native_batching(self) -> bool:
        # batch_generate() runs a single engine pass over the whole list.
        return True

    @property
    def supports_parallel_calls(self) -> bool:
        # Concurrent generate() calls from threads compete for GPU memory —
        # batching must go through batch_generate() instead.
        return False

    def __repr__(self) -> str:
        return f"VLLMBackend(model={self._model_name!r})"
