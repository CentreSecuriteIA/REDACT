"""Local vLLM backend for self-hosted inference.

Heavy init — loads model weights into GPU memory once. After that,
generate() and batch_generate() are fast. batch_generate() is
especially efficient because vLLM processes all prompts in a single
engine pass.

Reference: output dataset paraphraser.py (TheBloke/dolphin-2.2-70B-GPTQ).
"""

import multiprocessing
import os

from .base import LLMBackend
from .model_config import get_model_config


class VLLMBackend(LLMBackend):
    """Backend for local vLLM inference."""

    def __init__(
        self,
        model: str,
        quantization: str | None = None,
        chat_template: str | None = None,
        **vllm_kwargs,
    ):
        """Load a model with vLLM.

        Args:
            model: HuggingFace model ID or local path.
            quantization: Quantization method (e.g. "gptq", "awq").
            chat_template: Optional chat template string. If None, uses
                           a default ChatML template.
            **vllm_kwargs: Passed to vllm.LLM() (e.g. revision,
                           trust_remote_code, gpu_memory_utilization).
        """
        # vLLM v1 spawns engine core subprocesses. On Linux the default
        # multiprocessing start method is 'fork', which causes CUDA to fail
        # if it was already initialised in the parent (e.g. in a notebook).
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        from vllm import LLM  # Lazy import — vllm is heavy and optional

        self._model_name = model
        if "download_dir" not in vllm_kwargs and os.environ.get("HF_HOME"):
            vllm_kwargs["download_dir"] = os.environ["HF_HOME"]
        self._llm = LLM(
            model=model,
            quantization=quantization,
            **vllm_kwargs,
        )
        self._chat_template = chat_template

    def _format_messages(self, messages: list[dict]) -> str:
        """Convert chat messages to a prompt string using ChatML format."""
        if self._chat_template:
            return self._chat_template.format(messages=messages)

        # Default ChatML format (used by Dolphin, Hermes, etc.)
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        parts.append("<|im_start|>assistant")
        return "\n".join(parts)

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
            stop=["<|im_end|>"],  # prevent ChatML end-token from leaking into output
        )

        formatted = self._format_messages(messages)
        outputs = self._llm.generate([formatted], sampling_params)
        return outputs[0].outputs[0].text.strip()

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
            stop=["<|im_end|>"],  # prevent ChatML end-token from leaking into output
        )

        if not messages_list:
            return []
        formatted = [self._format_messages(msgs) for msgs in messages_list]
        outputs = self._llm.generate(formatted, sampling_params)
        return [out.outputs[0].text.strip() for out in outputs]

    @property
    def backend_name(self) -> str:
        return "vllm"

    def __repr__(self) -> str:
        return f"VLLMBackend(model={self._model_name!r})"
