"""Local raw-transformers backend for deep-internals logging.

Not a vLLM extension — vLLM's serving API (continuous batching + paged
KV-cache) discards intermediate activations by design, so there is no
supported way to recover hidden states or attention weights from it. This
backend is a standalone model load via HuggingFace ``transformers``, used
instead of vLLM for whichever run needs internals captured, not alongside a
live vLLM instance of the "same" weights.

Capture is a side effect keyed on caller-supplied ``internals_id``/
``internals_ids`` (see :meth:`LLMBackend.supports_internals`) — the backend
never invents its own ids. When no id is supplied, generation proceeds
normally with nothing captured.

Reference: llms/vllm_backend.py (lazy import + registry pattern).
"""

import json
from pathlib import Path

from .base import LLMBackend
from .model_config import get_model_config

_DEFAULT_CAPTURE = {"logprobs": True, "hidden_states": "last", "attention": False}


class TransformersIntrospectionBackend(LLMBackend):
    """Backend for local HF ``transformers`` inference with internals capture."""

    def __init__(
        self,
        model: str,
        log_dir: str | Path,
        capture: dict | None = None,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        **hf_kwargs,
    ):
        """Load a model with raw ``transformers`` for internals capture.

        Args:
            model: HuggingFace model ID or local path.
            log_dir: Root directory captured internals are written under
                (one subfolder per ``internals_id``).
            capture: Which internals to capture. Keys: ``"logprobs"`` (bool),
                ``"hidden_states"`` (``False`` / ``"last"`` / ``"all"``),
                ``"attention"`` (bool). Defaults are deliberately light
                (logprobs + last-layer hidden state only) — attention scales
                ``layers x heads x seq_len**2`` and this backend has no
                batching to amortize storage cost over.
            device_map / torch_dtype: Passed to ``AutoModelForCausalLM.from_pretrained``.
            **hf_kwargs: Extra kwargs passed to ``from_pretrained`` (e.g.
                ``trust_remote_code``).
        """
        import torch  # Lazy import — torch/transformers are heavy and optional
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._model_name = model
        self._log_dir = Path(log_dir)
        self._capture = {**_DEFAULT_CAPTURE, **(capture or {})}
        self._torch = torch

        self._tokenizer = AutoTokenizer.from_pretrained(model, **hf_kwargs)
        self._model = AutoModelForCausalLM.from_pretrained(
            model, device_map=device_map, torch_dtype=torch_dtype, **hf_kwargs,
        )
        self._model.eval()

    def _capture_dir(self, internals_id: str) -> Path:
        d = self._log_dir / internals_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def rename_capture(self, old_internals_id: str, new_internals_id: str) -> None:
        """Move a capture folder from a provisional id to its final one.

        No-op if nothing was captured under ``old_internals_id`` (e.g. the
        call wasn't given an ``internals_id`` at all).
        """
        old_dir = self._log_dir / old_internals_id
        if not old_dir.exists():
            return
        new_dir = self._log_dir / new_internals_id
        new_dir.parent.mkdir(parents=True, exist_ok=True)
        old_dir.rename(new_dir)

    def _save_meta(self, out_dir: Path, meta: dict) -> None:
        """Always-written companion JSON: resolved settings, prompt, and output.

        Not gated by ``capture`` config — cheap, and it's the *only* place some
        of this data survives at all. Multi-round jailbreak techniques discard
        intermediate prompts/completions once the final row is written (a
        rejected translation attempt, a scenario draft); this file is what
        keeps that data recoverable for analysis.
        """
        with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def _save_capture(self, internals_id: str, outputs, prompt_len: int, meta: dict) -> None:
        """Persist whatever ``capture`` config asks for from one generate() call."""
        torch = self._torch
        out_dir = self._capture_dir(internals_id)
        self._save_meta(out_dir, meta)
        gen_token_ids = outputs.sequences[0][prompt_len:]

        if self._capture.get("logprobs") and outputs.scores:
            logprobs = [
                torch.log_softmax(step_logits[0], dim=-1)
                for step_logits in outputs.scores
            ]
            record = {
                "tokens": self._tokenizer.convert_ids_to_tokens(gen_token_ids.tolist()),
                "token_ids": gen_token_ids.tolist(),
                "logprob": [
                    lp[tok].item() for lp, tok in zip(logprobs, gen_token_ids)
                ],
            }
            torch.save(record, out_dir / "logprobs.pt")

        hs_mode = self._capture.get("hidden_states")
        if hs_mode and outputs.hidden_states:
            # outputs.hidden_states: tuple (per generated token) of tuple (per layer)
            # of [batch, seq, hidden] tensors. Keep only the last generated token's
            # hidden state per layer per step (decode-step hidden states).
            layers = (
                [-1] if hs_mode == "last" else range(len(outputs.hidden_states[0]))
            )
            stacked = {
                li: torch.stack(
                    [step[li][0, -1, :].detach().cpu() for step in outputs.hidden_states]
                )
                for li in layers
            }
            torch.save(stacked, out_dir / "hidden_states.pt")

        if self._capture.get("attention") and outputs.attentions:
            stacked_attn = [
                torch.stack([layer[0].detach().cpu() for layer in step])
                for step in outputs.attentions
            ]
            torch.save(stacked_attn, out_dir / "attention.pt")

    def _generate_one(
        self,
        messages: list[dict],
        model: str,
        internals_id: str | None,
        max_tokens: int | None,
        temperature: float | None,
        top_p: float | None,
    ) -> str:
        config = get_model_config(model)
        resolved_max_tokens = max_tokens if max_tokens is not None else config.default_max_tokens
        resolved_temperature = temperature if temperature is not None else (config.default_temperature or 0.7)
        resolved_top_p = top_p if top_p is not None else 0.85

        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        prompt_len = inputs["input_ids"].shape[1]

        want_capture = internals_id is not None
        with self._torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=resolved_max_tokens,
                temperature=resolved_temperature,
                top_p=resolved_top_p,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=want_capture and self._capture.get("logprobs", False),
                output_hidden_states=want_capture and bool(self._capture.get("hidden_states")),
                output_attentions=want_capture and self._capture.get("attention", False),
            )

        text = self._tokenizer.decode(
            outputs.sequences[0][prompt_len:], skip_special_tokens=True,
        ).strip()

        if want_capture:
            meta = {
                "model": self._model_name,
                "settings": {
                    "max_tokens": resolved_max_tokens,
                    "temperature": resolved_temperature,
                    "top_p": resolved_top_p,
                    "capture": self._capture,
                },
                "messages": messages,
                "output": text,
            }
            self._save_capture(internals_id, outputs, prompt_len, meta)

        return text

    def generate(
        self,
        messages: list[dict],
        model: str,
        internals_id: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        **kwargs,
    ) -> str:
        """Generate a single response, optionally capturing internals.

        Args:
            messages: Chat messages in OpenAI format.
            model: Model identifier — validated against loaded model but not
                   used to switch models (this backend loads one model at init).
            internals_id: When given, captured internals (per ``capture``
                config) are written under ``{log_dir}/{internals_id}/``. When
                ``None``, generation proceeds with nothing captured.
            max_tokens / temperature / top_p: Sampling params.
        """
        return self._generate_one(messages, model, internals_id, max_tokens, temperature, top_p)

    def batch_generate(
        self,
        messages_list: list[list[dict]],
        model: str,
        internals_ids: list[str | None] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        **kwargs,
    ) -> list[str]:
        """Generate responses for multiple message lists.

        Sequential — this backend makes no claim of true batching (single
        GPU, no continuous batching). ``supports_native_batching`` is False,
        so ``BatchCaller`` normally reaches individual calls via ``generate()``
        instead; this override exists for callers using the backend directly.
        """
        ids = internals_ids if internals_ids is not None else [None] * len(messages_list)
        return [
            self._generate_one(msgs, model, iid, max_tokens, temperature, top_p)
            for msgs, iid in zip(messages_list, ids)
        ]

    @property
    def backend_name(self) -> str:
        return "transformers_introspect"

    @property
    def supports_native_batching(self) -> bool:
        # No true engine-level batch pass — single GPU, no continuous
        # batching. BatchCaller routes through run() instead, which already
        # zips internals_ids[i] into each sequential generate() call.
        return False

    @property
    def supports_parallel_calls(self) -> bool:
        # Concurrent generate() calls from threads compete for GPU memory,
        # same reasoning as VLLMBackend.
        return False

    @property
    def supports_internals(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f"TransformersIntrospectionBackend(model={self._model_name!r})"
