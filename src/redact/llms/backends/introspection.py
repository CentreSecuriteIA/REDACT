"""Local raw-transformers backend for deep-internals logging.

Not a vLLM extension — vLLM's serving API (continuous batching + paged
KV-cache) discards intermediate activations by design, so there is no
supported way to recover hidden states or attention weights from it. This
backend is a standalone model load via HuggingFace ``transformers``, used
instead of vLLM for whichever run needs internals captured. Its loaded-model
cache is therefore **separate from vLLM's**: the same ``hf_model_id`` under
the two backends is two different runtimes and two independent loads.

Capture is a side effect keyed on caller-supplied ``internals_ids`` (see
:attr:`LLMBackend.compute_config`'s ``supports_internals``) — the backend
never invents its own ids. A ``None`` entry means generation proceeds
normally with nothing captured for that batch item.

One instance **is one registered model**: sampling defaults, capture config
and log directory are bound in
:meth:`TransformersIntrospectionBackend.from_config`, not re-passed per call.

Reference: backends/vllm.py (lazy import + cache pattern).
"""

import json
import threading
import time
from pathlib import Path
from typing import ClassVar

from . import vram
from .base import ComputeConfig, LLMBackend

_DEFAULT_CAPTURE = {"logprobs": True, "hidden_states": "last", "attention": False}

# Sampling values with no ModelConfig field of their own — a backend default,
# not a registry lookup. Overridable per model via IntrospectConfig.sampling.
_DEFAULT_SAMPLING = {"top_p": 0.85}
_DEFAULT_TEMPERATURE = 0.7

# Loaded (tokenizer, model) pairs, keyed on what determines the load. Private
# to this class — never shared with vLLM's engine cache.
_models: dict[tuple, tuple] = {}
# Same reasoning as vllm.py's _engines_lock: preload runs off-thread, and a
# duplicate from_pretrained() would put two copies of the weights on the card.
_models_lock = threading.Lock()


def _load(hf_model_id: str, device_map: str, torch_dtype: str, hf_kwargs: dict):
    """Get or load the tokenizer + model pair for one checkpoint."""
    key = (hf_model_id, device_map, torch_dtype, repr(sorted(hf_kwargs.items())))
    if key in _models:
        return _models[key]

    with _models_lock:
        # Re-check: another thread may have finished the load while we waited.
        if key in _models:
            return _models[key]
        return _load_locked(key, hf_model_id, device_map, torch_dtype, hf_kwargs)


def _load_locked(key, hf_model_id: str, device_map: str, torch_dtype: str, hf_kwargs: dict):
    """The real load. Only ever called with ``_models_lock`` held."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError(
            "The 'torch' and 'transformers' packages are required for "
            "TransformersIntrospectionBackend. Install them with: "
            "pip install redact[transformers]"
        )

    tokenizer = AutoTokenizer.from_pretrained(hf_model_id, **hf_kwargs)
    # transformers renamed `torch_dtype` to `dtype` in 5.0 (4.x still
    # accepts torch_dtype but warns; 5.x warns on torch_dtype). Pick the
    # name the installed version actually wants, since pyproject allows
    # transformers>=4.40 — i.e. both sides of that rename.
    import transformers

    dtype_kwarg = (
        "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    )
    with vram.Measurement() as measured:
        model = AutoModelForCausalLM.from_pretrained(
            hf_model_id, device_map=device_map, **{dtype_kwarg: torch_dtype}, **hf_kwargs,
        )
    model.eval()

    # Unlike vLLM this preallocates no KV pool, so the measured delta really is
    # the model's need — vram.planning_gb() relies on the recorded backend name
    # to tell the two cases apart.
    from ... import paths  # local import: paths is above llms in the tree

    measured.record(
        paths.vram_cache_json(),
        f"introspect:{hf_model_id}",
        {"device_map": device_map, "torch_dtype": torch_dtype},
        backend="introspect",
    )
    _models[key] = (tokenizer, model)
    return _models[key]


class TransformersIntrospectionBackend(LLMBackend):
    """One model on local HF ``transformers``, with internals capture."""

    # No true engine-level batch pass — single GPU, no continuous
    # batching (native_batching=False). BatchCaller routes through run()
    # instead, which already zips internals_ids[i] into each sequential
    # generate() call. Concurrent generate() calls from threads compete
    # for GPU memory, same reasoning as VLLMBackend (parallel_calls=False).
    #
    # native_batching stays False *even though* generate() below accepts and
    # loops over a full list, so it would technically survive being handed the
    # whole batch. Two reasons not to:
    #
    #   1. The flag means "one transport call is one engine pass", which is
    #      what lets a caller read the trace: vLLM fuses 32 prompts into a
    #      single `call` event with n_items=32. This backend runs 32 forward
    #      passes and correctly emits 32 events. Claiming native batching
    #      would leave a "native" transport emitting per-item events, and the
    #      telemetry contract stops describing anything.
    #   2. ModelClient's native branch fires on_complete only after the whole
    #      call returns, so progress would jump 0 -> 100% at the end — on the
    #      slowest backend here, where a batch takes minutes. Nothing
    #      checkpoints off on_complete (resume is ledger-driven per chunk), so
    #      this is display only, but it is display exactly where it matters.
    #
    # What going native would actually save is the BatchCaller wrap and N-1
    # redundant _prepare()/_resolve() calls — noise next to GPU generation.
    compute_config: ClassVar[ComputeConfig] = ComputeConfig(
        supports_native_batching=False,
        supports_parallel_calls=False,
        supports_internals=True,
    )

    def __init__(
        self,
        model: str,
        hf_model_id: str,
        log_dir: str | Path,
        *,
        capture: dict | None = None,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        sampling: dict | None = None,
        hf_kwargs: dict | None = None,
        **identity,
    ):
        """Bind one registered model to a locally-loaded HF model.

        Args:
            model: Registry name of the model.
            hf_model_id: HuggingFace model ID or local path.
            log_dir: Root directory captured internals are written under
                (one subfolder per ``internals_id``).
            capture: Which internals to capture. Keys: ``"logprobs"`` (bool),
                ``"hidden_states"`` (``False`` / ``"last"`` / ``"all"``),
                ``"attention"`` (bool). Defaults are deliberately light
                (logprobs + last-layer hidden state only) — attention scales
                ``layers x heads x seq_len**2`` and this backend has no
                batching to amortize storage cost over.
            device_map / torch_dtype: Passed to ``from_pretrained``.
            sampling: Extra generation defaults for this model (``top_p``,
                ...), merged over :data:`_DEFAULT_SAMPLING`.
            hf_kwargs: Extra kwargs for ``from_pretrained`` (e.g.
                ``trust_remote_code``).
            **identity: Forwarded to :meth:`LLMBackend.__init__`.
        """
        try:
            import torch  # Lazy import — torch is heavy and optional
        except ImportError:
            raise ImportError(
                "The 'torch' and 'transformers' packages are required for "
                "TransformersIntrospectionBackend. Install them with: "
                "pip install redact[transformers]"
            )

        super().__init__(model, **identity)
        # A real number is required for sampling; ModelConfig's is optional.
        if self.default_temperature is None:
            self.default_temperature = _DEFAULT_TEMPERATURE
        self.hf_model_id = hf_model_id
        self._log_dir = Path(log_dir)
        self._capture = {**_DEFAULT_CAPTURE, **(capture or {})}
        self._sampling = {**_DEFAULT_SAMPLING, **(sampling or {})}
        self._torch = torch
        self._tokenizer, self._model = _load(
            hf_model_id, device_map, torch_dtype, hf_kwargs or {}
        )

    @classmethod
    def from_config(cls, config) -> "TransformersIntrospectionBackend":
        """Build from a registry entry's ``.introspect`` setup.

        Raises:
            ValueError: If the entry has no ``.introspect`` setup.
        """
        introspect = config.introspect
        if introspect is None:
            raise ValueError(
                f"Model {config.name!r} has no introspect setup. Register it "
                f"with register_model(..., introspect=IntrospectConfig("
                f"hf_model_id='...', log_dir='...'))."
            )
        return cls(
            hf_model_id=introspect.hf_model_id,
            log_dir=introspect.log_dir,
            capture=introspect.capture,
            device_map=introspect.device_map,
            torch_dtype=introspect.torch_dtype,
            sampling=introspect.sampling,
            hf_kwargs=introspect.extra_kwargs,
            **cls._identity(config),
        )

    @classmethod
    def clear_cache(cls) -> None:
        """Drop the loaded models — frees GPU memory between local models."""
        _models.clear()

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

    def generate(
        self,
        messages_list: list[list[dict]],
        *,
        system_prompts: str | list[str | None] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        internals_ids: list[str | None] | None = None,
        **kwargs,
    ) -> list[str]:
        """Generate responses for a batch, optionally capturing internals per item.

        Sequential — this backend makes no claim of true batching (single
        GPU, no continuous batching; ``compute_config.supports_native_batching
        =False``). A single sample is just a batch of one, same code path.

        The per-item work is inline in the loop, matching ``openai.py`` and
        ``anthropic.py``: every non-fusing backend here is a batch-shaped
        ``generate()`` wrapping its own item loop, and only ``vllm.py`` differs
        because its pass genuinely is fused. Keeping the body here also keeps
        it *behind* ``_prepare()`` — the system-prompt split/fold has to run
        over the batch before any item is touched, so there is deliberately no
        single-item entry point for a caller to reach past it.

        Args:
            messages_list: One chat message list per batch item.
            system_prompts: Optional system prompt(s) — re-inserted before the
                tokenizer's chat template is applied.
            max_tokens: Overrides this model's default.
            temperature: Overrides this model's default.
            internals_ids: One internals-capture id (or None) per item. When
                an item's id is given, captured internals (per ``capture``
                config) are written under ``{log_dir}/{internals_id}/``.
            **kwargs: Extra generation parameters (e.g. top_k,
                repetition_penalty), merged over this model's sampling
                defaults.

        Returns:
            Generated text, one per batch item, same order as ``messages_list``.
        """
        if not messages_list:
            return []
        prompts, resolved = self._prepare(messages_list, system_prompts)
        max_tok, temp = self._resolve(max_tokens, temperature)
        ids = internals_ids if internals_ids is not None else [None] * len(messages_list)
        sampling = {**self._sampling, **kwargs}

        results: list[str] = []
        for messages, system_prompt, internals_id in zip(resolved, prompts, ids):
            chat_messages = (
                [{"role": "system", "content": system_prompt}, *messages]
                if system_prompt else messages
            )
            prompt = self._tokenizer.apply_chat_template(
                chat_messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            prompt_len = inputs["input_ids"].shape[1]

            want_capture = internals_id is not None
            started = time.perf_counter()
            with self._torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tok,
                    temperature=temp,
                    do_sample=True,
                    return_dict_in_generate=True,
                    output_scores=want_capture and self._capture.get("logprobs", False),
                    output_hidden_states=(
                        want_capture and bool(self._capture.get("hidden_states"))
                    ),
                    output_attentions=want_capture and self._capture.get("attention", False),
                    **sampling,
                )

            # One forward pass = one event; this backend never fuses a batch,
            # so a batch of N emits N events rather than one with n_items=N.
            self._record_call(
                n_items=1, started=started,
                in_tok=prompt_len,
                out_tok=int(outputs.sequences[0].shape[0]) - prompt_len,
                max_tokens=max_tok, temperature=temp,
            )

            text = self._tokenizer.decode(
                outputs.sequences[0][prompt_len:], skip_special_tokens=True,
            ).strip()

            if want_capture:
                meta = {
                    "model": self.model,
                    "hf_model_id": self.hf_model_id,
                    "settings": {
                        "max_tokens": max_tok,
                        "temperature": temp,
                        **sampling,
                        "capture": self._capture,
                    },
                    "messages": chat_messages,
                    "output": text,
                }
                self._save_capture(internals_id, outputs, prompt_len, meta)

            results.append(text)

        return results

    @property
    def backend_name(self) -> str:
        return "introspect"

    def __repr__(self) -> str:
        return (
            f"TransformersIntrospectionBackend(model={self.model!r}, "
            f"hf_model_id={self.hf_model_id!r})"
        )
