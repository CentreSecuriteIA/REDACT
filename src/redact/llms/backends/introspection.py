"""Local transformers backend that can capture model internals.

Loads the model through HuggingFace ``transformers`` instead of vLLM, which
does not expose hidden states or attention weights. Its model cache is
separate from vLLM's: the same ``hf_model_id`` under both backends is loaded
twice.

Capture is requested per item with caller-supplied ``internals_ids``. An item
whose id is ``None`` is generated without capturing anything.
"""

import atexit
import json
import logging
import shutil
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from .. import observe
from ..resources import measure
from .base import ComputeConfig, LLMBackend

if TYPE_CHECKING:
    from ..model_config import ModelConfig

logger = logging.getLogger(__name__)

_DEFAULT_CAPTURE = {"logprobs": True, "hidden_states": "last", "attention": False}
# The values each capture key accepts.
_CAPTURE_VALUES = {
    "logprobs": (True, False),
    "hidden_states": (False, "last", "all"),
    "attention": (True, False),
}

# Backend defaults for sampling values that have no ModelConfig field.
# Override them per model with IntrospectConfig.sampling.
_DEFAULT_SAMPLING = {"top_p": 0.85}
_DEFAULT_TEMPERATURE = 0.7

# Loaded (tokenizer, model) pairs, keyed on the load arguments.
_models: dict[tuple, tuple] = {}
# Held during a load, so a preload thread and a real call cannot load the same
# checkpoint twice.
_models_lock = threading.Lock()

# Load time of each checkpoint. Lifetime is timed per checkpoint, not per
# backend instance, and reported as a "local" release event.
_models_loaded_at: dict[tuple, float] = {}


def validate_capture(capture: dict | None) -> None:
    """Check the keys and values of a ``capture`` dict.

    Raises:
        ValueError: An unknown key, or a value that key does not accept.
    """
    for key, value in (capture or {}).items():
        allowed = _CAPTURE_VALUES.get(key)
        if allowed is None:
            raise ValueError(
                f"Unknown capture key {key!r}. Expected one of {sorted(_CAPTURE_VALUES)}."
            )
        # The type check keeps 1 and 0 from passing as True and False.
        if type(value) not in (bool, str) or value not in allowed:
            raise ValueError(
                f"capture[{key!r}] must be one of {list(allowed)}, got {value!r}."
            )


# Same caching as vllm._engine(). The load is split into _load_locked() so
# tests can replace it without loading a model.
#TODO: With the memory-management work, check the naming/structure inconsistency between this and vllm._engine().
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
    """Load the tokenizer and model. Call with ``_models_lock`` held."""
    started = time.perf_counter()
    # torch is not imported here: __init__ already did, before any load.
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError(
            "The 'torch' and 'transformers' packages are required for "
            "TransformersIntrospectionBackend. Install them with: "
            "pip install redact[transformers]"
        )

    tokenizer = AutoTokenizer.from_pretrained(hf_model_id, **hf_kwargs)
    # transformers renamed the `torch_dtype` keyword to `dtype`. Pass `dtype`
    # on version 5 and later and `torch_dtype` on 4.x; pyproject allows both.
    import transformers

    dtype_kwarg = (
        "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    )
    with measure.Measurement() as measured:
        model = AutoModelForCausalLM.from_pretrained(
            hf_model_id, device_map=device_map, **{dtype_kwarg: torch_dtype}, **hf_kwargs,
        )
    model.eval()

    # The load runs in this process with no preallocated pool, so the measured
    # figures describe the model itself. They are used for telemetry only.
    _models[key] = (tokenizer, model)
    _models_loaded_at[key] = time.perf_counter()
    observe.record({
        "ev": "local",
        "phase": "load",
        "backend": "introspect",
        "hf_model_id": hf_model_id,
        "device_map": device_map,
        "torch_dtype": torch_dtype,
        "load_ms": round((_models_loaded_at[key] - started) * 1000, 1),
        "claimed_gib": measured.claimed_gib,
        "weights_gib": measured.weights_gib,
    })
    return _models[key]


def _release_models() -> None:
    """Emit a lifetime event per loaded checkpoint, then forget the timers.

    Called from :meth:`TransformersIntrospectionBackend.clear_cache` and at
    exit, so a process that just exits still reports how long each
    checkpoint was held.
    """
    #TODO: Release checks as in vllm
    now = time.perf_counter()
    for key, loaded_at in list(_models_loaded_at.items()):
        observe.record({
            "ev": "local",
            "phase": "release",
            "backend": "introspect",
            "hf_model_id": key[0],
            "held_s": round(now - loaded_at, 1),
        })
    _models_loaded_at.clear()


atexit.register(_release_models)


class TransformersIntrospectionBackend(LLMBackend):
    """One model on local HF ``transformers``, with internals capture."""

    # supports_native_batching stays False even though generate() loops over
    # a list. Each item is its own forward pass and emits its own `call`
    # telemetry event, where a native batch emits one event for the whole
    # list. ModelClient's native path also reports progress only after the
    # whole call returns, so it would jump from 0 to 100%.
    # Concurrent generate() calls would compete for GPU memory, so parallel
    # calls are off.
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
        """Bind one registered model to a locally loaded HF model.

        Args:
            model: Registry name of the model.
            hf_model_id: HuggingFace model ID or local path.
            log_dir: Root directory for captured internals, with one
                subfolder per ``internals_id``.
            capture: Which internals to capture. Keys: ``"logprobs"`` (bool),
                ``"hidden_states"`` (``False`` / ``"last"`` / ``"all"``),
                ``"attention"`` (bool). Defaults to logprobs and the
                last-layer hidden state. Attention is large: it scales with
                ``layers x heads x seq_len**2``.
            device_map: Passed to ``from_pretrained``.
            torch_dtype: Passed to ``from_pretrained``.
            sampling: Generation defaults for this model (``top_p``, ...),
                merged over :data:`_DEFAULT_SAMPLING`.
            hf_kwargs: Extra kwargs for ``from_pretrained`` (e.g.
                ``trust_remote_code``).
            **identity: Forwarded to :meth:`LLMBackend.__init__`.

        Raises:
            ValueError: ``capture`` has an unknown key or value.
        """
        validate_capture(capture)
        try:
            import torch  # lazy: torch is heavy and optional
        except ImportError:
            raise ImportError(
                "The 'torch' and 'transformers' packages are required for "
                "TransformersIntrospectionBackend. Install them with: "
                "pip install redact[transformers]"
            )

        super().__init__(model, **identity)
        # Sampling needs a number, so fill in a default when the model has
        # none.
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
    def from_config(cls, config: "ModelConfig") -> "TransformersIntrospectionBackend":
        """Build from a registry entry's ``.introspect`` setup.

        Raises:
            ValueError: The entry has no ``.introspect`` setup.
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
        """Drop the loaded models, after reporting how long each was held."""
        #TODO: Release checks as in vllm
        _release_models()
        _models.clear()

    def _capture_dir(self, internals_id: str) -> Path:
        d = self._log_dir / internals_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def rename_capture(self, old_internals_id: str, new_internals_id: str) -> None:
        """Move a capture folder from a provisional id to its final one.

        Does nothing if no folder exists under ``old_internals_id``. A folder
        already at ``new_internals_id`` is replaced.
        """
        old_dir = self._log_dir / old_internals_id
        if not old_dir.exists():
            # Logged because a mistyped provisional id looks the same as a
            # call that captured nothing.
            logger.debug("[internals] nothing to rename at %s", old_dir)
            return
        new_dir = self._log_dir / new_internals_id
        if new_dir == old_dir:
            return
        new_dir.parent.mkdir(parents=True, exist_ok=True)
        if new_dir.exists():
            # A re-run produced the same id. rename() would raise, so keep the
            # newer capture.
            shutil.rmtree(new_dir)
        old_dir.rename(new_dir)
        logger.info("[internals] %s -> %s", old_internals_id, new_internals_id)

    def _save_meta(self, out_dir: Path, meta: dict) -> None:
        """Write ``meta.json``: resolved settings, input messages and output.

        Always written for a captured item, whatever the ``capture`` settings.
        """
        with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def _save_capture(self, internals_id: str, outputs, prompt_len: int, meta: dict) -> None:
        """Write ``meta.json`` and the internals that ``capture`` asks for."""
        torch = self._torch
        out_dir = self._capture_dir(internals_id)
        self._save_meta(out_dir, meta)
        gen_token_ids = outputs.sequences[0][prompt_len:]

        if self._capture.get("logprobs") and outputs.logits:
            logprobs = [
                torch.log_softmax(step_logits[0], dim=-1)
                for step_logits in outputs.logits
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
            # outputs.hidden_states is a tuple (per generated token) of tuples
            # (per layer) of [batch, seq, hidden] tensors. Keep the last
            # position of each selected layer at each step.
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

        written = sorted(p.name for p in out_dir.iterdir() if p.is_file())
        logger.info("[internals] saved %s -> %s", ", ".join(written), out_dir)

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
        """Generate replies one item at a time, capturing internals where asked.

        Args:
            messages_list: One chat message list per item.
            system_prompts: Inserted as the leading message of each item.
            max_tokens: Overrides the model's default.
            temperature: Overrides the model's default.
            internals_ids: One capture id (or ``None``) per item. Internals
                for an item with an id are written under
                ``{log_dir}/{internals_id}/``.
            **kwargs: Extra generation parameters (e.g. ``top_k``,
                ``repetition_penalty``), merged over the model's sampling
                defaults.

        Returns:
            Generated text, one per item, in the order of ``messages_list``.
        """
        if not messages_list:
            return []
        prompts, resolved = self._prepare(messages_list, system_prompts)
        max_tok, temp = self._resolve(max_tokens, temperature)
        ids = internals_ids if internals_ids is not None else [None] * len(messages_list)
        sampling = {**self._sampling, **kwargs}

        results: list[str] = []
        # Runs once via ModelClient (one item per call); the list is just the return contract.
        for messages, system_prompt, internals_id in zip(resolved, prompts, ids):
            chat_messages = (
                [{"role": "system", "content": system_prompt}, *messages]
                if system_prompt else messages
            )
            prompt = self._tokenizer.apply_chat_template(
                chat_messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = self._tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False,
            ).to(self._model.device)
            prompt_len = inputs["input_ids"].shape[1]

            want_capture = internals_id is not None
            started = time.perf_counter()
            with self._torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tok,
                    temperature=float(temp),
                    do_sample=temp > 0,
                    return_dict_in_generate=True,
                    output_logits=want_capture and self._capture.get("logprobs", False),
                    output_hidden_states=(
                        want_capture and bool(self._capture.get("hidden_states"))
                    ),
                    output_attentions=want_capture and self._capture.get("attention", False),
                    **sampling,
                )

            # One telemetry event per forward pass.
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
