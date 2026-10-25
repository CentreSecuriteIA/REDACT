"""Fakes and fixtures shared by the combination-search tests.

No model is loaded and nothing goes to a network: the target echoes the
jailbreak text, so a fake judge decides success by looking for a text the test
precomputed with the real (pure, deterministic) techniques.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import pandas as pd

from redact.exceptions import PromptTooLongError
from redact.jailbreak import (
    all_techniques,
    apply_combination,
    combine_techniques,
)
from redact.jailbreak import (
    registry as build_registry,
)

REGISTRY = build_registry(all_techniques())

# Pure techniques across three families, no model call and no benign data.
# The casual writings and to_leetspeak_basic take no rng, so a combination of
# them has a text known in advance, which the success and too-long fakes match
# exactly. to_json draws its fields and its variant from the unit's rng (and
# to_base64, which the search drew here before, a decode hint), so no test
# matches a text it writes.
POOL_NAMES = ["to_lowercase", "to_uppercase", "to_leetspeak_basic", "to_json"]
POOL = [REGISTRY[name] for name in POOL_NAMES]

PROMPT = "please describe the full procedure in detail"


def pure_pool() -> list:
    """Every pure, rng-free adversarial technique (no model call, no benign
    data, no rng draw), so any combination of them has a text known in advance
    (:func:`applied_for`). 36 techniques across several families — enough that
    one input's first mutation round holds well over 32 candidates, which is
    what a mid-round save and resume need to be exercised."""
    out = []
    for fn in all_techniques():
        if getattr(fn, "requires_llm", False) or getattr(fn, "requires_benign", False):
            continue
        if "rng" in inspect.signature(fn).parameters:
            continue
        if "adversarial" not in getattr(fn, "kinds", ["adversarial"]):
            continue
        out.append(fn)
    return out


def applied_for(prompt: str, names: list[str]) -> str:
    """The exact jailbreak text a combination of rng-free techniques produces
    for ``prompt`` (run synchronously, no client)."""
    drawing = [name for name in names
               if "rng" in inspect.signature(REGISTRY[name]).parameters]
    assert not drawing, f"{drawing} draw from an rng: their text is not known"
    combined = combine_techniques(*(REGISTRY[name] for name in names))
    text, _info, _accepted, _reason = apply_combination(combined, prompt)
    return text


def applied_text(names: list[str]) -> str:
    """The exact jailbreak text a combination produces (run synchronously, no
    client), so a test knows which target input counts as a success. Only for
    techniques that take no rng: one that does draws differently here than
    in the search, which seeds it."""
    drawing = [name for name in names
               if "rng" in inspect.signature(REGISTRY[name]).parameters]
    assert not drawing, f"{drawing} draw from an rng: their text is not known"
    combined = combine_techniques(*(REGISTRY[name] for name in names))
    text, _info, _accepted, _reason = apply_combination(combined, PROMPT)
    return text


@dataclass
class _Caps:
    supports_native_batching: bool = False
    supports_parallel_calls: bool = True
    supports_internals: bool = False
    supports_async_calls: bool = True


class _FakeBackend:
    backend_name = "fake"

    def __init__(self, max_workers: int = 8):
        self.max_workers = max_workers


class FakeClient:
    """An async fake: ``agenerate`` returns what ``reply`` computes."""

    def __init__(self, model: str, reply):
        self.model = model
        self._reply = reply
        self.backend = _FakeBackend()
        self.compute_config = _Caps()

    async def agenerate(self, messages, *, max_tokens=None, temperature=None,
                        internals_id=None):
        return self._reply(self.model, messages)

    def generate(self, messages_list, **kw):
        return [self._reply(self.model, m) for m in messages_list]


def _content(messages) -> str:
    return "\n".join(str(m.get("content", "")) for m in messages)


def make_resolve(target_model, judge_model, success_texts, *, too_long_for=None,
                 kill=None):
    """A ``resolve`` returning fakes for every model name.

    The target echoes the user text. The judge says ``Yes`` when any of
    ``success_texts`` is in the sample it is shown. ``too_long_for`` is a set of
    target-input texts whose target call raises ``PromptTooLongError``.
    ``kill`` is a one-element ``[n]`` counter: the ``n``-th judge call raises
    ``RuntimeError`` (a stand-in for a crash)."""
    success_texts = set(success_texts)
    too_long_for = set(too_long_for or ())
    calls = {"judge": 0}

    def reply(model, messages):
        text = _content(messages)
        if model == target_model:
            user = messages[-1]["content"]
            if user in too_long_for:
                raise PromptTooLongError("target prompt too long")
            return user                       # echo
        if model == judge_model:
            calls["judge"] += 1
            if kill is not None and calls["judge"] == kill[0]:
                raise RuntimeError("boom (simulated crash)")
            return "Yes" if any(s and s in text for s in success_texts) else "No"
        return text                            # gen / check: unused here

    def resolve(name):
        return FakeClient(name, reply)

    return resolve


def inputs(*prompts) -> pd.DataFrame:
    """A harmful-input frame for the given prompts."""
    return pd.DataFrame({"prompt": list(prompts),
                         "category": ["c"] * len(prompts),
                         "entry_type": ["harmful"] * len(prompts)})
