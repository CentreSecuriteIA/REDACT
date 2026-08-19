"""Paraphrasing / fingerprint removal.

Batched paraphrase pass: rephrase generated samples to strip the stylistic
fingerprints of the generating model while preserving meaning. The prompt is
loaded from ``prompts/content_moderation/paraphrase/template.json`` (a 1:1
"rephrase this" instruction).

The real defingerprinting model is trained in a separate repository; here any
capable model (the ``paraphraser`` role) drives the prompt. Whether a paraphrase
actually preserved meaning is a *separate* concern — validated by a dedicated
checker (:func:`redact.content_moderation.checker.build_paraphrase_checker`),
never by the paraphraser itself.

Dispatch goes through a capability-aware :class:`BatchCaller` (one vLLM engine
pass / parallel API thread-pool / series-only sequential), so paraphrasing is
rate-limited and batched like every other stage.

Also owns the full ``generate_paraphrases`` pipeline orchestration (model-pool
resolution, the plan/ledger/execute loop over inputs and outputs) — moved here
from ``pipelines.py`` so that file stays a thin dispatcher; ``pipelines.py``'s
``generate_paraphrases()`` just resolves the model pool and calls
:func:`run_paraphrase_target` per target.
"""

import logging
from pathlib import Path

import pandas as pd

from .. import paths
from ..dataset import Ledger, Manifest, merge_all
from ..dataset.io import _hash_text
from ..llms import get_backend, get_router
from ..llms.base import LLMBackend
from ..llms.calls import batch_check_samples, generate_sample
from ..llms.model_config import default_model_for_role, get_models_by_role
from ..llms.prompts import build_messages, load_prompt
from ..llms.wrappers import BatchCaller, RateLimiter
from .checker import build_paraphrase_checker, paraphrase_check_payload

logger = logging.getLogger(__name__)

_PROMPT_DIR = None  # Uses load_prompt() default (package-relative)


def _load_paraphrase_prompt(prompt_dir: str | None = _PROMPT_DIR) -> dict:
    """Load the paraphrase prompt config from JSON."""
    return load_prompt("content_moderation", "paraphrase", prompt_dir=prompt_dir)


def paraphrase_batch(
    backend: LLMBackend,
    model: str,
    samples: list[str],
    rate_limiter: RateLimiter | None = None,
    prompt_dir: str | None = _PROMPT_DIR,
    progress: str | None = None,
    **kwargs,
) -> list[str]:
    """Paraphrase a batch of samples in one capability-aware dispatch.

    Args:
        backend: Paraphraser backend.
        model: Paraphraser model identifier.
        samples: Texts to paraphrase.
        rate_limiter: Optional shared rate limiter.
        prompt_dir: Root prompt directory (defaults to the package prompts/).
        progress: Optional progress label for the batch dispatch.
        **kwargs: Passed to ``BatchCaller.batch_generate`` / the backend.

    Returns:
        Paraphrased texts, one per input, in order.
    """
    if not samples:
        return []
    prompt_config = _load_paraphrase_prompt(prompt_dir)
    messages_list = [build_messages(prompt_config, sample=s) for s in samples]
    caller = BatchCaller.from_model(backend, model, rate_limiter=rate_limiter)
    return caller.batch_generate(messages_list, model, progress=progress, **kwargs)


def paraphrase_sample(
    backend: LLMBackend,
    model: str,
    sample: str,
    rate_limiter: RateLimiter | None = None,
    system_prompt: str | None = None,
    prompt_dir: str | None = _PROMPT_DIR,
    **kwargs,
) -> str:
    """Paraphrase a single sample to remove stylistic fingerprints.

    Thin wrapper over :func:`paraphrase_batch`. Pass ``system_prompt`` to override
    the JSON template's system prompt for this call.

    Args:
        backend: Paraphraser backend.
        model: Paraphraser model identifier.
        sample: Text to paraphrase.
        rate_limiter: Optional shared rate limiter.
        system_prompt: Custom system prompt (overrides the JSON template).
        prompt_dir: Root prompt directory.
        **kwargs: Passed to the backend.

    Returns:
        Paraphrased text.
    """
    if system_prompt is not None:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sample},
        ]
        return generate_sample(backend, model, messages, rate_limiter, **kwargs)
    return paraphrase_batch(
        backend, model, [sample], rate_limiter=rate_limiter,
        prompt_dir=prompt_dir, **kwargs,
    )[0]


# ---------------------------------------------------------------------------
# generate_paraphrases() pipeline orchestration — moved here from pipelines.py
# so that module stays a thin path/model-resolution wrapper. pipelines.py's
# generate_paraphrases() calls paraphrase_pool() and run_paraphrase_target()
# below directly.
# ---------------------------------------------------------------------------


def _paraphrase_state_path(out_path: Path) -> Path:
    """Sidecar state-ledger path beside a paraphrase artifact (``*.state.jsonl``)."""
    return out_path.with_name(out_path.stem + ".state.jsonl")


def _paraphrase_ledger(state_path: Path) -> Ledger:
    """Shared resume-ledger for paraphrase units, keyed ``(input_id, iteration)``."""
    return Ledger(state_path, key_fields=("input_id", "iteration"),
                  casters={"input_id": str, "iteration": int})


def _read_paraphrase_state(state_path: Path) -> set[tuple[str, int]]:
    """Return ``(input_id, iteration)`` units already recorded in the ledger."""
    return _paraphrase_ledger(state_path).completed()


def _append_paraphrase_state(state_path: Path, records: list[dict]) -> None:
    """Append attempted-unit records (``input_id``/``iteration``/``paraphrase_model``/``status``)."""
    _paraphrase_ledger(state_path).record(records)


def paraphrase_pool(paraphraser: str | None) -> list[str]:
    """Resolve the paraphraser model pool.

    ``None`` → the ``paraphraser`` role default; a model name → just that model;
    ``"distribution"`` → all models registered with ``role="paraphraser"``.
    """
    if paraphraser == "distribution":
        seen: set[str] = set()
        pool = [
            c.name for c in get_models_by_role("paraphraser")
            if not (c.name in seen or seen.add(c.name))
        ]
        if not pool:
            raise ValueError("paraphraser='distribution' but no model has role='paraphraser'.")
        return pool
    if paraphraser:
        return [paraphraser]
    return [default_model_for_role("paraphraser")]


def _assign_paraphraser(base_id: str, k: int, pool: list[str], seed: int) -> str:
    """Deterministic round-robin pick (content-hash of base_id + k + seed)."""
    import hashlib
    if len(pool) == 1:
        return pool[0]
    h = hashlib.sha256(f"{seed}:{base_id}:{k}".encode()).hexdigest()
    return pool[int(h, 16) % len(pool)]


def run_paraphrase_target(
    tgt, source, data_dir, pool, check, check_model, K, resume, batch_size,
    seed, prompt_dir, out_path_override, verbose,
) -> pd.DataFrame:
    """Paraphrase one target (``"inputs"`` or ``"outputs"``) → its artifact CSV."""
    # ---- Load source rows (base_id, text, category, entry_type) --------------
    if tgt == "inputs":
        src = source if source is not None else merge_all(paths.datasets(data_dir), accepted_only=True)
        src = src.copy()
        text_col = "sample" if "sample" in src.columns else "prompt"
        id_col = "sample_id"
        out_path = Path(out_path_override) if out_path_override else paths.paraphrases_inputs_csv(data_dir)
    else:  # outputs
        if source is not None:
            src = source.copy()
        else:
            rp = paths.output_responses_csv(data_dir)
            src = pd.read_csv(rp) if rp.exists() else pd.DataFrame()
        if not src.empty and "accepted" in src.columns:
            src = src[src["accepted"] == True]  # noqa: E712 — accepted base outputs only
        text_col = "output_response" if "output_response" in getattr(src, "columns", []) else "sample"
        id_col = "input_id"
        out_path = Path(out_path_override) if out_path_override else paths.paraphrases_outputs_csv(data_dir)

    if src is None or src.empty:
        if verbose:
            logger.info("[paraphrase:%s] no source rows; skipping.", tgt)
        return pd.DataFrame()

    src = src.reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _base_id(row) -> str:
        existing = str(row.get(id_col, "")).strip()
        if existing and existing.lower() != "nan":
            return existing
        return _hash_text(str(row[text_col]))

    # ---- Resume via a sidecar STATE LEDGER (like constitution/output) ---------
    # The ledger records every *attempted* (base_id, iteration) unit + its model +
    # outcome, so a re-run skips them all — including drops (deduped/rejected), which
    # the output CSV alone would silently re-attempt. The CSV is still read for
    # content dedup (seen_texts), but completion is the ledger's job.
    state_path = _paraphrase_state_path(out_path)
    manifest = Manifest.sidecar(out_path)
    if not resume:
        manifest.reset()
        for p in (out_path, state_path):
            if p.exists():
                p.unlink()
    completed: set[tuple[str, int]] = _read_paraphrase_state(state_path) if resume else set()
    seen_texts: set[str] = set()
    if out_path.exists():
        prev = pd.read_csv(out_path)
        if not prev.empty and "sample" in prev.columns:
            seen_texts = set(prev["sample"].astype(str))

    # ---- Plan the FULL run -> manifest = the whole (base_id, k -> model) mapping
    all_units: list[tuple] = []  # (base_id, k, model, original_text, category, entry_type)
    for _, row in src.iterrows():
        bid = _base_id(row)
        otext = str(row[text_col])
        cat, et = str(row.get("category", "")), str(row.get("entry_type", ""))
        for k in range(K):
            all_units.append((bid, k, _assign_paraphraser(bid, k, pool, seed), otext, cat, et))

    manifest.write(
        {
            "input_id": bid, "iteration": k, "paraphrase_model": m,
            "category": cat, "entry_type": et, "status": "planned",
        }
        for bid, k, m, _o, cat, et in all_units
    )

    units = [u for u in all_units if (u[0], u[1]) not in completed]
    if not units:
        if verbose:
            logger.info("[paraphrase:%s] nothing to do (all %d units done).", tgt, len(all_units))
        return pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()

    # ---- Execute grouped by model (one paraphraser loaded at a time) ---------
    from collections import defaultdict
    groups: dict[str, list[tuple]] = defaultdict(list)
    for u in units:
        groups[u[2]].append(u)

    rate_limiter = get_router().rate_limiter
    check_backend = get_backend(check_model) if check else None
    written = 0
    for model, gunits in groups.items():
        backend = get_backend(model)
        # Internals capture: paraphrase's own output text (its eventual sample_id)
        # isn't known until the call returns, but internals_id has to be supplied
        # before it — capture under a pre-call-known provisional id (bid/k, always
        # unique) and relabel to the real sample_id once it's computed below.
        capture_paraphrase = getattr(backend, "supports_internals", False)
        for start in range(0, len(gunits), batch_size):
            chunk = gunits[start : start + batch_size]
            texts = [u[3] for u in chunk]
            provisional_ids = [f"{bid}/paraphrase/attempt_{k}" for bid, k, *_ in chunk]
            paraphrased = paraphrase_batch(
                backend, model, texts, rate_limiter=rate_limiter,
                prompt_dir=prompt_dir,
                progress=f"paraphrase:{tgt} ({model})" if verbose else None,
                internals_ids=provisional_ids if capture_paraphrase else None,
            )
            if check:
                payloads = [paraphrase_check_payload(u[3], p) for u, p in zip(chunk, paraphrased)]
                checks = batch_check_samples(
                    check_backend, check_model, payloads,
                    build_paraphrase_checker(prompt_dir=prompt_dir),
                    batch_size=batch_size, rate_limiter=rate_limiter,
                    progress=f"paraphrase-check:{tgt}" if verbose else None,
                )
            else:
                checks = [(True, "")] * len(chunk)

            rows, ledger = [], []
            for (bid, k, m, otext, cat, et), ptext, (acc, reason), prov_id in zip(
                chunk, paraphrased, checks, provisional_ids
            ):
                pt = str(ptext).strip()
                if not pt or pt == otext or pt in seen_texts:
                    ledger.append({"input_id": bid, "iteration": k,
                                   "paraphrase_model": m, "status": "dropped_dedup"})
                    continue  # dedup: drop no-op / duplicate paraphrases
                seen_texts.add(pt)
                sample_id = _hash_text(pt)
                if capture_paraphrase:
                    backend.rename_capture(prov_id, f"{bid}/paraphrase/{sample_id}")
                rows.append({
                    "sample_id": sample_id, "input_id": bid, "iteration": k,
                    "sample": pt, "category": cat, "entry_type": et,
                    "paraphrase_model": m, "accepted": bool(acc),
                    "reasoning": reason, "source": f"paraphrase_{tgt}",
                })
                ledger.append({"input_id": bid, "iteration": k, "paraphrase_model": m,
                               "status": "accepted" if acc else "rejected"})
            if rows:
                cdf = pd.DataFrame(rows)
                cdf.to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
                written += len(rows)
            # Record every attempted unit only after the CSV append — a crash mid-chunk
            # leaves those units un-acked so they re-run next time (never duplicated).
            _append_paraphrase_state(state_path, ledger)

    if verbose:
        n = len(pd.read_csv(out_path)) if out_path.exists() else 0
        logger.info("[paraphrase:%s] +%d rows (artifact now %d) -> %s", tgt, written, n, out_path.name)
    return pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
