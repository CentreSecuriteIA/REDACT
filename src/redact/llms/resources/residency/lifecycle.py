"""Load, keep and release the local engines a plan describes."""

import logging
import threading
from typing import TYPE_CHECKING

from redact.llms import observe
from redact.llms.model_config import get_model_config

from .footprint import _LOCAL_SETUPS, _engine_id, _local_setup

if TYPE_CHECKING:
    from .plan import ResidencyPlan

logger = logging.getLogger(__name__)


def apply_plan(plan: "ResidencyPlan", replace: bool = True) -> None:
    """Hand the plan's ``gpu_memory_utilization`` values to the vLLM loader.

    Args:
        plan: The plan to load under.
        replace: ``False`` keeps values already set, for a caller that plans
            a subset inside a planned run.
    """
    from redact.llms.backends import vllm

    # Keyed like the engine cache: the engine id without the setup.
    vllm.set_planned_utilization(
        {fp.engine[1:]: fp.planned_utilization for fp in plan.footprints
         if fp.setup == "vllm" and fp.planned_utilization is not None},
        replace=replace)


def preload(
    models: list[str],
    backend_types: dict | None = None,
    verbose: bool = True,
) -> threading.Thread | None:
    """Start loading local models on a background thread and return at once.

    Models load one after another, in the order given, so a long load can
    overlap with an earlier stage's API calls. A call that has to load an
    engine waits for the load in progress, because the backends lock their
    engine caches.

    A failed load does not stop the run. It is logged at ERROR and recorded
    as a ``preload_failed`` event, and the stage that needs the model fails
    when it tries to use it.

    Args:
        models: Model names to load. Unregistered and non-local ones are
            skipped.
        backend_types: Optional per-model setup override.

    Returns:
        The daemon thread, or ``None`` when there is nothing local to load.
    """
    from redact.llms.client import ModelClient

    backend_types = backend_types or {}
    local = []
    for m in dict.fromkeys(models):
        try:
            if _local_setup(m, backend_types.get(m)) is not None:
                local.append(m)
        except (KeyError, ValueError) as exc:
            logger.debug("Not preloading %s (%s)", m, exc)
    if not local:
        return None

    def _load_all() -> None:
        for name in local:
            try:
                if verbose:
                    logger.info("[preload] loading %s ...", name)
                ModelClient.create(name, backend_type=backend_types.get(name))
                if verbose:
                    logger.info("[preload] %s ready", name)
            except Exception as exc:  # noqa: BLE001 — must not kill the run
                logger.error(
                    "[preload] %s failed to load (%s: %s). The run continues; "
                    "the stage that needs it will fail at that point.",
                    name, type(exc).__name__, exc,
                )
                observe.record({
                    "ev": "preload_failed",
                    "model": name,
                    "error": f"{type(exc).__name__}: {exc}",
                })

    thread = threading.Thread(target=_load_all, name="redact-preload", daemon=True)
    thread.start()
    return thread


def _engine_keys(models) -> dict[str, set]:
    """The backend cache keys of the models' local setups, per setup."""
    keys: dict[str, set] = {setup: set() for setup in _LOCAL_SETUPS}
    for name in models or ():
        try:
            config = get_model_config(name)
        except KeyError:
            continue
        # Every local setup: the model may be in use under a non-default one.
        for setup in _LOCAL_SETUPS:
            local = getattr(config, setup, None)
            if local is not None:
                # The backend's cache key is the engine id without the setup.
                keys[setup].add(_engine_id(local, setup)[1:])
    return keys


def loaded_engines(exclude: list[str] | None = None) -> dict[str, set]:
    """The cache keys of the engines loaded now, per setup.

    Args:
        exclude: Model names whose engines are left out.
    """
    from redact.llms.backends import introspection, vllm

    skip = _engine_keys(exclude)
    return {"vllm": set(vllm._engines) - skip["vllm"],
            "introspect": set(introspection._models) - skip["introspect"]}


def unload_local(keep: list[str] | None = None, engines: dict | None = None) -> None:
    """Release the cached local engines, except those the kept models use.

    A released engine is shut down and its memory freed, and its lifetime
    meter stops. Its planned grant stays, so a reload gets the same share. A
    client built on it raises until the checkpoint is loaded again, and a
    call in flight on it is not waited for. A vLLM version without a shutdown
    hook returns the card only when the engine is collected, or at process
    exit.

    Args:
        keep: Model names still in use. Their engines stay cached and timed,
            so a later client for the same model reuses the load.
        engines: Engines to keep as well, as :func:`loaded_engines` returns.
    """
    from redact.llms.backends.introspection import TransformersIntrospectionBackend
    from redact.llms.backends.vllm import VLLMBackend

    kept = _engine_keys(keep)
    for setup, keys in (engines or {}).items():
        kept[setup] |= keys
    VLLMBackend.clear_cache(keep=frozenset(kept["vllm"]))
    TransformersIntrospectionBackend.clear_cache(keep=frozenset(kept["introspect"]))
