"""Local vLLM backend.

The first backend built for a checkpoint loads its weights onto the GPU. The
engine is cached per checkpoint (``hf_model_id``, quantization and engine
kwargs), so two registry entries that agree on all three share one load.

The engine is vLLM's ``AsyncLLM``. Every item is its own request, awaited on
the shared event loop (``llms/aio.py``), and the engine batches whatever is
in flight: ``agenerate()`` makes one request, ``generate()`` one per item,
blocking until all have answered. Prompts go through the chat template of
the engine's tokenizer.
"""

import asyncio
import atexit
import contextlib
import itertools
import logging
import multiprocessing
import os
import threading
import time
from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING, ClassVar

from ...exceptions import PromptTooLongError
from .. import aio, observe
from ..resources import estimate, measure
from .base import ComputeConfig, LLMBackend, free_memory

if TYPE_CHECKING:
    from ..model_config import ModelConfig

logger = logging.getLogger(__name__)

# Backend defaults for sampling values that have no ModelConfig field.
# Override them per model with VLLMConfig.sampling.
_DEFAULT_SAMPLING = {"top_p": 0.85}
_DEFAULT_TEMPERATURE = 0.7

# Loaded engines, keyed by _engine_key(). This holds the only reference to
# each: a backend keeps the key, so a released engine can be collected.
_engines: dict[tuple, object] = {}
# Held during a load, so two threads cannot load one checkpoint twice.
_engines_lock = threading.Lock()
# When each engine's load started. Lifetime is timed per checkpoint, not per
# backend instance: the GPU is held for as long as the engine exists.
_engine_loaded_at: dict[tuple, float] = {}
# gpu_memory_utilization per engine key, set by the residency plan. Applied
# at load and kept out of the key, so it never splits a shared engine.
_planned_utilization: dict[tuple, float] = {}
# The cards per engine key, set by the residency plan and kept out of the
# key like the share. Indices over the cards this process can see.
_planned_devices: dict[tuple, tuple[int, ...]] = {}
# The gpu_memory_utilization and cards each loaded engine was built with,
# per engine key, as the plan numbers the cards.
_engine_grants: dict[tuple, tuple[float, tuple[int, ...]]] = {}
# The child processes each loaded engine's build started, per engine key:
# its engine cores and, with copies, vLLM's coordinator, as multiprocessing
# handles. Their own children, such as tensor-parallel workers, are found
# when they are ended (terminate_engine_processes). A child that another
# thread starts during a build is counted as the build's.
_engine_procs: dict[tuple, list] = {}
# The engine key of the load that holds _engines_lock.
_loading: tuple | None = None
# The build in progress: its engine key, and the pids of the child processes
# that existed when it started.
_building: tuple[tuple, frozenset[int]] | None = None
# Engine keys whose build a release on the way out ended (_end_load).
_ended: set[tuple] = set()
# Set by the exit hook: no load starts after it.
_exiting = False
# The child processes of builds that failed, which no engine holds.
_stray: list = []
# Seconds an engine process gets to end after SIGTERM, and again after SIGKILL.
_END_WAIT_S = 5.0
# How vLLM names its engine processes: EngineCore, EngineCore_DP<n> for a
# copy, and VLLM_DP_Coordinator.
_ENGINE_PROCESS_NAMES = ("EngineCore", "VLLM_DP_Coordinator")
#: Seconds the exit waits for a load in progress before it ends the load's
#: engine processes. A release mid-run waits for the load instead: a first
#: load can take 15 minutes.
EXIT_WAIT_S = 60.0
# Seconds to wait for the engine lock again once a build's processes are
# ended: the load then fails and lets go of it.
_GRACE_S = 10.0
# Set once the exit hook is registered, at the start of the first load (see _engine).
_exit_hook_registered = False
# Numbers the requests: an engine needs each id unique in the process.
_request_ids = itertools.count()


def set_planned_utilization(planned: dict[tuple, float], replace: bool = True) -> None:
    """Set the ``gpu_memory_utilization`` each engine key loads with.

    ``replace=False`` keeps the values already set.
    """
    global _planned_utilization
    # One assignment, so a load never reads a half-filled plan.
    _planned_utilization = {**planned, **({} if replace else _planned_utilization)}


def set_planned_devices(
    planned: dict[tuple, tuple[int, ...]], replace: bool = True
) -> None:
    """Set the cards each engine key loads on, as indices over the visible cards.

    ``replace=False`` keeps the values already set.
    """
    global _planned_devices
    _planned_devices = {**planned, **({} if replace else _planned_devices)}


def engine_grants() -> dict[tuple, tuple[float, tuple[int, ...]]]:
    """The ``gpu_memory_utilization`` and cards each loaded engine was built
    with, per engine key. The cards are indices over the visible ones."""
    return {key: grant for key, grant in list(_engine_grants.items())
            if key in _engines}


def _physical_ids(devices: tuple[int, ...]) -> list[str]:
    """The planned cards as the engine's processes are to see them.

    The plan counts over the visible cards, so with the variable set a
    planned index is a position in it. Each card is named by the UUID
    ``nvidia-smi`` gives it, so the engine gets the card the plan sized
    whatever order CUDA counts in; without one, as the variable names it,
    or by its index. ``[]`` when the plan names a card past the visible
    ones: such a plan does not fit, and the engine is left the cards it can
    see already, where vLLM fails for want of the rest.
    """
    visible = measure.visible_gpu_ids()
    last = max(devices, default=-1)
    if visible is not None and last >= len(visible):
        return []
    cards = measure.gpu_cards(visible=True)
    if cards is not None and last < len(cards):
        return [cards[d].uuid for d in devices]
    if visible is None:
        return [str(d) for d in devices]
    return [visible[d] for d in devices]


class EngineLoadEnded(RuntimeError):
    """A load that a release on the way out ended, because it had not
    returned within the release's wait."""


@contextlib.contextmanager
def _loading_of(key: tuple):
    """Mark ``key`` as the load that holds the engine lock, inside."""
    global _loading
    _loading = key
    try:
        yield
    finally:
        _loading = None


def _started_since(before: frozenset[int]) -> list:
    """The child processes alive now that were not in ``before`` (pids)."""
    return [p for p in multiprocessing.active_children() if p.pid not in before]


def terminate_engine_processes(
    processes=(), *, all: bool = False, quiet: bool = False,
) -> list[int]:
    """End engine processes this process started, and their children.

    Each gets SIGTERM, and SIGKILL when it is alive :data:`_END_WAIT_S`
    later: an engine hung in its start-up does not act on SIGTERM. Their
    children, such as tensor-parallel workers, are listed before any is
    signalled, since a worker may outlive its engine. Needs ``psutil``,
    which vLLM installs. Logged unless ``quiet``; never raises.

    Args:
        processes: ``multiprocessing`` handles of engine processes, as a
            build recorded them. Those that ended already are left alone.
        all: End every engine process instead: those of every build,
            loaded, in progress or failed, and every child process named as
            vLLM names its engine processes. For a forced exit; no other
            child of this process is touched.
        quiet: Log nothing, for a signal handler: a logging call there may
            re-enter a write the handler interrupted, and raise.

    Returns:
        The pids still alive afterwards: empty when every one ended.
    """
    roots = list(processes)
    if all:
        building = _building
        roots += [p for procs in list(_engine_procs.values()) for p in procs]
        roots += _stray + (_started_since(building[1]) if building else [])
        roots += [p for p in multiprocessing.active_children()
                  if (p.name or "").startswith(_ENGINE_PROCESS_NAMES)]
    roots = list({p.pid: p for p in roots if p.is_alive()}.values())
    if not roots:
        return []
    try:
        import psutil
    except ImportError:
        if not quiet:
            logger.error(
                "[vllm] psutil is not installed (vLLM requires it), so the engine "
                "processes %s are left running.", " ".join(str(p.pid) for p in roots))
        return [p.pid for p in roots]
    children: dict[int, object] = {}
    for root in roots:
        with contextlib.suppress(psutil.Error):
            for child in psutil.Process(root.pid).children(recursive=True):
                children.setdefault(child.pid, child)
    if not quiet:
        logger.warning(
            "[vllm] ending engine processes %s and their children %s",
            ", ".join(f"{p.pid} ({p.name})" for p in roots),
            " ".join(map(str, children)) or "(none)")

    def wait(roots, children):
        """Wait up to _END_WAIT_S for all; return those still alive."""
        deadline = time.monotonic() + _END_WAIT_S
        for root in roots:
            root.join(max(0.0, deadline - time.monotonic()))
        _, alive = psutil.wait_procs(
            children, timeout=max(0.0, deadline - time.monotonic()))
        return [p for p in roots if p.is_alive()], alive

    # A root is a multiprocessing child: it is signalled, waited for and
    # reaped through its handle, so that its pid stays its own until then.
    for signal_name in ("terminate", "kill"):
        for process in [*roots, *children.values()]:
            with contextlib.suppress(psutil.Error, OSError):
                getattr(process, signal_name)()
        roots, alive = wait(roots, list(children.values()))
        children = {p.pid: p for p in alive}
        if not roots and not children:
            return []
    left = [p.pid for p in roots] + list(children)
    if not quiet:
        logger.error(
            "[vllm] engine processes %s are still alive after SIGKILL; end them "
            "with kill -9 %s and check that nvidia-smi shows the cards free.",
            left, " ".join(map(str, left)))
    return left


def _engine_key(hf_model_id: str, quantization: str | None, vllm_kwargs: dict) -> tuple:
    return (hf_model_id, quantization, repr(sorted(vllm_kwargs.items())))


def _prepare_environment() -> None:
    """Set up the process environment before vLLM is constructed."""
    # CUDA fails in a forked child when the parent already initialised it.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    # The engine's processes count the cards as the plan does.
    measure.bus_order()

    # WSL2 workaround: vLLM's V2 model runner fails every load there with
    # "UVA is not available" (vllm-project/vllm#47387, fix proposed in
    # #47579), so default to the V1 runner. An explicit
    # VLLM_USE_V2_MODEL_RUNNER still wins. Remove once the fix ships.
    with contextlib.suppress(OSError), open("/proc/version") as f:
        if "microsoft" in f.read().lower():
            os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")


def _build(key: tuple, ids: list[str], make) -> tuple[object, list]:
    """Build an engine with ``make()``, with only the cards ``ids`` visible.

    The child processes the build starts are recorded: returned with the
    engine, or, when the build fails, kept as stray.

    Returns:
        The engine, and the child processes its build started.
    """
    global _building
    before = frozenset(p.pid for p in multiprocessing.active_children())
    try:
        # Waits for a transformers load that is starting CUDA.
        with measure.visible_only(ids):
            _building = (key, before)
            engine = make()
    except BaseException:
        _stray.extend(_started_since(before))
        raise
    finally:
        _building = None
    return engine, _started_since(before)


def _end_stray() -> list[int]:
    """End what failed builds left running; return the pids still alive."""
    alive = terminate_engine_processes(_stray)
    _stray[:] = [p for p in _stray if p.is_alive()]
    return alive


def _was_ended(key: tuple, hf_model_id: str) -> EngineLoadEnded | None:
    """The error for a build a release on the way out ended, if it did."""
    if key not in _ended:
        return None
    _ended.discard(key)
    return EngineLoadEnded(
        f"The load of {hf_model_id} was ended: the process is exiting, and "
        f"the load had not returned in time.")


def _engine(hf_model_id: str, quantization: str | None, vllm_kwargs: dict):
    """Get or load the vLLM engine for one checkpoint.

    The child processes a build starts are recorded under the engine's key.
    A failed build's are ended. When a build with the plan's copies fails,
    one copy on the engine's first cards is built instead, once.
    """
    global _exit_hook_registered
    key = _engine_key(hf_model_id, quantization, vllm_kwargs)
    # One read: clear_cache() may drop the key between a test and a lookup.
    engine = _engines.get(key)
    if engine is not None:
        return engine

    with _engines_lock, _loading_of(key):
        # Re-check: another thread may have finished the load while we waited.
        if key in _engines:
            return _engines[key]
        if _exiting:
            # Its engine processes would keep the interpreter from exiting.
            raise EngineLoadEnded(
                f"The load of {hf_model_id} was not started: the process is exiting.")

        _prepare_environment()
        try:
            # lazy: vllm is heavy and optional
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.usage.usage_lib import UsageContext
            from vllm.v1.engine.async_llm import AsyncLLM
        except ImportError:
            raise ImportError(
                "The 'vllm' package is required for VLLMBackend. "
                "Install it with: pip install redact[vllm]"
            )

        if not _exit_hook_registered:
            # Before the load, under the lock: an exit during a load then waits
            # on the lock for the load, at most EXIT_WAIT_S, and shuts the
            # engine down. atexit runs hooks last-registered first, and
            # multiprocessing's, registered when multiprocessing.util is first
            # imported, joins the engine's child process and never returns
            # while the engine runs. Import it here so ours is registered
            # after it: left to vllm, it may first be imported while the
            # engine is built, after this point.
            import multiprocessing.util as _  # noqa: F401
            atexit.register(_exit)
            _exit_hook_registered = True

        # No download_dir is passed: the hub cache is $HF_HOME/hub, so HF_HOME would download again.
        kwargs = dict(vllm_kwargs)
        # The plan's share, else the planner's whole-card one: vLLM's own
        # default is more than the plan assumes.
        kwargs.setdefault("gpu_memory_utilization", _planned_utilization.get(
            key, estimate.DEFAULT_GPU_MEMORY_UTILIZATION))
        # After the key is built, so the default never splits a shared engine.
        if not kwargs.get("max_model_len"):
            kwargs["max_model_len"] = estimate.effective_max_model_len(hf_model_id)
        # vllm.LLM's default. Left unset, the engine logs a stats line every
        # 10 seconds.
        kwargs.setdefault("disable_log_stats", True)
        # The plan's cards. Without a plan the engine takes the first visible
        # card(s), and the environment is left alone.
        devices = _planned_devices.get(key, ())
        # More cards than one copy spans are further copies, one engine still.
        span = kwargs.get("tensor_parallel_size") or 1
        copies = len(devices) // span
        if copies > 1:
            kwargs.setdefault("data_parallel_size", copies)
        ids = _physical_ids(devices)

        def make(kwargs: dict):
            return lambda: AsyncLLM.from_engine_args(
                AsyncEngineArgs(model=hf_model_id, quantization=quantization, **kwargs),
                # vllm.LLM's context: vLLM's defaults for running sequences
                # and batched tokens depend on it, and the engine's own has
                # lower ones.
                usage_context=UsageContext.LLM_CLASS,
            )

        started = time.perf_counter()
        fallback = False
        # Not measured: vLLM preallocates its memory grant in a separate
        # process, so a measurement would restate the setting.
        try:
            engine, procs = _build(key, ids, make(kwargs))
        except Exception as exc:
            ended = _was_ended(key, hf_model_id)
            if ended is not None:
                raise ended from exc
            alive = _end_stray()
            if copies <= 1:
                raise
            # The copies come from the plan, so the error names them. One
            # copy on the first cards is tried instead, unless the copies were
            # handed to the engine directly, the process is exiting, or the
            # failed build's processes live on to compete with a second one.
            n = kwargs["data_parallel_size"]
            each = "one per card" if span == 1 else f"each on {span} cards"
            failed = (f"The vLLM engine of {hf_model_id} was being built with {n} "
                      f"copies, {each} (data_parallel_size={n}), and the build "
                      f"failed ({type(exc).__name__}: {exc}).")
            if "data_parallel_size" in vllm_kwargs or _exiting or alive:
                raise RuntimeError(
                    f'{failed} To load one copy, set "copies": 1 in the model\'s '
                    f'entry under "models" in the recipe, or copies={{model: 1}} '
                    f"on the planner."
                    + (f" One copy was not tried: processes the build started "
                       f"are still alive ({', '.join(map(str, alive))})."
                       if alive else "")
                ) from exc
            kwargs = {k: v for k, v in kwargs.items() if k != "data_parallel_size"}
            ids, devices = ids[:span], devices[:span]
            home = ",".join(ids) or "the cards it can see"
            logger.warning(
                "[vllm] The engine of %s failed to build with %d copies (%s: %s); "
                "loading one copy on %s instead. To make this permanent, set "
                '"copies": 1 in the model\'s entry under "models" in the recipe.',
                hf_model_id, n, type(exc).__name__, exc, home)
            try:
                engine, procs = _build(key, ids, make(kwargs))
            except Exception as retry:
                ended = _was_ended(key, hf_model_id)
                if ended is not None:
                    raise ended from retry
                _end_stray()
                raise RuntimeError(
                    f"{failed} One copy on {home} failed as well "
                    f"({type(retry).__name__}: {retry})."
                ) from retry
            fallback = True
        # A build that returned as its processes were ended is shut down
        # as any other engine.
        _ended.discard(key)
        _engines[key] = engine
        _engine_procs[key] = procs
        # Without a plan the engine takes the first cards it can see.
        _engine_grants[key] = (kwargs["gpu_memory_utilization"], devices or tuple(
            range(span * kwargs.get("data_parallel_size", 1))))
        # Held from the start of the load; set after it, so a failed load
        # leaves no timer.
        _engine_loaded_at[key] = started

        observe.record({
            "ev": "local",
            "phase": "load",
            "backend": "vllm",
            "hf_model_id": hf_model_id,
            "quantization": quantization,
            "tensor_parallel_size": kwargs.get("tensor_parallel_size", 1),
            "data_parallel_size": kwargs.get("data_parallel_size", 1),
            # The plan's copies, and whether one was built instead.
            "planned_copies": max(copies, 1),
            "fallback": fallback,
            "gpu_memory_utilization": kwargs.get("gpu_memory_utilization"),
            # As handed to the engine's processes; empty when none were.
            "devices": ids,
            "load_ms": round((time.perf_counter() - started) * 1000, 1),
        })
        return _engines[key]


def _end_load(waited: float) -> bool:
    """Make a load that has held the engine lock for ``waited`` seconds let
    go of it.

    Ends the engine processes its build started, and those of failed
    builds: vLLM's start-up wait fails as soon as an engine process exits.
    A load that has not started its build has none to end. Logged at ERROR.

    Returns:
        Whether the caller holds the lock now.
    """
    loading, building = _loading, _building
    if building is not None:
        key, before = building
        logger.error(
            "[vllm] The load of %s has not returned in %g s. Ending the engine "
            "processes its build started, so that the process can exit.",
            key[0], waited)
        _ended.add(key)
        terminate_engine_processes(_started_since(before) + _stray)
    elif loading is not None:
        logger.error(
            "[vllm] The load of %s has not started its build in %g s (it reads "
            "its model's config, or waits for a transformers load to start "
            "CUDA), so there is no engine process to end.", loading[0], waited)
    else:
        logger.error("[vllm] Another release has held the engine lock for %g s.",
                     waited)
    if _engines_lock.acquire(timeout=_GRACE_S):
        return True
    logger.error("[vllm] The engine lock is still held %g s later; the engines "
                 "are left as they are.", _GRACE_S)
    return False


def _exit() -> None:
    """The exit hook: release every engine, waiting for a load in progress
    at most :data:`EXIT_WAIT_S`. No load starts after it."""
    global _exiting
    _exiting = True
    VLLMBackend.clear_cache(wait=EXIT_WAIT_S)


def _shut_down(engine) -> None:
    """Call the engine's ``shutdown()``.

    On the caller's thread, not the shared loop: the call is synchronous and
    safe from any thread. Never raises. An engine without the hook gives its
    memory back only when the object is collected.
    """
    hook = getattr(engine, "shutdown", None)
    if not callable(hook):
        logger.warning(
            "[vllm] %s has no shutdown hook; left to garbage collection",
            type(engine).__name__,
        )
        return
    try:
        hook()
    except Exception as exc:  # noqa: BLE001 — a release must not fail
        logger.debug("[vllm] shutdown() failed (%s: %s)", type(exc).__name__, exc)
        return
    logger.debug("[vllm] engine shut down")


def _release_engines(keep: Collection[tuple] = ()) -> None:
    """Emit a lifetime event per loaded engine, then forget the timers.

    Called from :meth:`VLLMBackend.clear_cache`, at exit too, so a process
    that just exits still reports how long each engine was held. Engines
    whose key is in ``keep`` stay timed.
    """
    now = time.perf_counter()
    for key, loaded_at in list(_engine_loaded_at.items()):
        if key in keep:
            continue
        del _engine_loaded_at[key]
        observe.record({
            "ev": "local",
            "phase": "release",
            "backend": "vllm",
            "hf_model_id": key[0],
            "quantization": key[1],
            "held_s": round(now - loaded_at, 1),
        })


def loaded_since() -> list[float]:
    """``time.perf_counter()`` at the start of each loaded engine's load."""
    return list(_engine_loaded_at.values())


def held_seconds() -> dict[str, float]:
    """Seconds each loaded engine has been held so far, per ``hf_model_id``."""
    now = time.perf_counter()
    held: dict[str, float] = {}
    for key, loaded_at in list(_engine_loaded_at.items()):
        held[key[0]] = held.get(key[0], 0.0) + now - loaded_at
    return held


class VLLMBackend(LLMBackend):
    """One model served by a locally-loaded vLLM engine."""

    # generate() takes the whole list, so the client hands it over undivided.
    # Every item is its own request to one engine on one loop, which makes
    # concurrent calls safe: the engine batches whatever is in flight.
    compute_config: ClassVar[ComputeConfig] = ComputeConfig(
        supports_native_batching=True,
        supports_parallel_calls=True,
        supports_internals=False,
        supports_async_calls=True,
    )

    def __init__(
        self,
        model: str,
        hf_model_id: str,
        *,
        quantization: str | None = None,
        vllm_kwargs: dict | None = None,
        sampling: dict | None = None,
        **identity,
    ):
        """Bind one registered model to a vLLM engine, loading it if needed.

        Args:
            model: Registry name of the model.
            hf_model_id: HuggingFace model ID or local path to load.
            quantization: Quantization method (e.g. "gptq", "awq").
            vllm_kwargs: Engine arguments, the fields of vLLM's
                ``AsyncEngineArgs`` (e.g. ``max_model_len``,
                ``tokenizer_mode``).
            sampling: ``SamplingParams`` defaults for this model (``top_p``,
                ``top_k``, ...), merged over :data:`_DEFAULT_SAMPLING`.
            **identity: Forwarded to :meth:`LLMBackend.__init__`.
                ``max_workers`` defaults to the number of sequences the
                engine runs at once, over all its copies.
        """
        engine = _engine(hf_model_id, quantization, vllm_kwargs or {})
        # What the async lane keeps in flight for this model. The engine
        # queues anything above its own limit, so more would only wait there.
        # The limit is per copy: each has a scheduler of its own.
        parallel = getattr(
            getattr(engine, "vllm_config", None), "parallel_config", None)
        identity.setdefault("max_workers", engine.scheduler_config.max_num_seqs
                            * getattr(parallel, "data_parallel_size", 1))
        super().__init__(model, **identity)
        # SamplingParams needs a number, so fill in a default when the model
        # has none.
        if self.default_temperature is None:
            self.default_temperature = _DEFAULT_TEMPERATURE
        self.hf_model_id = hf_model_id
        self._sampling = {**_DEFAULT_SAMPLING, **(sampling or {})}
        self._key = _engine_key(hf_model_id, quantization, vllm_kwargs or {})

    @property
    def _llm(self):
        """This model's engine, read from the cache on each use.

        Raises:
            RuntimeError: The engine was released and not loaded again.
        """
        engine = _engines.get(self._key)
        if engine is None:
            raise RuntimeError(
                f"The vLLM engine of {self.model!r} ({self.hf_model_id}) was "
                f"released. Build a new client with ModelClient.create()."
            )
        return engine

    @classmethod
    def from_config(cls, config: "ModelConfig") -> "VLLMBackend":
        """Build from a registry entry's ``.vllm`` setup.

        Raises:
            ValueError: The entry has no ``.vllm`` setup.
        """
        vllm = config.vllm
        if vllm is None:
            raise ValueError(
                f"Model {config.name!r} has no vllm setup. Register it with "
                f"register_model(..., vllm=VLLMConfig(hf_model_id='...'))."
            )
        return cls(
            hf_model_id=vllm.hf_model_id,
            quantization=vllm.quantization,
            vllm_kwargs=vllm.engine_kwargs,
            sampling=vllm.sampling,
            **cls._identity(config),
        )

    @classmethod
    def clear_cache(
        cls, keep: Collection[tuple] = (), *, wait: float | None = None,
    ) -> None:
        """Release the loaded engines and free their memory.

        Each is shut down, any of its processes the shutdown leaves running
        is ended, and its hold time is reported. Its planned utilization
        and cards stay until a plan is set again. A backend built on a released
        engine raises on ``generate()`` until the checkpoint is loaded again.
        A call in flight on a released engine is not waited for.
        Also the exit hook (:func:`_exit`), registered when the first load
        starts: an engine's child process would otherwise keep the
        interpreter from exiting.

        Args:
            keep: Engine keys (:func:`_engine_key`) to leave loaded and timed.
                A collection of keys, not one key.
            wait: Seconds to wait for a load in progress, for a release on
                the way out. Past them the engine processes of its build are
                ended, which makes the load fail and let go of the lock, and
                the release ends what failed builds left running too. If the
                load still holds the lock, the engines are left as they are
                and only their hold time is reported. ``None`` waits for the
                load however long it takes.
        """
        if any(not isinstance(key, tuple) for key in keep):
            raise TypeError(f"keep takes a collection of engine keys, got {keep!r}.")
        # Locked: a preload may be inserting an engine, and the next load
        # must wait until this memory is back.
        if wait is None:
            _engines_lock.acquire()
        elif not (_engines_lock.acquire(timeout=wait) or _end_load(wait)):
            _release_engines(keep)
            return
        try:
            released = [k for k in _engines if k not in keep]
            if released:
                # vLLM stops an engine before its own output task, which
                # then reports the dead engine as an error.
                logger.info(
                    "[vllm] shutting down %d engine(s). vLLM may log 'AsyncLLM "
                    "output_handler failed' with an EngineDeadError here: that is "
                    "this shutdown, not a failed call.", len(released))
            for key in released:
                _shut_down(_engines.pop(key))
                _engine_grants.pop(key, None)
            if released:
                free_memory()
            left = [p for key in [k for k in _engine_procs if k not in keep]
                    for p in _engine_procs.pop(key)]
            if wait is not None:
                left += _stray
                _stray.clear()
            if left:
                terminate_engine_processes(left)
            # After the teardown, so held_s covers it.
            _release_engines(keep)
        finally:
            _engines_lock.release()

    def _clean_output(self, text: str) -> str:
        """Strip whitespace and a leading ``|>`` tokenizer artifact."""
        text = text.strip()
        if text.startswith("|>"):
            text = text[2:].lstrip()
        return text

    def generate(
        self,
        messages_list: list[list[dict]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        internals_ids: list[str | None] | None = None,
        **kwargs,
    ) -> list[str]:
        """Generate replies for the whole batch, one engine request per item.

        Blocks until every reply is in. The requests run on the shared loop
        (``llms/aio.py``), and the engine batches them with whatever else it
        has in flight. When one fails, the others are cancelled.

        Args:
            messages_list: One chat message list per item.
            max_tokens: Overrides the model's default.
            temperature: Overrides the model's default.
            internals_ids: Ignored. This backend does not capture internals.
            **kwargs: ``SamplingParams`` overrides for this call (e.g.
                ``top_k``, ``presence_penalty``), merged over the model's
                sampling defaults.

        Returns:
            Generated text, one per item, in the order of ``messages_list``.
        """
        if not messages_list:
            return []
        max_tok, temp = self._resolve(max_tokens, temperature)
        return aio.run(self._many(messages_list, max_tok, temp, kwargs))

    async def agenerate(
        self,
        messages: list[dict],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        internals_id: str | None = None,
        **kwargs,
    ) -> str:
        """Generate one reply with one engine request, without blocking the loop.

        Runs on the shared loop (``llms/aio.py``).

        Args:
            messages: Chat messages of the one item.
            max_tokens: Overrides the model's default.
            temperature: Overrides the model's default.
            internals_id: Ignored. This backend does not capture internals.
            **kwargs: ``SamplingParams`` overrides for this call, merged over
                the model's sampling defaults.

        Returns:
            The generated text.
        """
        max_tok, temp = self._resolve(max_tokens, temperature)
        return await self._one(messages, max_tok, temp, kwargs)

    async def _many(
        self, messages_list: list[list[dict]], max_tok: int, temp: float, extra: dict
    ) -> list[str]:
        """Send each item as its own request and await them all.

        Returns:
            One reply per item, in the order of ``messages_list``.
        """
        tasks = [asyncio.ensure_future(self._one(messages, max_tok, temp, extra))
                 for messages in messages_list]
        try:
            return await asyncio.gather(*tasks)
        except BaseException:
            # A failed batch must not keep the engine working: a cancelled
            # request is aborted there.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def _one(
        self, messages: list[dict], max_tok: int, temp: float, extra: dict
    ) -> str:
        """Send one request to the engine and await its reply.

        Runs on the shared loop, as every await on an engine must: an engine
        hands its outputs to the loop of its first request. A reply vLLM ends
        with ``"length"``, its word for both caps (``max_tokens`` and
        ``max_model_len``), is recorded as cut.

        Returns:
            The reply's text.

        Raises:
            PromptTooLongError: The prompt is longer than the model's context.
            RuntimeError: The engine was released, sent no output, or
                aborted the request and sent only part of a reply.
        """
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind

        engine = self._llm  # raises before any work if the engine was released
        (system_prompt,), (chat,) = self._prepare([messages])
        if system_prompt:
            chat = [{"role": "system", "content": system_prompt}, *chat]
        # The engine takes token ids, so the chat template is applied here.
        ids = engine.get_tokenizer().apply_chat_template(
            chat, add_generation_prompt=True, tokenize=True)
        # Some tokenizers return the ids, some a mapping that holds them.
        if isinstance(ids, Mapping):
            ids = ids["input_ids"]
        params = SamplingParams(
            max_tokens=max_tok,
            temperature=temp,
            # One output per request, once the reply is complete.
            output_kind=RequestOutputKind.FINAL_ONLY,
            **{**self._sampling, **extra},
        )
        request_id = f"redact-{next(_request_ids)}"

        # One telemetry event per request: the engine batches continuously,
        # so the request is the transport call.
        started = time.perf_counter()
        try:
            final = None
            async for output in engine.generate(
                    {"prompt_token_ids": ids}, params, request_id):
                final = output
            if final is None:
                raise RuntimeError(
                    f"The vLLM engine of {self.model!r} sent no output for "
                    f"request {request_id}."
                )
            # An engine released with the request in flight does not raise:
            # it ends the request with the text it had so far.
            if final.outputs[0].finish_reason == "abort":
                raise RuntimeError(
                    f"The vLLM engine of {self.model!r} aborted request "
                    f"{request_id}: the engine was released or shut down "
                    f"while the request was in flight."
                )
        # A cancelled request was sent all the same.
        except (Exception, asyncio.CancelledError) as exc:
            self._record_call(
                n_items=1, started=started, max_tokens=max_tok,
                temperature=temp, error=f"{type(exc).__name__}: {exc}",
            )
            # The engine fails this one request and goes on with the others.
            # Read off the error itself: vLLM's validation error names the
            # parameter it rejected, and older versions have no such class.
            if getattr(exc, "parameter", None) == "input_tokens":
                raise PromptTooLongError(str(exc)) from exc
            raise
        reply = final.outputs[0]
        out_tok = len(reply.token_ids)
        self._record_call(
            n_items=1, started=started,
            in_tok=len(final.prompt_token_ids), out_tok=out_tok, out_tok_max=out_tok,
            max_tokens=max_tok, temperature=temp, finish_reason=reply.finish_reason,
            cut=reply.finish_reason == "length",
        )
        return self._clean_output(reply.text)

    @property
    def backend_name(self) -> str:
        return "vllm"

    def __repr__(self) -> str:
        return f"VLLMBackend(model={self.model!r}, hf_model_id={self.hf_model_id!r})"
