"""Run telemetry: what was called, how long it took, and what it cost.

``llms/`` emits facts and knows nothing about where they go (see
``llms/observe.py``). This module is the other half: it owns configuration,
aggregation, cost, and the sinks. Keeping the split that way is what lets
``llms/`` import nothing upward and carry no optional dependency.

**Relationship to ``logging``** — the library's contract (see
``redact/__init__.py``) is that every module logs through
``getLogger(__name__)`` under the ``redact`` root, and that the *application*
configures handlers and level. This module honours that exactly:

- The JSONL sink is a plain file writer, **never a ``logging.Handler``**.
  Attaching one to the ``redact`` root would hijack the caller's configuration
  and make the trace depend on their log level.
- Human-readable summaries *do* go through ``logging`` at INFO (cost roll-up,
  preload status, residency plan) and DEBUG (per-call detail), so
  ``scripts/run.py``'s existing ``--quiet``/``--debug`` already control them.
- Nothing here ever calls ``basicConfig()`` or ``setLevel()``.

Two channels, complementary: ``logging`` for a human reading a run, JSONL (or
W&B) for the machine-readable record.

Usage::

    from redact import telemetry

    telemetry.install(data_dir="./runs/training")
    with telemetry.stage("inputs"):
        ...                       # every call event is labelled "inputs"
    telemetry.summary()           # -> dict, also logged at INFO
"""

import json
import logging
import os
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from . import paths
from .llms import observe
from .llms.resources import measure

logger = logging.getLogger(__name__)

#: Sink chosen by ``REDACT_TRACE``. "jsonl" is the default: it needs no
#: account, works offline, and means the trace is already there when a run goes
#: wrong rather than only when someone remembered to switch it on.
_VALID_SINKS = ("jsonl", "wandb", "off")
DEFAULT_SINK = "jsonl"


# ---------------------------------------------------------------------------
# GPU rates
# ---------------------------------------------------------------------------

_warned_default_provider = False

#: Rate set by :func:`set_gpu_rate`, overriding the pricing table entirely.
_gpu_rate_override: float | None = None


def set_gpu_rate(usd_per_hour: float | None) -> None:
    """Price this machine's GPUs directly, without editing the pricing table.

    ``configs/llm/gpu_pricing.json`` ships inside the package, so editing it to
    say what your own card costs means editing an installed file — fine for a
    provider's published rates, wrong for a number only you know (your
    electricity, your amortization, your lab's internal chargeback).

    Wins over both ``REDACT_GPU_PROVIDER`` and the table, and applies to
    whatever GPU is detected. Setting it also silences the unset-provider
    warning: a rate you chose is an answer, including ``0.0``.

    Args:
        usd_per_hour: Hourly USD per **single** GPU — the same unit the table
            uses, since billing multiplies by the number of GPUs rented.
            ``None`` clears the override and restores table lookup.

    Raises:
        ValueError: If negative.
    """
    global _gpu_rate_override
    if usd_per_hour is not None:
        usd_per_hour = float(usd_per_hour)
        if usd_per_hour < 0:
            raise ValueError(f"GPU rate must be >= 0, got {usd_per_hour}")
    _gpu_rate_override = usd_per_hour


def _env_gpu_rate() -> float | None:
    """``REDACT_GPU_HOURLY_USD``, for setting a rate without importing anything.

    Ignores an unparseable value rather than failing a run over a typo in an
    env var — the rate is reporting, not correctness.
    """
    raw = os.environ.get("REDACT_GPU_HOURLY_USD")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        logger.debug("REDACT_GPU_HOURLY_USD=%r is not a number; ignoring", raw)
        return None


def _warn_if_unpriced_by_default(
    rented: int, held_h: float, rate: float | None
) -> None:
    """Warn once when a $0 GPU bill is the default talking, not a measurement.

    Every other unpriced case degrades to ``None`` — "time only, no guess".
    This one doesn't: ``REDACT_GPU_PROVIDER`` is *declared*, not detected, and
    defaults to ``local``, whose ``"*"`` entry is a real rate of 0.0. So
    forgetting the variable on a rented pod produces a confident
    ``local_cost_usd: 0.0`` — the one wrong number the cost model can emit,
    reached by the likeliest mistake. Nothing is changed here; the zero is
    just made legible.
    """
    global _warned_default_provider
    if _warned_default_provider or rate != 0.0:
        return
    # A rate the caller chose is an answer, 0.0 included — only the default
    # talking is worth a warning.
    if _gpu_rate_override is not None or _env_gpu_rate() is not None:
        return
    if not (rented and held_h > 0) or os.environ.get("REDACT_GPU_PROVIDER"):
        return
    _warned_default_provider = True
    logger.warning(
        "GPU cost reported as $0: no GPU price is set, so pricing fell back to "
        "provider 'local'. %d GPU(s) held for %.2fh. To price it: call "
        "redact.telemetry.set_gpu_rate(<usd_per_hour>), or set "
        "REDACT_GPU_HOURLY_USD, or set REDACT_GPU_PROVIDER if this machine is "
        "rented from a provider in configs/llm/gpu_pricing.json.",
        rented, held_h,
    )


def gpu_hourly_rate(name: str | None, provider: str | None = None) -> float | None:
    """Hourly rate for one GPU of this model, from ``configs/llm/gpu_pricing.json``.

    Args:
        name: GPU model as ``nvidia-smi`` reports it.
        provider: Key into the pricing table; defaults to ``REDACT_GPU_PROVIDER``,
            then ``"local"`` (rate 0 — running on your own machine reports
            GPU-seconds without inventing a dollar figure).

    Checked in order: :func:`set_gpu_rate`, ``REDACT_GPU_HOURLY_USD``, then
    the table — so a machine-specific rate never requires editing a file that
    ships inside the package.

    Returns:
        The rate, or ``None`` when the provider or GPU isn't in the table.
        ``None`` means *report time only* — a missing entry must never be
        guessed at, since a wrong rate is worse than no rate.
    """
    if _gpu_rate_override is not None:
        return _gpu_rate_override
    env_rate = _env_gpu_rate()
    if env_rate is not None:
        return env_rate
    provider = provider or os.environ.get("REDACT_GPU_PROVIDER") or "local"
    path = paths.gpu_pricing_json()
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("Could not read %s (%s); GPU time will be reported without cost", path, exc)
        return None
    rates = table.get(provider)
    if not isinstance(rates, dict):
        logger.debug("No GPU pricing for provider %r; reporting time only", provider)
        return None
    if name is not None and name in rates:
        return float(rates[name])
    if "*" in rates:
        return float(rates["*"])
    logger.debug("GPU %r not priced under provider %r; reporting time only", name, provider)
    return None


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------


class Collector:
    """Accumulates events, writes the trace, and rolls up cost.

    Thread-safe by the same reasoning as :class:`~redact.llms.progress.ProgressReporter`:
    ``BatchCaller`` dispatches from a thread pool, so events arrive concurrently.
    """

    def __init__(self, trace_path: Path | None, sinks: list):
        self.trace_path = trace_path
        self._sinks = sinks
        self._lock = threading.Lock()
        self.events: list[dict] = []
        # Per-model usage (every backend, local ones included), and
        # per-checkpoint local hold time. Two meters because two different
        # things are being bought: tokens from an endpoint, wall-clock from a
        # leased card.
        self.tokens: dict[str, dict[str, int]] = defaultdict(
            lambda: {"in": 0, "out": 0, "calls": 0, "items": 0, "ms": 0.0}
        )
        self.local_s: dict[str, float] = defaultdict(float)
        self.errors: list[dict] = []

    def __call__(self, event: dict) -> None:
        """Receive one event from ``observe.record``."""
        event.setdefault("ts", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        with self._lock:
            self.events.append(event)
            if event.get("ev") == "call":
                acc = self.tokens[event.get("model", "?")]
                acc["in"] += event.get("in_tok") or 0
                acc["out"] += event.get("out_tok") or 0
                acc["calls"] += 1
                acc["items"] += event.get("n_items") or 0
                acc["ms"] += event.get("ms") or 0.0
                if event.get("error"):
                    self.errors.append(event)
            elif event.get("ev") == "local" and event.get("phase") == "release":
                self.local_s[event.get("hf_model_id", "?")] += event.get("held_s") or 0.0
        for sink in self._sinks:
            try:
                sink(event)
            except Exception as exc:  # noqa: BLE001 — a sink must not break a run
                logger.debug("Telemetry sink failed (%s: %s)", type(exc).__name__, exc)

    def summary(self) -> dict:
        """Roll up totals and cost. Also logged at INFO."""
        from .llms.model_config import MODEL_REGISTRY

        api_cost = 0.0
        priced = False
        per_model = {}
        with self._lock:
            for model, acc in self.tokens.items():
                row = dict(acc)
                cfg = MODEL_REGISTRY.get(model)
                api = getattr(cfg, "api", None) if cfg else None
                pin = getattr(api, "price_per_1m_input", None) if api else None
                pout = getattr(api, "price_per_1m_output", None) if api else None
                if pin is not None or pout is not None:
                    cost = (acc["in"] / 1e6) * (pin or 0.0) + (acc["out"] / 1e6) * (pout or 0.0)
                    row["cost_usd"] = round(cost, 4)
                    api_cost += cost
                    priced = True
                per_model[model] = row
            local_s = dict(self.local_s)
            n_errors = len(self.errors)
        # Engines still loaded have sent no release event yet.
        from .llms.backends import introspection, vllm
        for backend in (vllm, introspection):
            for hf_id, held in backend.held_seconds().items():
                local_s[hf_id] = local_s.get(hf_id, 0.0) + held

        name, rented = measure.detect_gpus()
        rate = gpu_hourly_rate(name)
        held_h = sum(local_s.values()) / 3600.0
        _warn_if_unpriced_by_default(rented, held_h, rate)
        # Billed on GPUs *rented*, not the one an engine occupies: you rent the
        # whole pod. Reporting both is what makes idle capacity visible.
        local_cost = (
            round(rate * rented * held_h, 4) if (rate is not None and rented) else None
        )

        out = {
            "models": per_model,
            "local_s": {k: round(v, 1) for k, v in local_s.items()},
            "gpu": {"name": name, "rented": rented, "hourly_rate_usd": rate},
            "api_cost_usd": round(api_cost, 4) if priced else None,
            "local_cost_usd": local_cost,
            "errors": n_errors,
        }
        _log_summary(out)
        return out


def _log_summary(s: dict) -> None:
    """Human-readable roll-up through the standard logger."""
    for model, row in s["models"].items():
        cost = f"  ${row['cost_usd']}" if "cost_usd" in row else ""
        logger.info(
            "[telemetry] %s: %d calls / %d items, %d in + %d out tok, %.1fs%s",
            model, row["calls"], row["items"], row["in"], row["out"],
            row["ms"] / 1000.0, cost,
        )
    for hf_id, secs in s["local_s"].items():
        logger.info("[telemetry] local %s held %.1fs", hf_id, secs)
    gpu = s["gpu"]
    if gpu["rented"]:
        rate = (
            f" @ ${gpu['hourly_rate_usd']}/h each"
            if gpu["hourly_rate_usd"] is not None else " (unpriced)"
        )
        logger.info("[telemetry] GPUs rented: %d x %s%s", gpu["rented"], gpu["name"], rate)
    if s["api_cost_usd"] is not None or s["local_cost_usd"] is not None:
        logger.info(
            "[telemetry] cost: api $%s, local $%s",
            s["api_cost_usd"] if s["api_cost_usd"] is not None else "?",
            s["local_cost_usd"] if s["local_cost_usd"] is not None else "?",
        )
    if s["errors"]:
        logger.info("[telemetry] %d failed call(s) recorded in the trace", s["errors"])


# ---------------------------------------------------------------------------
# Sinks
# ---------------------------------------------------------------------------


def _jsonl_sink(path: Path):
    """Append events to a JSONL trace beside the other sidecars.

    Deliberately a plain file writer, not a ``logging.Handler`` — see the
    module docstring. Serialization failures are swallowed: a trace is a
    convenience, never a reason for a generation run to die.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def write(event: dict) -> None:
        with lock, path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    return write


def _wandb_sink():
    """Stream events to Weights & Biases. Imports the SDK lazily.

    A thin replay of the same events the JSONL sink gets — no separate
    instrumentation, so the two can never disagree. Returns ``None`` (and logs)
    when ``wandb`` isn't installed, so ``REDACT_TRACE=wandb`` degrades to the
    local trace rather than failing the run.
    """
    try:
        import wandb
    except ImportError:
        logger.warning(
            "REDACT_TRACE=wandb but the 'wandb' package isn't installed; "
            "falling back to the local JSONL trace only. pip install wandb"
        )
        return None
    if wandb.run is None:
        wandb.init(project=os.environ.get("WANDB_PROJECT", "redact"))

    def write(event: dict) -> None:
        wandb.log({f"redact/{event.get('ev', 'event')}": event})

    return write


# ---------------------------------------------------------------------------
# Install / stage
# ---------------------------------------------------------------------------

_collector: Collector | None = None


def install(data_dir: str | Path | None = None, sink: str | None = None) -> Collector:
    """Start collecting, and register the emitter ``llms/observe`` calls.

    Args:
        data_dir: Working root; the trace lands in
            ``<data_dir>/Datasets/<run_id>.trace.jsonl``. ``None`` falls back to
            :func:`redact.get_output_dir`, which is what direct API callers who
            never went through ``run_pipeline`` get.
        sink: ``"jsonl"`` / ``"wandb"`` / ``"off"``. ``None`` reads
            ``REDACT_TRACE``, defaulting to ``"jsonl"``.

    Returns:
        The installed :class:`Collector`.
    """
    global _collector

    sink = (sink or os.environ.get("REDACT_TRACE") or DEFAULT_SINK).lower()
    if sink not in _VALID_SINKS:
        logger.warning(
            "REDACT_TRACE=%r is not one of %s; using %r",
            sink, ", ".join(_VALID_SINKS), DEFAULT_SINK,
        )
        sink = DEFAULT_SINK

    if sink == "off":
        observe.set_emitter(None)
        _collector = None
        return Collector(None, [])

    run_id = datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S")
    trace_path = paths.datasets(data_dir) / f"{run_id}.trace.jsonl"
    sinks = [_jsonl_sink(trace_path)]
    if sink == "wandb":
        wb = _wandb_sink()
        if wb is not None:
            sinks.append(wb)

    _collector = Collector(trace_path, sinks)
    observe.set_emitter(_collector)
    logger.info("[telemetry] tracing to %s", trace_path)
    return _collector


def uninstall() -> None:
    """Stop collecting. Mostly for tests and notebooks."""
    global _collector
    observe.set_emitter(None)
    observe.set_stage(None)
    _collector = None


def collector() -> Collector | None:
    """The installed collector, or ``None`` when telemetry is off."""
    return _collector


def summary() -> dict:
    """Roll up the current run. Empty dict when telemetry is off."""
    return _collector.summary() if _collector is not None else {}


@contextmanager
def stage(label: str):
    """Label every event emitted inside this block with a pipeline stage.

    Also emits a ``stage`` event carrying the block's duration, so the trace
    shows where wall-clock went even for stages that make no LLM calls.
    """
    previous = observe.current_stage()
    observe.set_stage(label)
    started = time.perf_counter()
    try:
        yield
    finally:
        observe.record({
            "ev": "stage", "stage": label,
            "duration_s": round(time.perf_counter() - started, 1),
        })
        observe.set_stage(previous)
