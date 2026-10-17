"""Run telemetry: what was called, how long it took, and what it cost.

``llms/observe.py`` emits the events; this module collects, prices and sinks
them. The JSONL sink is a plain file writer, never a ``logging.Handler``, and
nothing here configures logging: the roll-up goes to this module's logger at
INFO, so the application's log level decides whether it shows.
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

# The values of ``REDACT_TRACE``. JSONL is the default: it needs no account, so
# the trace is there when a run goes wrong.
_VALID_SINKS = ("jsonl", "wandb", "off")
_DEFAULT_SINK = "jsonl"

_warned_default_provider = False

# Set by set_gpu_rate; wins over the pricing table.
_gpu_rate_override: float | None = None


def set_gpu_rate(usd_per_hour: float | None) -> None:
    """Price this machine's GPUs, over ``REDACT_GPU_PROVIDER`` and the pricing table.

    A rate set here, ``0.0`` included, also silences the unset-provider
    warning.

    Args:
        usd_per_hour: Hourly USD per single GPU, the table's unit. ``None``
            clears the override.

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
    """``REDACT_GPU_HOURLY_USD``, or ``None`` when it is unset or not a number."""
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
    """Warn once when a $0 GPU bill comes from the default provider.

    ``REDACT_GPU_PROVIDER`` defaults to ``local``, priced at 0.0, so a rented
    pod with no provider set would report ``local_cost_usd: 0.0``.
    """
    global _warned_default_provider
    if _warned_default_provider or rate != 0.0:
        return
    # A rate the caller set, 0.0 included, is an answer.
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
    """Hourly USD for one GPU of this model, or ``None`` to report time only.

    Checked in order: :func:`set_gpu_rate`, ``REDACT_GPU_HOURLY_USD``, then
    ``configs/llm/gpu_pricing.json``. A provider or GPU missing from the table
    gives ``None``, never a guess.

    Args:
        name: GPU model as ``nvidia-smi`` reports it.
        provider: Key into the pricing table; defaults to
            ``REDACT_GPU_PROVIDER``, then ``"local"`` (rate 0).
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


# Backends that run a model on this machine: paid for as engine time, so their
# tokens are counted and not priced.
_LOCAL_BACKENDS = frozenset({"vllm", "introspect"})


class Collector:
    """Accumulates events, writes them to the sinks, and rolls up cost.

    Thread-safe: events arrive from worker threads.
    """

    def __init__(self, trace_path: Path | None, sinks: list):
        self.trace_path = trace_path
        self._sinks = sinks
        self._lock = threading.Lock()
        self.events: list[dict] = []
        # Per-model usage, local backends included.
        self.tokens: dict[str, dict[str, int]] = defaultdict(
            lambda: {"in": 0, "out": 0, "calls": 0, "items": 0, "ms": 0.0}
        )
        # Per-model tokens of hosted calls: the ones a price applies to.
        self.priced: dict[str, dict[str, int]] = defaultdict(
            lambda: {"in": 0, "out": 0}
        )
        # Per-checkpoint seconds a released engine was held.
        self.local_s: dict[str, float] = defaultdict(float)
        # When each released engine was held, on time.perf_counter().
        self.local_spans: list[tuple[float, float]] = []
        self.errors: list[dict] = []
        # Replies cut at the token limit, per stage and model.
        self.cut: dict[str | None, dict[str, int]] = defaultdict(
            lambda: defaultdict(int))

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
                if event.get("backend") not in _LOCAL_BACKENDS:
                    priced = self.priced[event.get("model", "?")]
                    priced["in"] += event.get("in_tok") or 0
                    priced["out"] += event.get("out_tok") or 0
                if event.get("error"):
                    self.errors.append(event)
                if event.get("cut"):
                    self.cut[event.get("stage")][event.get("model", "?")] += 1
            elif event.get("ev") == "local" and event.get("phase") == "release":
                held = event.get("held_s") or 0.0
                self.local_s[event.get("hf_model_id", "?")] += held
                end = time.perf_counter()
                self.local_spans.append((end - held, end))
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
                # A model with no hosted call has no row in self.priced.
                if model in self.priced and (pin is not None or pout is not None):
                    hosted = self.priced[model]
                    cost = ((hosted["in"] / 1e6) * (pin or 0.0)
                            + (hosted["out"] / 1e6) * (pout or 0.0))
                    row["cost_usd"] = round(cost, 4)
                    api_cost += cost
                    priced = True
                per_model[model] = row
            all_local = bool(self.tokens) and not self.priced
            local_s = dict(self.local_s)
            spans = list(self.local_spans)
            n_errors = len(self.errors)
            cut = {stage: dict(models) for stage, models in self.cut.items()}
        # Engines still loaded have sent no release event yet.
        from .llms.backends import introspection, vllm
        now = time.perf_counter()
        for backend in (vllm, introspection):
            for hf_id, held in backend.held_seconds().items():
                local_s[hf_id] = local_s.get(hf_id, 0.0) + held
            spans += [(since, now) for since in backend.loaded_since()]

        name, rented = measure.detect_gpus()
        rate = gpu_hourly_rate(name)
        # Time two engines share is billed once.
        billed_s = _union_seconds(spans)
        held_h = billed_s / 3600.0
        _warn_if_unpriced_by_default(rented, held_h, rate)
        # Billed on every GPU rented: the whole pod is leased.
        local_cost = (
            round(rate * rented * held_h, 4) if (rate is not None and rented) else None
        )

        out = {
            "models": per_model,
            "local_s": {k: round(v, 1) for k, v in local_s.items()},
            "local_billed_s": round(billed_s, 1),
            "gpu": {"name": name, "rented": rented, "hourly_rate_usd": rate},
            # 0.0 when every call was local. None: no priced model was called.
            "api_cost_usd": round(api_cost, 4) if priced or all_local else None,
            "local_cost_usd": local_cost,
            "errors": n_errors,
            # Replies cut at the token limit, per stage and model.
            "cut": cut,
        }
        _log_summary(out)
        return out

    def cut_in(self, stage: str | None) -> dict[str, int]:
        """The replies cut at the token limit in ``stage``, per model."""
        with self._lock:
            return dict(self.cut.get(stage, {}))


def _union_seconds(spans: list[tuple[float, float]]) -> float:
    """Total time covered by ``(start, end)`` spans, overlaps counted once."""
    total, covered_to = 0.0, float("-inf")
    for start, end in sorted(spans):
        if end > covered_to:
            total += end - max(start, covered_to)
            covered_to = end
    return total


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
    if len(s["local_s"]) > 1:
        logger.info("[telemetry] local time billed %.1fs", s["local_billed_s"])
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


def _free_trace_path(directory: Path, run_id: str) -> Path:
    """The first of ``<run_id>.trace.jsonl``, ``<run_id>-2.trace.jsonl``, ...
    that does not exist, created empty so that no later run takes it."""
    directory.mkdir(parents=True, exist_ok=True)
    path, n = directory / f"{run_id}.trace.jsonl", 1
    while True:
        try:
            path.open("x", encoding="utf-8").close()
        except FileExistsError:
            n += 1
            path = directory / f"{run_id}-{n}.trace.jsonl"
        else:
            return path


def _jsonl_sink(path: Path):
    """Return a sink that appends each event to ``path`` as one JSON line."""
    lock = threading.Lock()

    def write(event: dict) -> None:
        with lock, path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    return write


def _wandb_sink():
    """Return a sink that logs each event to Weights & Biases.

    Returns ``None``, with a warning, when ``wandb`` is not installed; the JSONL
    trace is still written.
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


_collector: Collector | None = None


def install(data_dir: str | Path | None = None, sink: str | None = None) -> Collector:
    """Start collecting, and register the emitter ``llms/observe`` calls.

    Args:
        data_dir: Working root; the trace lands in
            ``<data_dir>/Datasets/<run_id>.trace.jsonl``, with ``-2``,
            ``-3``, ... after the run id when an earlier run of the same
            second has that name. ``None`` uses :func:`redact.get_output_dir`.
        sink: ``"jsonl"`` / ``"wandb"`` / ``"off"``. ``None`` reads
            ``REDACT_TRACE``, defaulting to ``"jsonl"``.

    Returns:
        The installed :class:`Collector`.
    """
    global _collector

    sink = (sink or os.environ.get("REDACT_TRACE") or _DEFAULT_SINK).lower()
    if sink not in _VALID_SINKS:
        logger.warning(
            "REDACT_TRACE=%r is not one of %s; using %r",
            sink, ", ".join(_VALID_SINKS), _DEFAULT_SINK,
        )
        sink = _DEFAULT_SINK

    if sink == "off":
        observe.set_emitter(None)
        _collector = None
        return Collector(None, [])

    run_id = datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S")
    trace_path = _free_trace_path(paths.datasets(data_dir), run_id)
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
    shows where wall-clock went even for stages that make no LLM calls, and
    logs how many replies of each model were cut at the token limit in it.
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
        cut = _collector.cut_in(label) if _collector is not None else {}
        for model, n in cut.items():
            replies = "1 reply was" if n == 1 else f"{n} replies were"
            logger.warning("[%s] %s: %s cut at the token limit and %s incomplete.",
                           label, model, replies, "is" if n == 1 else "are")
