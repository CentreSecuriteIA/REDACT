"""Model registry with rate limits and provider-specific defaults.

Centralizes model metadata that was previously hardcoded across modules
(e.g. MODEL_RPMS in the jailbreak library's _RateLimitedClient).

Split into an identity (backend-agnostic facts) plus per-backend-type setup
configs, since most fields only mean something for one backend type:
- ``APIConfig`` — rate limiting/concurrency, only meaningful for backends
  that reach a model over the network (openai, anthropic).
- ``VLLMConfig`` — which local weights to load via vLLM, and how.
- ``IntrospectConfig`` — same, plus where captured internals get written.

**One entry may populate more than one setup.** A model that exists both as a
hosted endpoint and as local weights is one model, so it gets one entry with
both ``.api`` and ``.vllm`` rather than two near-duplicate rows;
``ModelClient.create(name, backend_type=...)`` picks which setup to bind.
``backend_type`` is the selector for which setup a row *defaults* to, not an
identity fact.

Which setup is active decides which budget applies: a backend built from the
``.vllm`` setup carries ``rpm=None``, so selecting the local setup on a
dual-setup entry does not inherit the endpoint's RPM — or its provider params.
Both are bound when the backend is built (see
:func:`redact.llms.backends.capabilities.backend_for`), never looked up per call.
"""

import dataclasses
import json
from dataclasses import dataclass, field

from .. import paths
from .backends import (
    API_BACKEND_TYPES,
    SETUP_TYPES,
    ComputeConfig,
    compute_config_for,
    transport_for,
    validate_concurrency,
)

# Whose budget an APIConfig's `rpm` describes. "model" = this model alone
# (Venice meters per model); "endpoint" = every model behind the same
# provider+URL+key shares one window (Anthropic meters the account).
RATE_LIMIT_SCOPES: frozenset[str] = frozenset({"model", "endpoint"})


def _require(condition: bool, message: str, exc: type[Exception] = ValueError) -> None:
    """Reject an invalid config field.

    A raise, not ``assert``: ``python -O`` strips asserts, and these validate
    caller-supplied input at a public API boundary — exactly the checks that
    must survive optimization.

    Args:
        condition: What must hold.
        message: Shown when it doesn't.
        exc: ``ValueError`` for a bad value, ``TypeError`` for a bad type —
            the ``_check_*`` helpers below pick the right one per field.
    """
    if not condition:
        raise exc(message)


# The four field shapes this module validates. Each separates "wrong kind of
# thing" (TypeError) from "right kind, bad value" (ValueError), so a caller
# who passes rpm="60" is told it isn't an int rather than that it isn't
# positive.

def _check_str(label: str, value, *, optional: bool = False) -> None:
    """Non-empty string (or ``None`` when optional)."""
    if optional and value is None:
        return
    _require(isinstance(value, str),
             f"{label} must be a string, got {type(value).__name__}", TypeError)
    _require(value.strip(), f"{label} must not be empty")


def _check_int(label: str, value, *, minimum: int) -> None:
    """Integer at or above ``minimum``. ``bool`` is rejected: it subclasses
    ``int``, so ``rpm=True`` would otherwise pass as the number 1."""
    _require(isinstance(value, int) and not isinstance(value, bool),
             f"{label} must be an int, got {type(value).__name__}", TypeError)
    _require(value >= minimum, f"{label} must be >= {minimum}, got {value!r}")


def _check_bool(label: str, value) -> None:
    _require(isinstance(value, bool),
             f"{label} must be a bool, got {type(value).__name__}", TypeError)


def _check_number(label: str, value, *, minimum: float = 0.0) -> None:
    """Optional non-negative number (or ``None``). Rejects ``bool``, which
    subclasses ``int`` and would otherwise pass as 1."""
    if value is None:
        return
    _require(isinstance(value, int | float) and not isinstance(value, bool),
             f"{label} must be a number, got {type(value).__name__}", TypeError)
    _require(value >= minimum, f"{label} must be >= {minimum}, got {value!r}")


def _check_dict(label: str, value) -> None:
    """Optional dict — every dict field in this module may be ``None``."""
    _require(value is None or isinstance(value, dict),
             f"{label} must be a dict or None, got {type(value).__name__}", TypeError)


@dataclass
class APIConfig:
    """A hosted endpoint: which provider, where it is, and its budget.

    Self-sufficient by design. ``ModelConfig.backend_type`` says which *setup*
    an entry prefers, not which provider its API is — so everything needed to
    actually reach the endpoint lives here. Without that, a dual-setup entry
    defaulting to ``"vllm"`` would have no way to say what its ``.api`` even
    was, and every OpenAI-compatible model would silently inherit one
    hardcoded provider's URL and key.

    Attributes:
        backend_type: Which API transport serves this — "openai" (any
            OpenAI-compatible endpoint) or "anthropic".
        api_key_env: Name of the environment variable holding the API key.
        rpm: Requests-per-minute limit enforced by ``RateLimiter``.
        base_url: Endpoint URL. Required for "openai", since that transport
            works against any compatible provider and has no meaningful
            default. Unused for "anthropic", whose SDK knows its own endpoint.
        api_model_id: The identifier sent upstream, when the provider's slug
            differs from this entry's registry name. ``None`` means they match.
            The registry name is the library's *stable* identity — it keys
            roles, the client cache, telemetry rows and every price lookup — so
            it must survive a provider's version bumps. Venice renaming
            ``venice-uncensored`` to ``venice-uncensored-1-2`` is exactly that
            case: without this field the choice would be renaming the entry
            (churning roles, docs, ledgers, and orphaning past traces) or
            calling a stale model. Same role as ``hf_model_id`` on the local
            setups: what the runtime is actually asked for.
        rate_limit_scope: Whose budget ``rpm`` describes — ``"model"``
            (default) or ``"endpoint"``. Venice meters each model separately,
            so three models on one key hold three independent windows and
            ``"model"`` is correct. Anthropic meters the *account*, so every
            model behind one key draws on one budget; ``"endpoint"`` makes
            them share a single window keyed on ``backend_type`` + ``base_url``
            + ``api_key_env``. Without it, N models each get the full budget
            and collectively blow it — two entries at ``rpm=5`` issuing 10/min
            against a 5/min account. Only rate limiting is shared this way:
            ``recommended_max_workers`` is pipeline depth (3 workers saturate
            60 RPM fine), not a budget that adds up across models.
        recommended_max_workers: Concurrency the backend binds as its
            ``max_workers``, clamped there against the transport's own
            parallelism capability. Also checked against the provider at
            registration time by :class:`ModelConfig`.
        default_extra_body: Provider-specific parameters sent with every call.
            Bound only when this setup is the one built, so a local binding of
            the same entry never inherits them.
        price_per_1m_input: USD per 1M prompt tokens, for cost roll-up. ``None``
            means unpriced — telemetry then reports tokens without a dollar
            figure rather than guessing.
        price_per_1m_output: USD per 1M completion tokens.
    """

    backend_type: str
    api_key_env: str
    rpm: int
    base_url: str | None = None
    api_model_id: str | None = None
    rate_limit_scope: str = "model"
    recommended_max_workers: int = 1
    default_extra_body: dict | None = None
    price_per_1m_input: float | None = None
    price_per_1m_output: float | None = None

    def __post_init__(self) -> None:
        _check_str("APIConfig.backend_type", self.backend_type)
        _require(
            self.backend_type in API_BACKEND_TYPES,
            f"APIConfig.backend_type must be one of {sorted(API_BACKEND_TYPES)}, "
            f"got {self.backend_type!r}",
        )
        _check_str("APIConfig.api_key_env", self.api_key_env)
        _check_int("APIConfig.rpm", self.rpm, minimum=1)
        if self.backend_type == "openai":
            _check_str("APIConfig.base_url", self.base_url)
        _check_str("APIConfig.api_model_id", self.api_model_id, optional=True)
        _check_str("APIConfig.rate_limit_scope", self.rate_limit_scope)
        _require(
            self.rate_limit_scope in RATE_LIMIT_SCOPES,
            f"APIConfig.rate_limit_scope must be one of "
            f"{sorted(RATE_LIMIT_SCOPES)}, got {self.rate_limit_scope!r}",
        )
        _check_int("APIConfig.recommended_max_workers",
                   self.recommended_max_workers, minimum=1)
        _check_dict("APIConfig.default_extra_body", self.default_extra_body)
        _check_number("APIConfig.price_per_1m_input", self.price_per_1m_input)
        _check_number("APIConfig.price_per_1m_output", self.price_per_1m_output)

    @property
    def endpoint_id(self) -> str:
        """Identity of the endpoint this setup talks to.

        The unit a per-account budget belongs to: same provider, same URL,
        same key. Deliberately the same triple ``_sdk_client`` caches its HTTP
        pool on, so "one connection pool" and "one rate budget" can never
        disagree about what counts as the same endpoint.
        """
        return f"{self.backend_type}:{self.base_url}:{self.api_key_env}"


@dataclass
class VLLMConfig:
    """Which local weights to load via vLLM, and how.

    ``hf_model_id`` is required rather than validated later: a vLLM setup
    without weights isn't a setup, so the invalid state is unrepresentable.

    Attributes:
        hf_model_id: HuggingFace model ID or local path.
        quantization: Quantization method, e.g. "gptq", "awq".
        vllm_kwargs: Extra kwargs passed to ``vllm.LLM()`` (e.g.
            ``gpu_memory_utilization``, ``tokenizer_mode``).
        sampling: Per-model ``SamplingParams`` defaults with no identity field
            of their own (``top_p``, ``top_k``, ...). Bound at construction so
            they are never magic numbers inside ``generate()``.
        vram_gb: Estimated VRAM this model *needs*, for pre-flight residency
            planning. An estimate: real usage moves with ``max_model_len``,
            quantization and KV cache. ``None`` means the planner falls back to
            a measured value (``Data_cache/vram.json``) or plans optimistically.
            NOTE: this is the model's need, **not** what vLLM reserves —
            ``gpu_memory_utilization`` makes it claim a fraction of the whole
            card regardless of model size. See ``redact.residency``.
        min_gpus: Whole devices this model claims, i.e. ``tensor_parallel_size``.
            A model too large for one card takes N cards outright, which is a
            different packing problem from "does it fit in the leftover GB".
    """

    hf_model_id: str
    quantization: str | None = None
    vllm_kwargs: dict | None = None
    sampling: dict | None = None
    vram_gb: float | None = None
    min_gpus: int = 1

    def __post_init__(self) -> None:
        _check_str("VLLMConfig.hf_model_id", self.hf_model_id)
        _check_str("VLLMConfig.quantization", self.quantization, optional=True)
        _check_dict("VLLMConfig.vllm_kwargs", self.vllm_kwargs)
        _check_dict("VLLMConfig.sampling", self.sampling)
        _check_number("VLLMConfig.vram_gb", self.vram_gb)
        _check_int("VLLMConfig.min_gpus", self.min_gpus, minimum=1)


@dataclass
class IntrospectConfig:
    """Which local weights to load for internals capture, and where to write it.

    Both ``hf_model_id`` and ``log_dir`` are required: capture with nowhere to
    write is as broken as weights that don't exist, so neither is optional.

    Attributes:
        hf_model_id: HuggingFace model ID or local path.
        log_dir: Root dir captured internals are written under.
        capture: Which internals to capture — logprobs / hidden_states /
            attention flags.
        device_map: Passed to ``AutoModelForCausalLM.from_pretrained``.
        torch_dtype: Passed to ``AutoModelForCausalLM.from_pretrained``.
        sampling: Per-model generation defaults with no identity field of
            their own (``top_p``, ``top_k``, ...), bound at construction.
        vram_gb: Estimated VRAM need, for residency planning. Unlike vLLM this
            backend preallocates nothing, so a measured figure here really is
            the model's need.
        min_gpus: Whole devices this model claims.
        extra_kwargs: Extra kwargs for ``from_pretrained``, e.g.
            trust_remote_code.
    """

    hf_model_id: str
    log_dir: str
    capture: dict | None = None
    device_map: str = "auto"
    torch_dtype: str = "auto"
    sampling: dict | None = None
    vram_gb: float | None = None
    min_gpus: int = 1
    extra_kwargs: dict | None = None

    def __post_init__(self) -> None:
        _check_str("IntrospectConfig.hf_model_id", self.hf_model_id)
        _check_str("IntrospectConfig.log_dir", self.log_dir)
        _check_dict("IntrospectConfig.capture", self.capture)
        _check_str("IntrospectConfig.device_map", self.device_map)
        _check_str("IntrospectConfig.torch_dtype", self.torch_dtype)
        _check_dict("IntrospectConfig.sampling", self.sampling)
        _check_number("IntrospectConfig.vram_gb", self.vram_gb)
        _check_int("IntrospectConfig.min_gpus", self.min_gpus, minimum=1)
        _check_dict("IntrospectConfig.extra_kwargs", self.extra_kwargs)


@dataclass
class ModelConfig:
    """One model: backend-agnostic identity plus the setups that can serve it.

    Validates itself on construction, so any ``ModelConfig`` in the registry
    is coherent — the setup fields are already valid by their own
    construction, and this checks the identity fields plus the cross-setup
    rules that only make sense once assembled.

    **More than one setup may be populated.** A model that exists both as a
    hosted endpoint and as local weights is one model: give it ``api`` *and*
    ``vllm``. ``backend_type`` then names which setup it *defaults* to; the
    others stay reachable via
    ``ModelClient.create(name, backend_type=...)``, which checks the
    corresponding field is present.

    Attributes:
        name: Model identifier used at call sites.
        roles: Logical roles this model can fill, for
            :func:`default_model_for_role` lookup. A model may hold several.
        is_uncensored: True for uncensored generation models.
        supports_system_prompt: False if the model ignores system messages —
            drives the system-prompt fold the backend applies in
            ``LLMBackend._prepare()``.
        default_max_tokens: Fallback when a caller passes no max_tokens.
        default_temperature: Fallback sampling temperature (None = provider
            default).
        backend_type: Which **setup** this entry prefers — "api", "vllm", or
            "introspect", each naming the field that holds it. ``None`` means
            "the only setup present". Which API *provider* an ``.api`` setup
            uses is that config's own ``backend_type``, not this one.
        api: Network budget, or None for a local-only model.
        vllm: vLLM weight-loading setup, or None.
        introspect: Internals-capture setup, or None.
    """

    name: str
    roles: list[str] = field(default_factory=list)
    is_uncensored: bool = False
    supports_system_prompt: bool = True
    default_max_tokens: int = 2000
    default_temperature: float | None = None
    backend_type: str | None = None

    api: APIConfig | None = None
    vllm: VLLMConfig | None = None
    introspect: IntrospectConfig | None = None

    def __post_init__(self) -> None:
        _check_str("ModelConfig.name", self.name)
        _require(isinstance(self.roles, list),
                 f"{self.name!r}: roles must be a list, got {type(self.roles).__name__}",
                 TypeError)
        for role in self.roles:
            _check_str(f"{self.name!r}: role", role)
            _require(role in _ROLES,
                     f"{self.name!r}: unknown role {role!r}; declare it in "
                     f"configs/llm/roles.json (known: {sorted(_ROLES)})")
        _check_bool(f"{self.name!r}: is_uncensored", self.is_uncensored)
        _check_bool(f"{self.name!r}: supports_system_prompt",
                    self.supports_system_prompt)
        _check_int(f"{self.name!r}: default_max_tokens",
                   self.default_max_tokens, minimum=1)
        if self.default_temperature is not None:
            _require(
                isinstance(self.default_temperature, int | float)
                and not isinstance(self.default_temperature, bool),
                f"{self.name!r}: default_temperature must be a number, "
                f"got {type(self.default_temperature).__name__}",
                TypeError,
            )
            _require(
                0.0 <= self.default_temperature <= 2.0,
                f"{self.name!r}: default_temperature must be within 0.0-2.0, "
                f"got {self.default_temperature!r}",
            )

        for field_name, value, cls in (
            ("api", self.api, APIConfig),
            ("vllm", self.vllm, VLLMConfig),
            ("introspect", self.introspect, IntrospectConfig),
        ):
            _require(
                value is None or isinstance(value, cls),
                f"{self.name!r}: {field_name} must be a {cls.__name__} or None, "
                f"got {type(value).__name__}",
                TypeError,
            )

        if self.backend_type is None:
            return  # inferred from the name later; nothing to cross-check yet

        _require(
            self.backend_type in SETUP_TYPES,
            f"{self.name!r}: unknown backend_type {self.backend_type!r}; "
            f"expected one of {sorted(SETUP_TYPES)}",
        )
        # Every setup name is the field that holds it, so this needs no
        # lookup table — that uniformity is why the names were chosen.
        _require(
            getattr(self, self.backend_type) is not None,
            f"{self.name!r}: backend_type={self.backend_type!r} needs a "
            f"{self.backend_type} config — pass {self.backend_type}=...",
        )
        # Concurrency is a property of the API provider, which the APIConfig
        # names itself — so this holds whichever setup the entry prefers. A
        # local setup alongside it ignores the value entirely (a
        # native-batching client never builds a BatchCaller), which is what
        # makes a dual-setup entry with workers>1 legitimate.
        if self.api is not None:
            validate_concurrency(
                self.name, self.api.backend_type, self.api.recommended_max_workers
            )


DEFAULT_RPM = 20

_SETUP_CLASSES = {"api": APIConfig, "vllm": VLLMConfig, "introspect": IntrospectConfig}


def available_roles() -> dict[str, str]:
    """Every role a model may claim, mapped to what it is for.

    Loaded from ``configs/llm/roles.json``. This is the list :func:`register_model`
    validates against, so a typo'd role is rejected at load rather than
    surfacing much later as "no model registered for role".
    """
    return dict(_ROLES)


def _load_roles() -> dict[str, str]:
    path = paths.roles_json()
    try:
        roles = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read roles from {path}: {exc}") from exc
    _require(isinstance(roles, dict), f"{path}: expected a JSON object of role -> description")
    return roles


def _build_entry(name: str, raw: dict) -> ModelConfig:
    """Turn one JSON entry into a validated :class:`ModelConfig`.

    Setups are constructed through their own dataclasses, so a file entry gets
    exactly the validation a hand-written registration does — no second,
    weaker schema check that could drift from it.
    """
    _require(isinstance(raw, dict), f"model {name!r}: entry must be a JSON object", TypeError)
    fields = {k: v for k, v in raw.items() if k not in _SETUP_CLASSES and k != "notes"}
    unknown = set(fields) - {f.name for f in dataclasses.fields(ModelConfig)}
    _require(not unknown, f"model {name!r}: unknown field(s) {sorted(unknown)}")

    setups = {
        key: cls(**raw[key]) for key, cls in _SETUP_CLASSES.items() if key in raw
    }
    return ModelConfig(name=name, **fields, **setups)


def _load_registry() -> dict[str, ModelConfig]:
    """Build the registry from ``configs/llm/models.json``.

    Every entry is validated, and **all** failures are collected before
    raising — one malformed model would otherwise hide the rest, so you'd fix
    them one import at a time.
    """
    path = paths.models_json()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read models from {path}: {exc}") from exc
    _require(isinstance(raw, dict), f"{path}: expected a JSON object of model -> config")

    registry: dict[str, ModelConfig] = {}
    problems: list[str] = []
    for name, entry in raw.items():
        try:
            registry[name] = _build_entry(name, entry)
        except (ValueError, TypeError) as exc:
            problems.append(f"  {name}: {exc}")
    if problems:
        raise ValueError(
            f"{len(problems)} invalid model entr"
            f"{'y' if len(problems) == 1 else 'ies'} in {path}:\n"
            + "\n".join(problems)
        )
    return registry


_ROLES: dict[str, str] = _load_roles()

#: Every model the library knows about, loaded from ``configs/llm/models.json``.
#: Add or edit entries there — they persist across sessions, unlike a runtime
#: :func:`register_model` call.
MODEL_REGISTRY: dict[str, ModelConfig] = _load_registry()


def get_model_config(model: str) -> ModelConfig:
    """Look up model config from registry.

    Args:
        model: Registered model name.

    Returns:
        The model's :class:`ModelConfig`.

    Raises:
        KeyError: If ``model`` isn't registered — there is no generic-default
            fallback, so a typo'd name errors rather than silently resolving.
            Register it first via :func:`register_model`.
    """
    if model not in MODEL_REGISTRY:
        raise KeyError(
            f"Model {model!r} is not registered in MODEL_REGISTRY. "
            f"Register it first via register_model(...)."
        )
    return MODEL_REGISTRY[model]


def model_compute_config(
    model: str, backend_type: str | None = None
) -> ComputeConfig:
    """Capability flags for a registered model, **without building anything**.

    Answers "could this model capture internals / be called in parallel?" by
    reading the flags off the backend *class* that would serve it. Constructing
    the backend to ask would load model weights for a local setup — so any
    caller that only needs the answer (e.g. ``jailbreak/chain.py`` deciding
    whether to tag a request for capture) should use this instead.

    Args:
        model: Registered model name.
        backend_type: Optional setup override, as for ``ModelClient.create``.

    Returns:
        The serving backend class's :class:`ComputeConfig`.

    Raises:
        KeyError: If the model isn't registered.
        ValueError: If the setup can't be resolved.
    """
    return compute_config_for(transport_for(get_model_config(model), backend_type))


def register_model(
    name: str,
    backend_type: str | None = None,
    roles: list[str] | None = None,
    is_uncensored: bool = False,
    supports_system_prompt: bool = True,
    default_max_tokens: int = 2000,
    default_temperature: float | None = None,
    api: APIConfig | None = None,
    vllm: VLLMConfig | None = None,
    introspect: IntrospectConfig | None = None,
) -> None:
    """Add or update a model in the registry at runtime.

    Build the setups you need and pass them in. Each config validates itself
    on construction, so by the time it reaches here it is already known-good;
    this only assembles them and checks the rules that need the whole picture
    (does the default setup exist, does the concurrency suit the provider).

    **Pass more than one setup** for a model that exists both as a hosted
    endpoint and as local weights — the same shape the built-in
    ``venice-uncensored`` entry uses::

        register_model(
            "my-model",
            backend_type="openai",                        # the default setup
            api=APIConfig(backend_type="openai", api_key_env="MY_API_KEY",
                          base_url="https://api.example.com/v1",
                          rpm=60, recommended_max_workers=3),
            vllm=VLLMConfig(hf_model_id="org/my-model"),  # also reachable
        )

    ``ModelClient.create("my-model")`` then uses the API;
    ``ModelClient.create("my-model", backend_type="vllm")`` runs it locally.

    Args:
        name: Model identifier.
        backend_type: Which **setup** this entry defaults to — "api",
            "vllm" or "introspect", each naming the field that holds it.
            ``None`` means "the only setup present"; an entry with more than
            one must say. Which API *provider* an ``.api`` setup uses is that
            config's own ``backend_type``.
        roles: Logical roles this model can fill, for
            :func:`default_model_for_role` lookup. A model may hold several.
        is_uncensored: True for uncensored generation models.
        supports_system_prompt: False if the model ignores system messages.
        default_max_tokens: Default max tokens for generation.
        default_temperature: Default sampling temperature (None = provider
            default). Must be within 0.0-2.0.
        api: :class:`APIConfig` — rate limit and concurrency for a networked
            model.
        vllm: :class:`VLLMConfig` — which local weights to load via vLLM.
        introspect: :class:`IntrospectConfig` — local weights plus where
            captured internals are written.

    Raises:
        ValueError: If an identity field is invalid, the default
            ``backend_type`` has no matching config, or the concurrency
            doesn't suit the provider. (Field-level problems inside a setup
            surface earlier, when that config is constructed.)
    """
    MODEL_REGISTRY[name] = ModelConfig(
        name=name,
        roles=list(roles or []),
        is_uncensored=is_uncensored,
        supports_system_prompt=supports_system_prompt,
        default_max_tokens=default_max_tokens,
        default_temperature=default_temperature,
        backend_type=backend_type,
        api=api,
        vllm=vllm,
        introspect=introspect,
    )


def get_models_by_role(role: str) -> list[ModelConfig]:
    """Return every registered model that can fill ``role``.

    A model may hold several roles, so it can appear under more than one.

    Args:
        role: Logical role to match (e.g. "uncensored_gen").

    Returns:
        Matching configs in registration order; empty if none match.
    """
    return [cfg for cfg in MODEL_REGISTRY.values() if role in cfg.roles]


def default_model_for_role(role: str) -> str:
    """Return the first registered model name for a role.

    Args:
        role: Logical role to look up (e.g. "constitution_gen").

    Returns:
        Name of the first registered model with that role.

    Raises:
        KeyError: If no model in the registry has this role.
    """
    matches = get_models_by_role(role)
    if not matches:
        raise KeyError(f"No model registered for role {role!r}")
    return matches[0].name
