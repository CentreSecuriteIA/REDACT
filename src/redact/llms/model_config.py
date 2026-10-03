"""Model registry: one entry per model, with the setups that can serve it.

A :class:`ModelConfig` holds the backend-independent facts about a model
(name, roles, generation defaults) plus the setups that can serve it:

- ``APIConfig``: a hosted endpoint, with its rate limit and concurrency.
- ``VLLMConfig``: local weights loaded through vLLM.
- ``IntrospectConfig``: local weights loaded through transformers, with
  internals capture.

An entry may have more than one setup. ``backend_type`` names the default
one, and ``ModelClient.create(name, backend_type=...)`` selects another. A
backend built from a local setup has ``rpm=None`` and none of the endpoint's
provider parameters.
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
from .backends.introspection import validate_capture

# Whose budget an APIConfig's rpm is: "model" for this model alone, "endpoint"
# for every model behind the same provider, URL and key.
RATE_LIMIT_SCOPES: frozenset[str] = frozenset({"model", "endpoint"})


def _require(condition: bool, message: str, exc: type[Exception] = ValueError) -> None:
    """Raise ``exc(message)`` unless ``condition`` holds.

    Used instead of ``assert``, which ``python -O`` strips.
    """
    if not condition:
        raise exc(message)


# Field checks. Each raises TypeError for the wrong type and ValueError for a
# bad value.

def _check_str(label: str, value, *, optional: bool = False) -> None:
    """Non-empty string (or ``None`` when optional)."""
    if optional and value is None:
        return
    _require(isinstance(value, str),
             f"{label} must be a string, got {type(value).__name__}", TypeError)
    _require(value.strip(), f"{label} must not be empty")


def _check_int(label: str, value, *, minimum: int) -> None:
    """Integer at or above ``minimum``. Rejects ``bool``, which subclasses ``int``."""
    _require(isinstance(value, int) and not isinstance(value, bool),
             f"{label} must be an int, got {type(value).__name__}", TypeError)
    _require(value >= minimum, f"{label} must be >= {minimum}, got {value!r}")


def _check_bool(label: str, value) -> None:
    _require(isinstance(value, bool),
             f"{label} must be a bool, got {type(value).__name__}", TypeError)


def _check_number(label: str, value, *, minimum: float = 0.0) -> None:
    """Number at or above ``minimum``, or ``None``. Rejects ``bool``."""
    if value is None:
        return
    _require(isinstance(value, int | float) and not isinstance(value, bool),
             f"{label} must be a number, got {type(value).__name__}", TypeError)
    _require(value >= minimum, f"{label} must be >= {minimum}, got {value!r}")


def _check_dict(label: str, value) -> None:
    """Dict or ``None``."""
    _require(value is None or isinstance(value, dict),
             f"{label} must be a dict or None, got {type(value).__name__}", TypeError)


@dataclass
class APIConfig:
    """A hosted endpoint: provider, location and budget.

    Attributes:
        backend_type: API transport: "openai" (any OpenAI-compatible
            endpoint) or "anthropic".
        api_key_env: Name of the environment variable holding the API key.
        rpm: Requests-per-minute limit.
        base_url: Endpoint URL. Required for "openai", unused for
            "anthropic".
        api_model_id: Model identifier sent to the provider when it differs
            from the registry name (``venice-uncensored`` is sent as
            ``venice-uncensored-1-2``). ``None`` sends the registry name.
            The registry name stays the key for roles, telemetry and pricing.
        rate_limit_scope: Whose budget ``rpm`` is. ``"model"`` (default)
            gives each model its own window. ``"endpoint"`` makes every model
            with the same :attr:`endpoint_id` share one window, for providers
            that meter the account (Anthropic).
        recommended_max_workers: Concurrency the backend uses as its
            ``max_workers``. Use 1 for a provider that cannot take parallel
            calls (Anthropic).
        default_extra_body: Provider-specific parameters sent with every
            call. Only the OpenAI-compatible backend reads it.
        price_per_1m_input: USD per 1M prompt tokens, for cost reporting.
            ``None`` means unpriced.
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
        """Identity of the endpoint: provider, URL and API-key variable.

        Models with the same id share one rate-limit window under
        ``rate_limit_scope="endpoint"``.
        """
        return f"{self.backend_type}:{self.base_url}:{self.api_key_env}"


@dataclass
class VLLMConfig:
    """Local weights to load through vLLM.

    Attributes:
        hf_model_id: HuggingFace model ID or local path.
        quantization: Quantization method, e.g. "gptq", "awq".
        vllm_kwargs: Extra kwargs for ``vllm.LLM()`` (e.g.
            ``gpu_memory_utilization``, ``tokenizer_mode``).
        sampling: Default ``SamplingParams`` values for this model
            (``top_p``, ``top_k``, ...).
        vram_gb: VRAM the model needs, in GiB, for residency planning.
            ``None`` lets the planner estimate it from the checkpoint's HF
            config; set it to override that estimate. This is the model's
            need, not what vLLM reserves, which is ``gpu_memory_utilization``
            of the whole card.
        min_gpus: Whole devices the model claims. Above 1 it is passed to
            vLLM as ``tensor_parallel_size``, unless ``vllm_kwargs`` sets
            that itself.
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

    @property
    def engine_kwargs(self) -> dict:
        """The kwargs ``vllm.LLM()`` is built with, ``min_gpus`` included."""
        kwargs = dict(self.vllm_kwargs or {})
        if self.min_gpus > 1:
            kwargs.setdefault("tensor_parallel_size", self.min_gpus)
        return kwargs


@dataclass
class IntrospectConfig:
    """Local weights to load through transformers, and where captures go.

    Attributes:
        hf_model_id: HuggingFace model ID or local path.
        log_dir: Root directory for captured internals.
        capture: Which internals to capture: ``logprobs``, ``hidden_states``
            and ``attention`` settings.
        device_map: Passed to ``AutoModelForCausalLM.from_pretrained``.
        torch_dtype: Passed to ``AutoModelForCausalLM.from_pretrained``.
        sampling: Default generation values for this model (``top_p``,
            ``top_k``, ...).
        vram_gb: VRAM the model needs, in GiB, for residency planning.
            ``None`` lets the planner estimate it.
        min_gpus: Whole devices the model claims.
        extra_kwargs: Extra kwargs for ``from_pretrained`` (e.g.
            ``trust_remote_code``).
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
        validate_capture(self.capture)
        _check_str("IntrospectConfig.device_map", self.device_map)
        _check_str("IntrospectConfig.torch_dtype", self.torch_dtype)
        _check_dict("IntrospectConfig.sampling", self.sampling)
        _check_number("IntrospectConfig.vram_gb", self.vram_gb)
        _check_int("IntrospectConfig.min_gpus", self.min_gpus, minimum=1)
        _check_dict("IntrospectConfig.extra_kwargs", self.extra_kwargs)


@dataclass
class ModelConfig:
    """One model: its identity and the setups that can serve it.

    Validated on construction. When several setups are present,
    ``backend_type`` names the default and the others are reachable through
    ``ModelClient.create(name, backend_type=...)``.

    Attributes:
        name: Model identifier used at call sites.
        roles: Logical roles this model can fill, for
            :func:`default_model_for_role`.
        is_uncensored: True for uncensored generation models.
        supports_system_prompt: False if the model ignores system messages.
            The backend then folds system content into the first user
            message.
        default_max_tokens: Used when a caller passes no ``max_tokens``.
        default_temperature: Used when a caller passes no temperature.
            ``None`` leaves it to the backend or provider.
        backend_type: The default setup: "api", "vllm" or "introspect".
            ``None`` means the only setup present. The API provider is named
            by ``api.backend_type``, not here.
        api: Hosted-endpoint setup, or None.
        vllm: vLLM setup, or None.
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

        # The worker count must suit the API provider, whichever setup is the
        # default. A local setup on the same entry does not use it.
        if self.api is not None:
            validate_concurrency(
                self.name, self.api.backend_type, self.api.recommended_max_workers
            )

        if self.backend_type is None:
            return  # no default declared; the setup checks below are skipped

        _require(
            self.backend_type in SETUP_TYPES,
            f"{self.name!r}: unknown backend_type {self.backend_type!r}; "
            f"expected one of {sorted(SETUP_TYPES)}",
        )
        # Each setup name is also the name of the field that holds it.
        _require(
            getattr(self, self.backend_type) is not None,
            f"{self.name!r}: backend_type={self.backend_type!r} needs a "
            f"{self.backend_type} config — pass {self.backend_type}=...",
        )


_SETUP_CLASSES = {"api": APIConfig, "vllm": VLLMConfig, "introspect": IntrospectConfig}


def available_roles() -> dict[str, str]:
    """Every role a model may claim, mapped to its description.

    Loaded from ``configs/llm/roles.json``. A role outside this list is
    rejected when a model is registered.
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
    """Turn one JSON entry into a validated :class:`ModelConfig`."""
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

    Every invalid entry is collected and reported in one ``ValueError``.
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
#: Edit that file to add a model permanently. A :func:`register_model` call
#: lasts only for the process.
MODEL_REGISTRY: dict[str, ModelConfig] = _load_registry()


def get_model_config(model: str) -> ModelConfig:
    """Return the :class:`ModelConfig` of a registered model.

    Args:
        model: Registered model name.

    Raises:
        KeyError: ``model`` is not registered.
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
    """Capability flags for a registered model, without building a backend.

    Reads the flags from the backend class that would serve the model, so no
    weights are loaded.

    Args:
        model: Registered model name.
        backend_type: Optional setup override, as for ``ModelClient.create``.

    Raises:
        KeyError: The model is not registered.
        ValueError: The setup cannot be resolved.
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
    """Add or replace a model in the registry for this process.

    Build the setup configs and pass them in. Pass more than one for a model
    that is both a hosted endpoint and local weights::

        register_model(
            "my-model",
            backend_type="api",                           # the default setup
            api=APIConfig(backend_type="openai", api_key_env="MY_API_KEY",
                          base_url="https://api.example.com/v1",
                          rpm=60, recommended_max_workers=3),
            vllm=VLLMConfig(hf_model_id="org/my-model"),  # also available
        )

    ``ModelClient.create("my-model")`` then uses the API, and
    ``ModelClient.create("my-model", backend_type="vllm")`` runs it locally.

    Args:
        name: Model identifier.
        backend_type: The default setup: "api", "vllm" or "introspect".
            ``None`` means the only setup present.
        roles: Logical roles this model can fill.
        is_uncensored: True for uncensored generation models.
        supports_system_prompt: False if the model ignores system messages.
        default_max_tokens: Default max tokens for generation.
        default_temperature: Default sampling temperature, within 0.0-2.0.
            ``None`` leaves it to the backend or provider.
        api: Hosted-endpoint setup.
        vllm: vLLM setup.
        introspect: Internals-capture setup.

    Raises:
        TypeError: A field has the wrong type.
        ValueError: A field is invalid, the default ``backend_type`` has no
            matching config, or the concurrency does not suit the provider.
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

    Args:
        role: Logical role to match (e.g. "uncensored_gen").

    Returns:
        Matching configs in registration order; empty if none match.
    """
    return [cfg for cfg in MODEL_REGISTRY.values() if role in cfg.roles]


def default_model_for_role(role: str) -> str:
    """Return the name of the first registered model with a role.

    Args:
        role: Logical role to look up (e.g. "constitution_gen").

    Raises:
        KeyError: No registered model has this role.
    """
    matches = get_models_by_role(role)
    if not matches:
        raise KeyError(f"No model registered for role {role!r}")
    return matches[0].name
