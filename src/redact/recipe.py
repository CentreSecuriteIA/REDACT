"""What a recipe and its params file say.

A recipe says what to run: ``dataset_type`` (``"eval"`` or ``"training"``;
a training run's inputs are seeded from a constitution, an eval run's when
it has one), ``data_dir``, ``models`` per role (a registry name, or an
object that also says how the model is served, see
:func:`serving_of`), ``taxonomy``/``taxonomy_dir``/``prompt_dir``, an ordered
``stages`` list, ``augmentations`` (the augmentation section, see
:func:`augmentations_of`), a ``params_file`` and ``resume``. Its params file
holds each stage's tunables under the stage's name.
"""

import json
import warnings
from pathlib import Path

from redact.stages import STAGES

_DEFAULT_MODELS = {
    "gen": None, "check": None, "translation": None, "constitution": None,
    "paraphraser": None, "paraphrase_check": None,
}

# Recipe role -> the registry role of its default model.
_REGISTRY_ROLES = {role: registry
                   for stage in STAGES.values() for role, registry in stage.roles}

# The keys of a role's object under ``models`` that say how its model is served.
_SERVING_KEYS = ("backend_type", "copies", "workers")

# The keys a recipe may hold, beside comments ("_" first).
_RECIPE_KEYS = ("dataset_type", "data_dir", "models", "taxonomy", "taxonomy_dir",
                "prompt_dir", "stages", "augmentations", "params_file", "resume")


def load_recipe(recipe: str | Path | dict) -> dict:
    """Load + validate a recipe (JSON path or already-parsed dict)."""
    if isinstance(recipe, dict):
        data = dict(recipe)
        data.setdefault("_base_dir", None)
    else:
        path = Path(recipe)
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        data["_base_dir"] = str(path.parent)

    dtype = data.get("dataset_type", "eval")
    if dtype not in ("eval", "training"):
        raise ValueError(f"dataset_type must be 'eval' or 'training', got {dtype!r}")

    # `is None`, not `or`: an explicit `stages: []` means "run nothing", and
    # `or` turned that into the full default pipeline — the opposite of what
    # was asked for, with no error to notice.
    stages = data.get("stages")
    if stages is None:
        stages = ["inputs", "outputs", "jailbreaks", "build"]
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise ValueError(f"Unknown stage(s) {unknown}; valid: {list(STAGES)}")
    if len(set(stages)) != len(stages):
        raise ValueError(f"A stage is listed more than once in {stages}.")
    if data.get("copies") is not None:
        raise ValueError(
            'The top-level "copies" key is no longer read; write each count in '
            'its role\'s entry under "models" instead, e.g. '
            '"gen": {"name": "llama-3.2-3b-debug", "copies": 2}.')
    # "copies" is no longer read; a null one says nothing.
    unknown = [key for key in data
               if key not in (*_RECIPE_KEYS, "copies") and not key.startswith("_")]
    if unknown:
        raise ValueError(
            f"The recipe has unknown key(s) {unknown}; write only "
            f"{list(_RECIPE_KEYS)}.")
    # For their errors. The entries stay as written, and a run reads them again.
    serving_of(data.get("models") or {})
    data["augmentations"] = augmentations_of(data)
    data["stages"] = stages
    data["dataset_type"] = dtype
    return data


def load_params(
    params: str | Path | dict | None, base_dir: str | Path | None = None,
) -> dict:
    """Load an input-params file (JSON path or dict). ``base_dir`` resolves a
    relative ``params_file`` from the recipe's directory."""
    if params is None:
        return {}
    if isinstance(params, dict):
        data = dict(params)
    else:
        p = Path(params)
        if not p.is_absolute() and base_dir is not None:
            p = Path(base_dir) / p
        with p.open(encoding="utf-8") as fh:
            data = json.load(fh)
    _check_jailbreak_params(data)
    return data


def _check_jailbreak_params(params: dict) -> None:
    """Refuse a key of ``params["jailbreaks"]`` that the augmentation section
    holds, and warn for one of today's flat keywords, read as before for one
    release.

    Raises:
        ValueError: A key only the section holds.
    """
    from .jailbreak.augmentation import FLAT_KEYWORDS, defaults

    given = params.get("jailbreaks") or {}
    section_keys = {key for kind in defaults().values() for key in kind}
    only_section = [key for key in given
                    if key in section_keys and key not in FLAT_KEYWORDS]
    if only_section:
        raise ValueError(
            f"params['jailbreaks'] sets {only_section}, which the augmentation "
            "section holds; write them under the recipe's augmentations, in its "
            "adversarial or natural kind.")
    flat = [key for key in given if key in FLAT_KEYWORDS]
    if flat:
        warnings.warn(
            f"params['jailbreaks'] sets {flat}, which the augmentation section "
            "now holds; write them under the recipe's augmentations.adversarial. "
            "They are read as before for this release.",
            DeprecationWarning, stacklevel=3)


def augmentations_of(recipe: dict, jailbreak_params: dict | None = None) -> dict:
    """The recipe's augmentation section, each kind resolved over its default.

    A kind given as null is off and a kind left out is the default
    (:func:`redact.jailbreak.augmentation.resolve`). Today's flat form, the
    keywords of ``generate_jailbreaks`` the section replaced written at the top
    of ``augmentations``, is read as the adversarial kind, the natural kind
    then the default, with a ``DeprecationWarning``. Those keywords in
    ``jailbreak_params``, the params file's entry for the jailbreak stage, are
    applied over the adversarial kind, so the section is the one the stage
    runs.

    Returns:
        ``{"adversarial": kind or None, "natural": kind or None}``.

    Raises:
        ValueError: The section breaks a rule, or ``jailbreak_params`` sets a
            keyword of the adversarial kind the recipe switches off. The
            message names the kind and the key and says what to write.
    """
    from .jailbreak.augmentation import (
        FLAT_KEYWORDS,
        KINDS,
        apply_flat_keywords,
        resolve,
    )

    section = recipe.get("augmentations")
    section = {} if section is None else section
    if not isinstance(section, dict):
        raise ValueError(
            f"augmentations must be an object of {list(KINDS)}, got {section!r}.")
    flat = {key: value for key, value in section.items()
            if key not in KINDS and not key.startswith("_")}
    if flat:
        kinds = [kind for kind in KINDS if kind in section]
        if kinds:
            raise ValueError(
                f"augmentations gives the kinds {kinds} and the flat keys "
                f"{list(flat)}; write the flat keys inside the adversarial kind.")
        other = [key for key in flat if key not in FLAT_KEYWORDS]
        if other:
            raise ValueError(
                f"augmentations names {other}, which are not augmentation keys; "
                "write a parameter of generate_jailbreaks under "
                "params['jailbreaks'], and a key of the section inside its "
                "adversarial or natural kind.")
        warnings.warn(
            f"augmentations holds the flat keys {list(flat)}, which are deprecated; "
            "write them in augmentations.adversarial (include_<layer> as a layers "
            "deny, include_translation as translation, sampling_probs as "
            "layer_rates, settings_per_iteration as rounds).",
            DeprecationWarning, stacklevel=3)
        section = {"adversarial": apply_flat_keywords(
            resolve(None, "adversarial"), flat)}
    kinds = {kind: resolve(section, kind) for kind in KINDS}
    params = {key: value for key, value in (jailbreak_params or {}).items()
              if key in FLAT_KEYWORDS and value is not None}
    if params and kinds["adversarial"] is None:
        raise ValueError(
            f"params['jailbreaks'] sets {list(params)}, which the adversarial kind "
            "takes, and the recipe switches that kind off "
            "(augmentations.adversarial is null); leave them out of the params, "
            "or switch the kind on.")
    if params:
        kinds["adversarial"] = resolve({"adversarial": apply_flat_keywords(
            kinds["adversarial"], params)}, "adversarial")
    return kinds


def _role_names(models: dict) -> dict:
    """A recipe's ``models`` with each object reduced to its model name."""
    return {role: entry.get("name") if isinstance(entry, dict) else entry
            for role, entry in models.items()}


def resolve_role(
    models: dict, role: str, augmentations: dict | None = None,
) -> list[str]:
    """The models a recipe role stands for, resolved as its stage resolves it.

    ``paraphraser`` may name the pool of every paraphraser, ``check`` left
    out is the ``gen`` model, and ``translation`` is no model when no kind of
    ``augmentations`` (the recipe's section, in any form
    :func:`augmentations_of` reads) draws a translation technique; not asked
    when ``augmentations`` is None.

    Raises:
        KeyError, ValueError: No model is registered for the role.
    """
    from .content_moderation.paraphrase import paraphrase_pool
    from .jailbreak.pool import technique_pool, uses_translation
    from .llms.model_config import default_model_for_role

    name = models.get(role)
    if role == "paraphraser":
        return paraphrase_pool(name)
    if role == "check":
        return [name] if name else resolve_role(models, "gen", augmentations)
    if role == "translation" and augmentations is not None:
        kinds = augmentations_of({"augmentations": augmentations}).values()
        if not any(uses_translation(technique_pool(kind))
                   for kind in kinds if kind is not None):
            return []
    return [name or default_model_for_role(_REGISTRY_ROLES[role])]


def serving_of(models: dict) -> dict[str, dict]:
    """How a recipe's ``models`` say its models are served, per model name.

    A role's entry may be an object in place of a name:

    - ``name``: the model's registry name. Left out, the role's default
      model, as for ``null``.
    - ``backend_type``: the setup of the registry entry that serves it:
      ``"api"``, ``"vllm"`` or ``"introspect"``. Left out, the entry's default.
    - ``copies``: copies of a local vLLM model, one per card. Left out, a
      copy on each idle card; ``1`` turns that off.
    - ``workers``: requests a hosted model gets at once. Left out, the
      entry's ``recommended_max_workers``.

    One model is served one way in a run. A role that gives none of the last
    three keys takes what another role gives its model, so a ``check`` left
    out is served as the generation model is.

    Returns:
        The keys given, per model name. A model no role gives a key for is
        left out.

    Raises:
        ValueError: An object breaks a rule. The message names the role and
            says what to write instead.
    """
    from .llms.backends import SETUP_TYPES, resolve_setup, validate_concurrency
    from .llms.model_config import get_model_config

    names = {**_DEFAULT_MODELS, **_role_names(models)}
    serving: dict[str, dict] = {}
    said_by: dict[str, str] = {}
    for role, entry in models.items():
        if not isinstance(entry, dict):
            continue
        where = f"models[{role!r}]"
        if role not in _DEFAULT_MODELS:
            raise ValueError(
                f"{where} is not a model role; write one of {list(_DEFAULT_MODELS)}.")
        unknown = [key for key in entry if key not in ("name", *_SERVING_KEYS)]
        if unknown:
            raise ValueError(
                f"{where} has unknown key(s) {unknown}; write only "
                f"{['name', *_SERVING_KEYS]}.")
        if role == "paraphraser" and entry.get("name") == "distribution":
            raise ValueError(
                f"{where} names the pool of all paraphrasers, which takes no "
                f'object; write "distribution", or the name of one paraphraser.')
        # A null says what leaving the key out says.
        given = {key: entry[key] for key in _SERVING_KEYS
                 if entry.get(key) is not None}
        for key in ("copies", "workers"):
            if key in given and (type(given[key]) is not int or given[key] < 1):
                raise ValueError(
                    f"{where}: {key} must be a whole number of at least 1, got "
                    f"{given[key]!r}; write one, or leave {key} out.")
        if not given:
            continue

        # The rules below need the registry entry of the role's model.
        try:
            (name,) = resolve_role(names, role)
            config = get_model_config(name)
        except (KeyError, TypeError, ValueError):
            asked = repr(names[role]) if names[role] else "the role's default model"
            raise ValueError(
                f"{where}: {asked} is not a registered model, so how it is "
                f'served cannot be set; write a registered "name".') from None
        present = [s for s in sorted(SETUP_TYPES) if getattr(config, s) is not None]
        try:
            setup = resolve_setup(config, given.get("backend_type"))
        except (TypeError, ValueError):
            raise ValueError(
                f"{where}: backend_type {given.get('backend_type')!r} is not a "
                f"setup of {name!r}; write one of {present}.") from None
        if "copies" in given and setup != "vllm":
            raise ValueError(
                f"{where}: copies are for a model served by vLLM, and {name!r} is "
                f"served by its {setup} setup; leave copies out"
                + (', or write "backend_type": "vllm".' if "vllm" in present else "."))
        if "workers" in given and setup != "api":
            raise ValueError(
                f"{where}: workers are for a hosted model, and {name!r} is served "
                f"by its {setup} setup, where requests at once are automatic; "
                f"leave workers out"
                + (', or write "backend_type": "api".' if "api" in present else "."))
        if "workers" in given:
            try:
                validate_concurrency(name, config.api.backend_type, given["workers"])
            except ValueError:
                raise ValueError(
                    f"{where}: {name!r} takes one request at a time (its "
                    f"{config.api.backend_type} backend makes no parallel calls); "
                    f'write "workers": 1, or leave workers out.') from None
        if serving.setdefault(name, given) != given:
            raise ValueError(
                f"models[{said_by[name]!r}] and {where} both resolve to {name!r} "
                f"but serve it differently ({serving[name]} and {given}); one "
                f"model is served one way in a run, so write the same keys in "
                f"both, or in one of them only.")
        said_by.setdefault(name, role)
    return serving
