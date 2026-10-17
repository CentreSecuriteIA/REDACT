"""What a recipe and its params file say.

A recipe says what to run: ``dataset_type`` (``"eval"`` or ``"training"``,
required; see :func:`load_dataset_type`), ``data_dir``,
``models`` per role (a registry name, or an object that also says how the
model is served, see :func:`serving_of`),
``taxonomy``/``taxonomy_dir``/``prompt_dir``,
``benign_taxonomy`` (see :func:`benign_taxonomy_of`), an ordered
``stages`` list, ``augmentations`` (the augmentation section, see
:func:`augmentations_of`), a ``params_file``, ``resume`` and a
``config_dir`` (a folder of your own configs, read relative to the recipe,
see :mod:`redact.config_overlay`). Its params file holds each stage's
tunables under the stage's name.

The dataset type's file gives the defaults the recipe leaves out: its
``stages`` (the recipe's replace them), its ``augmentations`` (the recipe's
overlay them kind by kind) and its ``params`` (the params file's overlay them
key by key per stage).
"""

import copy
import json
import warnings
from functools import cache
from pathlib import Path

from redact.config_overlay import DATASET_TYPES, config_file, taxonomy_folder
from redact.stages import STAGES, includes_general_benign

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
                "benign_taxonomy", "prompt_dir", "stages", "augmentations",
                "params_file", "resume", "config_dir")

# The former name of the packaged taxonomy "harmful", read for one release.
_FORMER_TAXONOMY = "content_moderation_categories"

# The keys a dataset type file may hold, beside comments ("_" first).
_TYPE_KEYS = ("stages", "params", "augmentations")


@cache
def _read_type_file(path: str, modified: int) -> dict:
    with Path(path).open(encoding="utf-8") as fh:
        data = json.load(fh)
    keys = [key for key in data if not key.startswith("_")]
    if sorted(keys) != sorted(_TYPE_KEYS):
        raise ValueError(
            f"{path} has the keys {keys}; write exactly {list(_TYPE_KEYS)}.")
    load_params(data["params"])     # for its errors
    return data


def _dataset_type_file(name: str, config_dir: str | Path | None) -> dict:
    """The type file, read again when it changes."""
    if name not in DATASET_TYPES:
        raise ValueError(f'dataset_type must be "eval" or "training", got {name!r}.')
    path = config_file(f"dataset_types/{name}.json", config_dir)
    return _read_type_file(str(path), path.stat().st_mtime_ns)


def load_dataset_type(name: str, config_dir: str | Path | None = None) -> dict:
    """The file ``configs/dataset_types/<name>.json``, ``config_dir``'s when
    it has one, else the packaged one: the default ``stages``, ``params``
    per stage and ``augmentations`` overlay of a run of that type, as a copy
    the caller may change.

    Raises:
        ValueError: ``name`` is not ``"eval"`` or ``"training"``, the file
            does not hold exactly those three keys, or :func:`load_params`
            refuses its params.
    """
    return copy.deepcopy(_dataset_type_file(name, config_dir))


def dataset_type_hash(name: str, config_dir: str | Path | None = None) -> str:
    """The content id of the dataset type's file, as parsed: what a run
    record names the defaults it ran with by."""
    from .dataset import content_id

    return content_id(
        json.dumps(_dataset_type_file(name, config_dir), sort_keys=True))


def stage_params(
    stage: str, dataset_type: str, params: dict | None = None,
    config_dir: str | Path | None = None,
) -> dict:
    """The params of ``stage``: the dataset type file's, with the entry of
    ``params`` (the params file's) laid over them key by key."""
    typed = load_dataset_type(dataset_type, config_dir)["params"].get(stage)
    return {**(typed or {}), **((params or {}).get(stage) or {})}


def _config_dir(data: dict) -> str | None:
    """The recipe's ``config_dir``, relative to the recipe's folder.

    Raises:
        ValueError: It is not a folder.
    """
    given = data.get("config_dir")
    if given is None:
        return None
    if not isinstance(given, str | Path):
        raise ValueError(f"config_dir must be a folder's path, got {given!r}.")
    folder = Path(given)
    if not folder.is_absolute() and data["_base_dir"] is not None:
        folder = Path(data["_base_dir"]) / folder
    folder = folder.absolute()      # so a loaded recipe loads again the same
    if not folder.is_dir():
        raise ValueError(
            f"config_dir {str(folder)!r} is not a folder; write the folder of "
            "your dataset_types/, taxonomy/ and jailbreak/ files, relative to "
            "the recipe.")
    return str(folder)


def load_recipe(
    recipe: str | Path | dict, params: str | Path | dict | None = None,
) -> dict:
    """Load + validate a recipe (JSON path or already-parsed dict).

    ``params`` stands in for the recipe's ``params_file``, as a run's does:
    the benign taxonomy is checked with its params over the dataset type's.
    """
    if isinstance(recipe, dict):
        data = dict(recipe)
        data.setdefault("_base_dir", None)
    else:
        path = Path(recipe)
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        data["_base_dir"] = str(path.parent)

    dtype = data.get("dataset_type")
    if dtype is None:
        raise ValueError(
            'The recipe names no dataset_type; write "dataset_type": "eval" or '
            '"training".')
    data["config_dir"] = _config_dir(data)
    dataset_type = load_dataset_type(dtype, data["config_dir"])

    # `is None`, not `or`: an explicit `stages: []` means "run nothing", and
    # `or` turned that into the full default pipeline — the opposite of what
    # was asked for, with no error to notice.
    stages = data.get("stages")
    if stages is None:
        stages = dataset_type["stages"]
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
    # A string "false" would read as true.
    resume = data.get("resume", True)
    if type(resume) is not bool:
        raise ValueError(f"resume must be true or false, got {resume!r}.")
    if data.get("taxonomy") == _FORMER_TAXONOMY:
        warnings.warn(
            f'The recipe names the taxonomy "{_FORMER_TAXONOMY}", now "harmful"; '
            'write "taxonomy": "harmful". The former name is read for this release.',
            DeprecationWarning, stacklevel=2)
        if data.get("taxonomy_dir") is None:
            data["taxonomy"] = "harmful"
    # For their errors. The entries stay as written, and a run reads them again.
    serving_of(data.get("models") or {})
    data["augmentations"] = augmentations_of(data)
    # Read as written: the run checks the params, and warns, when it loads them.
    given = _read_params(data.get("params_file") if params is None else params,
                         data["_base_dir"])
    data["benign_taxonomy"] = benign_taxonomy_of(data, stages, given)
    data["stages"] = stages
    data["dataset_type"] = dtype
    return data


def _read_params(params: str | Path | dict | None,
                 base_dir: str | Path | None = None) -> dict:
    """``params`` as written: the dict, or the JSON file it names, a relative
    path read from ``base_dir``; none for ``None``."""
    if params is None:
        return {}
    if isinstance(params, dict):
        return dict(params)
    path = Path(params)
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def load_params(
    params: str | Path | dict | None, base_dir: str | Path | None = None,
) -> dict:
    """Load an input-params file (JSON path or dict). ``base_dir`` resolves a
    relative ``params_file`` from the recipe's directory.

    Raises:
        ValueError: A key is not a stage's name, nor a comment (``_`` first),
            or ``params["jailbreaks"]`` sets a key of the augmentation section.
    """
    data = _read_params(params, base_dir)
    _check_jailbreak_params(data)
    unknown = [key for key in data if key not in STAGES and not key.startswith("_")]
    if unknown:
        raise ValueError(
            f"The params file has key(s) {unknown}, which are not stages; write "
            f"only {list(STAGES)}, or a comment key starting with _.")
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


def benign_taxonomy_of(
    recipe: dict, stages: list[str], params: dict | None = None,
) -> dict | str:
    """The recipe's ``benign_taxonomy``: the taxonomy file, under ``taxonomy_dir``,
    whose categories the constitution's general-benign part is built from and
    whose subcategories the benign Q&A pairs of the fsh and dap techniques are
    generated for; ``"benign"``, the packaged one, when left out or null.

    The file is read when the recipe runs the constitution stage and its
    params (the dataset type's, with ``params``, the params file's, over
    them) switch the general-benign part on, by its name or its former one,
    or runs the jailbreak stage and a kind of its ``augmentations``
    (resolved) draws a technique that needs benign data.

    Raises:
        ValueError: The key is not a name or an object, or the taxonomy it
            names is missing, does not parse or has no category.
    """
    from .dataset.taxonomy import load_taxonomy
    from .jailbreak.manipulation.benign import DEFAULT_TAXONOMY, benign_categories
    from .jailbreak.pool import technique_pool

    taxonomy = recipe.get("benign_taxonomy")
    if taxonomy is None:
        taxonomy = DEFAULT_TAXONOMY
    if not isinstance(taxonomy, dict) and not (
            isinstance(taxonomy, str) and taxonomy.strip()):
        raise ValueError(
            f"benign_taxonomy must name a taxonomy file, got {taxonomy!r}; write "
            f'a name such as "{DEFAULT_TAXONOMY}", or leave it out.')
    uses = []
    config_dir = recipe.get("config_dir")
    if "constitution" in stages and includes_general_benign(stage_params(
            "constitution", recipe["dataset_type"], params, config_dir)):
        uses.append("the constitution builds its general-benign part from its "
                    "categories")
    kinds = [kind for kind in recipe["augmentations"].values() if kind is not None]
    if "jailbreaks" in stages and any(
            getattr(fn, "requires_benign", False)
            for kind in kinds for fn in technique_pool(kind)):
        uses.append("the jailbreak stage generates the benign Q&A pairs of its "
                    "fsh and dap techniques for its subcategories")
    if not uses:
        return taxonomy
    where = repr(taxonomy) if isinstance(taxonomy, str) else "(an object)"
    try:
        folder = taxonomy_folder(taxonomy, recipe.get("taxonomy_dir"), config_dir)
        loaded = (load_taxonomy(taxonomy, config_dir=folder)
                  if isinstance(taxonomy, str) else taxonomy)
        categories = benign_categories(loaded)
    except (OSError, ValueError, AttributeError, TypeError) as error:
        raise ValueError(
            f"benign_taxonomy {where} cannot be read: {error}. "
            f"{' and '.join(uses).capitalize()}; name a taxonomy file in "
            "taxonomy_dir (when it is left out, the config_dir's taxonomy/, "
            "then the packaged configs/taxonomy, which holds benign.json).") from None
    if not categories:
        raise ValueError(
            f"benign_taxonomy {where} has no category; write its categories, "
            "each with its subcategories.")
    return taxonomy


def augmentations_of(recipe: dict, jailbreak_params: dict | None = None) -> dict:
    """The recipe's augmentation section, each kind resolved over its default.

    The default is the recipe's dataset type's section over the packaged
    one; with no ``dataset_type`` in ``recipe``, the packaged one. Both are
    read through the recipe's ``config_dir`` when it has one. A kind
    given as null is off and a kind left out is the default
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

    config_dir = recipe.get("config_dir")
    base = (load_dataset_type(recipe["dataset_type"], config_dir)["augmentations"]
            if recipe.get("dataset_type") else None)
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
        section = {"adversarial": apply_flat_keywords(resolve(
            {"adversarial": {}}, "adversarial", base=base, config_dir=config_dir),
            flat)}
    kinds = {kind: resolve(section, kind, base=base, config_dir=config_dir)
             for kind in KINDS}
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
            kinds["adversarial"], params)}, "adversarial", config_dir=config_dir)
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
        ValueError: A key is not a model role, or an object breaks a rule.
            The message names the role and says what to write instead.
    """
    from .llms.backends import SETUP_TYPES, resolve_setup, validate_concurrency
    from .llms.model_config import get_model_config

    names = {**_DEFAULT_MODELS, **_role_names(models)}
    serving: dict[str, dict] = {}
    said_by: dict[str, str] = {}
    for role, entry in models.items():
        where = f"models[{role!r}]"
        if role.startswith("_"):    # a comment
            continue
        if role not in _DEFAULT_MODELS:
            raise ValueError(
                f"{where} is not a model role; write one of {list(_DEFAULT_MODELS)}.")
        if not isinstance(entry, dict):
            continue
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
