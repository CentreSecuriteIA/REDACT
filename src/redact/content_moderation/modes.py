"""The answer modes of the outputs stage: how a response answers its input.

A mode is ``harmful`` or ``helpful``. It picks the generation prompt
(``prompts/output/generation/<mode>/``) and the checker
(``prompts/output/quality_check/<mode>/``), and gives the response its label,
``<entry_type>_<output_label>``: the input's entry type, then the mode's
output label, ``harmful`` for the harmful mode and ``benign`` for the helpful
one. The modes an entry type allows are ``outputs.modes``, laid over
:data:`DEFAULT_MODES` entry type by entry type.

The mode is not a generation axis. It selects the prompt, the checker and the
label, where an axis is a line of the response spec; it must hold with
``axes: false``, since without it the stage has no prompt to run; and no
augmentation family varies it, so the dimensions table has nothing to say
about it.

The deal (:func:`deal_modes`) follows the axes' (``axes.deal_specs``): one
dealer per ``(category, entry_type)`` group, its units in ``(input_id,
iteration)`` order, an input's units taking a mode at most ceil(k / modes)
times, k its units. At an even k every input takes each of two modes k / 2
times, so the group's counts are level at every input's first unit and an
input's deal is a function of its own key alone: a resume with more inputs
keeps it. At k = 1 and an odd k the group alternates, so inputs that join a
group would change the modes of units dealt before them; the stage passes
the modes its done units were written with as ``fixed``, which the deal
counts and deals the other units around.
"""

import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence

from ..dataset.dealing import Dealer, stable_seed
from ..types import ALL_ENTRY_TYPES

#: The answer modes, each a folder of the generation and checker prompts.
MODES = ("harmful", "helpful")

#: Each mode's output label: the second part of a response's label.
OUTPUT_LABEL = {"harmful": "harmful", "helpful": "benign"}

#: The modes each entry type allows when ``outputs.modes`` does not say.
DEFAULT_MODES = {
    "harmful": ["harmful"],
    "dual_use_harmful": ["harmful", "helpful"],
    "dual_use_benign": ["harmful", "helpful"],
    "benign": ["helpful"],
}

Group = tuple[str, str]     # (category, entry_type)
Unit = tuple[Group, tuple[str, int]]    # (group, (input_id, iteration))


def _value(entry_type) -> str:
    return getattr(entry_type, "value", entry_type)


def resolve_modes(given: dict | None = None) -> dict[str, list[str]]:
    """The modes each entry type allows: :data:`DEFAULT_MODES` with ``given``
    laid over it, entry type by entry type.

    ``given`` maps an entry type to a non-empty list of distinct modes; a key
    starting with ``_`` is a comment. ``None`` gives the default.

    Returns:
        Every entry type, in their order, with its list of modes.

    Raises:
        ValueError: ``given`` is not an object, names a key that is not an
            entry type, or gives one a value that is not a non-empty list of
            distinct modes. The message names ``outputs.modes``.
    """
    resolved = {_value(t): list(DEFAULT_MODES[_value(t)]) for t in ALL_ENTRY_TYPES}
    if given is None:
        return resolved
    if not isinstance(given, dict):
        raise ValueError(
            "outputs.modes must be an object of entry type: [mode, ...], got "
            f"{given!r}.")
    for entry_type, allowed in given.items():
        if isinstance(entry_type, str) and entry_type.startswith("_"):
            continue
        if entry_type not in resolved:
            raise ValueError(
                f"outputs.modes names no entry type {entry_type!r}; the entry "
                f"types are {list(resolved)}.")
        if (not isinstance(allowed, list) or not allowed
                or any(mode not in MODES for mode in allowed)
                or len(set(allowed)) != len(allowed)):
            raise ValueError(
                f"outputs.modes.{entry_type} must be a non-empty list of distinct "
                f"modes of {list(MODES)}, got {allowed!r}.")
        resolved[entry_type] = list(allowed)
    return resolved


def check_outputs_per_input(value) -> int:
    """``value`` when it is a whole number of at least 1: the responses the
    outputs stage writes per input, each a unit of its own.

    Raises:
        ValueError: Anything else, a bool too. The message names
            ``outputs.outputs_per_input``.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            "outputs.outputs_per_input must be a whole number of at least 1, got "
            f"{value!r}.")
    return value


def response_label(entry_type, mode: str) -> str:
    """A response's label: its input's ``entry_type``, then the output label
    of its ``mode``, as ``dual_use_benign_harmful``. The parts hold
    underscores, so the label splits on its last one only.

    Raises:
        ValueError: ``mode`` is not one of :data:`MODES`.
    """
    if mode not in OUTPUT_LABEL:
        raise ValueError(f"The answer mode must be one of {list(MODES)}, got {mode!r}.")
    return f"{_value(entry_type)}_{OUTPUT_LABEL[mode]}"


def deal_modes(
    modes: dict[str, list[str]], units: Sequence[Unit], *, seed: int,
    fixed: Mapping[int, str] | None = None,
) -> list[str]:
    """One mode for each unit, dealt per group from the modes its entry type
    allows.

    ``units`` are ``(group, (input_id, iteration))`` pairs, ``group`` being
    ``(category, entry_type)``, the shape ``axes.deal_specs`` takes; an
    input's units are its siblings. The units of a group are dealt in sorted
    order from one dealer: each takes the mode of ``modes[entry_type]`` with
    room and the lowest count in the group, ties broken by the unit's rng,
    seeded by ``(seed, *group, input_id, iteration)``, so the deal does not
    depend on the units' order. An input of k siblings takes a mode at most
    ceil(k / len(allowed)) times, so from k = 2 an input whose entry type
    allows two modes gets one answer of each.

    ``fixed`` maps the index of a unit in ``units`` to the mode it already
    has, as a done unit's row records it: the unit keeps it, and it counts
    in the group and for its siblings as a dealt mode does, so the other
    units are dealt around it. Each unit's rng is its own, so fixing one
    changes no other unit's draws.

    Returns:
        Each unit's mode, in the order of ``units``.
    """
    fixed = fixed or {}
    siblings = Counter((group, entry) for group, (entry, _) in units)
    taken: dict[tuple, Counter] = defaultdict(Counter)
    dealers: dict[Group, Dealer] = {}
    dealt = [""] * len(units)
    for i in sorted(range(len(units)), key=lambda i: units[i]):
        group, (entry, iteration) = units[i]
        allowed = modes[group[1]]
        dealer = dealers.setdefault(
            group, Dealer(dict.fromkeys(MODES, 1), key=(seed, *group, "mode")))
        rng = random.Random(stable_seed(seed, *group, entry, iteration, "mode"))
        own = taken[(group, entry)]
        cap = math.ceil(siblings[(group, entry)] / len(allowed))
        room = [mode for mode in allowed if own[mode] < cap]
        mode = fixed[i] if i in fixed else dealer.pick(rng, room or list(allowed))
        dealer.counts[mode] += 1
        own[mode] += 1
        dealt[i] = mode
    return dealt
