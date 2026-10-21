"""A technique's applicability: the plan draws a technique only for an input
it applies to.

A technique may carry ``applicable(text, entry_type)``, set with
``technique.applicable_when``; ``plan_run`` filters the pool for each input
before the draw. A technique without one applies to every input, so a pool
whose techniques all apply draws what it drew before (the dealing digests of
``test_dealing.py`` pin it too).
"""

import pandas as pd
import pytest

import redact.jailbreak  # noqa: F401  (tags the techniques)
from redact.dataset import Manifest
from redact.jailbreak.plan import combination_of, plan_run
from redact.jailbreak.pool import all_techniques
from redact.jailbreak.technique import applicable_when, applies
from tests.jailbreak import simulate


def _technique(name: str, family: str, layer: str = "obfuscation", predicate=None):
    """A pure technique of complexity 0 tagged by hand, as ``technique.tag``
    tags one, with ``predicate`` as its applicability."""
    def fn(prompt, **kwargs):
        return f"{prompt} {name}", ""
    fn.__name__ = name
    fn.families = [family]
    fn.layer = layer
    fn.within_layer_order = 0
    fn.complexity = 0
    return fn if predicate is None else applicable_when(predicate)(fn)


def _marked(text: str, entry_type: str) -> bool:
    return "x" in text


def _benign_only(text: str, entry_type: str) -> bool:
    return entry_type in ("benign", "dual_use_benign")


def _inputs(texts: list[str], entry_types: list[str] | None = None) -> pd.DataFrame:
    """One input per text, of one category; ``u00``, ``u01``, ... as ids, so
    the group is dealt in the texts' order."""
    return pd.DataFrame({
        "sample_id": [f"u{i:02d}" for i in range(len(texts))],
        "prompt": texts,
        "category": ["Cyber"] * len(texts),
        "entry_type": entry_types or ["harmful"] * len(texts),
    })


def _plan(tmp_path, inputs: pd.DataFrame, pool: list, **settings) -> list[dict]:
    """The manifest rows of a one-round plan of ``inputs`` drawn from ``pool``."""
    manifest = tmp_path / "jb.manifest.jsonl"
    plan_run(inputs, pool, manifest_path=manifest, seed=7, sample_kwargs=settings,
             verbose=False)
    return Manifest(manifest).load()


def test_a_technique_without_an_applicability_applies_to_every_input():
    assert applies(_technique("to_any", "alpha"), "", "harmful")


def test_applicable_when_returns_the_technique_with_its_predicate():
    def to_role(prompt):
        return prompt, ""

    assert applicable_when(_benign_only)(to_role) is to_role
    assert to_role.applicable is _benign_only
    assert applies(to_role, "a text", "benign")
    assert not applies(to_role, "a text", "harmful")


_MODES = {"budget": {}, "count": {"exact_techniques": 1}}


@pytest.mark.parametrize("mode", sorted(_MODES))
def test_a_technique_is_drawn_only_for_the_inputs_it_applies_to(tmp_path, mode):
    pool = [_technique("to_marked", "alpha", predicate=_marked),
            _technique("to_b", "beta"), _technique("to_c", "gamma")]
    texts = [f"input {i}" + (" x" if i % 2 else "") for i in range(30)]
    rows = _plan(tmp_path, _inputs(texts), pool, **_MODES[mode])
    text_of = dict(zip(_inputs(texts)["sample_id"], texts))
    drawn_for = [text_of[row["sample_id"]] for row in rows
                 if "to_marked" in row["combination"]]
    assert drawn_for and all("x" in text for text in drawn_for)


def test_a_technique_for_benign_inputs_reads_the_entry_type(tmp_path):
    """As a benign role would: drawn for benign and dual-use benign inputs,
    never for harmful ones."""
    pool = [_technique("to_benign_role", "alpha", predicate=_benign_only),
            _technique("to_other", "beta")]
    entry_types = ["harmful", "benign", "dual_use_benign"] * 10
    rows = _plan(tmp_path, _inputs([f"input {i}" for i in range(30)], entry_types),
                 pool, exact_techniques=1)
    drawn = {row["entry_type"] for row in rows
             if row["combination"] == ["to_benign_role"]}
    assert drawn == {"benign", "dual_use_benign"}


def test_a_layer_goes_to_the_units_its_techniques_apply_to(tmp_path):
    """No request technique applies to an input without ``x``: that unit
    takes no request, and the dealer, which counts what each unit drew, gives
    the layer to the next unit that can take it, so the group still meets
    its target of 10."""
    ask = _technique("to_ask", "asking", layer="requests", predicate=_marked)
    texts = [f"input {i}" + (" x" if i % 2 else "") for i in range(20)]
    rows = _plan(tmp_path, _inputs(texts), [ask],
                 sampling_probs={"hacking": 0, "manipulation": 0, "requests": 0.5})
    assert [row["combination"] for row in rows] == [[], ["to_ask"]] * 10


def test_an_input_no_technique_applies_to_gets_the_empty_combination(tmp_path):
    """The identity, as from an empty pool: the natural kind rejects the
    unchanged text, and the build leaves out an adversarial no-op."""
    never = _technique("to_never", "alpha", predicate=lambda text, entry_type: False)
    rows = _plan(tmp_path, _inputs(["one", "two"]), [never], exact_techniques=1)
    assert [row["combination"] for row in rows] == [[], []]


def test_a_predicate_true_for_every_input_changes_no_draw(monkeypatch):
    rounds = [{}, {"exact_techniques": 2}]
    pool = all_techniques()
    before = simulate.deal(pool, 60, groups=2, rounds=rounds)
    for fn in pool:
        monkeypatch.setattr(fn, "applicable", lambda text, entry_type: True,
                            raising=False)
    assert simulate.deal(pool, 60, groups=2, rounds=rounds) == before


def test_a_kept_plan_replays_without_asking_the_predicate():
    def refuse(text, entry_type):
        raise AssertionError("the replay asked the predicate")

    never = _technique("to_never", "alpha", predicate=refuse)
    combined = combination_of(["to_never"], {"to_never": never})
    assert [fn.__name__ for fn in combined.techniques] == ["to_never"]
