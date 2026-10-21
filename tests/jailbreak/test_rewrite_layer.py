"""The rewrite layer: the spec's first layer, which holds no technique yet.

A stand-in technique, tagged by hand, shows what the layer does once it has
one; nothing here edits a file.
"""

import redact.jailbreak  # noqa: F401  (tags the techniques)
from redact.jailbreak.pool import all_techniques, registry
from redact.jailbreak.rewrite import get_all_rewrite_functions
from redact.jailbreak.rules import compatible
from redact.jailbreak.technique import load_spec

BY_NAME = registry(all_techniques())


def _stand_in(name: str = "to_stand_in_rewrite", family: str = "stand_in_rewrite"):
    """A rewrite technique of ``family`` that returns the prompt unchanged."""
    def technique(prompt, *, rng=None):
        return prompt, ""

    technique.__name__ = name
    technique.families = [family]
    technique.layer = "rewrite"
    technique.within_layer_order = 0
    technique.complexity = 1
    technique.kinds = ["adversarial"]
    return technique


def test_the_rewrite_layer_comes_first_and_holds_no_technique_yet():
    spec = load_spec()
    assert spec["layer_order"] == [
        "rewrite", "hacking", "manipulation", "obfuscation", "requests"]
    assert get_all_rewrite_functions() == []
    assert not [name for name, family in spec["families"].items()
                if family["layer"] == "rewrite"]


def test_the_rewrite_layer_takes_one_pick_per_combination():
    first = _stand_in("to_stand_in_rewrite_a", "stand_in_rewrite_a")
    second = _stand_in("to_stand_in_rewrite_b", "stand_in_rewrite_b")
    framing = BY_NAME["to_fictional_world"]
    assert load_spec()["layer_max_picks"]["rewrite"] == 1
    assert compatible([first], [second, framing]) == [framing]
    assert compatible([second], [first]) == []


def test_a_technique_of_a_rewrite_family_is_tagged_with_the_layer(monkeypatch):
    import redact.jailbreak.technique as T

    spec = load_spec()
    monkeypatch.setattr(T, "load_spec", lambda: {
        **spec,
        "families": {**spec["families"], "stand_in_rewrite": {"layer": "rewrite"}},
        "techniques": {**spec["techniques"], "to_stand_in_rewrite": {
            "families": ["stand_in_rewrite"], "complexity": 1}},
    })

    def to_stand_in_rewrite(prompt):
        return prompt, ""

    T.tag([to_stand_in_rewrite])
    assert to_stand_in_rewrite.layer == "rewrite"
    assert to_stand_in_rewrite.within_layer_order == 0
    assert to_stand_in_rewrite.kinds == ["adversarial"]
