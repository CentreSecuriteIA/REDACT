"""The technique pool: every registered technique, the run's filters, the registry."""

import pytest

import redact.jailbreak.pool as JPool
from redact.jailbreak.hacking import get_all_hacking_functions
from redact.jailbreak.manipulation import get_all_manipulation_functions
from redact.jailbreak.obfuscation import get_all_obfuscation_functions
from redact.jailbreak.pool import (
    all_techniques,
    registry,
    technique_pool,
    uses_translation,
)
from redact.jailbreak.requests import get_all_request_functions
from redact.jailbreak.technique import load_spec


def _in(family: str, pool: list) -> list:
    return [fn for fn in pool if family in fn.families]


def test_all_techniques_are_the_four_layers_in_registry_order():
    assert all_techniques() == (
        get_all_obfuscation_functions() + get_all_hacking_functions()
        + get_all_manipulation_functions() + get_all_request_functions())


def test_the_default_pool_is_every_technique():
    assert technique_pool() == all_techniques()


def test_include_translation_false_drops_exactly_the_translation_family():
    full = technique_pool()
    assert _in("translation", full)

    pool = technique_pool(include_translation=False)
    assert pool == [fn for fn in full if fn not in _in("translation", full)]


def test_a_layer_flag_drops_that_layer_only():
    pool = technique_pool(include_hacking=False, include_requests=False)
    assert pool == get_all_obfuscation_functions() + get_all_manipulation_functions()


def test_pure_only_drops_model_calls_and_benign_data():
    pool = technique_pool(pure_only=True)
    assert pool
    assert not any(fn.requires_llm or fn.requires_benign for fn in pool)
    assert pool == [fn for fn in all_techniques()
                    if not fn.requires_llm and not fn.requires_benign]


def test_registry_names_every_technique_it_is_given():
    techniques = all_techniques()
    by_name = registry(techniques)
    assert list(by_name) == [fn.__name__ for fn in techniques]
    assert all(by_name[fn.__name__] is fn for fn in techniques)


def test_uses_translation():
    assert uses_translation(technique_pool())
    assert not uses_translation(technique_pool(include_translation=False))
    assert not uses_translation(technique_pool(include_obfuscation=False))


def test_the_registered_techniques_match_the_spec():
    JPool._check_against_spec(all_techniques())


@pytest.mark.parametrize(("drop_from", "missing"), [
    ("spec", "registered but not in the spec: to_base64"),
    ("registry", "in the spec but not registered: to_base64"),
])
def test_a_name_the_spec_and_the_registry_do_not_share_raises(
        monkeypatch, drop_from, missing):
    spec = load_spec()
    techniques = all_techniques()
    if drop_from == "spec":
        trimmed = {k: v for k, v in spec["techniques"].items() if k != "to_base64"}
        monkeypatch.setattr(JPool, "load_spec", lambda: {**spec, "techniques": trimmed})
    else:
        techniques = [fn for fn in techniques if fn.__name__ != "to_base64"]

    with pytest.raises(ValueError, match=missing):
        JPool._check_against_spec(techniques)
