"""A library that adds and retires techniques: what a kind's pool draws.

The spec dates a technique with ``since``, the version that added it, and
``deprecated``, the one that retired it. The library has retired none yet, so
these tests retire a real technique by hand.
"""

import pytest

from redact.jailbreak.augmentation import resolve
from redact.jailbreak.pool import all_techniques, registry, technique_pool


def _retire(monkeypatch, name: str = "to_rot13", version: str = "1.1"):
    """The technique ``name``, retired by spec version ``version``."""
    fn = registry(all_techniques())[name]
    monkeypatch.setattr(fn, "deprecated", version)
    return fn


def test_a_deprecated_technique_leaves_the_pool(monkeypatch):
    retired = _retire(monkeypatch)
    pool = technique_pool(resolve(None, "adversarial"))
    assert pool and retired not in pool


def test_allowing_its_family_does_not_keep_a_deprecated_technique(monkeypatch):
    retired = _retire(monkeypatch)
    pool = technique_pool(resolve(
        {"adversarial": {"families": {"allow": ["encode"]}, "translation": False}},
        "adversarial"))
    assert retired not in pool
    assert [fn for fn in pool if "encode" in fn.families]


def test_naming_a_deprecated_technique_in_techniques_allow_draws_it_with_a_warning(
        monkeypatch):
    retired = _retire(monkeypatch)
    with pytest.warns(DeprecationWarning,
                      match="'to_rot13', which spec version 1.1 deprecated"):
        pool = technique_pool(resolve(
            {"adversarial": {"techniques": {"allow": ["to_rot13"]},
                             "translation": False}}, "adversarial"))
    assert pool == [retired]
