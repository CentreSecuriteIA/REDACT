"""A library that adds and retires techniques: what a kind's pool draws, and
what a kept plan does on resume.

The spec dates a technique with ``since``, the version that added it, and
``deprecated``, the one that retired it; the plan file records the spec
version its pool was drawn at. A kept plan whose pool differs only by what
the library added or retired after that version resumes with a WARNING;
any other difference raises. The library has added and retired nothing yet,
so these tests add a dummy technique, or retire a real one, by hand.
"""

import pytest

import redact.jailbreak.pipeline as JPipe
import redact.jailbreak.pool as JPool
import redact.jailbreak.technique as T
from redact.dataset import Manifest
from redact.dataset.planfile import read_plan, write_plan
from redact.jailbreak.augmentation import resolve
from redact.jailbreak.pool import all_techniques, registry, technique_pool
from redact.stages import generate_jailbreaks
from tests.jailbreak.test_pipeline_jailbreaks import _inputs, _pure


def _retire(monkeypatch, name: str = "to_rot13", version: str = "1.1"):
    """The technique ``name``, retired by spec version ``version``."""
    fn = registry(all_techniques())[name]
    monkeypatch.setattr(fn, "deprecated", version)
    return fn


def _add(monkeypatch, since: str):
    """``to_reversed``, a pure encoding whose spec entry says the library
    added it in version ``since``, tagged and registered after every other."""
    def to_reversed(prompt):
        return prompt[::-1], ""

    spec = T.load_spec()
    entry = {"families": ["encode"], "complexity": 1, "encode_weight": "light",
             "since": since}
    with monkeypatch.context() as patch:
        patch.setattr(T, "load_spec", lambda: {
            **spec, "techniques": {**spec["techniques"], "to_reversed": entry}})
        T.tag([to_reversed])
    registered = all_techniques()
    monkeypatch.setattr(JPool, "all_techniques", lambda: [*registered, to_reversed])
    return to_reversed


def _planned(out, monkeypatch, **kind) -> list[dict]:
    """The plan of a run of the pure obfuscations into ``out``, stopped before
    its first unit, as its manifest holds it."""
    with monkeypatch.context() as patch:
        patch.setattr(JPipe, "drive_units", lambda units, **kwargs: None)
        generate_jailbreaks(inputs=_inputs(), output_path=out, seed=1, **_pure(**kind))
    return Manifest.sidecar(out).load()


def _resume(out, seed=1, **kind):
    return generate_jailbreaks(inputs=_inputs(), output_path=out, seed=seed,
                               **_pure(**kind))


def _warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records]


# --- the pool ----------------------------------------------------------------


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


# --- a kept plan on resume -----------------------------------------------------


def test_a_technique_added_after_the_plan_warns_once_and_keeps_the_plan(
        tmp_path, monkeypatch, caplog):
    out = tmp_path / "jb.csv"
    plan = _planned(out, monkeypatch)
    _add(monkeypatch, since="1.1")
    with caplog.at_level("WARNING"):
        rows = _resume(out)
    (warning,) = _warnings(caplog)
    assert "added ['to_reversed'] and retired []" in warning
    assert "does not draw the added techniques" in warning
    assert "resume=False" in warning
    # Every planned unit ran as planned; none drew the new technique.
    assert Manifest.sidecar(out).load() == plan
    assert len(rows) == len(plan)
    assert not rows["technique"].str.contains("to_reversed").any()


def test_a_technique_added_without_a_later_since_raises(tmp_path, monkeypatch):
    """A name the plan's version already had, as one a section newly allows."""
    out = tmp_path / "jb.csv"
    _planned(out, monkeypatch)
    _add(monkeypatch, since="1.0")
    with pytest.raises(ValueError, match=r"\['pool'\].*resume=False"):
        _resume(out)


def test_a_technique_no_longer_registered_raises(tmp_path, monkeypatch):
    out = tmp_path / "jb.csv"
    _planned(out, monkeypatch)
    registered = [fn for fn in all_techniques() if fn.__name__ != "to_rot13"]
    monkeypatch.setattr(JPool, "all_techniques", lambda: registered)
    with pytest.raises(ValueError, match=r"\['pool'\].*resume=False"):
        _resume(out)


def test_a_technique_retired_after_the_plan_warns_and_still_replays(
        tmp_path, monkeypatch, caplog):
    out = tmp_path / "jb.csv"
    plan = _planned(out, monkeypatch)
    unit = next(row for row in plan if row["combination"])
    retired = _retire(monkeypatch, unit["combination"][0])
    with caplog.at_level("WARNING"):
        rows = _resume(out)
    (warning,) = _warnings(caplog)
    assert f"added [] and retired ['{retired.__name__}']" in warning
    assert "resume=False" in warning
    replayed = rows[(rows["input_id"] == unit["sample_id"])
                    & (rows["iteration"] == unit["iteration"])]
    assert list(replayed["technique"]) == ["+".join(unit["combination"])]


def test_a_technique_retired_by_the_plan_s_own_version_raises(tmp_path, monkeypatch):
    """Retired no later than the plan was made, it left the pool by the
    section's doing, not the library's."""
    out = tmp_path / "jb.csv"
    _planned(out, monkeypatch)
    _retire(monkeypatch, "to_rot13", version="1.0")
    with pytest.raises(ValueError, match=r"\['pool'\].*resume=False"):
        _resume(out)


def _without_spec_version(out) -> None:
    """Make the plan file beside ``out`` one written before it recorded
    ``spec_version``."""
    manifest = Manifest.sidecar(out).path
    made_with = read_plan(manifest)
    del made_with["spec_version"]
    write_plan(manifest, made_with)


def test_a_plan_file_without_a_spec_version_resumes_quietly(
        tmp_path, monkeypatch, caplog):
    out = tmp_path / "jb.csv"
    plan = _planned(out, monkeypatch)
    _without_spec_version(out)
    with caplog.at_level("WARNING"):
        rows = _resume(out)
    assert _warnings(caplog) == [] and len(rows) == len(plan)


def test_a_plan_file_without_a_spec_version_was_made_at_the_first_version(
        tmp_path, monkeypatch, caplog):
    out = tmp_path / "jb.csv"
    _planned(out, monkeypatch)
    _without_spec_version(out)
    _add(monkeypatch, since="1.1")
    with caplog.at_level("WARNING"):
        _resume(out)
    (warning,) = _warnings(caplog)
    assert "spec version 1.0" in warning and "added ['to_reversed']" in warning


@pytest.mark.parametrize(("seed", "kind", "named"), [
    (2, {}, "seed"),
    (1, {"rounds": [{"max_complexity": 3}]}, "rounds"),
    (1, {"family_weights": {"encode": 2}}, "family_weights"),
    (1, {"layer_rates": {"requests": 0.1}}, "layer_rates"),
], ids=["seed", "rounds", "family_weights", "layer_rates"])
def test_another_seed_or_section_raises_though_the_library_grew(
        tmp_path, monkeypatch, seed, kind, named):
    out = tmp_path / "jb.csv"
    _planned(out, monkeypatch)
    _add(monkeypatch, since="1.1")
    with pytest.raises(ValueError, match=rf"this run's \[{named!r}\].*resume=False"):
        _resume(out, seed=seed, **kind)
