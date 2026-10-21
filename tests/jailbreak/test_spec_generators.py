"""The spec's generators: families whose techniques come one per row of a
data file, expanded into technique entries by ``load_spec``."""

import json
import shutil

import pytest

import redact.jailbreak.pool as JPool
import redact.jailbreak.technique as T
from redact import paths
from redact.jailbreak.pool import all_techniques

_PACKAGED = paths.jailbreak_configs_dir()


def test_the_language_generators_make_the_entries_the_spec_wrote_out():
    techniques = T.load_spec()["techniques"]
    assert techniques["to_swahili"] == {
        "families": ["translation"], "requires_llm": True, "complexity": 2}
    assert techniques["to_ask_answer_in_french"] == {
        "families": ["answer_language"], "requires_llm": False, "complexity": 0}
    # Maori has an answer-language technique and no translation.
    assert "to_ask_answer_in_maori" in techniques and "to_maori" not in techniques


def _spec(tmp_path, rows: list[dict], **generator) -> dict:
    """The packaged spec, copied into ``tmp_path`` with a stand-in family,
    ``stand_in``, whose techniques come from ``hacking/stand_ins.json``
    holding ``rows``, and read back."""
    shutil.copy(_PACKAGED / "languages.json", tmp_path / "languages.json")
    spec = json.loads((_PACKAGED / "combination_spec.json").read_text(encoding="utf-8"))
    spec["families"]["stand_in"] = {
        "layer": "hacking", "dimension": "role_framing", "within_layer_order": 0,
        "cross_incompatible_with": []}
    spec["generators"]["stand_in"] = {
        "file": "hacking/stand_ins.json", "name": "to_{slug}_stand_in",
        "complexity": 2, "requires_llm": False, "kinds": ["adversarial"], **generator}
    (tmp_path / "hacking").mkdir()
    (tmp_path / "hacking" / "stand_ins.json").write_text(
        json.dumps({"rows": rows}), encoding="utf-8")
    path = tmp_path / "combination_spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return T.read_spec(path)


def _functions(rows: list[dict]) -> list:
    """One technique per row, as the family's module would build them."""
    fns = []
    for row in rows:
        def fn(prompt: str, *, rng=None) -> tuple[str, str]:
            return prompt, ""
        fn.__name__ = fn.__qualname__ = f"to_{row['slug']}_stand_in"
        fns.append(fn)
    return fns


def _register(monkeypatch, spec: dict, fns: list) -> None:
    """Tag ``fns`` from ``spec`` and run the import checks of ``pool.py`` over
    the registry with them."""
    monkeypatch.setattr(T, "load_spec", lambda: spec)
    monkeypatch.setattr(JPool, "load_spec", lambda: spec)
    T.tag(fns)
    JPool._check_against_spec(all_techniques() + fns)
    JPool._check_dimensions()


def test_a_generated_family_whose_file_has_no_rows_registers_with_no_technique(
        tmp_path, monkeypatch):
    spec = _spec(tmp_path, [])
    assert "stand_in" in spec["families"]
    assert [name for name in spec["techniques"]
            if name.endswith("_stand_in")] == []
    _register(monkeypatch, spec, [])


def test_each_row_registers_one_technique_with_what_the_generator_gives(
        tmp_path, monkeypatch):
    rows = [{"slug": "first"}, {"slug": "second"}]
    spec = _spec(tmp_path, rows)
    fns = _functions(rows)
    _register(monkeypatch, spec, fns)
    assert [fn.__name__ for fn in fns] == [
        "to_first_stand_in", "to_second_stand_in"]
    for fn in fns:
        assert fn.families == ["stand_in"]
        assert fn.layer == "hacking"
        assert fn.complexity == 2      # noqa: PLR2004
        assert fn.kinds == ["adversarial"]
        assert (fn.since, fn.deprecated) == ("1.0", None)


def test_a_row_s_function_missing_from_the_registry_fails_the_import_check(
        tmp_path, monkeypatch):
    rows = [{"slug": "first"}, {"slug": "second"}]
    spec = _spec(tmp_path, rows)
    with pytest.raises(ValueError, match="in the spec but not registered: "
                       "to_second_stand_in"):
        _register(monkeypatch, spec, _functions(rows[:1]))


def test_a_row_s_complexity_since_and_deprecated_win_over_the_generator_s(tmp_path):
    spec = _spec(tmp_path, [{"slug": "first", "complexity": 1, "since": "1.2",
                             "deprecated": "1.3"}])
    assert spec["techniques"]["to_first_stand_in"] == {
        "families": ["stand_in"], "complexity": 1, "requires_llm": False,
        "kinds": ["adversarial"], "since": "1.2", "deprecated": "1.3"}


def test_where_keeps_only_the_rows_whose_column_is_true(tmp_path):
    spec = _spec(tmp_path, [{"slug": "first", "drawn": True},
                            {"slug": "second", "drawn": False}], where="drawn")
    names = [name for name in spec["techniques"] if name.endswith("_stand_in")]
    assert names == ["to_first_stand_in"]


def test_a_generated_name_the_spec_has_already_raises(tmp_path):
    with pytest.raises(ValueError, match="'to_base64', which the spec has already"):
        _spec(tmp_path, [{"slug": "base64"}], name="to_{slug}")
