"""The spec's generators: families whose techniques come one per row of a
data file, expanded into technique entries by ``load_spec``."""

import json
import shutil
from collections.abc import Callable

import pytest

import redact.jailbreak.pool as JPool
import redact.jailbreak.technique as T
from redact import paths
from redact.jailbreak.pool import all_techniques

_PACKAGED = paths.jailbreak_configs_dir()


def _copy_data_files(tmp_path) -> None:
    """Copy every top-level data file the packaged spec's generators name into
    ``tmp_path`` (languages.json and the language-mix files), so a spec read
    from there finds them. A generator whose file is under a subfolder (the
    test's own stand-in) is written by the test itself."""
    spec = json.loads((_PACKAGED / "combination_spec.json").read_text(encoding="utf-8"))
    for generator in spec.get("generators", {}).values():
        if not isinstance(generator, dict):
            continue
        name = generator.get("file", "")
        if name and "/" not in name and (_PACKAGED / name).exists():
            shutil.copy(_PACKAGED / name, tmp_path / name)


def test_the_language_generators_make_the_entries_the_spec_wrote_out():
    techniques = T.load_spec()["techniques"]
    assert techniques["to_swahili"] == {
        "families": ["translation"], "requires_llm": True, "complexity": 2}
    assert techniques["to_ask_answer_in_french"] == {
        "families": ["answer_language"], "requires_llm": False, "complexity": 0}
    # Maori has an answer-language technique and no translation.
    assert "to_ask_answer_in_maori" in techniques and "to_maori" not in techniques


def _copy_data_files(folder) -> None:
    """Copy the data files the packaged spec's generators read into ``folder``,
    at their places beside the spec."""
    for generator in T.load_spec()["generators"].values():
        if isinstance(generator, dict):
            (folder / generator["file"]).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(_PACKAGED / generator["file"], folder / generator["file"])


def _spec(tmp_path, rows: list[dict], **generator) -> dict:
    """The packaged spec, copied into ``tmp_path`` with a stand-in family,
    ``stand_in``, whose techniques come from ``hacking/stand_ins.json``
    holding ``rows``, and read back."""
    _copy_data_files(tmp_path)
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


def test_a_row_s_family_since_dates_that_family_s_technique_alone(tmp_path):
    """A language that joined a second family later: the row's ``since`` dates
    the first, its ``<family>_since`` the second."""
    spec = _spec(tmp_path, [{"slug": "first", "since": "1.2", "stand_in_since": "1.3"}])
    assert spec["techniques"]["to_first_stand_in"]["since"] == "1.3"
    packaged = T.load_spec()["techniques"]
    assert packaged["to_chinese"]["since"] == "1.3"
    assert packaged["to_ask_answer_in_chinese"]["since"] == "1.2"


def test_where_keeps_only_the_rows_whose_column_is_true(tmp_path):
    spec = _spec(tmp_path, [{"slug": "first", "drawn": True},
                            {"slug": "second", "drawn": False}], where="drawn")
    names = [name for name in spec["techniques"] if name.endswith("_stand_in")]
    assert names == ["to_first_stand_in"]


def test_a_generated_name_the_spec_has_already_raises(tmp_path):
    with pytest.raises(ValueError, match="'to_base64', which the spec has already"):
        _spec(tmp_path, [{"slug": "base64"}], name="to_{slug}")


@pytest.mark.parametrize(("rows", "generator", "message"), [
    ([{"slug": "first", "drawn": "yes"}], {"where": "drawn"},
     "hacking/stand_ins.json: row 'first' must give 'drawn' true or false"),
    ([{"slug": "first"}], {"where": "drawn"},
     "hacking/stand_ins.json: row 'first' must give 'drawn' true or false"),
    ([{"name": "first"}], {}, "hacking/stand_ins.json: row 0 has no slug"),
    ([{"slug": "first", "complexity": "low"}], {},
     "hacking/stand_ins.json: the complexity of row 'first' must be a whole number "
     "from 0 to 3, got 'low'"),
    ([{"slug": "first", "stand_ins_since": "1.3"}], {},
     "hacking/stand_ins.json: row 'first' gives 'stand_ins_since', which names no "
     "family made from this file"),
    ([{"slug": "first", "stand_in_since": "1.x"}], {},
     "hacking/stand_ins.json: the stand_in_since of row 'first' must be a version"),
], ids=["where text", "where missing", "no slug", "complexity text",
        "misspelled family since", "bad family since"])
def test_a_row_that_breaks_a_rule_raises_naming_its_file(tmp_path, rows, generator,
                                                        message):
    with pytest.raises(ValueError) as raised:
        _spec(tmp_path, rows, **generator)
    assert str(raised.value).startswith(message)


def _read_changed(tmp_path, change) -> dict:
    """The packaged spec with ``change`` applied, read from ``tmp_path``."""
    _copy_data_files(tmp_path)
    spec = json.loads((_PACKAGED / "combination_spec.json").read_text(encoding="utf-8"))
    change(spec)
    path = tmp_path / "combination_spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return T.read_spec(path)


@pytest.mark.parametrize(("change", "message"), [
    (lambda s: s["families"]["encode"].__setitem__("within_layer_ordr", 3),
     "combination_spec.json: family 'encode' has unknown key(s) ['within_layer_ordr']"),
    (lambda s: s["families"]["encode"].__setitem__("layer", "bogus"),
     "combination_spec.json: the layer of family 'encode' must be one of layer_order"),
    (lambda s: s["layer_rates"].__setitem__("hacking", 2),
     "combination_spec.json: layer_rates['hacking'] must be a number from 0 to 1"),
    (lambda s: s["layer_rates"].__setitem__("hackin", 0.5),
     "combination_spec.json: layer_rates['hackin'] must be a number from 0 to 1"),
    (lambda s: s["generators"].__setitem__(
        "translatoin", s["generators"].pop("translation")),
     "combination_spec.json: generators has 'translatoin', which is not a family"),
    (lambda s: s["request_obfuscation_refuses"].__setitem__("translatoin", ["a"]),
     "combination_spec.json: request_obfuscation_refuses['translatoin'] must"),
    (lambda s: s["request_obfuscation_refuses"].__setitem__("translation", ["suffix"]),
     "combination_spec.json: request_obfuscation_refuses['translation'] must"),
    (lambda s: s["request_obfuscation_refuses"].__setitem__("translation", "suffixes"),
     "combination_spec.json: request_obfuscation_refuses['translation'] must"),
], ids=["family key", "family layer", "rate 2", "rate of no layer",
        "generator of no family", "refuses of no family", "refuses a non-family",
        "refuses not a list"])
def test_a_spec_that_breaks_a_rule_raises_when_read(tmp_path, change, message):
    with pytest.raises(ValueError) as raised:
        _read_changed(tmp_path, change)
    assert str(raised.value).startswith(message)


def _refuses(table) -> Callable[[dict], None]:
    return lambda s: s.__setitem__("request_obfuscation_refuses", table)


@pytest.mark.parametrize(("change", "message"), [
    (_refuses(["suffixes"]),
     "combination_spec.json: request_obfuscation_refuses must be an object"),
    (_refuses({"encode_light": ["suffixes"]}),
     "combination_spec.json: request_obfuscation_refuses['encode_light'] must be "
     "keyed by a family a request obfuscation is drawn from, one of ['encode', "
     "'translation']"),
    (_refuses({"framing": ["suffixes"]}),
     "combination_spec.json: request_obfuscation_refuses['framing'] must be keyed"),
    (lambda s: s.__setitem__("request_obfuscation_allowed_families", ["translation"])
     or s["request_obfuscation_refuses"].__setitem__("encode", ["structural"]),
     "combination_spec.json: request_obfuscation_refuses['encode'] must be keyed "
     "by a family a request obfuscation is drawn from, one of ['translation']"),
    (_refuses({"translation": ["suffixes", "framing"]}),
     "combination_spec.json: request_obfuscation_refuses['translation'] must be a "
     "list of obfuscation-layer families"),
    (_refuses({"translation": ["fsh"]}),
     "combination_spec.json: request_obfuscation_refuses['translation'] must be a "
     "list of obfuscation-layer families"),
    (_refuses({"translation": ["answer_format"]}),
     "combination_spec.json: request_obfuscation_refuses['translation'] must be a "
     "list of obfuscation-layer families"),
], ids=["not an object", "encode_light", "a family never drawn", "encode not allowed",
        "a hacking family", "a manipulation family", "a request family"])
def test_a_refuses_table_that_could_not_act_raises_when_read(tmp_path, change, message):
    """Phase 5 draws the request obfuscation from a translation or a light
    encoding (``encode``, the light sensitive-word encodings among its
    members), after every other layer's pick but the obfuscations': another
    key is never read, and a family of another layer is drawn already."""
    with pytest.raises(ValueError) as raised:
        _read_changed(tmp_path, change)
    assert str(raised.value).startswith(message)


def test_a_light_encoding_drawn_as_the_request_obfuscation_may_refuse_too(tmp_path):
    table = {"translation": ["suffixes"], "encode": ["structural", "neutral_container"]}
    spec = _read_changed(tmp_path, _refuses(table))
    assert spec["request_obfuscation_refuses"] == table


def test_the_packaged_spec_reads_as_it_is(tmp_path):
    assert _read_changed(tmp_path, lambda spec: None) == T.load_spec()
