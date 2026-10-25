"""The flip-token family: the rows of
``configs/jailbreak/obfuscation/flip_tokens.json`` and the technique each row
makes.

The packaged file ships with no row (the rows are to come from a
guard-in-the-loop search run on a pod, which the library does not hold yet),
so the family registers with no technique. The mechanism is tested, both
ways, with stand-in rows (no token
found against any guard) in a copy of the package, so that the spec's
generator, the module and the import checks of ``pool.py`` read the same file
as they do when ``redact.jailbreak`` is imported.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import redact.jailbreak.pool as JPool
import redact.jailbreak.technique as T
from redact import paths
from redact.jailbreak.augmentation import resolve
from redact.jailbreak.obfuscation import flip_tokens as F
from redact.jailbreak.pool import all_techniques, technique_pool
from redact.recipe import load_dataset_type

_SRC = Path(__file__).parents[3] / "src"
_FIXTURES = Path(__file__).parents[2] / "fixtures"
_ROWS = [
    {"slug": "alpha", "tokens": "zz alpha", "guard": "a stand-in guard",
     "flips": 7, "n_harmful": 10, "n_benign": 12, "benign_flips": 1,
     "notes": "not a real flip token", "since": "1.5"},
    {"slug": "beta_2", "tokens": "=qq", "guard": "another stand-in guard",
     "flips": 0, "n_harmful": 3, "n_benign": 0, "separator": "",
     "complexity": 2, "since": "1.5"},
]


def _write_rows(folder: Path, rows: list[dict]) -> None:
    file = folder / F.FLIP_TOKEN_FILE
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps({"rows": rows}), encoding="utf-8")


def _package_with(tmp_path: Path, rows: list[dict]) -> Path:
    """A copy of the package whose flip-token file holds ``rows``: its ``src``."""
    src = tmp_path / "src"
    shutil.copytree(_SRC / "redact", src / "redact",
                    ignore=shutil.ignore_patterns("__pycache__"))
    _write_rows(src / "redact" / "configs" / "jailbreak", rows)
    return src


def _copy_spec_and_data(folder: Path) -> None:
    """The packaged spec and every other data file its generators read, at
    their places, into ``folder``; the flip-token file is written by the test."""
    configs = paths.jailbreak_configs_dir()
    shutil.copy(configs / "combination_spec.json", folder)
    for generator in T.load_spec()["generators"].values():
        if isinstance(generator, dict) and generator["file"] != F.FLIP_TOKEN_FILE:
            (folder / generator["file"]).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(configs / generator["file"], folder / generator["file"])


_REPORT = """
import json
import redact.jailbreak
from redact.jailbreak.augmentation import resolve
from redact.jailbreak.chain import in_layer_order
from redact.jailbreak.pool import all_techniques, registry, technique_pool
from redact.jailbreak.rules import after_request_obfuscation, compatible
from redact.recipe import load_dataset_type
every = registry(all_techniques())
made = [fn for fn in all_techniques() if fn.__name__.startswith("to_flip_tokens_")]
art, b64, js = every["to_ascii_art"], every["to_base64"], every["to_json"]

def both(fn, other):
    return other in compatible([fn], [other]) and fn in compatible([other], [fn])

pools = {"adversarial": resolve(None, "adversarial"),
         "natural": resolve(None, "natural"),
         "eval": resolve(load_dataset_type("eval")["augmentations"], "adversarial")}
print(json.dumps({
    "techniques": {fn.__name__: {
        "families": fn.families, "kinds": fn.kinds, "complexity": fn.complexity,
        "since": fn.since, "requires_llm": fn.requires_llm, "layer": fn.layer,
        "order": fn.within_layer_order, "applied": fn("a prompt"),
        "with_ascii_art": both(fn, art), "with_base64": both(fn, b64),
        "with_json": both(fn, js),
        "after_translation": [fn] == after_request_obfuscation(
            every["to_french"], [fn]),
        "after_light_encoding": [fn] == after_request_obfuscation(
            every["to_rot13"], [fn]),
        "applies_in": [f.__name__ for f in in_layer_order([js, fn, b64])],
    } for fn in made},
    "pools": {name: sorted(fn.__name__ for fn in technique_pool(kind)
                           if fn.__name__.startswith("to_flip_tokens_"))
              for name, kind in pools.items()},
}))
"""


def _import(src: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(src)}
    return subprocess.run([sys.executable, "-c", _REPORT], capture_output=True,
                          text=True, env=env, timeout=180, check=False)


@pytest.fixture(scope="module")
def report(tmp_path_factory) -> dict:
    done = _import(_package_with(tmp_path_factory.mktemp("flip"), _ROWS))
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_each_row_registers_an_attack_only_technique_when_the_package_imports(report):
    assert list(report["techniques"]) == [
        "to_flip_tokens_alpha", "to_flip_tokens_beta_2"]
    alpha = report["techniques"]["to_flip_tokens_alpha"]
    assert {key: alpha[key] for key in ("families", "kinds", "complexity", "since",
                                        "requires_llm", "layer", "order")} == {
        "families": ["flip_tokens"], "kinds": ["adversarial"], "complexity": 1,
        "since": "1.5", "requires_llm": False, "layer": "obfuscation", "order": 75}
    # A row's complexity is read over the generator's.
    assert report["techniques"]["to_flip_tokens_beta_2"]["complexity"] == 2  # noqa: PLR2004


def test_a_row_s_technique_appends_its_tokens_after_its_separator(report):
    assert report["techniques"]["to_flip_tokens_alpha"]["applied"] == [
        "a prompt zz alpha", "slug=alpha"]
    # The separator the search appended with: here none.
    assert report["techniques"]["to_flip_tokens_beta_2"]["applied"] == [
        "a prompt=qq", "slug=beta_2"]


def test_the_rows_join_the_attacks_and_neither_the_natural_nor_the_eval_pool(report):
    """The proof that a row added later changes neither pinned pool: the
    natural kind draws only what the spec marks natural, and the eval type
    allows families by name, flip_tokens not among them."""
    assert report["pools"] == {
        "adversarial": ["to_flip_tokens_alpha", "to_flip_tokens_beta_2"],
        "natural": [], "eval": []}


def test_a_flip_token_refuses_ascii_art_both_ways_and_meets_the_rest(report):
    """It works only as the literal string the guard was searched with: drawn
    as art it is gone. An encoding applies before it and a wrapper after it,
    keeping it literal, so both are admitted."""
    for technique in report["techniques"].values():
        assert technique["with_ascii_art"] is False
        assert technique["with_base64"] is True
        assert technique["with_json"] is True


def test_a_translation_drawn_as_the_request_obfuscation_keeps_flip_tokens_out(report):
    """It would translate the tokens, as it would a published suffix; a light
    encoding drawn so keeps them, as it keeps a suffix."""
    for technique in report["techniques"].values():
        assert technique["after_translation"] is False
        assert technique["after_light_encoding"] is True


def test_flip_tokens_apply_after_the_encodings_and_before_the_wrappers(report):
    for name, technique in report["techniques"].items():
        assert technique["applies_in"] == ["to_base64", name, "to_json"]


def test_a_row_without_its_guard_fails_the_import_naming_the_row(tmp_path):
    rows = [_ROWS[0], {key: value for key, value in _ROWS[1].items() if key != "guard"}]
    done = _import(_package_with(tmp_path, rows))
    assert done.returncode != 0
    assert ("ValueError: obfuscation/flip_tokens.json: row 'beta_2' must give "
            "['guard']") in done.stderr


def test_the_packaged_file_ships_empty_and_registers_the_family_alone():
    """Until rows found against a guard on a pod are copied in."""
    doc = json.loads((paths.jailbreak_configs_dir() / F.FLIP_TOKEN_FILE)
                     .read_text(encoding="utf-8"))
    assert doc["rows"] == [] and F.flip_token_rows() == []
    spec = T.load_spec()
    assert "flip_tokens" in spec["families"]
    assert [name for name in spec["techniques"] if name.startswith(F.PREFIX)] == []
    assert F.get_flip_token_functions() == []
    assert [fn for fn in all_techniques() if "flip_tokens" in fn.families] == []


def test_an_empty_file_registers_the_family_with_no_technique(tmp_path, monkeypatch):
    _copy_spec_and_data(tmp_path)
    _write_rows(tmp_path, [])
    spec = T.read_spec(tmp_path / "combination_spec.json")
    assert [name for name in spec["techniques"] if name.startswith(F.PREFIX)] == []
    monkeypatch.setattr(JPool, "load_spec", lambda: spec)
    JPool._check_against_spec(all_techniques())
    JPool._check_dimensions()


def test_the_packaged_pools_hold_no_flip_token_and_the_pinned_pools_are_unchanged():
    eval_kind = resolve(load_dataset_type("eval")["augmentations"], "adversarial")
    natural = technique_pool(resolve(None, "natural"))
    pinned_eval = json.loads((_FIXTURES / "eval_pool.json").read_text(encoding="utf-8"))
    pinned_natural = json.loads((_FIXTURES / "natural_pool.json")
                                .read_text(encoding="utf-8"))
    assert "flip_tokens" not in json.dumps(eval_kind)
    assert not any(fn.__name__.startswith(F.PREFIX)
                   for fn in technique_pool(eval_kind) + natural)
    assert "flip_tokens" not in json.dumps(pinned_eval)
    assert "flip_tokens" not in json.dumps(pinned_natural)


def test_the_spec_gives_the_family_its_place_and_its_refusals():
    spec = T.load_spec()
    family = spec["families"]["flip_tokens"]
    orders = {name: entry["within_layer_order"]
              for name, entry in spec["families"].items()
              if entry["layer"] == "obfuscation"}
    assert family["kinds"] == ["adversarial"]
    assert family["cross_incompatible_with"] == ["ascii_art"]
    assert family["dimension"] == "surface_noise"
    # After every obfuscation that rewrites characters, before the wrappers.
    assert max(orders["payload_split"], orders["encode"], orders["suffixes"]) \
        < orders["flip_tokens"] < min(orders["structural"], orders["neutral_container"])
    assert "flip_tokens" in spec["request_obfuscation_refuses"]["translation"]
    generator = spec["generators"]["flip_tokens"]
    assert (generator["file"], generator["name"], generator["kinds"]) == (
        F.FLIP_TOKEN_FILE, "to_flip_tokens_{slug}", ["adversarial"])
    assert "kinds" not in T._ROW_KEYS


def _without(field: str) -> dict:
    return {key: value for key, value in _ROWS[0].items() if key != field}


@pytest.mark.parametrize(("row", "message"), [
    *[(_without(field), f"row 'alpha' must give ['{field}']")
      for field in F.REQUIRED_FIELDS],
    ({**_ROWS[0], "tokens": "  "}, "row 'alpha' must give 'tokens' as a non-blank"),
    ({**_ROWS[0], "guard": 3}, "row 'alpha' must give 'guard' as a non-blank"),
    ({**_ROWS[0], "kinds": ["natural"]},
     "row 'alpha' gives unknown field(s) ['kinds']"),
    ({**_ROWS[0], "flips": 11}, "the flips of row 'alpha' must be a whole number of "
                                "at least 0 and at most 10"),
    ({**_ROWS[0], "flips": -1}, "the flips of row 'alpha' must be a whole number"),
    ({**_ROWS[0], "flips": 2.0}, "the flips of row 'alpha' must be a whole number"),
    ({**_ROWS[0], "flips": True}, "the flips of row 'alpha' must be a whole number"),
    ({**_ROWS[0], "n_harmful": 0, "flips": 0},
     "the n_harmful of row 'alpha' must be a whole number of at least 1"),
    ({**_ROWS[0], "n_benign": -2}, "the n_benign of row 'alpha' must be a whole"),
    ({**_ROWS[0], "benign_flips": 13}, "the benign_flips of row 'alpha' must be a "
                                       "whole number of at least 0 and at most 12"),
    ({**_ROWS[0], "separator": None}, "the separator of row 'alpha' must be a string"),
    ({**_ROWS[0], "notes": ["a"]}, "the notes of row 'alpha' must be a string"),
    ({**_ROWS[0], "slug": "Alpha-1"}, "row 'Alpha-1' must have a slug of lower-case"),
    (_without("slug"), "row 0 has no slug"),
], ids=[*F.REQUIRED_FIELDS, "blank tokens", "guard not text", "kinds", "flips over",
        "flips negative", "flips float", "flips bool", "no harmful", "benign negative",
        "benign flips over", "separator", "notes", "slug case", "no slug"])
def test_a_row_that_breaks_a_rule_raises_naming_it(tmp_path, row, message):
    _write_rows(tmp_path, [row])
    with pytest.raises(ValueError) as raised:
        F.flip_token_rows(tmp_path)
    assert str(raised.value).startswith(f"{F.FLIP_TOKEN_FILE}: {message}")


def test_the_stand_in_rows_pass_the_module_s_check(tmp_path):
    _write_rows(tmp_path, _ROWS)
    assert [row["slug"] for row in F.flip_token_rows(tmp_path)] == ["alpha", "beta_2"]


def test_a_row_named_twice_is_one_the_spec_has_already(tmp_path):
    _copy_spec_and_data(tmp_path)
    _write_rows(tmp_path, [_ROWS[0], _ROWS[0]])
    with pytest.raises(ValueError, match="'to_flip_tokens_alpha', which the spec has "
                       "already"):
        T.read_spec(tmp_path / "combination_spec.json")
