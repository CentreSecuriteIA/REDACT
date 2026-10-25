"""The published adversarial suffixes: rows of
``configs/jailbreak/obfuscation/suffix_strings.json``, one technique each.

The rows here are stand-ins (no published string): the mechanism is what is
tested, both ways, in a copy of the package so that the spec's generator, the
module and the import checks of ``pool.py`` read the same file as they do
when ``redact.jailbreak`` is imported.
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
from redact.jailbreak.obfuscation import suffixes as S
from redact.jailbreak.pool import all_techniques, technique_pool

_SRC = Path(__file__).parents[3] / "src"
_ROWS = [
    {"slug": "alpha", "suffix": "zz alpha omega", "source": "Stand-in 2026, arXiv:0000.00000",
     "optimized_against": "a stand-in model", "notes": "not a real suffix", "since": "1.4"},
    {"slug": "beta_2", "suffix": "qq beta", "source": "Stand-in repository",
     "optimized_against": "two stand-in models", "complexity": 2, "since": "1.4",
     "kinds": ["adversarial", "natural"]},
]


def _write_rows(folder: Path, rows: list[dict]) -> None:
    file = folder / S.SUFFIX_FILE
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps({"rows": rows}), encoding="utf-8")


def _package_with(tmp_path: Path, rows: list[dict]) -> Path:
    """A copy of the package whose suffix file holds ``rows``: its ``src``."""
    src = tmp_path / "src"
    shutil.copytree(_SRC / "redact", src / "redact",
                    ignore=shutil.ignore_patterns("__pycache__"))
    _write_rows(src / "redact" / "configs" / "jailbreak", rows)
    return src


_REPORT = """
import json
import redact.jailbreak
from redact.jailbreak.augmentation import resolve
from redact.jailbreak.pool import all_techniques, technique_pool
from redact.recipe import load_dataset_type
made = [fn for fn in all_techniques() if fn.__name__.startswith("to_adversarial_suffix_")]
pools = {"adversarial": resolve(None, "adversarial"), "natural": resolve(None, "natural"),
         "eval": resolve(load_dataset_type("eval")["augmentations"], "adversarial")}
print(json.dumps({
    "techniques": {fn.__name__: {"families": fn.families, "kinds": fn.kinds,
                                 "complexity": fn.complexity, "since": fn.since,
                                 "requires_llm": fn.requires_llm,
                                 "applied": fn("a prompt")} for fn in made},
    "pools": {name: sorted(fn.__name__ for fn in technique_pool(kind)
                           if fn.__name__.startswith("to_adversarial_suffix_"))
              for name, kind in pools.items()},
}))
"""


def _import(src: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(src)}
    return subprocess.run([sys.executable, "-c", _REPORT], capture_output=True,
                          text=True, env=env, timeout=120, check=False)


def test_each_row_registers_an_attack_only_technique_when_the_package_imports(tmp_path):
    done = _import(_package_with(tmp_path, _ROWS))
    assert done.returncode == 0, done.stderr
    report = json.loads(done.stdout)
    noise = ["to_adversarial_suffix_punctuation", "to_adversarial_suffix_fragments",
             "to_adversarial_suffix_unicode", "to_adversarial_suffix_emoji"]
    # In the family, after its four noise suffixes, in the file's order.
    assert list(report["techniques"]) == [*noise, "to_adversarial_suffix_alpha",
                                          "to_adversarial_suffix_beta_2"]
    alpha = report["techniques"]["to_adversarial_suffix_alpha"]
    assert alpha == {"families": ["suffixes"], "kinds": ["adversarial"], "complexity": 1,
                     "since": "1.4", "requires_llm": False,
                     "applied": ["a prompt zz alpha omega",
                                 "slug=alpha;source=Stand-in 2026, arXiv:0000.00000"]}
    beta = report["techniques"]["to_adversarial_suffix_beta_2"]
    # A row's complexity is read; its kinds are not: no row is natural.
    assert (beta["complexity"], beta["kinds"]) == (2, ["adversarial"])
    made = ["to_adversarial_suffix_alpha", "to_adversarial_suffix_beta_2"]
    assert set(made) <= set(report["pools"]["adversarial"])
    # The natural kind keeps its two noise suffixes; the eval type draws no suffix.
    assert report["pools"]["natural"] == ["to_adversarial_suffix_emoji",
                                          "to_adversarial_suffix_punctuation"]
    assert report["pools"]["eval"] == []


def test_a_row_without_its_source_fails_the_import_naming_the_row(tmp_path):
    rows = [_ROWS[0], {key: value for key, value in _ROWS[1].items() if key != "source"}]
    done = _import(_package_with(tmp_path, rows))
    assert done.returncode != 0
    assert ("ValueError: obfuscation/suffix_strings.json: row 'beta_2' must give "
            "['source']") in done.stderr


def test_the_packaged_file_s_rows_are_attack_only_suffix_techniques():
    """Whatever rows the packaged file holds (none until the strings land)."""
    rows = S.suffix_rows()
    made = [fn for fn in all_techniques()
            if fn.__name__ in {f"to_adversarial_suffix_{row['slug']}" for row in rows}]
    assert len(made) == len(rows)
    for fn, row in zip(made, rows, strict=True):
        assert (fn.families, fn.kinds, fn.since) == (["suffixes"], ["adversarial"],
                                                     row["since"])
        assert fn("p") == (f"p {row['suffix']}",
                           f"slug={row['slug']};source={row['source']}")
    assert S.get_suffix_functions() == [*S.NOISE_SUFFIXES, *made]


def test_an_empty_file_registers_the_family_with_its_noise_suffixes_alone(
        tmp_path, monkeypatch):
    shutil.copy(paths.jailbreak_configs_dir() / "combination_spec.json", tmp_path)
    shutil.copy(paths.jailbreak_configs_dir() / "languages.json", tmp_path)
    _write_rows(tmp_path, [])
    spec = T.read_spec(tmp_path / "combination_spec.json")
    assert S.suffix_rows(tmp_path) == []
    assert [name for name in spec["techniques"]
            if name.startswith("to_adversarial_suffix_")] == [
        fn.__name__ for fn in S.NOISE_SUFFIXES]
    monkeypatch.setattr(JPool, "load_spec", lambda: spec)
    JPool._check_against_spec(all_techniques())


def test_a_row_s_technique_appends_its_suffix_after_a_space():
    fn = S.make_suffix_fn(_ROWS[0])
    assert fn.__name__ == "to_adversarial_suffix_alpha"
    assert fn("a prompt") == ("a prompt zz alpha omega",
                              "slug=alpha;source=Stand-in 2026, arXiv:0000.00000")


def _without(field: str) -> dict:
    return {key: value for key, value in _ROWS[0].items() if key != field}


@pytest.mark.parametrize(("row", "message"), [
    *[(_without(field), f"row 'alpha' must give ['{field}']")
      for field in S.REQUIRED_FIELDS],
    ({**_ROWS[0], "suffix": "  "}, "row 'alpha' must give ['suffix']"),
    ({**_ROWS[0], "optimized_against": 7}, "row 'alpha' must give ['optimized_against']"),
    ({**_ROWS[0], "source": "a; b"}, "the source of row 'alpha' may not hold a ';'"),
    ({**_ROWS[0], "notes": ["a"]}, "the notes of row 'alpha' must be a string"),
    ({**_ROWS[0], "slug": "Alpha-1"}, "row 'Alpha-1' must have a slug of lower-case"),
    ({**_ROWS[0], "slug": "emoji"}, "row 'emoji' must have a slug of lower-case"),
    (_without("slug"), "row 0 has no slug"),
], ids=[*S.REQUIRED_FIELDS, "blank suffix", "not a string", "semicolon", "notes",
        "slug case", "a noise slug", "no slug"])
def test_a_row_that_breaks_a_rule_raises_naming_it(tmp_path, row, message):
    _write_rows(tmp_path, [row])
    with pytest.raises(ValueError) as raised:
        S.suffix_rows(tmp_path)
    assert str(raised.value).startswith(f"{S.SUFFIX_FILE}: {message}")


def test_a_row_named_as_a_noise_suffix_is_one_the_spec_has_already(tmp_path):
    """The spec's generator refuses it too, as for any name it has."""
    shutil.copy(paths.jailbreak_configs_dir() / "combination_spec.json", tmp_path)
    shutil.copy(paths.jailbreak_configs_dir() / "languages.json", tmp_path)
    _write_rows(tmp_path, [{**_ROWS[0], "slug": "emoji"}])
    with pytest.raises(ValueError, match="'to_adversarial_suffix_emoji', which the "
                       "spec has already"):
        T.read_spec(tmp_path / "combination_spec.json")


def test_the_kinds_of_a_row_s_technique_are_the_generator_s():
    """The generator says adversarial, whatever the family's kinds become:
    the natural kind's allow list naming one raises."""
    generator = T.load_spec()["generators"]["suffixes"]
    assert generator["kinds"] == ["adversarial"]
    assert "kinds" not in T._ROW_KEYS
    assert technique_pool(resolve(None, "natural"))
