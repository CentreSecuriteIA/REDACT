"""The published adversarial suffixes: the rows of
``configs/jailbreak/obfuscation/suffix_strings.json`` and the technique each
row makes.

The file's own rows are checked for their shape, and each published string
is pinned by digest (no suffix is quoted here: the strings live in the data
file). The mechanism is tested, both ways, with stand-in rows (no published
string) in a copy of the package, so that the spec's generator, the module
and the import checks of ``pool.py`` read the same file as they do when
``redact.jailbreak`` is imported.
"""

import copy
import hashlib
import json
import os
import re
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
    {"slug": "alpha", "suffix": "zz alpha omega",
     "source": "Stand-in 2026, arXiv:0000.00000",
     "optimized_against": "a stand-in model", "notes": "not a real suffix",
     "since": "1.4"},
    {"slug": "beta_2", "suffix": "qq beta", "source": "Stand-in repository",
     "optimized_against": "two stand-in models", "complexity": 2, "since": "1.4",
     "kinds": ["adversarial", "natural"]},
]


def _write_rows(folder: Path, rows: list[dict]) -> None:
    file = folder / S.SUFFIX_FILE
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps({"rows": rows}), encoding="utf-8")


def _copy_spec_and_data(folder: Path) -> None:
    """The packaged spec and every top-level data file its generators read
    (languages.json and the spec 1.4 language-mix files) into ``folder``, so
    ``read_spec`` there finds them; the suffix file is written by the test."""
    configs = paths.jailbreak_configs_dir()
    shutil.copy(configs / "combination_spec.json", folder)
    spec = json.loads((configs / "combination_spec.json").read_text(encoding="utf-8"))
    for generator in spec.get("generators", {}).values():
        name = generator.get("file", "") if isinstance(generator, dict) else ""
        if name and "/" not in name and (configs / name).exists():
            shutil.copy(configs / name, folder)


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
made = [fn for fn in all_techniques()
        if fn.__name__.startswith("to_adversarial_suffix_")]
pools = {"adversarial": resolve(None, "adversarial"),
         "natural": resolve(None, "natural"),
         "eval": resolve(load_dataset_type("eval")["augmentations"], "adversarial")}
print(json.dumps({
    "techniques": {fn.__name__: {"families": fn.families, "kinds": fn.kinds,
                                 "complexity": fn.complexity, "since": fn.since,
                                 "requires_llm": fn.requires_llm,
                                 "refuses": list(fn.cross_incompatible_with or ()),
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
    assert alpha == {"families": ["suffixes"], "kinds": ["adversarial"],
                     "complexity": 1, "since": "1.4", "requires_llm": False,
                     "refuses": ["ascii_art"],
                     "applied": ["a prompt zz alpha omega", "slug=alpha"]}
    beta = report["techniques"]["to_adversarial_suffix_beta_2"]
    # A row's complexity is read; its kinds are not: no row is natural.
    assert (beta["complexity"], beta["kinds"]) == (2, ["adversarial"])
    made = ["to_adversarial_suffix_alpha", "to_adversarial_suffix_beta_2"]
    assert set(made) <= set(report["pools"]["adversarial"])
    # The natural kind keeps its two noise suffixes; the eval type (eval pool
    # v2) draws the published ones, a row added later too, and no noise suffix.
    assert report["pools"]["natural"] == ["to_adversarial_suffix_emoji",
                                          "to_adversarial_suffix_punctuation"]
    assert report["pools"]["eval"] == made


def test_a_row_without_its_source_fails_the_import_naming_the_row(tmp_path):
    rows = [_ROWS[0],
            {key: value for key, value in _ROWS[1].items() if key != "source"}]
    done = _import(_package_with(tmp_path, rows))
    assert done.returncode != 0
    assert ("ValueError: obfuscation/suffix_strings.json: row 'beta_2' must give "
            "['source']") in done.stderr


def test_the_packaged_file_s_rows_are_attack_only_suffix_techniques():
    """Whatever rows the packaged file holds (24 published strings)."""
    rows = S.suffix_rows()
    made = [fn for fn in all_techniques()
            if fn.__name__ in {f"to_adversarial_suffix_{row['slug']}" for row in rows}]
    assert len(made) == len(rows)
    for fn, row in zip(made, rows, strict=True):
        assert (fn.families, fn.kinds, fn.since) == (["suffixes"], ["adversarial"],
                                                     row["since"])
        assert fn("p") == (f"p {row['suffix']}", f"slug={row['slug']}")
    assert S.get_suffix_functions() == [*S.NOISE_SUFFIXES, *made]


def test_an_empty_file_registers_the_family_with_its_noise_suffixes_alone(
        tmp_path, monkeypatch):
    _copy_spec_and_data(tmp_path)
    _write_rows(tmp_path, [])
    spec = T.read_spec(tmp_path / "combination_spec.json")
    assert S.suffix_rows(tmp_path) == []
    assert [name for name in spec["techniques"]
            if name.startswith("to_adversarial_suffix_")] == [
        fn.__name__ for fn in S.NOISE_SUFFIXES]
    monkeypatch.setattr(JPool, "load_spec", lambda: spec)
    # The registry less the packaged rows' techniques is what an empty file
    # registers: the spec made from one names exactly those.
    published = {f"to_adversarial_suffix_{row['slug']}" for row in S.suffix_rows()}
    JPool._check_against_spec(
        [fn for fn in all_techniques() if fn.__name__ not in published])


def test_a_row_s_technique_appends_its_suffix_after_a_space():
    fn = S.make_suffix_fn(_ROWS[0])
    assert fn.__name__ == "to_adversarial_suffix_alpha"
    assert fn("a prompt") == ("a prompt zz alpha omega", "slug=alpha")


def test_a_row_s_info_is_its_slug_alone_whatever_its_source_says():
    """The source and the model stay in the data file: a dataset row's
    ``technique_info`` names the slug, and a source may hold any text."""
    row = {**_ROWS[0], "source": "a; b = c, retrieved 2026-10-25"}
    _, info = S.make_suffix_fn(row)("a prompt")
    assert info == "slug=alpha"
    assert "source" not in info and "a; b" not in info


def test_the_published_suffixes_refuse_ascii_art_both_ways():
    """A published suffix works only as the literal token string at the end
    of the prompt, and drawing the prompt as art removes it (its characters
    are plain ASCII, which the fonts draw: not the unicode and emoji noise
    suffixes' reason). 24 published x to_ascii_art pairs are refused
    whichever is picked first, and the punctuation and fragment suffixes
    still pair with it (26 of the family's 28 refuse it)."""
    from redact.jailbreak.rules import compatible

    every = {fn.__name__: fn for fn in all_techniques()}
    art = every["to_ascii_art"]
    published = [every[f"to_adversarial_suffix_{row['slug']}"]
                 for row in S.suffix_rows()]
    family = [fn for fn in all_techniques() if fn.families[0] == "suffixes"]

    def refused(fn):
        return art not in compatible([fn], [art]) and fn not in compatible([art], [fn])

    def accepted(fn):
        return art in compatible([fn], [art]) and fn in compatible([art], [fn])

    assert len(published) == 24 and all(refused(fn) for fn in published)   # noqa: PLR2004
    assert [fn.__name__ for fn in family if accepted(fn)] == [
        "to_adversarial_suffix_punctuation", "to_adversarial_suffix_fragments"]
    assert sum(refused(fn) for fn in family) == 26     # noqa: PLR2004


def _without(field: str) -> dict:
    return {key: value for key, value in _ROWS[0].items() if key != field}


@pytest.mark.parametrize(("row", "message"), [
    *[(_without(field), f"row 'alpha' must give ['{field}']")
      for field in S.REQUIRED_FIELDS],
    ({**_ROWS[0], "suffix": "  "}, "row 'alpha' must give ['suffix']"),
    ({**_ROWS[0], "optimized_against": 7},
     "row 'alpha' must give ['optimized_against']"),
    ({**_ROWS[0], "notes": ["a"]}, "the notes of row 'alpha' must be a string"),
    ({**_ROWS[0], "slug": "Alpha-1"}, "row 'Alpha-1' must have a slug of lower-case"),
    ({**_ROWS[0], "slug": "emoji"}, "row 'emoji' must have a slug of lower-case"),
    (_without("slug"), "row 0 has no slug"),
], ids=[*S.REQUIRED_FIELDS, "blank suffix", "not a string", "notes",
        "slug case", "a noise slug", "no slug"])
def test_a_row_that_breaks_a_rule_raises_naming_it(tmp_path, row, message):
    _write_rows(tmp_path, [row])
    with pytest.raises(ValueError) as raised:
        S.suffix_rows(tmp_path)
    assert str(raised.value).startswith(f"{S.SUFFIX_FILE}: {message}")


def test_a_row_named_as_a_noise_suffix_is_one_the_spec_has_already(tmp_path):
    """The spec's generator refuses it too, as for any name it has."""
    _copy_spec_and_data(tmp_path)
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


# --- the packaged rows ------------------------------------------------------

_FILE = paths.jailbreak_configs_dir() / "obfuscation" / "suffix_strings.json"
#: What every packaged row gives: the mechanism's required fields
#: (``S.REQUIRED_FIELDS`` and the slug) and the notes and complexity, which
#: every published row writes out; ``deprecated`` may be added to retire one.
_FIELDS = ("slug", "suffix", "source", "optimized_against", "notes", "complexity",
           "since")
_OPTIONAL = ("deprecated",)
_TEXT_FIELDS = ("slug", "suffix", "source", "optimized_against", "notes", "since")
#: The complexity scale's top, and the fewest published rows the file holds
#: (the plan asked for six).
_MAX_COMPLEXITY = 3
_AT_LEAST = 6
#: The models the rows name, each spelled as its Hugging Face id.
_MODEL_IDS = ("vicuna-13b-v1.5", "Llama-2-7b-chat-hf")

# Short SHA-256 of each published suffix, as copied (and checked against its
# source again when the file was made). A new row is added here with its digest.
_DIGESTS = {
    "gcg_site_example": "47ab045b2a9630ac",
    "gcg_repo_demo_llama2": "740ad8d156d72db8",
    "gcg_smoothllm_llama2_1": "a28f1113e0e48a35",
    "gcg_smoothllm_llama2_2": "3d247d72efa526f9",
    "gcg_smoothllm_vicuna_1": "f87cf5de907749b0",
    "gcg_smoothllm_vicuna_2": "4b7933be3b5ebed0",
    "gcg_jbb_llama2_65": "31065e1ade69774b",
    "gcg_jbb_llama2_75": "eeb6842e122cf490",
    "gcg_jbb_vicuna_gpt4_78": "73fdd44e643686e6",
    "gcg_jbb_vicuna_gpt4_86": "481488128eee878c",
    "gcg_jbb_vicuna_gpt35_1": "f22743b7b3a706fb",
    "gcg_jbb_vicuna_gpt35_48": "90f721c7fb8be1b9",
    "gcg_jbb_vicuna_2": "16530081ffad809a",
    "gcg_jbb_vicuna_6": "81f2a41dc434dede",
    "gcg_jbb_vicuna_26": "39e73762a18f03b0",
    "gcg_jbb_vicuna_32": "6ceea7e19b3f625e",
    "gcg_jbb_vicuna_38": "54f742bc17fe2acd",
    "gcg_jbb_vicuna_45": "d15f9c0c9d97a35e",
    "gcg_jbb_vicuna_54": "684bf1bfb8d66648",
    "gcg_jbb_vicuna_76": "f50f376e98de2ad4",
    "gcg_jbb_vicuna_77": "6109e362e5105086",
    "gcg_jbb_vicuna_90": "1bb9c9b1c8170fd1",
    "gcg_jbb_vicuna_95": "e29a50ac00148524",
    "gcg_jbb_vicuna_97": "91a004769217020c",
}


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def problems(doc: object) -> list[str]:
    """What is wrong with a suffix file's content; an empty list when nothing is."""
    if not isinstance(doc, dict) or not isinstance(doc.get("rows"), list):
        return ["the file is not an object with a list under 'rows'"]
    found: list[str] = []
    seen_slugs: set[str] = set()
    seen_suffixes: set[str] = set()
    for i, row in enumerate(doc["rows"]):
        if not isinstance(row, dict):
            found.append(f"row {i} is not an object")
            continue
        missing = [f for f in _FIELDS if f not in row]
        if missing:
            found.append(f"row {i} lacks {missing}")
            continue
        extra = sorted(set(row) - set(_FIELDS) - set(_OPTIONAL))
        if extra:
            found.append(f"row {i} has unknown fields {extra}")
        for field in _TEXT_FIELDS:
            if not isinstance(row[field], str) or not row[field].strip():
                found.append(f"row {i} has an empty or non-text {field}")
        complexity = row["complexity"]
        if not isinstance(complexity, int) or isinstance(complexity, bool):
            found.append(f"row {i} has a complexity that is not a whole number")
        elif not 0 <= complexity <= _MAX_COMPLEXITY:
            found.append(f"row {i} has a complexity outside 0 to 3")
        slug = row["slug"]
        if isinstance(slug, str):
            if not re.fullmatch(r"[a-z][a-z0-9_]*", slug):
                found.append(f"row {i}: slug is not lowercase snake_case")
            if slug in seen_slugs:
                found.append(f"row {i}: slug repeats")
            seen_slugs.add(slug)
        for field in ("optimized_against", "notes"):
            for model in _MODEL_IDS:
                for spelled in re.findall(re.escape(model), str(row[field]), re.I):
                    if spelled != model:
                        found.append(f"row {i}: {field} spells {model} as {spelled}")
        if isinstance(row["suffix"], str):
            if row["suffix"] in seen_suffixes:
                found.append(f"row {i}: suffix repeats another row's")
            seen_suffixes.add(row["suffix"])
    return found


@pytest.fixture(scope="module")
def doc() -> dict:
    return json.loads(_FILE.read_text(encoding="utf-8"))


class TestFile:
    def test_it_parses_and_has_nothing_wrong(self, doc):
        assert problems(doc) == []

    def test_at_least_six_rows(self, doc):
        assert len(doc["rows"]) >= _AT_LEAST

    def test_the_comment_says_these_are_a_surface_pattern_family(self, doc):
        comment = doc["_comment"].lower()
        assert "optimized against" in comment
        assert "transfer" in comment and "surface-pattern" in comment

    def test_every_row_names_where_it_was_copied_from(self, doc):
        for row in doc["rows"]:
            assert re.search(r"github\.com/|https?://", row["source"]), row["slug"]

    def test_every_published_string_is_pinned(self, doc):
        by_slug = {r["slug"]: r for r in doc["rows"]}
        assert set(_DIGESTS) <= set(by_slug), sorted(set(_DIGESTS) - set(by_slug))
        for slug, digest in _DIGESTS.items():
            assert _digest(by_slug[slug]["suffix"]) == digest, (
                f"{slug}: the string was edited")


class TestTheCheckCanFail:
    """Each way a row can be wrong is found by ``problems``."""

    def _rows(self, doc) -> dict:
        return copy.deepcopy(doc)

    def test_a_missing_field(self, doc):
        bad = self._rows(doc)
        del bad["rows"][0]["source"]
        assert problems(bad)

    def test_an_unknown_field(self, doc):
        bad = self._rows(doc)
        bad["rows"][0]["extra"] = "x"
        assert problems(bad)

    @pytest.mark.parametrize("field", _TEXT_FIELDS)
    def test_an_empty_text_field(self, doc, field):
        bad = self._rows(doc)
        bad["rows"][1][field] = "   "
        assert problems(bad)

    def test_a_repeated_slug(self, doc):
        bad = self._rows(doc)
        bad["rows"][1]["slug"] = bad["rows"][0]["slug"]
        assert problems(bad)

    def test_a_repeated_suffix(self, doc):
        bad = self._rows(doc)
        bad["rows"][1]["suffix"] = bad["rows"][0]["suffix"]
        assert problems(bad)

    def test_a_slug_that_is_not_snake_case(self, doc):
        bad = self._rows(doc)
        bad["rows"][0]["slug"] = "Not-Snake"
        assert problems(bad)

    def test_a_model_spelled_otherwise(self, doc):
        bad = self._rows(doc)
        bad["rows"][0]["optimized_against"] = "Vicuna-13B-v1.5, one behavior"
        assert problems(bad)

    def test_a_row_without_its_version(self, doc):
        bad = self._rows(doc)
        del bad["rows"][0]["since"]
        assert problems(bad)

    def test_a_complexity_that_is_not_a_number(self, doc):
        bad = self._rows(doc)
        bad["rows"][0]["complexity"] = "1"
        assert problems(bad)

    def test_a_file_that_is_not_a_list_of_rows(self):
        assert problems({"rows": {}}) and problems([])

    def test_an_edited_string_fails_the_digest(self, doc):
        by_slug = {r["slug"]: r for r in doc["rows"]}
        for slug, digest in _DIGESTS.items():
            assert _digest(by_slug[slug]["suffix"] + " ") != digest
            assert _digest(by_slug[slug]["suffix"][:-1]) != digest
