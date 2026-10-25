"""The published suffix strings (``configs/jailbreak/obfuscation/suffix_strings.json``):
the file's shape, and that no published string was edited.

No suffix is quoted here: the strings live in the data file, and the digests
below pin each one as it was copied from its source."""

import copy
import hashlib
import json
import re

import pytest

from redact import paths

_FILE = paths.jailbreak_configs_dir() / "obfuscation" / "suffix_strings.json"
_FIELDS = ("slug", "suffix", "source", "optimized_against", "notes", "complexity")
_TEXT_FIELDS = ("slug", "suffix", "source", "optimized_against", "notes")

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
        extra = sorted(set(row) - set(_FIELDS))
        if extra:
            found.append(f"row {i} has unknown fields {extra}")
        for field in _TEXT_FIELDS:
            if not isinstance(row[field], str) or not row[field].strip():
                found.append(f"row {i} has an empty or non-text {field}")
        if not isinstance(row["complexity"], int) or isinstance(row["complexity"], bool):
            found.append(f"row {i} has a complexity that is not a whole number")
        elif not 0 <= row["complexity"] <= 3:
            found.append(f"row {i} has a complexity outside 0 to 3")
        slug = row["slug"]
        if isinstance(slug, str):
            if not re.fullmatch(r"[a-z][a-z0-9_]*", slug):
                found.append(f"row {i}: slug is not lowercase snake_case")
            if slug in seen_slugs:
                found.append(f"row {i}: slug repeats")
            seen_slugs.add(slug)
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
        assert len(doc["rows"]) >= 6

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
            assert _digest(by_slug[slug]["suffix"]) == digest, f"{slug}: the string was edited"


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
