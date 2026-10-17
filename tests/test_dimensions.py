"""The dimensions table, and the tag each augmentation family carries."""

import json
import re

import pytest

import redact.jailbreak.pool as JPool
from redact import dimensions, paths
from redact.jailbreak.technique import load_spec

_FAMILY_TAGS = {
    "language": ["translation", "answer_language"],
    "surface_noise": ["typos", "casual_writing", "suffixes"],
    "encoding": ["encode", "sensitive_words", "tokenbreak", "ascii_art"],
    "container": ["structural"],
    "role_framing": ["persona", "framing", "cognitive", "impersonation"],
    "request_shape": ["indirect", "distractor_prefix", "distractor_suffix",
                      "continuation", "asking", "temporal"],
    "answer_shape": ["answer_format"],
    "example_priming": ["fsh", "dap"],
}


@pytest.fixture
def table_at(tmp_path, monkeypatch):
    """Point the table at a file of the test's own."""
    monkeypatch.setattr(paths, "configs_dir", lambda: tmp_path)
    dimensions.table.cache_clear()
    yield lambda entries: (tmp_path / "dimensions.json").write_text(
        json.dumps(entries), encoding="utf-8")
    dimensions.table.cache_clear()


class TestTheTable:
    def test_every_tag_has_a_side_and_a_definition(self):
        for entry in dimensions.table().values():
            assert entry["side"] in dimensions.SIDES
            assert entry["definition"].endswith(".")

    def test_the_augmentation_tags_are_the_eight_of_the_plan(self):
        assert dimensions.tags_of("augmentation") == list(_FAMILY_TAGS)

    @pytest.mark.parametrize(("entry", "message"), [
        ({"side": "both", "definition": "A thing."}, "side must be one of"),
        ({"side": "generation", "definition": " "}, "must be a sentence"),
        ({"side": "generation"}, "and nothing else"),
        ({"side": "generation", "definition": "A thing.", "x": 1}, "and nothing else"),
    ])
    def test_a_malformed_tag_raises(self, table_at, entry, message):
        table_at({"_comment": "skipped", "tag": entry})
        with pytest.raises(ValueError, match=message):
            dimensions.table()


class TestTheFamilies:
    def test_each_family_carries_the_tag_of_the_plan(self):
        assert {name: family["dimension"]
                for name, family in load_spec()["families"].items()} == {
            family: tag for tag, families in _FAMILY_TAGS.items()
            for family in families}

    def test_the_spec_passes_the_check(self):
        JPool._check_dimensions()

    @pytest.mark.parametrize(("tag", "named"), [
        (None, "fsh must carry"),
        ("example", "fsh ('example') must carry"),
    ])
    def test_a_family_without_an_augmentation_tag_raises(self, monkeypatch, tag, named):
        spec = load_spec()
        families = {**spec["families"], "fsh": {**spec["families"]["fsh"]}}
        families["fsh"].pop("dimension")
        if tag is not None:
            families["fsh"]["dimension"] = tag
        monkeypatch.setattr(JPool, "load_spec", lambda: {**spec, "families": families})
        with pytest.raises(ValueError, match=rf"families: {re.escape(named)}"):
            JPool._check_dimensions()
