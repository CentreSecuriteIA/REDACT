"""Tests for constitution generation pipeline."""

import pandas as pd

from redact.constitution.generation import (
    ALL_ENTRY_TYPES,
    ConstitutionEntry,
    ConstitutionPipeline,
    ConstitutionResult,
    EntryType,
    _append_state,
    _read_state,
    _save_entries_csv,
    _state_path,
    _strip_end_marker,
    _unit_key,
)
from tests.conftest import MockBackend, make_client

# Valid 3-layer markdown that parse_constitution can parse
_VALID_CONSTITUTION_OUTPUT = """\
## 1. Category One
### 1.1 Subcategory Alpha
- (Sample description one)
- (Sample description two)
### 1.2 Subcategory Beta
- (Sample description three)

## 2. Category Two
### 2.1 Subcategory Gamma
- (Sample description four)

[END]
"""


class TestEntryType:
    def test_four_values(self):
        assert len(EntryType) == 4

    def test_values(self):
        assert EntryType.HARMFUL.value == "harmful"
        assert EntryType.DUAL_USE_HARMFUL.value == "dual_use_harmful"
        assert EntryType.DUAL_USE_BENIGN.value == "dual_use_benign"
        assert EntryType.BENIGN.value == "benign"

    def test_all_entry_types(self):
        assert len(ALL_ENTRY_TYPES) == 4
        assert EntryType.HARMFUL in ALL_ENTRY_TYPES

    def test_str_enum(self):
        assert str(EntryType.HARMFUL) == "EntryType.HARMFUL"
        assert EntryType.HARMFUL == "harmful"


class TestConstitutionEntry:
    def test_fields(self):
        entry = ConstitutionEntry(
            category="Cat", subcategory="Sub", sample="Sample",
            entry_type="harmful", source_category="Violence",
            source_group_tag="general",
        )
        assert entry.category == "Cat"
        assert entry.subcategory == "Sub"
        assert entry.sample == "Sample"
        assert entry.entry_type == "harmful"
        assert entry.source_category == "Violence"
        assert entry.source_group_tag == "general"


class TestConstitutionResult:
    def test_empty_to_dataframe(self):
        result = ConstitutionResult()
        df = result.to_dataframe()
        assert df.empty
        assert "constitution_category" in df.columns

    def test_to_dataframe(self):
        result = ConstitutionResult(entries=[
            ConstitutionEntry("Cat", "Sub", "Sample", "harmful", "Violence", "general"),
            ConstitutionEntry("Cat2", "Sub2", "Sample2", "benign", "Privacy", "group1"),
        ])
        df = result.to_dataframe()
        assert len(df) == 2
        assert list(df.columns) == [
            "sample_id",
            "constitution_category", "constitution_subcategory",
            "sample_description", "entry_type",
            "source_category", "source_group_tag",
        ]
        assert df.iloc[0]["constitution_category"] == "Cat"
        assert df.iloc[1]["entry_type"] == "benign"
        # sample_id is a content-hash of the entry's own text (own identity,
        # distinct from source_category/group_tag which describe its origin).
        assert df.iloc[0]["sample_id"] and df.iloc[0]["sample_id"] != df.iloc[1]["sample_id"]


class TestStripEndMarker:
    def test_found(self):
        text, complete = _strip_end_marker("some content\n[END]")
        assert complete is True
        assert "[END]" not in text

    def test_missing(self):
        text, complete = _strip_end_marker("some content without marker")
        assert complete is False
        assert text == "some content without marker"

    def test_with_trailing_whitespace(self):
        text, complete = _strip_end_marker("content\n[END]  ")
        assert complete is True


class TestSaveEntriesCsv:
    def test_creates_csv(self, tmp_path):
        entries = [
            ConstitutionEntry("Cat", "Sub", "Sample", "harmful", "Src", "tag"),
        ]
        path = tmp_path / "out" / "test.csv"
        _save_entries_csv(entries, path, "test-model")
        assert path.exists()
        df = pd.read_csv(path)
        assert len(df) == 1
        assert df.iloc[0]["constitution_category"] == "Cat"
        assert df.iloc[0]["model"] == "test-model"

    def test_append_mode(self, tmp_path):
        entries = [ConstitutionEntry("A", "B", "C", "harmful", "D", "E")]
        path = tmp_path / "append.csv"
        _save_entries_csv(entries, path, "m")
        _save_entries_csv(entries, path, "m", append=True)
        df = pd.read_csv(path)
        assert len(df) == 2


class TestConstitutionPipeline:
    def _make_pipeline(self, backend, tmp_path):
        return ConstitutionPipeline(client=make_client(backend, "test-model"),
            output_dir=tmp_path / "constitution",
        )

    def test_format_category_info(self, tmp_path):
        pipeline = self._make_pipeline(MockBackend(), tmp_path)
        info = pipeline._format_category_info("Violence", {
            "description": "Physical harm",
            "subcategories": ["Assault", "Weapons"],
            "group_tag": "harmful",
        })
        assert info["Category"] == "Violence"
        assert "Physical harm" in info["description"]
        assert "Assault" in info["subcategories"]
        assert info["group_tag"] == "harmful"

    def test_format_category_info_dict_subcats(self, tmp_path):
        pipeline = self._make_pipeline(MockBackend(), tmp_path)
        info = pipeline._format_category_info("Cat", {
            "subcategories": {"Sub1": "desc1", "Sub2": "desc2"},
        })
        assert "Sub1" in info["subcategories"]

    def test_generate_for_type(self, tmp_path):
        backend = MockBackend(_VALID_CONSTITUTION_OUTPUT)
        pipeline = self._make_pipeline(backend, tmp_path)
        entries, raw = pipeline.generate_for_type(
            "Violence",
            {"description": "Physical harm", "subcategories": [], "group_tag": "general"},
            EntryType.HARMFUL,
            num_categories=5,
        )
        assert len(entries) > 0
        assert all(e.entry_type == "harmful" for e in entries)
        assert all(e.source_category == "Violence" for e in entries)
        assert "[END]" in raw

    def test_generate_for_type_all_fail(self, tmp_path):
        backend = MockBackend("no parseable content here")
        pipeline = self._make_pipeline(backend, tmp_path)
        entries, raw = pipeline.generate_for_type(
            "Cat", {"description": ""}, EntryType.BENIGN,
        )
        assert entries == []
        assert raw == ""

    def test_generate_for_category(self, tmp_path):
        backend = MockBackend(_VALID_CONSTITUTION_OUTPUT)
        pipeline = self._make_pipeline(backend, tmp_path)
        result = pipeline.generate_for_category(
            "Violence",
            {"description": "Physical harm", "subcategories": []},
            entry_types=[EntryType.HARMFUL, EntryType.BENIGN],
        )
        assert len(result.entries) > 0
        types_present = set(e.entry_type for e in result.entries)
        assert "harmful" in types_present
        assert "benign" in types_present

    def test_run_saves_csvs(self, tmp_path):
        backend = MockBackend(_VALID_CONSTITUTION_OUTPUT)
        pipeline = self._make_pipeline(backend, tmp_path)
        taxonomy = {
            "categories": {
                "TestCat": {
                    "description": "Test category",
                    "subcategories": ["Sub1"],
                },
            },
        }
        result = pipeline.run(
            taxonomy=taxonomy,
            entry_types=[EntryType.HARMFUL],
            num_categories=3,
            save=True,
            verbose=False,
        )
        assert len(result.entries) > 0
        out_dir = tmp_path / "constitution"
        assert (out_dir / "harmful.csv").exists()
        assert (out_dir / "merged.csv").exists()

    def test_generate_general_benign(self, tmp_path):
        backend = MockBackend(_VALID_CONSTITUTION_OUTPUT)
        pipeline = self._make_pipeline(backend, tmp_path)
        entries, raw = pipeline.generate_general_benign(num_categories=3)
        assert len(entries) > 0
        assert all(e.entry_type == "general_benign" for e in entries)
        assert all(e.source_category == "general" for e in entries)


class TestStateLedger:
    """Sidecar resume-state ledger for constitution generation."""

    def test_state_path_is_sidecar(self, tmp_path):
        sp = _state_path(tmp_path / "constitution")
        assert sp.name == "constitution.state.jsonl"
        assert sp.parent == tmp_path / "constitution"

    def test_unit_key(self):
        assert _unit_key("Violence", "harmful") == "Violence::harmful"

    def test_read_missing_returns_empty(self, tmp_path):
        assert _read_state(tmp_path / "nope.state.jsonl") == set()

    def test_append_then_read_roundtrip(self, tmp_path):
        sp = tmp_path / "s.state.jsonl"
        _append_state(sp, ["A::harmful", "A::benign"])
        _append_state(sp, ["B::harmful"])
        assert _read_state(sp) == {"A::harmful", "A::benign", "B::harmful"}

    def test_read_ignores_malformed_lines(self, tmp_path):
        sp = tmp_path / "s.state.jsonl"
        sp.write_text(
            '{"unit": "x"}\nnot json\n{"no_unit": 1}\n{"unit": "y"}\n',
            encoding="utf-8",
        )
        assert _read_state(sp) == {"x", "y"}


class TestRunResume:
    """run() resume behaviour + incremental save + CSV-backed result."""

    _TAXONOMY = {
        "categories": {
            "CatA": {"description": "A", "subcategories": ["S"]},
            "CatB": {"description": "B", "subcategories": ["S"]},
        },
    }

    def _make_pipeline(self, backend, tmp_path):
        return ConstitutionPipeline(client=make_client(backend, "test-model"),
            output_dir=tmp_path / "constitution",
        )

    def test_run_writes_ledger_units(self, tmp_path):
        backend = MockBackend(_VALID_CONSTITUTION_OUTPUT)
        pipeline = self._make_pipeline(backend, tmp_path)
        pipeline.run(
            taxonomy=self._TAXONOMY,
            entry_types=[EntryType.HARMFUL, EntryType.BENIGN],
            num_categories=2, save=True, verbose=False,
        )
        units = _read_state(_state_path(tmp_path / "constitution"))
        assert units == {
            "CatA::harmful", "CatA::benign",
            "CatB::harmful", "CatB::benign",
        }

    def test_run_writes_manifest_units(self, tmp_path):
        backend = MockBackend(_VALID_CONSTITUTION_OUTPUT)
        pipeline = self._make_pipeline(backend, tmp_path)
        pipeline.run(
            taxonomy=self._TAXONOMY,
            entry_types=[EntryType.HARMFUL, EntryType.BENIGN],
            num_categories=2, save=True, verbose=False,
        )
        from redact.dataset import Manifest
        rows = Manifest(tmp_path / "constitution" / "constitution.manifest.jsonl").load()
        assert {(r["source_category"], r["entry_type"]) for r in rows} == {
            ("CatA", "harmful"), ("CatA", "benign"),
            ("CatB", "harmful"), ("CatB", "benign"),
        }
        assert all(r["status"] == "planned" for r in rows)

    def test_resume_skips_completed_and_no_duplicates(self, tmp_path):
        backend = MockBackend(_VALID_CONSTITUTION_OUTPUT)
        pipeline = self._make_pipeline(backend, tmp_path)
        types = [EntryType.HARMFUL, EntryType.BENIGN]

        first = pipeline.run(
            taxonomy=self._TAXONOMY, entry_types=types,
            num_categories=2, save=True, verbose=False,
        )
        calls_after_first = len(backend.calls)
        rows_after_first = len(first.entries)
        assert calls_after_first == 4  # 2 categories x 2 types

        # Re-run: everything is already in the ledger -> no new LLM calls,
        # and the CSV-backed result does not double.
        second = pipeline.run(
            taxonomy=self._TAXONOMY, entry_types=types,
            num_categories=2, save=True, resume=True, verbose=False,
        )
        assert len(backend.calls) == calls_after_first  # no new calls
        assert len(second.entries) == rows_after_first  # no duplicate rows

    def test_resume_continues_after_partial(self, tmp_path):
        backend = MockBackend(_VALID_CONSTITUTION_OUTPUT)
        out_dir = tmp_path / "constitution"

        # Pre-seed the ledger as if CatA was already completed.
        _append_state(_state_path(out_dir), ["CatA::harmful"])

        pipeline = self._make_pipeline(backend, tmp_path)
        pipeline.run(
            taxonomy=self._TAXONOMY, entry_types=[EntryType.HARMFUL],
            num_categories=2, save=True, resume=True, verbose=False,
        )
        # Only CatB should have been generated (CatA skipped).
        assert len(backend.calls) == 1
        assert _read_state(_state_path(out_dir)) == {"CatA::harmful", "CatB::harmful"}

    def test_resume_false_wipes_csvs_and_ledger(self, tmp_path):
        backend = MockBackend(_VALID_CONSTITUTION_OUTPUT)
        out_dir = tmp_path / "constitution"
        pipeline = self._make_pipeline(backend, tmp_path)

        pipeline.run(
            taxonomy=self._TAXONOMY, entry_types=[EntryType.HARMFUL],
            num_categories=2, save=True, verbose=False,
        )
        rows_first = len(pd.read_csv(out_dir / "merged.csv"))

        # resume=False should wipe and regenerate, not append.
        result = pipeline.run(
            taxonomy=self._TAXONOMY, entry_types=[EntryType.HARMFUL],
            num_categories=2, save=True, resume=False, verbose=False,
        )
        assert len(pd.read_csv(out_dir / "merged.csv")) == rows_first
        assert len(result.entries) == rows_first

    def test_result_is_csv_backed(self, tmp_path):
        backend = MockBackend(_VALID_CONSTITUTION_OUTPUT)
        out_dir = tmp_path / "constitution"
        pipeline = self._make_pipeline(backend, tmp_path)
        result = pipeline.run(
            taxonomy=self._TAXONOMY, entry_types=[EntryType.HARMFUL],
            num_categories=2, save=True, verbose=False,
        )
        merged = pd.read_csv(out_dir / "merged.csv")
        assert len(result.entries) == len(merged)

    def test_save_false_keeps_in_memory_and_writes_nothing(self, tmp_path):
        backend = MockBackend(_VALID_CONSTITUTION_OUTPUT)
        out_dir = tmp_path / "constitution"
        pipeline = self._make_pipeline(backend, tmp_path)
        result = pipeline.run(
            taxonomy=self._TAXONOMY, entry_types=[EntryType.HARMFUL],
            num_categories=2, save=False, verbose=False,
        )
        assert len(result.entries) > 0
        assert not (out_dir / "merged.csv").exists()
        assert not _state_path(out_dir).exists()
