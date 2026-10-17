"""Tests for constitution generation pipeline."""

import json

import pandas as pd
import pytest

from redact.constitution import (
    ALL_ENTRY_TYPES,
    ConstitutionPipeline,
    EntryType,
    read_constitution,
)
from redact.constitution.parse import strip_end_marker
from redact.constitution.units import (
    Unit,
    _prompt_kwargs,
    messages_of,
    plan_units,
    rows_of,
)
from redact.dataset import Ledger
from tests.conftest import MockBackend, make_client


def _ledger(output_dir):
    """The constitution ledger of the run that writes to ``output_dir``."""
    return Ledger.sidecar(
        output_dir / "merged.csv", ("unit",), {"unit": str},
        name="constitution.state.jsonl",
    )


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
# The same reply cut off before its end marker.
_TRUNCATED_OUTPUT = _VALID_CONSTITUTION_OUTPUT.replace("[END]", "")
# The samples in it, and the generations a unit makes before it is given up.
_SAMPLES = 4
_ATTEMPTS = 3
# A benign taxonomy of two categories.
_BENIGN = {"categories": {
    "Cooking": {"description": "Food at home", "subcategories": ["Bread", "Soup"]},
    "Travel": {"description": "Trips", "subcategories": ["Trains"]},
}}


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


class TestRowsOf:
    def test_a_reply_gives_one_row_per_sample_with_the_unit_tags(self):
        unit = Unit("Violence", "harmful", "general")
        rows = rows_of(unit, _VALID_CONSTITUTION_OUTPUT, "test-model")
        assert len(rows) == _SAMPLES
        assert list(rows[0]) == [
            "sample_id", "constitution_category", "constitution_subcategory",
            "sample_description", "entry_type", "source_category",
            "source_taxonomy", "source_group_tag", "model",
        ]
        assert {r["source_taxonomy"] for r in rows} == {"harmful"}
        assert rows[0]["constitution_category"] == "Category One"
        assert rows[0]["constitution_subcategory"] == "Subcategory Alpha"
        assert rows[0]["sample_description"] == "Sample description one"
        assert {r["entry_type"] for r in rows} == {"harmful"}
        assert {r["source_category"] for r in rows} == {"Violence"}
        assert {r["model"] for r in rows} == {"test-model"}
        # sample_id is a content-hash of the entry's own text.
        assert rows[0]["sample_id"] and rows[0]["sample_id"] != rows[1]["sample_id"]

    def test_a_reply_that_does_not_parse_gives_no_row(self):
        unit = Unit("Cat", "benign", "general")
        assert rows_of(unit, "no parseable content here\n[END]", "m") == []

    def test_a_reply_without_its_end_marker_is_rejected_whole(self):
        unit = Unit("Cat", "benign", "general")
        assert rows_of(unit, _TRUNCATED_OUTPUT, "m") == []


class TestReadConstitution:
    def test_no_file_gives_the_columns(self, tmp_path):
        frame = read_constitution(tmp_path)
        assert frame.empty
        assert "constitution_category" in frame.columns

    def test_an_empty_cell_is_text(self, tmp_path):
        pd.DataFrame([{"sample_id": "0123", "constitution_category": "Cat",
                       "constitution_subcategory": None, "sample_description": "s",
                       "entry_type": "harmful", "source_category": "Src",
                       "source_group_tag": None, "model": "m"}]
                     ).to_csv(tmp_path / "merged.csv", index=False)
        row = read_constitution(tmp_path).iloc[0]
        assert row["sample_id"] == "0123"
        assert row["constitution_subcategory"] == ""


class TestStripEndMarker:
    def test_found(self):
        text, complete = strip_end_marker("some content\n[END]")
        assert complete is True
        assert "[END]" not in text

    def test_missing(self):
        text, complete = strip_end_marker("some content without marker")
        assert complete is False
        assert text == "some content without marker"

    def test_with_trailing_whitespace(self):
        text, complete = strip_end_marker("content\n[END]  ")
        assert complete is True


class TestConstitutionPipeline:
    def _make_pipeline(self, backend, tmp_path):
        return ConstitutionPipeline(client=make_client(backend, "test-model"),
            output_dir=tmp_path / "constitution",
        )

    def test_format_category_info(self):
        info = _prompt_kwargs("Violence", {
            "description": "Physical harm",
            "subcategories": ["Assault", "Weapons"],
            "group_tag": "harmful",
        })
        assert info["Category"] == "Violence"
        assert "Physical harm" in info["description"]
        assert "Assault" in info["subcategories"]
        assert info["group_tag"] == "harmful"

    def test_format_category_info_dict_subcats(self):
        info = _prompt_kwargs("Cat", {
            "subcategories": {"Sub1": "desc1", "Sub2": "desc2"},
        })
        assert "Sub1" in info["subcategories"]

    def test_a_unit_is_one_call_tagged_with_its_category(self, tmp_path):
        backend = MockBackend(_VALID_CONSTITUTION_OUTPUT)
        pipeline = self._make_pipeline(backend, tmp_path)
        result = pipeline.run(
            {"categories": {"Violence": {"description": "Physical harm",
                                         "subcategories": [], "group_tag": "general"}}},
            entry_types=[EntryType.HARMFUL], num_categories=5, verbose=False,
        )
        assert len(backend.calls) == 1
        assert len(result) > 0
        assert set(result["entry_type"]) == {"harmful"}
        assert set(result["source_category"]) == {"Violence"}

    def test_a_unit_that_never_parses_is_acked_as_rejected(self, tmp_path):
        backend = MockBackend("no parseable content here")
        pipeline = self._make_pipeline(backend, tmp_path)
        result = pipeline.run({"categories": {"Cat": {"description": ""}}},
                              entry_types=[EntryType.BENIGN], verbose=False)
        assert len(backend.calls) == _ATTEMPTS
        assert result.empty
        assert _ledger(tmp_path / "constitution").completed() == {"Cat::benign"}

    def test_a_repeated_entry_type_is_planned_once(self):
        units = plan_units({"categories": {"Cat": {"description": ""}}},
                           entry_types=[EntryType.HARMFUL, EntryType.HARMFUL])
        assert [u.key for u in units] == ["Cat::harmful"]

    def test_a_truncated_reply_is_regenerated(self, tmp_path):
        backend = MockBackend([_TRUNCATED_OUTPUT, _VALID_CONSTITUTION_OUTPUT])
        pipeline = self._make_pipeline(backend, tmp_path)
        result = pipeline.run({"categories": {"Cat": {"description": ""}}},
                              entry_types=[EntryType.HARMFUL], verbose=False)
        assert len(backend.calls) == 2
        assert len(result) == _SAMPLES
        assert _ledger(tmp_path / "constitution").completed() == {"Cat::harmful"}

    def test_a_unit_truncated_every_time_is_acked_rejected_and_skipped_on_resume(
            self, tmp_path):
        out_dir = tmp_path / "constitution"
        taxonomy = {"categories": {"Cat": {"description": ""}}}
        backend = MockBackend(_TRUNCATED_OUTPUT)
        result = self._make_pipeline(backend, tmp_path).run(
            taxonomy, entry_types=[EntryType.HARMFUL], verbose=False)
        assert len(backend.calls) == _ATTEMPTS
        assert result.empty
        assert not (out_dir / "merged.csv").exists()
        acks = [json.loads(line) for line in _ledger(out_dir).path.read_text(
            encoding="utf-8").splitlines()]
        assert acks == [{"unit": "Cat::harmful", "status": "rejected",
                         "attempts": _ATTEMPTS}]

        backend = MockBackend(_VALID_CONSTITUTION_OUTPUT)
        self._make_pipeline(backend, tmp_path).run(
            taxonomy, entry_types=[EntryType.HARMFUL], verbose=False)
        assert backend.calls == []

    def test_each_entry_type_is_a_unit(self, tmp_path):
        backend = MockBackend(_VALID_CONSTITUTION_OUTPUT)
        pipeline = self._make_pipeline(backend, tmp_path)
        result = pipeline.run(
            {"categories": {"Violence": {"description": "Physical harm",
                                         "subcategories": []}}},
            entry_types=[EntryType.HARMFUL, EntryType.BENIGN], verbose=False,
        )
        assert set(result["entry_type"]) == {"harmful", "benign"}

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
            verbose=False,
        )
        assert len(result) > 0
        out_dir = tmp_path / "constitution"
        assert (out_dir / "harmful.csv").exists()
        assert (out_dir / "merged.csv").exists()

    def test_the_general_benign_units_are_one_per_benign_category_and_come_last(self):
        units = plan_units({"categories": {"Cat": {}}}, [EntryType.HARMFUL],
                           include_general_benign=True,
                           general_benign_categories=3,
                           benign_taxonomy=_BENIGN)
        assert [u.key for u in units] == [
            "Cat::harmful", "Cooking::general_benign", "Travel::general_benign"]
        unit = units[1]
        assert unit.prompt == "general_benign"
        assert unit.kwargs == {
            "Category": "Cooking", "description": "Food at home",
            "subcategories": "Bread, Soup", "group_tag": "general",
            "num_categories": "3"}
        rows = rows_of(unit, _VALID_CONSTITUTION_OUTPUT, "m")
        assert {r["entry_type"] for r in rows} == {"benign"}
        assert {r["source_category"] for r in rows} == {"Cooking"}
        assert {r["source_taxonomy"] for r in rows} == {"benign"}

    def test_a_benign_category_named_as_a_harmful_one_is_a_unit_of_its_own(self):
        units = plan_units({"categories": {"Cooking": {}}}, [EntryType.BENIGN],
                           include_general_benign=True, benign_taxonomy=_BENIGN)
        assert [u.key for u in units][:2] == [
            "Cooking::benign", "Cooking::general_benign"]
        assert units[0].manifest_row == {
            "source_category": "Cooking", "entry_type": "benign",
            "prompt": "benign", "status": "planned"}
        assert units[1].manifest_row == {
            "source_category": "Cooking", "entry_type": "benign",
            "prompt": "general_benign", "status": "planned"}

    def test_num_taxonomy_categories_limits_the_benign_taxonomy_too(self):
        units = plan_units({"categories": {"A": {}, "B": {}}}, [EntryType.HARMFUL],
                           num_taxonomy_categories=1, include_general_benign=True,
                           benign_taxonomy=_BENIGN)
        assert [u.key for u in units] == ["A::harmful", "Cooking::general_benign"]

    def test_no_general_benign_unit_unless_asked(self):
        units = plan_units({"categories": {"Cat": {}}}, [EntryType.HARMFUL],
                           benign_taxonomy=_BENIGN)
        assert [u.key for u in units] == ["Cat::harmful"]

    def test_the_general_benign_rows_are_saved_to_their_own_file(self, tmp_path):
        pipeline = self._make_pipeline(MockBackend(_VALID_CONSTITUTION_OUTPUT), tmp_path)
        pipeline.run({"categories": {}}, include_general_benign=True,
                     benign_taxonomy=_BENIGN, verbose=False)
        out_dir = tmp_path / "constitution"
        saved = pd.read_csv(out_dir / "general_benign.csv")
        assert len(saved) == 2 * _SAMPLES
        assert set(saved["entry_type"]) == {"benign"}
        assert set(saved["source_taxonomy"]) == {"benign"}
        assert not (out_dir / "benign.csv").exists()
        assert len(pd.read_csv(out_dir / "merged.csv")) == 2 * _SAMPLES
        assert _ledger(out_dir).completed() == {
            "Cooking::general_benign", "Travel::general_benign"}

    def test_the_packaged_general_benign_prompt_renders_the_benign_category(self):
        unit = plan_units({"categories": {}}, include_general_benign=True,
                          benign_taxonomy=_BENIGN)[0]
        rendered = " ".join(m["content"] for m in messages_of(unit))
        # Only the values are compared: the prompt's own text is not shown.
        missing = [value for value in ("Cooking", "Food at home", "Bread, Soup")
                   if value not in rendered]
        assert missing == []

    def test_a_resume_of_a_constitution_without_the_source_taxonomy_column_stops(
            self, tmp_path):
        out_dir = tmp_path / "constitution"
        out_dir.mkdir()
        pd.DataFrame([{"sample_id": "1", "entry_type": "harmful",
                       "source_category": "Cat"}]).to_csv(
            out_dir / "merged.csv", index=False)
        backend = MockBackend(_VALID_CONSTITUTION_OUTPUT)
        with pytest.raises(ValueError, match="run with resume=False") as raised:
            self._make_pipeline(backend, tmp_path).run(
                {"categories": {"Cat": {}}}, entry_types=[EntryType.HARMFUL],
                verbose=False)
        assert "'source_taxonomy'" in str(raised.value)
        assert backend.calls == []
        # A fresh run writes it anew.
        self._make_pipeline(backend, tmp_path).run(
            {"categories": {"Cat": {}}}, entry_types=[EntryType.HARMFUL],
            resume=False, verbose=False)
        assert set(pd.read_csv(out_dir / "merged.csv")["source_taxonomy"]) == {
            "harmful"}


class TestStateLedger:
    """Sidecar resume-state ledger for constitution generation."""

    def test_state_path_is_sidecar(self, tmp_path):
        sp = _ledger(tmp_path / "constitution").path
        assert sp.name == "constitution.state.jsonl"
        assert sp.parent == tmp_path / "constitution"

    def test_unit_key(self):
        assert Unit("Violence", "harmful", "general").key == "Violence::harmful"

    def test_read_missing_returns_empty(self, tmp_path):
        assert _ledger(tmp_path / "nope").completed() == set()

    def test_append_then_read_roundtrip(self, tmp_path):
        ledger = _ledger(tmp_path)
        ledger.record([{"unit": "A::harmful"}, {"unit": "A::benign"}])
        ledger.record([{"unit": "B::harmful"}])
        assert ledger.completed() == {"A::harmful", "A::benign", "B::harmful"}

    def test_read_ignores_malformed_lines(self, tmp_path):
        ledger = _ledger(tmp_path)
        ledger.path.write_text(
            '{"unit": "x"}\nnot json\n{"no_unit": 1}\n{"unit": "y"}\n',
            encoding="utf-8",
        )
        assert ledger.completed() == {"x", "y"}


class _DiesAfter(MockBackend):
    """Answers ``calls`` requests, then fails as a transport does."""

    def __init__(self, calls):
        super().__init__(_VALID_CONSTITUTION_OUTPUT)
        self.left = calls

    def generate(self, messages_list, **kwargs):
        if self.left < len(messages_list):
            raise ConnectionError("the endpoint is down")
        self.left -= len(messages_list)
        return super().generate(messages_list, **kwargs)


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
            num_categories=2, verbose=False,
        )
        units = _ledger(tmp_path / "constitution").completed()
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
            num_categories=2, verbose=False,
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
            num_categories=2, verbose=False,
        )
        calls_after_first = len(backend.calls)
        rows_after_first = len(first)
        assert calls_after_first == 4  # 2 categories x 2 types

        # Re-run: everything is already in the ledger -> no new LLM calls,
        # and the CSV-backed result does not double.
        second = pipeline.run(
            taxonomy=self._TAXONOMY, entry_types=types,
            num_categories=2, resume=True, verbose=False,
        )
        assert len(backend.calls) == calls_after_first  # no new calls
        assert len(second) == rows_after_first  # no duplicate rows

    def test_resume_continues_after_partial(self, tmp_path):
        backend = MockBackend(_VALID_CONSTITUTION_OUTPUT)
        out_dir = tmp_path / "constitution"

        # Pre-seed the ledger as if CatA was already completed.
        _ledger(out_dir).record([{"unit": "CatA::harmful"}])

        pipeline = self._make_pipeline(backend, tmp_path)
        pipeline.run(
            taxonomy=self._TAXONOMY, entry_types=[EntryType.HARMFUL],
            num_categories=2, resume=True, verbose=False,
        )
        # Only CatB should have been generated (CatA skipped).
        assert len(backend.calls) == 1
        assert _ledger(out_dir).completed() == {"CatA::harmful", "CatB::harmful"}

    def test_a_transport_error_stops_the_run_with_the_acked_units_kept(self, tmp_path):
        out_dir = tmp_path / "constitution"
        with pytest.raises(ConnectionError):
            self._make_pipeline(_DiesAfter(1), tmp_path).run(
                taxonomy=self._TAXONOMY, entry_types=[EntryType.HARMFUL],
                num_categories=2, verbose=False,
            )
        assert _ledger(out_dir).completed() == {"CatA::harmful"}
        assert set(pd.read_csv(out_dir / "merged.csv")["source_category"]) == {"CatA"}

        # The resume makes the one call the error stopped.
        backend = MockBackend(_VALID_CONSTITUTION_OUTPUT)
        self._make_pipeline(backend, tmp_path).run(
            taxonomy=self._TAXONOMY, entry_types=[EntryType.HARMFUL],
            num_categories=2, verbose=False,
        )
        assert len(backend.calls) == 1
        assert _ledger(out_dir).completed() == {"CatA::harmful", "CatB::harmful"}

    def test_resume_false_wipes_csvs_and_ledger(self, tmp_path):
        backend = MockBackend(_VALID_CONSTITUTION_OUTPUT)
        out_dir = tmp_path / "constitution"
        pipeline = self._make_pipeline(backend, tmp_path)

        pipeline.run(
            taxonomy=self._TAXONOMY, entry_types=[EntryType.HARMFUL],
            num_categories=2, verbose=False,
        )
        rows_first = len(pd.read_csv(out_dir / "merged.csv"))

        # resume=False should wipe and regenerate, not append.
        result = pipeline.run(
            taxonomy=self._TAXONOMY, entry_types=[EntryType.HARMFUL],
            num_categories=2, resume=False, verbose=False,
        )
        assert len(pd.read_csv(out_dir / "merged.csv")) == rows_first
        assert len(result) == rows_first

    def test_result_is_csv_backed(self, tmp_path):
        backend = MockBackend(_VALID_CONSTITUTION_OUTPUT)
        out_dir = tmp_path / "constitution"
        pipeline = self._make_pipeline(backend, tmp_path)
        result = pipeline.run(
            taxonomy=self._TAXONOMY, entry_types=[EntryType.HARMFUL],
            num_categories=2, verbose=False,
        )
        merged = pd.read_csv(out_dir / "merged.csv")
        assert len(result) == len(merged)

    def test_read_back_false_returns_an_empty_frame_and_saves(self, tmp_path):
        backend = MockBackend(_VALID_CONSTITUTION_OUTPUT)
        out_dir = tmp_path / "constitution"
        pipeline = self._make_pipeline(backend, tmp_path)
        result = pipeline.run(
            taxonomy=self._TAXONOMY, entry_types=[EntryType.HARMFUL],
            num_categories=2, verbose=False, read_back=False,
        )
        assert result.empty
        assert len(pd.read_csv(out_dir / "merged.csv")) > 0


class TestTheFront:
    """``generate_constitution``: the benign taxonomy and the former names."""

    def _run(self, tmp_path, monkeypatch, **kwargs):
        from redact import generate_constitution
        from redact.llms import ModelClient

        backend = MockBackend(_VALID_CONSTITUTION_OUTPUT)
        monkeypatch.setattr(ModelClient, "create",
                            lambda model, *a, **k: make_client(backend, model))
        result = generate_constitution(
            taxonomy={"categories": {}}, model="c-model", data_dir=tmp_path,
            verbose=False, **kwargs)
        return result, backend

    def test_the_general_benign_part_is_built_from_a_benign_taxonomy_by_name(
            self, tmp_path, monkeypatch):
        from redact.dataset import create_taxonomy

        create_taxonomy("mine", _BENIGN["categories"], taxonomy_dir=tmp_path,
                        verbose=False)
        result, backend = self._run(
            tmp_path, monkeypatch, include_general_benign=True,
            general_benign_categories=2, benign_taxonomy="mine",
            taxonomy_dir=tmp_path)
        assert len(backend.calls) == len(_BENIGN["categories"])
        assert set(result["source_category"]) == {"Cooking", "Travel"}
        assert set(result["source_taxonomy"]) == {"benign"}
        assert set(result["entry_type"]) == {"benign"}

    def test_without_the_switch_the_benign_taxonomy_is_not_read(
            self, tmp_path, monkeypatch):
        result, backend = self._run(tmp_path, monkeypatch, benign_taxonomy="missing")
        assert backend.calls == []
        assert result.empty

    def test_the_former_names_are_read_with_a_warning(self, tmp_path, monkeypatch):
        with pytest.warns(DeprecationWarning) as warned:
            _, backend = self._run(
                tmp_path, monkeypatch, include_standalone_benign=True,
                standalone_benign_categories=2, benign_taxonomy=_BENIGN)
        assert [str(w.message) for w in warned] == [
            "generate_constitution: include_standalone_benign and "
            "standalone_benign_categories are now include_general_benign and "
            "general_benign_categories; the former names are read for this "
            "release."]
        assert len(backend.calls) == len(_BENIGN["categories"])
        assert _ledger(tmp_path / "Data_cache" / "constitution").completed() == {
            "Cooking::general_benign", "Travel::general_benign"}

    def test_a_former_name_given_wins_over_the_new_one(self, tmp_path, monkeypatch):
        with pytest.warns(DeprecationWarning):
            _, backend = self._run(
                tmp_path, monkeypatch, include_general_benign=True,
                include_standalone_benign=False, benign_taxonomy=_BENIGN)
        assert backend.calls == []
