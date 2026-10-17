"""Tests for constitution parsing (constitution/parse.py)."""

import logging
import time

import pytest

from redact.constitution.parse import ParsedEntry, parse_constitution


class TestParseConstitution:
    def test_valid_hierarchy(self):
        text = (
            "## 1. Main Category\n"
            "### 1.1 Subcategory A\n"
            "- (Sample one)\n"
            "- (Sample two)\n"
            "### 1.2 Subcategory B\n"
            "- (Sample three)\n"
            "## 2. Second Category\n"
            "### 2.1 Sub\n"
            "- (Sample four)\n"
        )
        entries = parse_constitution(text)
        assert len(entries) == 4
        assert entries[0] == ParsedEntry("Main Category", "Subcategory A", "Sample one")
        assert entries[1].subcategory == "Subcategory A"
        assert entries[2].subcategory == "Subcategory B"
        assert entries[3].category == "Second Category"

    def test_sample_without_subcategory(self):
        text = "## 1. Cat\n- (Direct sample)\n"
        entries = parse_constitution(text)
        assert len(entries) == 1
        assert entries[0].subcategory == ""
        assert entries[0].sample == "Direct sample"

    def test_empty_input(self):
        assert parse_constitution("") == []

    def test_no_matching_structure(self):
        assert parse_constitution("Random text\nNo headers") == []

    def test_malformed_sample_ignored(self):
        text = "## 1. Cat\n### 1.1 Sub\n- No parens\n- (Valid)\n"
        entries = parse_constitution(text)
        assert len(entries) == 1
        assert entries[0].sample == "Valid"

    def test_end_marker_ignored(self):
        text = "## 1. Cat\n### 1.1 Sub\n- (Sample)\n[END]\n"
        entries = parse_constitution(text)
        assert len(entries) == 1


class TestParseConstitutionTolerance:
    def test_bold_headers_and_star_bullets_parse(self):
        text = "## **1. Main**\n### **1.1 Sub**\n* (one)\n- (two)"
        assert [(e.category, e.subcategory, e.sample) for e in parse_constitution(text)] == [
            ("Main", "Sub", "one"), ("Main", "Sub", "two"),
        ]

    def test_unparsed_header_drops_its_samples_instead_of_mislabelling(self):
        text = "## 1. Main\n- (kept)\n## Broken header\n- (dropped)\n## 2. Next\n- (kept too)"
        assert [(e.category, e.sample) for e in parse_constitution(text)] == [
            ("Main", "kept"), ("Next", "kept too"),
        ]

    def test_document_title_does_not_end_a_category(self):
        text = "## 1. Main\n# Title\n- (kept)"
        assert [e.sample for e in parse_constitution(text)] == ["kept"]


class TestCrlf:
    def test_constitution(self):
        text = "## 1. Main\r\n### 1.1 Sub\r\n- (one)\r\n"
        assert parse_constitution(text) == [ParsedEntry("Main", "Sub", "one")]


class TestParseConstitutionStrictness:
    @pytest.mark.parametrize("header, name", [
        ("## 1. **Main**", "Main"),
        ("## *1. Main*", "Main"),
        ("## __1. Main__", "Main"),
        ("## 1. Main:", "Main"),
        ("##   1.   Main   ", "Main"),
        ("## 1. Main (with note)", "Main (with note)"),
    ])
    def test_header_markup_is_stripped_from_the_name(self, header, name):
        entries = parse_constitution(f"{header}\n### 1.1 **Sub**:\n- (one)")
        assert entries == [ParsedEntry(name, "Sub", "one")]

    @pytest.mark.parametrize("bullet, sample", [
        ("- (text (with parens) more)", "text (with parens) more"),
        ("- (a (b) c (d) e)", "a (b) c (d) e"),
    ])
    def test_nested_parentheses_are_kept(self, bullet, sample):
        assert [e.sample for e in parse_constitution(f"## 1. M\n{bullet}")] == [sample]

    @pytest.mark.parametrize("bullet", [
        "- (one) and (two)",       # two groups: which is the sample is a guess
        "- (e.g.) bad (really)",
        "- (one) trailing note",
        "- (one (unbalanced)",
        "- ( )",
        "- ()",
        "1. (numbered)",
    ])
    def test_ambiguous_or_empty_sample_is_skipped(self, bullet):
        assert parse_constitution(f"## 1. M\n{bullet}\n- (kept)") == [
            ParsedEntry("M", "", "kept")
        ]

    @pytest.mark.parametrize("header", ["## 1. **", "## 1. :", "#### 1.1.1 Deep", "### Broken"])
    def test_unparsed_header_ends_the_category(self, header):
        text = f"## 1. A\n- (kept)\n{header}\n- (dropped)\n## 2. B\n- (kept too)"
        assert [(e.category, e.sample) for e in parse_constitution(text)] == [
            ("A", "kept"), ("B", "kept too"),
        ]

    def test_subcategory_before_any_category_has_no_samples(self):
        text = "### 1.1 Sub\n- (dropped)\n## 1. Main\n- (kept)"
        assert parse_constitution(text) == [ParsedEntry("Main", "", "kept")]

    def test_a_logged_line_is_cut_to_80_characters(self, caplog):
        text = f"## {'h' * 200}\n## 1. Main\n{'x' * 200}"
        with caplog.at_level(logging.DEBUG, logger="redact.constitution.parse"):
            parse_constitution(text)
        header = "## " + "h" * 77
        assert [r.getMessage() for r in caplog.records] == [
            f"Unrecognised constitution header {header!r}; "
            "dropping the samples under it.",
            f"Skipped constitution line {'x' * 80!r}",
        ]


class TestLinearTime:
    """A long run of whitespace or markers is parsed in linear time."""

    SIZE = 200_000

    @pytest.mark.parametrize("text", [
        "##" + " " * SIZE + "x",
        "## 1. a" + " " * SIZE + "b",
        "### 1.1 a" + " " * SIZE + "b",
    ], ids=["main-header", "main-name", "sub-name"])
    def test_whitespace_run_is_parsed_quickly(self, text):
        start = time.perf_counter()
        parse_constitution(text)
        assert time.perf_counter() - start < 2.0
