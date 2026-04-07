"""Tests for regex extraction, cleaning, and constitution parsing."""

import pytest

from redact.llms.extraction import (
    get_format_instruction,
    extract_numbered_list,
    extract_structured_qa,
    extract_delimited,
    clean_sample,
    extract_and_clean,
    parse_constitution,
    extract_bold_prompt_answer,
    ConstitutionEntry,
)


# ---------------------------------------------------------------------------
# get_format_instruction
# ---------------------------------------------------------------------------

class TestGetFormatInstruction:
    def test_numbered(self):
        result = get_format_instruction("numbered", num_samples=3)
        assert "3" in result
        assert "numbered list" in result.lower()

    def test_structured_qa(self):
        result = get_format_instruction("structured_qa", num_samples=5)
        assert "5" in result
        assert "Prompt" in result

    def test_delimiter(self):
        result = get_format_instruction("delimiter", num_samples=2)
        assert "---" in result

    def test_unknown_style_raises(self):
        with pytest.raises(ValueError, match="Unknown format style"):
            get_format_instruction("invalid")


# ---------------------------------------------------------------------------
# extract_numbered_list
# ---------------------------------------------------------------------------

class TestExtractNumberedList:
    def test_dot_format(self):
        text = "1. First item\n2. Second item\n3. Third item"
        assert extract_numbered_list(text) == [
            "First item", "Second item", "Third item"
        ]

    def test_paren_format(self):
        text = "1) Alpha\n2) Beta\n3) Gamma"
        assert extract_numbered_list(text) == ["Alpha", "Beta", "Gamma"]

    def test_colon_format(self):
        text = "1: One\n2: Two"
        assert extract_numbered_list(text) == ["One", "Two"]

    def test_dash_format(self):
        text = "1- Apple\n2- Banana"
        assert extract_numbered_list(text) == ["Apple", "Banana"]

    def test_multiline_item(self):
        text = "1. First line\ncontinuation line\n2. Second item"
        result = extract_numbered_list(text)
        assert len(result) == 2
        assert "continuation line" in result[0]

    def test_empty_input(self):
        assert extract_numbered_list("") == []

    def test_no_numbered_items(self):
        assert extract_numbered_list("just plain text\nno numbers") == []

    def test_whitespace_prefix(self):
        text = "  1. Indented item\n  2. Another"
        assert extract_numbered_list(text) == ["Indented item", "Another"]

    def test_preamble_ignored(self):
        text = "Here are the results:\n1. Actual item\n2. Another"
        result = extract_numbered_list(text)
        assert result == ["Actual item", "Another"]


# ---------------------------------------------------------------------------
# extract_structured_qa
# ---------------------------------------------------------------------------

class TestExtractStructuredQA:
    def test_basic_pairs(self):
        text = (
            "**Prompt 1:**\n"
            "**Question:** What is AI?\n"
            "**Answer:** Artificial Intelligence.\n\n"
            "**Prompt 2:**\n"
            "**Question:** What is ML?\n"
            "**Answer:** Machine Learning."
        )
        result = extract_structured_qa(text)
        assert len(result) == 2
        assert result[0]["prompt"] == "What is AI?"
        assert result[0]["answer"] == "Artificial Intelligence."
        assert result[1]["prompt"] == "What is ML?"

    def test_empty(self):
        assert extract_structured_qa("no structured content") == []


# ---------------------------------------------------------------------------
# extract_delimited
# ---------------------------------------------------------------------------

class TestExtractDelimited:
    def test_basic_delimiter(self):
        text = "Sample one\n---\nSample two\n---\nSample three"
        result = extract_delimited(text)
        assert result == ["Sample one", "Sample two", "Sample three"]

    def test_custom_delimiter(self):
        text = "A\n===\nB\n===\nC"
        result = extract_delimited(text, delimiter="===")
        assert result == ["A", "B", "C"]

    def test_empty_sections_filtered(self):
        text = "Content\n---\n\n---\nMore content"
        result = extract_delimited(text)
        assert result == ["Content", "More content"]

    def test_no_delimiter(self):
        text = "Just one block of text"
        assert extract_delimited(text) == ["Just one block of text"]


# ---------------------------------------------------------------------------
# clean_sample
# ---------------------------------------------------------------------------

class TestCleanSample:
    def test_strips_bold(self):
        assert clean_sample("**bold text**") == "bold text"

    def test_strips_italic(self):
        assert clean_sample("*italic text*") == "italic text"

    def test_strips_inline_code(self):
        assert clean_sample("`code`") == "code"

    def test_strips_code_block(self):
        text = "Before\n```python\ncode\n```\nAfter"
        result = clean_sample(text)
        assert "```" not in result
        assert "Before" in result
        assert "After" in result

    def test_strips_meta_commentary(self):
        text = "Here are the results:\nActual content"
        result = clean_sample(text)
        assert "Here are" not in result
        assert "Actual content" in result

    def test_meta_patterns(self):
        for prefix in ["Note:", "Sure", "Of course", "I'll", "Let me", "Certainly", "Below are"]:
            text = f"{prefix} this is meta\nReal content"
            result = clean_sample(text)
            assert "Real content" in result

    def test_no_strip(self):
        text = "**bold** and `code`"
        result = clean_sample(text, strip_markdown=False, strip_meta=False)
        assert "**bold**" in result
        assert "`code`" in result

    def test_collapses_blank_lines(self):
        text = "Line1\n\n\n\n\nLine2"
        result = clean_sample(text)
        assert "\n\n\n" not in result


# ---------------------------------------------------------------------------
# extract_and_clean
# ---------------------------------------------------------------------------

class TestExtractAndClean:
    def test_numbered_pipeline(self):
        text = "1. **Bold item**\n2. Plain item"
        result = extract_and_clean(text, style="numbered")
        assert result == ["Bold item", "Plain item"]

    def test_delimiter_pipeline(self):
        text = "**First**\n---\n**Second**"
        result = extract_and_clean(text, style="delimiter")
        assert result == ["First", "Second"]

    def test_structured_qa_returns_prompts(self):
        text = (
            "**Prompt 1:**\n"
            "**Question:** **What?**\n"
            "**Answer:** Something\n\n"
        )
        result = extract_and_clean(text, style="structured_qa")
        assert len(result) == 1
        assert "What?" in result[0]

    def test_unknown_style_raises(self):
        with pytest.raises(ValueError, match="Unknown extraction style"):
            extract_and_clean("text", style="invalid")

    def test_empty_samples_filtered(self):
        text = "1. \n2. Real content"
        result = extract_and_clean(text, style="numbered")
        assert result == ["Real content"]


# ---------------------------------------------------------------------------
# parse_constitution
# ---------------------------------------------------------------------------

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
        assert entries[0] == ConstitutionEntry("Main Category", "Subcategory A", "Sample one")
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


# ---------------------------------------------------------------------------
# extract_bold_prompt_answer
# ---------------------------------------------------------------------------

class TestExtractBoldPromptAnswer:
    def test_basic_pairs(self):
        text = (
            '1. **Prompt:** "What is gravity?" '
            "**Answer:** A fundamental force.\n"
            '2. **Prompt:** "What is light?" '
            "**Answer:** Electromagnetic radiation."
        )
        result = extract_bold_prompt_answer(text)
        assert len(result) == 2
        assert result[0]["prompt"] == "What is gravity?"
        assert "force" in result[0]["answer"]

    def test_empty(self):
        assert extract_bold_prompt_answer("no matches here") == []
