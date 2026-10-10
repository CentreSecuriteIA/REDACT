"""Tests for regex extraction, cleaning, and constitution parsing."""

import logging
import time

import pytest

from redact.llms.prompting import (
    EXTRACTION_STYLES,
    ConstitutionEntry,
    clean_sample,
    extract_and_clean,
    extract_numbered_list,
    extract_structured_qa,
    get_format_instruction,
    parse_constitution,
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

    @pytest.mark.parametrize("style", ["invalid", "delimiter"])
    def test_unknown_style_raises(self, style):
        with pytest.raises(ValueError, match="Unknown format style"):
            get_format_instruction(style, num_samples=2)

    @pytest.mark.parametrize("count", [0, -1, 2.5, None, "many", True])
    def test_bad_sample_count_raises(self, count):
        with pytest.raises(ValueError, match="num_samples must be an integer >= 1"):
            get_format_instruction("numbered", num_samples=count)

    def test_style_and_count_have_no_default(self):
        with pytest.raises(TypeError):
            get_format_instruction("numbered")
        with pytest.raises(TypeError):
            get_format_instruction(num_samples=2)

    def test_every_extraction_style_has_a_format_instruction(self):
        # EXTRACTION_STYLES is shared with extract_and_clean()'s own
        # validation — this is the "the two can't silently drift" guarantee.
        for style in EXTRACTION_STYLES:
            assert get_format_instruction(style, num_samples=1)

    @pytest.mark.parametrize("content, reason", [
        ('{"instruction": "Give {num_sample}"}', r"is missing \{num_sample\}"),
        ('{"system_prompt": "S", "template": "T"}', 'needs an "instruction"'),
    ], ids=["wrong-placeholder", "no-instruction-key"])
    def test_a_broken_override_is_named(self, tmp_path, content, reason):
        override = tmp_path / "format_instructions" / "numbered"
        override.mkdir(parents=True)
        (override / "template.json").write_text(content, encoding="utf-8")
        with pytest.raises(ValueError, match=reason) as exc:
            get_format_instruction("numbered", num_samples=2, prompt_dir=tmp_path)
        assert "instruction of format_instructions/numbered (override:" in str(exc.value)


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
        assert result[0]["question"] == "What is AI?"
        assert result[0]["answer"] == "Artificial Intelligence."
        assert result[1]["question"] == "What is ML?"

    def test_empty(self):
        assert extract_structured_qa("no structured content") == []


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

    @pytest.mark.parametrize("first_line", [
        "Here are the samples:", "Sure, coming up", "Let me try:", "I'll wait.",
    ])
    def test_first_line_is_kept(self, first_line):
        text = f"{first_line}\nReal content"
        assert clean_sample(text) == text

    def test_no_strip(self):
        text = "**bold** and `code`"
        result = clean_sample(text, strip_markdown=False)
        assert "**bold**" in result
        assert "`code`" in result

    def test_collapses_blank_lines(self):
        text = "Line1\n\n\n\n\nLine2"
        result = clean_sample(text)
        assert "\n\n\n" not in result


# ---------------------------------------------------------------------------
# extract_and_clean
# ---------------------------------------------------------------------------

class TestCleanSamplePreservesContent:
    """Markdown *markup* comes off; the text it wraps stays, including the
    code inside a fence."""

    def test_code_block_is_unwrapped_not_deleted(self):
        text = 'Write this:\n```python\nimport os\nos.system("x")\n```\nThen run it.'
        out = clean_sample(text)
        assert "import os" in out
        assert 'os.system("x")' in out
        assert "```" not in out

    def test_two_code_blocks_both_survive(self):
        out = clean_sample("A:\n```\nx=1\n```\nB:\n```js\ny=2\n```\ndone")
        assert "x=1" in out and "y=2" in out and "```" not in out

    def test_unclosed_fence_does_not_mangle_the_line(self):
        """What truncation at max_tokens looks like: an opener, no closer.

        The lone fence and its language tag are removed and the code line is
        left whole.
        """
        out = clean_sample("Write a script:\n```python\nimport os")
        assert out == "Write a script:\nimport os"
        assert "`" not in out

    def test_arithmetic_asterisks_are_left_alone(self):
        """'3 * 4 * 5' is multiplication, not emphasis."""
        assert clean_sample("Compute 3 * 4 * 5 and report") == "Compute 3 * 4 * 5 and report"

    def test_real_emphasis_is_still_unwrapped(self):
        assert clean_sample("This is *emphasised* text") == "This is emphasised text"

    def test_bold_and_inline_code_still_unwrapped(self):
        assert clean_sample("Explain **how** to run `ls -la`") == "Explain how to run ls -la"


class TestStructuredQAHeaderIsOptional:
    """The '**Prompt N:**' line is a pure index — never required.

    The prompts that produce this format put nothing after it (the question is
    on the next line), so pairs parse with or without it.
    """

    CANONICAL = (
        "**Prompt 1:**\n**Question:** What is X?\n**Answer:** It is Y.\n\n"
        "**Prompt 2:**\n**Question:** And Z?\n**Answer:** Sure."
    )

    def test_canonical_format_still_parses(self):
        pairs = extract_structured_qa(self.CANONICAL)
        assert pairs == [
            {"question": "What is X?", "answer": "It is Y."},
            {"question": "And Z?", "answer": "Sure."},
        ]

    def test_pairs_parse_without_the_index_header(self):
        text = self.CANONICAL.replace("**Prompt 1:**", "").replace("**Prompt 2:**", "")
        pairs = extract_structured_qa(text)
        assert len(pairs) == 2
        assert pairs[0]["question"] == "What is X?"
        assert pairs[1]["answer"] == "Sure."

    def test_missing_colon_in_the_header_still_parses(self):
        pairs = extract_structured_qa(self.CANONICAL.replace("Prompt 1:**", "Prompt 1**"))
        assert len(pairs) == 2

    def test_answers_still_stop_at_the_next_pair(self):
        """Without headers the terminator has to fall back to **Question:**."""
        text = self.CANONICAL.replace("**Prompt 1:**", "").replace("**Prompt 2:**", "")
        assert extract_structured_qa(text)[0]["answer"] == "It is Y."


class TestExtractAndClean:
    def test_numbered_pipeline(self):
        text = "1. **Bold item**\n2. Plain item"
        result = extract_and_clean(text, style="numbered")
        assert result == ["Bold item", "Plain item"]

    def test_paired_style_raises_and_names_the_right_function(self):
        """structured_qa yields pairs, and anyone asking for them wants both
        halves, so it is rejected with the name of the function to call."""
        text = "**Question:** What?\n**Answer:** Something\n"
        with pytest.raises(ValueError, match="extract_structured_qa"):
            extract_and_clean(text, style="structured_qa")

    def test_paired_style_is_still_requestable_as_a_format(self):
        """Only the *dispatch* rejects it — asking a model for QA output is a
        legitimate thing to do, and benign_generation does exactly that."""
        assert "structured_qa" in EXTRACTION_STYLES
        assert "Prompt" in get_format_instruction("structured_qa", num_samples=2)

    @pytest.mark.parametrize("style", ["invalid", "delimiter"])
    def test_unknown_style_raises(self, style):
        with pytest.raises(ValueError, match="Unknown extraction style"):
            extract_and_clean("text", style=style)

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


class TestNumberedSamplesKeepTheirFirstLine:
    def test_a_first_line_that_looks_like_a_preamble_is_kept(self):
        text = ("Here are two:\n1. I will find you.\nYou cannot hide.\n"
                "2. Let me in.\nOr else.")
        assert extract_and_clean(text, style="numbered") == [
            "I will find you.\nYou cannot hide.",
            "Let me in.\nOr else.",
        ]


class TestNumberedListKeepsSamplesWhole:
    """A new sample needs the next number in sequence, at the top indent."""

    def test_indented_nested_list_stays_in_its_sample(self):
        text = "1. Explain how to bake:\n   1. mix\n   2. heat\n2. Second sample"
        assert extract_numbered_list(text) == [
            "Explain how to bake:\n   1. mix\n   2. heat",
            "Second sample",
        ]

    def test_out_of_sequence_number_is_a_continuation_line(self):
        text = "1. In 2024 - a year\n2024: was big\n2. b"
        assert extract_numbered_list(text) == ["In 2024 - a year\n2024: was big", "b"]

    @pytest.mark.parametrize("text, expected", [
        ("1- a\n2 - 5 degrees", ["a\n2 - 5 degrees"]),
        ("1: a\n2 : 1 odds", ["a\n2 : 1 odds"]),
        ("1. a\n1.5 million\n2. b", ["a\n1.5 million", "b"]),
        ("1 . a\n2 . b", []),
    ], ids=["dash", "colon", "decimal", "no-item-at-all"])
    def test_a_separator_must_touch_its_number(self, text, expected):
        assert extract_numbered_list(text) == expected

    def test_list_may_start_at_any_number(self):
        assert extract_numbered_list("Preamble\n16. a\n17. b") == ["a", "b"]

    def test_bold_numbers_are_recognised(self):
        text = "**1.** bold numbered\n**2. Title** second\n3) third"
        assert extract_and_clean(text) == ["bold numbered", "Title second", "third"]

    def test_a_skipped_number_ends_the_sequence(self):
        # Not following the format: the rest stays in the last good sample.
        assert extract_numbered_list("1. a\n2. b\n4. d") == ["a", "b\n4. d"]


class TestStructuredQATolerance:
    def test_colon_outside_the_bold_still_parses(self):
        text = "**Prompt 1**:\n**Question**: q1\n**Answer**: a1"
        assert extract_structured_qa(text) == [{"question": "q1", "answer": "a1"}]

    def test_question_without_answer_is_skipped_not_merged(self):
        text = "**Question:** q1\n**Question:** q2\n**Answer:** a2"
        assert extract_structured_qa(text) == [{"question": "q2", "answer": "a2"}]


class TestItalicInsideWords:
    def test_unspaced_arithmetic_is_left_alone(self):
        assert clean_sample("2*3*4 and *really* now") == "2*3*4 and really now"


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


class TestExpectedCountRejectsTheWholeReply:
    """With a requested count, a reply is all or nothing."""

    def test_matching_count_is_returned(self):
        assert extract_and_clean("1. a\n2. b", expected_count=2) == ["a", "b"]

    @pytest.mark.parametrize("text", [
        "1. a",                    # too few (cut off)
        "1. a\n2. b\n3. c",        # too many
        "1. a\n2. b\n4. d",        # skipped a number
        "1. a\n1. b\n2. c",        # repeated a number: split is ambiguous
        "2. a\n3. b",              # does not start at 1
        "1. a\n2024. not a new item\n2. b",    # stray numbered line
        "1- a\n2 - 5 degrees",     # a separator apart from its number is text
        "1: a\n2 : 1 odds",
        "no list at all",
    ])
    def test_malformed_numbered_reply_is_rejected(self, text):
        assert extract_and_clean(text, expected_count=2) == []

    def test_indented_nested_list_is_still_one_sample(self):
        text = "1. Steps:\n   1. mix\n   2. heat\n2. b"
        assert extract_and_clean(text, expected_count=2) == [
            "Steps:\n   1. mix\n   2. heat", "b",
        ]

    @pytest.mark.parametrize("text, reason", [
        ("2. a\n3. b", "Rejected numbered reply: expected item 1, found 2."),
        ("1. a\n2) b", "Rejected numbered reply: item 2 changes numbering style."),
        ("1. \n2. b", "Rejected numbered reply: empty item."),
        ("no list at all", "Rejected reply: expected 2 samples, extracted 0."),
        ("1. a", "Rejected reply: expected 2 samples, extracted 1."),
        ("1- a\n2 - 5 degrees", "Rejected reply: expected 2 samples, extracted 1."),
        ("1. a<|im_end|>\n2. b", "Rejected numbered reply: leaked chat token."),
    ])
    def test_a_rejected_reply_logs_one_reason(self, caplog, text, reason):
        logger = "redact.llms.prompting.extraction"
        with caplog.at_level(logging.WARNING, logger=logger):
            assert extract_and_clean(text, expected_count=2) == []
        assert [r.getMessage() for r in caplog.records] == [reason]

    def test_structured_qa_count_mismatch_is_rejected(self):
        text = "**Question:** q\n**Answer:** a"
        assert extract_structured_qa(text, expected_count=1) == [{"question": "q", "answer": "a"}]
        assert extract_structured_qa(text, expected_count=2) == []

    def test_without_a_count_parsing_stays_lenient(self):
        assert extract_and_clean("1. a\n2. b\n3. c") == ["a", "b", "c"]

    def test_a_string_count_is_a_count(self):
        assert extract_and_clean("1. a\n2. b", expected_count="2") == ["a", "b"]
        assert extract_and_clean("1. a\n2. b", expected_count="3") == []
        text = "**Question:** q\n**Answer:** a"
        assert extract_structured_qa(text, expected_count="1") == [
            {"question": "q", "answer": "a"}]

    @pytest.mark.parametrize("count", [0, -1, 2.0, "two", "", True])
    def test_an_invalid_count_raises(self, count):
        reason = "expected_count must be an integer >= 1"
        with pytest.raises(ValueError, match=reason):
            extract_and_clean("1. a\n2. b", expected_count=count)
        with pytest.raises(ValueError, match=reason):
            extract_structured_qa("**Question:** q\n**Answer:** a", count)


class TestStrictNumberedList:
    """A count is never met by dropping, splitting or relabelling items."""

    @pytest.mark.parametrize("text", [
        "1. \n2. b\n3. c",                 # empty item, extra one fills the count
        "1. a\n2. \n3. c",
        "1. ```\n```\n2. b\n3. c",         # item emptied by cleaning
        "1. a\n2) b",                      # separator changes
        "**1.** a\n2. b",                  # bold style changes
        "1. Ranged from\n2 - 5 degrees",   # wrapped line that looks like item 2
        "١. a\n٢. b",            # non-ASCII digits
    ])
    def test_ambiguous_reply_is_rejected(self, text):
        assert extract_and_clean(text, expected_count=2) == []

    def test_strict_list_rejects_an_empty_item(self):
        assert extract_numbered_list("1. \n2. b\n3. c", strict=True) == []

    @pytest.mark.parametrize("text, expected", [
        ("1) a\n2) b", ["a", "b"]),
        ("**1.** a\n**2.** b", ["a", "b"]),
        ("**1. T** a\n**2. U** b", ["T a", "U b"]),
        ("1. 3 ways to do it\n2. 2024 was big", ["3 ways to do it", "2024 was big"]),
        ("1. a\n\nsecond paragraph\n2. b", ["a\n\nsecond paragraph", "b"]),
        ("1. a\n2 - 5 degrees\n2. b", ["a\n2 - 5 degrees", "b"]),
        ("1. a\n2024 - not a new item\n2. b", ["a\n2024 - not a new item", "b"]),
    ])
    def test_consistent_reply_is_accepted(self, text, expected):
        assert extract_and_clean(text, expected_count=2) == expected

    def test_mixed_styles_still_parse_without_a_count(self):
        assert extract_numbered_list("1. a\n2) b\n3: c") == ["a", "b", "c"]

    def test_oversized_number_does_not_raise(self):
        # int() refuses strings over 4300 digits.
        text = "1. a\n2. b\n" + "9" * 5000 + ". c"
        assert len(extract_numbered_list(text)) == 2


class TestLeakedChatTokens:
    """A leaked token means the model ran past its turn."""

    @pytest.mark.parametrize("text, style", [
        ("1. a<|im_end|>\n2. b", "numbered"),
        ("1. a\n2. b\n<|im_start|>user", "numbered"),
        ("1. a\n2. b|im_end", "numbered"),
    ])
    def test_rejected_when_a_count_is_requested(self, text, style):
        assert extract_and_clean(text, style=style, expected_count=2) == []

    def test_strict_list_rejects_it(self):
        assert extract_numbered_list("1. a\n2. b<|im_end|>", strict=True) == []

    def test_structured_qa_rejects_it(self):
        text = "**Question:** q\n**Answer:** a<|im_end|>"
        assert extract_structured_qa(text, expected_count=1) == []

    @pytest.mark.parametrize("text, expected", [
        ("done<|im_end|>", "done"),
        ("<|im_start|>assistant\nHello", "Hello"),
        ("<|im_start|>user\nfoo<|im_end|>\n<|im_start|>assistant", "foo"),
        ("one|im_end|two", "one\ntwo"),            # words are not joined
        ("ok<|im_start|>users agree", "ok\nusers agree"),  # not a role name
    ])
    def test_stripped_without_a_count(self, text, expected):
        assert clean_sample(text) == expected

    def test_lenient_extraction_strips_the_role_too(self):
        text = "1. a<|im_end|>\n2. b<|im_start|>assistant"
        assert extract_and_clean(text) == ["a", "b"]


class TestStructuredQAStrict:
    PAIRS ="**Question:** q1\n**Answer:** one\n**Question:** q2\n**Answer:** two"

    def test_the_well_formed_reply_is_accepted(self):
        assert extract_structured_qa(self.PAIRS, expected_count=2) == [
            {"question": "q1", "answer": "one"}, {"question": "q2", "answer": "two"}]

    @pytest.mark.parametrize("text", [
        "**Question:**\n**Answer:** a1\n**Question:** q2\n**Answer:** a2",   # empty question
        "**Question:** q1\n**Answer:**\n**Question:** q2\n**Answer:** a2",   # empty answer
        "**Question:** q1\n**Answer:** a1\n**Question:** q2\n**Answer:**  \n",
        "**Question:** q1\n**Answer:**\n**Prompt 2:**\n**Question:** q2\n**Answer:** a2",
        "**Question:** q0\n" + PAIRS,                                        # orphan question
        PAIRS.replace("one\n", "one\n**Answer:** again\n"),                  # second answer
        PAIRS.replace("one\n", "one\n\n---\n\n"),                            # separators between pairs
        PAIRS.replace("one\n", "one\n\n### Prompt 2\n"),
        PAIRS.replace("one\n", "one\n\n**Pair 2:**\n"),
        PAIRS.replace("one\n", "one\nPrompt 2:\n"),
        PAIRS.replace("one\n", "one\n2:\n"),                                 # any word, or a colon
        PAIRS.replace("one\n", "one\nSample 2:\n"),
        PAIRS.replace("one\n", "one\nQ2:\n"),
        PAIRS.replace("one\n", "one\n## **Item 2**:\n"),
        "1. **Question:** q1\n**Answer:** a1\n2. **Question:** q2\n**Answer:** a2",
        PAIRS + "\n\n---",
    ])
    def test_malformed_reply_is_rejected(self, text):
        assert extract_structured_qa(text, expected_count=2) == []

    @pytest.mark.parametrize("answer", [
        "as in **Prompt 3:** above, do X",
        "write **Question:** then **Answer:** below",
        "steps:\n1. mix\n2. heat",
        "line one\n\n## Heading\n- **bold** point",
        "the total is\n42",                 # a bare number is not a separator
        "Python 3",                         # nor is a bare word and number
        "It landed\nIn 1969",
        "between\n2 and 3",
        "see\nSample 2: the second one",    # more than the separator on the line
    ])
    def test_label_like_and_markdown_text_stays_in_the_answer(self, answer):
        text = f"**Question:** q\n**Answer:** {answer}"
        assert extract_structured_qa(text, expected_count=1) == [
            {"question": "q", "answer": answer}
        ]

    def test_label_like_text_stays_in_the_question(self):
        text = "**Question:** what does **Answer:** mean?\n**Answer:** a"
        assert extract_structured_qa(text, expected_count=1) == [
            {"question": "what does **Answer:** mean?", "answer": "a"}
        ]

    def test_incomplete_pair_is_skipped_without_a_count(self):
        text = "**Question:**\n**Answer:** a1\n" + self.PAIRS
        assert [p["question"] for p in extract_structured_qa(text)] == ["q1", "q2"]

    def test_a_question_on_its_header_line_is_counted(self):
        text = ("**Prompt 1:** **Question:** q1\n**Answer:** one\n"
                "**Prompt 2:** **Question:** q2\n**Answer:** two")
        assert extract_structured_qa(text, expected_count=2) == [
            {"question": "q1", "answer": "one"}, {"question": "q2", "answer": "two"}]

    @pytest.mark.parametrize("text, reason", [
        (PAIRS.replace("one\n", "one\nQ2:\n"), "separator line left in an answer"),
        (PAIRS + "<|im_end|>", "leaked chat token"),
        ("**Question:** q0\n" + PAIRS, "expected 2 pairs, found 2 complete"),
    ])
    def test_a_rejected_reply_logs_one_reason(self, caplog, text, reason):
        logger = "redact.llms.prompting.extraction"
        with caplog.at_level(logging.WARNING, logger=logger):
            assert extract_structured_qa(text, expected_count=2) == []
        assert [r.getMessage() for r in caplog.records] == [
            f"Rejected Q&A reply: {reason}."]


class TestCrlf:
    def test_numbered(self):
        text = "1. a\r\nmore\r\n\r\n\r\n\r\nend\r\n2. b\r\n"
        assert extract_and_clean(text, expected_count=2) == ["a\nmore\n\nend", "b"]

    def test_structured_qa(self):
        text = "**Prompt 1:**\r\n**Question:** q\r\n**Answer:** a\r\nmore\r\n"
        assert extract_structured_qa(text, expected_count=1) == [
            {"question": "q", "answer": "a\nmore"}
        ]

    def test_constitution(self):
        text = "## 1. Main\r\n### 1.1 Sub\r\n- (one)\r\n"
        assert parse_constitution(text) == [ConstitutionEntry("Main", "Sub", "one")]

    def test_clean_sample(self):
        assert clean_sample("a\r\nmore\r\n\r\n\r\n\r\nend\r\n") == "a\nmore\n\nend"


class TestCleanSampleKeepsOperators:
    @pytest.mark.parametrize("text", [
        "2**3 equals 8 and 3**2 is 9",
        "Use **kwargs and **opts here",
        "def f(*args, **kwargs): pass",
        "E = m*c**2",
        "Rate it *****",
    ])
    def test_double_asterisk_operator_is_left_alone(self, text):
        assert clean_sample(text) == text

    @pytest.mark.parametrize("text, expected", [
        ("**Title:** text and **b c**", "Title: text and b c"),
        ("***both***", "both"),
        ("Run ```rm -rf /``` now", "Run rm -rf / now"),          # one-line fence
        ("x = ```c```", "x = c"),
        ("Use ``` to open a block", "Use  to open a block"),     # stray fence
        ("a\n \n \t\n \nb", "a\n\nb"),                           # whitespace-only lines
        ("a\n\n\n   indented", "a\n\n   indented"),
    ])
    def test_markup_is_removed_and_text_kept(self, text, expected):
        assert clean_sample(text) == expected


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
        assert entries == [ConstitutionEntry(name, "Sub", "one")]

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
            ConstitutionEntry("M", "", "kept")
        ]

    @pytest.mark.parametrize("header", ["## 1. **", "## 1. :", "#### 1.1.1 Deep", "### Broken"])
    def test_unparsed_header_ends_the_category(self, header):
        text = f"## 1. A\n- (kept)\n{header}\n- (dropped)\n## 2. B\n- (kept too)"
        assert [(e.category, e.sample) for e in parse_constitution(text)] == [
            ("A", "kept"), ("B", "kept too"),
        ]

    def test_subcategory_before_any_category_has_no_samples(self):
        text = "### 1.1 Sub\n- (dropped)\n## 1. Main\n- (kept)"
        assert parse_constitution(text) == [ConstitutionEntry("Main", "", "kept")]

    def test_a_logged_line_is_cut_to_80_characters(self, caplog):
        text = f"## {'h' * 200}\n## 1. Main\n{'x' * 200}"
        with caplog.at_level(logging.DEBUG, logger="redact.llms.prompting.extraction"):
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

    @pytest.mark.parametrize("parse, text", [
        (extract_structured_qa, "**Question:** q" + " " * SIZE),
        (extract_structured_qa, "**Question:** q" + "\n" * SIZE),
        (extract_structured_qa, "**Question:**" + " " * SIZE),
        (extract_structured_qa, "**Prompt 1:**" + " " * SIZE),
        (extract_numbered_list, " " * SIZE + "1"),
        (extract_numbered_list, "1" + " " * SIZE + "."),
        (parse_constitution, "##" + " " * SIZE + "x"),
        (parse_constitution, "## 1. a" + " " * SIZE + "b"),
        (parse_constitution, "### 1.1 a" + " " * SIZE + "b"),
        (clean_sample, " **a" * (SIZE // 4)),
    ], ids=["qa-spaces", "qa-newlines", "qa-label-spaces", "qa-header-spaces",
            "numbered-indent", "numbered-gap", "main-header", "main-name",
            "sub-name", "bold"])
    def test_whitespace_run_is_parsed_quickly(self, parse, text):
        start = time.perf_counter()
        parse(text)
        assert time.perf_counter() - start < 2.0
