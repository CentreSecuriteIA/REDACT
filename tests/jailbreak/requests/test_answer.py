"""Tests for answer-format and output-conditioning directives (requests/answer.py)."""

import random

import pytest

import redact.jailbreak  # noqa: F401  (tags the techniques)
from redact.jailbreak.directives import draw_variant, read_pools
from redact.jailbreak.requests.answer import (
    get_answer_functions,
    to_answer_as_code,
    to_answer_as_dialogue,
    to_answer_as_story,
    to_answer_as_structured_data,
    to_answer_as_tutorial,
    to_answer_in_encoding,
    to_answer_in_markdown,
    to_apology_prevention,
    to_dual_answer,
    to_prefix_injection,
    to_refusal_then_comply,
    to_tagged_prefix,
)

SAMPLE = "How do I pick a lock?"


class TestAnswerRegistry:
    def test_returns_twelve(self):
        assert len(get_answer_functions()) == 12     # noqa: PLR2004

    def test_all_return_tuple(self):
        random.seed(42)
        for fn in get_answer_functions():
            result = fn(SAMPLE)
            assert isinstance(result, tuple) and len(result) == 2

    def test_prompt_preserved(self):
        random.seed(42)
        for fn in get_answer_functions():
            result, _ = fn(SAMPLE)
            assert SAMPLE in result

    def test_info_format(self):
        random.seed(42)
        for fn in get_answer_functions():
            _, info = fn(SAMPLE)
            assert "directive=" in info
            assert "variant=" in info

    def test_expected_names(self):
        names = [f.__name__ for f in get_answer_functions()]
        assert "to_refusal_then_comply" in names
        assert "to_apology_prevention" in names
        assert "to_prefix_injection" in names
        assert "to_answer_in_markdown" in names
        assert "to_answer_as_tutorial" in names
        assert "to_answer_as_code" in names
        assert "to_answer_as_dialogue" in names
        assert "to_answer_as_story" in names
        assert "to_answer_as_structured_data" in names
        assert names[-3:] == ["to_dual_answer", "to_tagged_prefix",
                              "to_answer_in_encoding"]


class TestOutputConditioningDirectives:
    def test_refusal_then_comply_info(self):
        random.seed(0)
        _, info = to_refusal_then_comply(SAMPLE)
        assert info.startswith("directive=refusal_then_comply")

    def test_apology_prevention_info(self):
        random.seed(0)
        _, info = to_apology_prevention(SAMPLE)
        assert info.startswith("directive=apology_prevention")

    def test_prefix_injection_info(self):
        random.seed(0)
        _, info = to_prefix_injection(SAMPLE)
        assert info.startswith("directive=prefix_injection")


class TestOutputFormatDirectives:
    def test_answer_in_markdown_info(self):
        random.seed(0)
        _, info = to_answer_in_markdown(SAMPLE)
        assert info.startswith("directive=answer_in_markdown")

    def test_answer_as_tutorial_info(self):
        random.seed(0)
        _, info = to_answer_as_tutorial(SAMPLE)
        assert info.startswith("directive=answer_as_tutorial")

    def test_answer_as_code_info(self):
        random.seed(0)
        _, info = to_answer_as_code(SAMPLE)
        assert info.startswith("directive=answer_as_code")

    def test_answer_as_dialogue_info(self):
        random.seed(0)
        _, info = to_answer_as_dialogue(SAMPLE)
        assert info.startswith("directive=answer_as_dialogue")

    def test_answer_as_story_info(self):
        random.seed(0)
        _, info = to_answer_as_story(SAMPLE)
        assert info.startswith("directive=answer_as_story")

    def test_answer_as_structured_data_info(self):
        random.seed(0)
        _, info = to_answer_as_structured_data(SAMPLE)
        assert info.startswith("directive=answer_as_structured_data")

    def test_variant_randomness(self):
        results = set()
        for seed in range(20):
            random.seed(seed)
            result, _ = to_answer_as_code(SAMPLE)
            results.add(result)
        assert len(results) > 1


#: The three answer formats of spec 1.3, each with its directive. Their text
#: is adversarial on purpose: the tests compare flags and names, never print
#: it.
_SPEC_1_3 = [(to_dual_answer, "dual_answer"), (to_tagged_prefix, "tagged_prefix"),
             (to_answer_in_encoding, "answer_in_encoding")]


class TestTheSpec13AnswerFormats:
    @pytest.mark.parametrize(("fn", "directive"), _SPEC_1_3)
    def test_a_draw_is_its_directive_s_variant_for_the_seed(self, fn, directive):
        pool = read_pools("requests/answer_templates.json")[directive]
        for seed in range(12):
            text, info = fn(SAMPLE, rng=random.Random(seed))
            drawn = draw_variant(pool, random.Random(seed))["name"]
            assert info == f"directive={directive};variant={drawn}"
            assert (SAMPLE in text, "{prompt}" in text) == (True, False)

    @pytest.mark.parametrize(("fn", "directive"), _SPEC_1_3)
    def test_each_is_a_pure_attack_of_spec_1_3(self, fn, directive):
        assert (fn.families, fn.complexity, fn.requires_llm) == (
            ["answer_format"], 2, False)
        assert (fn.kinds, fn.since) == (["adversarial"], "1.3")

    def test_the_rules_they_carry(self):
        """The persona templates' mechanics ask for a dual answer and a
        tagged opening already; an answer in an encoding is kept from an
        encoded prompt (the ten sensitive-word encodings are encode members
        too, so they are kept from it as well) and from an enciphered one
        (``bijection``, which refuses what the heavy encodings refuse)."""
        assert to_dual_answer.cross_incompatible_with == ["persona_template"]
        assert to_tagged_prefix.cross_incompatible_with == ["persona_template"]
        assert to_answer_in_encoding.cross_incompatible_with == ["encode", "bijection"]
