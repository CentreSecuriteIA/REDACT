"""Tests for jailbreak technique composition utilities."""

import random

from redact.jailbreak.utils import (
    combine_techniques,
    _parse_rejection_info,
    apply_combination,
    sample_combination,
    sample_exact_combination,
    default_escalation_schedule,
)


def _tech(name, layer, families, complexity=1, encode_weight=None):
    """Build a fake tagged technique function (mirrors tag_all_functions output)."""
    def fn(text, **kwargs):
        return text, name
    fn.__name__ = name
    fn.layer = layer
    fn.families = families
    fn.complexity = complexity
    fn.encode_weight = encode_weight
    fn.within_layer_order = 0
    return fn


class TestCombineTechniques:
    def test_single_technique(self):
        def tech1(text, **kwargs):
            return text.upper(), "upper"

        combined = combine_techniques(tech1)
        result, info = combined("hello")
        assert result == "HELLO"
        assert info == "upper"

    def test_two_techniques(self):
        def tech1(text, **kwargs):
            return text + "!", "exclaim"

        def tech2(text, **kwargs):
            return text.upper(), "upper"

        combined = combine_techniques(tech1, tech2)
        result, info = combined("hello")
        assert result == "HELLO!"
        assert info == "exclaim;upper"

    def test_noop_recorded(self):
        """No-op steps (text unchanged) are recorded as 'noop=<name>' in info."""
        def tech1(text, **kwargs):
            return text, ""

        def tech2(text, **kwargs):
            return text + "x", "added_x"

        combined = combine_techniques(tech1, tech2)
        _, info = combined("hi")
        assert info == "noop=tech1;added_x"

    def test_kwargs_passed_through(self):
        received = {}

        def tech1(text, **kwargs):
            received.update(kwargs)
            return text, ""

        combined = combine_techniques(tech1)
        combined("hi", backend="mock", model="test")
        assert received["backend"] == "mock"
        assert received["model"] == "test"

    def test_name_combined(self):
        def a(text, **kwargs):
            return text, ""

        def b(text, **kwargs):
            return text, ""

        combined = combine_techniques(a, b)
        assert combined.__name__ == "a+b"

    def test_three_techniques_chained(self):
        def add_a(text, **kwargs):
            return text + "a", "a"

        def add_b(text, **kwargs):
            return text + "b", "b"

        def add_c(text, **kwargs):
            return text + "c", "c"

        combined = combine_techniques(add_a, add_b, add_c)
        result, info = combined("")
        assert result == "abc"
        assert info == "a;b;c"


class TestParseRejectionInfo:
    def test_accepted_empty_info(self):
        accepted, reasoning = _parse_rejection_info("")
        assert accepted is True
        assert reasoning == ""

    def test_accepted_normal_info(self):
        accepted, reasoning = _parse_rejection_info("language=Swahili")
        assert accepted is True
        assert reasoning == ""

    def test_accepted_noop_info(self):
        accepted, reasoning = _parse_rejection_info("noop=to_rot13")
        assert accepted is True
        assert reasoning == ""

    def test_rejected_bare_discarded(self):
        accepted, reasoning = _parse_rejection_info("DISCARDED")
        assert accepted is False
        assert reasoning == "DISCARDED"

    def test_rejected_with_feedback(self):
        info = "DISCARDED; language=Scots Gaelic; feedback=translation refused"
        accepted, reasoning = _parse_rejection_info(info)
        assert accepted is False
        assert reasoning == info

    def test_rejected_with_technique_prefix(self):
        info = "DISCARDED; technique=to_scots_gaelic; feedback=refused"
        accepted, reasoning = _parse_rejection_info(info)
        assert accepted is False
        assert reasoning == info


class TestCombineTechniquesEarlyExit:
    def test_early_exit_on_discarded(self):
        """Technique chain stops when a step produces DISCARDED output."""
        called = []

        def failing_tech(text, **kwargs):
            called.append("failing")
            return text, "DISCARDED; feedback=refused"

        def should_not_run(text, **kwargs):
            called.append("should_not_run")
            return text + "_modified", "modified"

        combined = combine_techniques(failing_tech, should_not_run)
        result, info = combined("original")

        assert "should_not_run" not in called
        assert info.startswith("DISCARDED")
        assert "failing_tech" in info
        assert result == "original"  # text unchanged since failing_tech returned original

    def test_continues_after_success(self):
        """Technique chain continues normally when no DISCARDED output."""
        def tech1(text, **kwargs):
            return text + "_1", "step1"

        def tech2(text, **kwargs):
            return text + "_2", "step2"

        combined = combine_techniques(tech1, tech2)
        result, info = combined("x")
        assert result == "x_1_2"
        assert "DISCARDED" not in info

    def test_early_exit_preserves_pre_failure_result(self):
        """Text state before the failed technique is returned, not the failed output."""
        def tech1(text, **kwargs):
            return text + "_modified", "ok"

        def tech2(text, **kwargs):
            return "garbage", "DISCARDED; feedback=bad"

        combined = combine_techniques(tech1, tech2)
        result, info = combined("start")
        assert result == "start_modified"  # state before tech2 ran
        assert info.startswith("DISCARDED")


class TestApplyCombination:
    def test_returns_four_tuple_on_success(self):
        def pure_tech(text):
            return text.upper(), "upper"

        # combine_techniques wraps pure functions to accept **kwargs
        fn = combine_techniques(pure_tech)
        result, info, accepted, reasoning = apply_combination(fn, "hello")
        assert result == "HELLO"
        assert info == "upper"
        assert accepted is True
        assert reasoning == ""

    def test_returns_four_tuple_on_rejection(self):
        def failing_tech(text, **kwargs):
            return text, "DISCARDED; feedback=refused"

        failing_tech.techniques = [failing_tech]

        result, info, accepted, reasoning = apply_combination(failing_tech, "hello")
        assert accepted is False
        assert reasoning.startswith("DISCARDED")
        assert result == "hello"  # original preserved

    def test_accepted_false_propagates_from_combined(self):
        """apply_combination correctly detects rejection inside a combined function."""
        def failing(text, **kwargs):
            return text, "DISCARDED; feedback=bad translation"

        def unused(text, **kwargs):
            return text + "_extra", "extra"

        combined = combine_techniques(failing, unused)
        result, info, accepted, reasoning = apply_combination(combined, "test")
        assert accepted is False
        assert "DISCARDED" in reasoning
        assert "_extra" not in result  # unused technique was skipped


class TestSampleExactCombination:
    def _obfuscation_pool(self):
        # Three distinct obfuscation families, all high-complexity, so a budget
        # would block stacking — but the exact path ignores complexity.
        return [
            _tech("t_encode", "obfuscation", ["encode"], complexity=3),
            _tech("t_structural", "obfuscation", ["structural"], complexity=3),
            _tech("t_suffix", "obfuscation", ["suffixes"], complexity=1),
        ]

    def test_exactly_one(self):
        fn = sample_exact_combination(random.Random(0), self._obfuscation_pool(), 1)
        assert len(fn.techniques) == 1

    def test_exactly_two(self):
        fn = sample_exact_combination(random.Random(0), self._obfuscation_pool(), 2)
        assert len(fn.techniques) == 2

    def test_ignores_complexity(self):
        """High-complexity techniques still stack — count is the only constraint."""
        # 2 x complexity-3 = 6; a budget run would often reject the second pick.
        for seed in range(10):
            fn = sample_exact_combination(random.Random(seed), self._obfuscation_pool(), 2)
            assert len(fn.techniques) == 2

    def test_distinct_families(self):
        """The two picks come from different families (family dedup)."""
        fn = sample_exact_combination(random.Random(3), self._obfuscation_pool(), 2)
        fams = [t.families[0] for t in fn.techniques]
        assert len(set(fams)) == 2

    def test_layer_cap_best_effort(self):
        """Layer cap (1 hacking) means n=2 from a hacking-only pool yields 1."""
        pool = [
            _tech("h1", "hacking", ["framing"]),
            _tech("h2", "hacking", ["persona"]),
        ]
        fn = sample_exact_combination(random.Random(0), pool, 2)
        assert len(fn.techniques) == 1  # second hacking pick blocked by layer cap

    def test_include_flags_filter_pool(self):
        """include_* restrict the initial candidate pool by layer."""
        pool = self._obfuscation_pool() + [_tech("h1", "hacking", ["framing"])]
        fn = sample_exact_combination(
            random.Random(0), pool, 1, include_obfuscation=False
        )
        # Only the hacking technique was eligible.
        assert [t.__name__ for t in fn.techniques] == ["h1"]

    def test_reproducible(self):
        p = self._obfuscation_pool()
        a = sample_exact_combination(random.Random(7), p, 2)
        b = sample_exact_combination(random.Random(7), p, 2)
        assert [t.__name__ for t in a.techniques] == [t.__name__ for t in b.techniques]

    def test_zero_returns_identity(self):
        fn = sample_exact_combination(random.Random(0), self._obfuscation_pool(), 0)
        assert fn.techniques == []
        assert fn.__name__ == "identity"

    def test_sample_combination_dispatches(self):
        """sample_combination(exact_techniques=2) delegates to the exact path."""
        fn = sample_combination(
            random.Random(0), self._obfuscation_pool(), exact_techniques=2,
        )
        assert len(fn.techniques) == 2


class TestDefaultEscalationSchedule:
    def test_four_rounds(self):
        sched = default_escalation_schedule()
        assert len(sched) == 4

    def test_first_two_are_count_driven(self):
        sched = default_escalation_schedule()
        assert sched[0] == {"exact_techniques": 1}
        assert sched[1] == {"exact_techniques": 2}

    def test_last_two_are_budget_driven(self):
        sched = default_escalation_schedule()
        for r in sched[2:]:
            assert "exact_techniques" not in r
            assert "max_complexity" in r
        # Complexity escalates between the two budget rounds.
        assert sched[3]["max_complexity"] > sched[2]["max_complexity"]
