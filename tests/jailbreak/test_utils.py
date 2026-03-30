"""Tests for jailbreak technique composition utilities."""

from redact.jailbreak.utils import combine_techniques


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

    def test_empty_info_skipped(self):
        def tech1(text, **kwargs):
            return text, ""

        def tech2(text, **kwargs):
            return text + "x", "added_x"

        combined = combine_techniques(tech1, tech2)
        _, info = combined("hi")
        assert info == "added_x"

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
