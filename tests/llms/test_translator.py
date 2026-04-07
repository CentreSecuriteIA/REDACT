"""Tests for translation calls with mock backend."""

from tests.conftest import MockBackend
from redact.llms.translator import translate, check_translation, translate_with_check


class TestTranslate:
    def test_calls_backend(self):
        backend = MockBackend("translated text")
        result = translate(backend, "model", "hello", "French")
        assert result == "translated text"
        assert len(backend.calls) == 1
        assert "French" in backend.calls[0]["messages"][1]["content"]

    def test_includes_feedback(self):
        backend = MockBackend("fixed translation")
        translate(backend, "m", "hello", "French", feedback="wrong tone")
        user_msg = backend.calls[0]["messages"][1]["content"]
        assert "wrong tone" in user_msg
        assert "FEEDBACK" in user_msg

    def test_no_feedback_on_first_call(self):
        backend = MockBackend("trans")
        translate(backend, "m", "hello", "French")
        user_msg = backend.calls[0]["messages"][1]["content"]
        assert "FEEDBACK" not in user_msg

    def test_custom_system_prompt(self):
        backend = MockBackend("trans")
        translate(backend, "m", "hi", "German", system_prompt="Custom {language}")
        sys_msg = backend.calls[0]["messages"][0]["content"]
        assert sys_msg == "Custom German"


class TestCheckTranslation:
    def test_accepted(self):
        backend = MockBackend("Yes, looks good")
        accepted, reasoning = check_translation(
            backend, "m", "hello", "bonjour", "French"
        )
        assert accepted is True
        assert reasoning == ""

    def test_rejected(self):
        backend = MockBackend("No, the tone is wrong")
        accepted, reasoning = check_translation(
            backend, "m", "hello", "bad trans", "French"
        )
        assert accepted is False
        assert "tone" in reasoning


class TestTranslateWithCheck:
    def test_success_first_try(self):
        backend = MockBackend(["translated", "Yes"])
        result, info = translate_with_check(
            backend, "gen-m", "check-m", "hello", "French"
        )
        assert result == "translated"
        assert info == "French"

    def test_retries_on_rejection(self):
        backend = MockBackend(["bad trans", "No fix it", "good trans", "Yes"])
        result, info = translate_with_check(
            backend, "m", "m", "hello", "French", num_retries=3
        )
        assert result == "good trans"
        assert info == "French"

    def test_exhausted(self):
        backend = MockBackend(["bad", "No", "bad", "No"])
        result, info = translate_with_check(
            backend, "m", "m", "hello", "Swahili", num_retries=2
        )
        assert "DISCARDED" in info
        assert "Swahili" in info
