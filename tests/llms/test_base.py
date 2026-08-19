"""Tests for shared LLMBackend helpers (base.py)."""

from redact.llms.base import fold_system_into_first_message


class TestFoldSystemIntoFirstMessage:
    def test_noop_when_no_system_message(self):
        messages = [{"role": "user", "content": "hi"}]
        assert fold_system_into_first_message(messages) is messages

    def test_folds_system_into_first_user_message(self):
        messages = [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "Hello"},
        ]
        result = fold_system_into_first_message(messages)
        assert result == [{"role": "user", "content": "Be terse.\n\nHello"}]

    def test_multiple_system_messages_merged_in_order(self):
        messages = [
            {"role": "system", "content": "Rule 1."},
            {"role": "system", "content": "Rule 2."},
            {"role": "user", "content": "Hi"},
        ]
        result = fold_system_into_first_message(messages)
        assert result == [{"role": "user", "content": "Rule 1.\n\nRule 2.\n\nHi"}]

    def test_only_system_messages_become_single_user_message(self):
        messages = [{"role": "system", "content": "Just a system prompt."}]
        result = fold_system_into_first_message(messages)
        assert result == [{"role": "user", "content": "Just a system prompt."}]

    def test_preserves_later_messages_untouched(self):
        messages = [
            {"role": "system", "content": "Sys."},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ]
        result = fold_system_into_first_message(messages)
        assert result == [
            {"role": "user", "content": "Sys.\n\nQ1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ]

    def test_does_not_mutate_input(self):
        messages = [
            {"role": "system", "content": "Sys."},
            {"role": "user", "content": "Q"},
        ]
        original_first = messages[1]
        fold_system_into_first_message(messages)
        assert messages[1] is original_first
        assert messages[1]["content"] == "Q"
