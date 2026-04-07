"""Tests for meta-prompt generation functions."""

from tests.conftest import MockBackend
from redact.content_moderation.metaprompt import (
    generate_category_description,
    generate_seeds,
)


class TestGenerateCategoryDescription:
    def test_returns_description(self):
        backend = MockBackend("A detailed description of the violence category.")
        result = generate_category_description(backend, "model", "Violence")
        assert "violence" in result.lower()

    def test_retry_on_empty(self):
        backend = MockBackend(["", "   ", "Real description here"])
        result = generate_category_description(backend, "model", "Test", max_retries=3)
        assert result == "Real description here"

    def test_fallback_to_category_name(self):
        backend = MockBackend(["", ""])
        result = generate_category_description(backend, "model", "Fallback", max_retries=2)
        assert result == "Fallback"


class TestGenerateSeeds:
    def test_returns_numbered_list(self):
        backend = MockBackend("1. Seed one\n2. Seed two\n3. Seed three")
        result = generate_seeds(
            backend, "model", "Violence", "Description of violence"
        )
        assert "1." in result
        assert "Seed one" in result

    def test_retry_on_empty(self):
        backend = MockBackend(["No format here", "1. Real seed"])
        result = generate_seeds(
            backend, "model", "Test", "desc", max_retries=3
        )
        assert "Real seed" in result

    def test_fallback_to_empty(self):
        backend = MockBackend("No numbered items at all")
        result = generate_seeds(
            backend, "model", "Test", "desc", max_retries=1
        )
        assert result == ""
