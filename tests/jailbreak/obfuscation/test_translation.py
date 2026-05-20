"""Tests for translation obfuscation generators with a mock backend.

Translation techniques are now technique generators (they yield LLMRequests).
These tests drive them via ``protocol.run_sync``, routing each yielded request
to a MockBackend — the same way ``utils.apply_combination`` runs a single
sample.
"""

from tests.conftest import MockBackend
from redact.jailbreak.protocol import run_sync
from redact.jailbreak.obfuscation.translation import (
    LANGUAGES,
    to_zulu,
    to_scots_gaelic,
    to_bengali,
    to_swahili,
    to_thai,
    to_javanese,
    get_translation_functions,
)


def _run(fn, prompt, backend):
    """Drive a translation generator to completion against a MockBackend."""
    def call(request):
        return backend.generate(request.messages, request.model)
    return run_sync(fn(prompt), call)


class TestLanguages:
    def test_twenty_languages(self):
        assert len(LANGUAGES) == 20
        assert "Zulu" in LANGUAGES
        assert "Swahili" in LANGUAGES


class TestTranslationFunctions:
    def test_all_twenty_returned(self):
        funcs = get_translation_functions()
        assert len(funcs) == 20

    def test_to_zulu(self):
        backend = MockBackend(["translated zulu", "Yes"])
        result, info = _run(to_zulu, "hello", backend)
        assert result == "translated zulu"
        assert "Zulu" in info

    def test_to_swahili(self):
        backend = MockBackend(["jambo", "Yes"])
        result, info = _run(to_swahili, "hello", backend)
        assert result == "jambo"
        assert "Swahili" in info

    def test_to_scots_gaelic(self):
        backend = MockBackend(["halò", "Yes"])
        result, info = _run(to_scots_gaelic, "hello", backend)
        assert "Scots Gaelic" in info

    def test_to_bengali(self):
        backend = MockBackend(["হ্যালো", "Yes"])
        result, info = _run(to_bengali, "hello", backend)
        assert "Bengali" in info

    def test_to_thai(self):
        backend = MockBackend(["สวัสดี", "Yes"])
        result, info = _run(to_thai, "hello", backend)
        assert "Thai" in info

    def test_to_javanese(self):
        backend = MockBackend(["halo", "Yes"])
        result, info = _run(to_javanese, "hello", backend)
        assert "Javanese" in info

    def test_retries_then_discards(self):
        # Checker always rejects -> exhausts retries -> DISCARDED.
        backend = MockBackend(["bad", "No", "bad", "No", "bad", "No", "bad", "No"])
        result, info = _run(to_swahili, "hello", backend)
        assert info.startswith("DISCARDED")
        assert "Swahili" in info
