"""Tests for translation obfuscation with mock backend."""

from tests.conftest import MockBackend
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


class TestLanguages:
    def test_six_languages(self):
        assert len(LANGUAGES) == 6
        assert "Zulu" in LANGUAGES
        assert "Swahili" in LANGUAGES


class TestTranslationFunctions:
    def test_all_six_returned(self):
        funcs = get_translation_functions()
        assert len(funcs) == 6

    def test_to_zulu(self):
        backend = MockBackend(["translated zulu", "Yes"])
        result, info = to_zulu("hello", backend)
        assert result == "translated zulu"
        assert "Zulu" in info

    def test_to_swahili(self):
        backend = MockBackend(["jambo", "Yes"])
        result, info = to_swahili("hello", backend)
        assert result == "jambo"
        assert "Swahili" in info

    def test_to_scots_gaelic(self):
        backend = MockBackend(["halò", "Yes"])
        result, info = to_scots_gaelic("hello", backend)
        assert "Scots Gaelic" in info

    def test_to_bengali(self):
        backend = MockBackend(["হ্যালো", "Yes"])
        result, info = to_bengali("hello", backend)
        assert "Bengali" in info

    def test_to_thai(self):
        backend = MockBackend(["สวัสดี", "Yes"])
        result, info = to_thai("hello", backend)
        assert "Thai" in info

    def test_to_javanese(self):
        backend = MockBackend(["halo", "Yes"])
        result, info = to_javanese("hello", backend)
        assert "Javanese" in info
