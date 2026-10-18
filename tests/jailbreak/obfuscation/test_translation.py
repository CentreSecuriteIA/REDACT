"""Tests for translation obfuscation generators with a mock backend.

Translation techniques are now technique generators (they yield LLMRequests).
These tests drive them via ``redact.llm_pipeline.drive_sync``, routing each
yielded request to a MockBackend — the same way ``chain.apply_combination``
runs a single sample.
"""

from redact.jailbreak.obfuscation.translation import (
    LANGUAGES,
    get_translation_functions,
    to_bengali,
    to_javanese,
    to_scots_gaelic,
    to_swahili,
    to_thai,
    to_zulu,
)
from redact.llm_pipeline import drive_sync
from tests.conftest import MockBackend


def _run(fn, prompt, backend):
    """Drive a translation generator to completion against a MockBackend."""
    def call(request):
        return backend.generate([request.messages])[0]
    return drive_sync(fn(prompt), call)


class TestLanguages:
    def test_nineteen_languages(self):
        assert len(LANGUAGES) == 19
        assert "Zulu" in LANGUAGES
        assert "Swahili" in LANGUAGES

    def test_maori_is_not_among_them(self):
        """TranslateGemma invents Maori terms that pass every check."""
        assert "Maori" not in LANGUAGES


class TestTranslationFunctions:
    def test_all_nineteen_returned(self):
        funcs = get_translation_functions()
        assert len(funcs) == 19

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


class TestTranslationModel:
    def _models(self, **kwargs) -> list[str]:
        """The model of each request of one translation that is accepted."""
        gen = to_swahili("hello", **kwargs)
        models = [gen.send(None).model]
        models.append(gen.send("jambo").model)
        try:
            gen.send("Yes")
        except StopIteration:
            return models
        raise AssertionError("the translation made more than two calls")

    def test_the_runs_translation_model_translates_and_checks(self):
        assert self._models(translate_model="local-translator") == [
            "local-translator", "local-translator"]

    def test_without_one_both_calls_go_to_the_translation_role(self):
        from redact.jailbreak.obfuscation.translation import _resolve_translate_model

        role = _resolve_translate_model()
        assert self._models() == [role, role]


class TestMachineTranslationTables:
    """The code, accepted labels and median length ratio of every language."""

    def test_every_table_names_exactly_the_languages(self):
        from redact.jailbreak.obfuscation.translation import (
            ACCEPTED_LABELS,
            LANGUAGE_CODES,
            MEDIAN_RATIO,
        )

        for table in (LANGUAGE_CODES, ACCEPTED_LABELS, MEDIAN_RATIO):
            assert list(table) == LANGUAGES

    def test_codes_are_iso_639_1(self):
        from redact.jailbreak.obfuscation.translation import LANGUAGE_CODES

        assert LANGUAGE_CODES["Scots Gaelic"] == "gd"
        assert LANGUAGE_CODES["Khmer"] == "km"
        assert all(len(code) == 2 and code.islower()  # noqa: PLR2004
                   for code in LANGUAGE_CODES.values())
        assert len(set(LANGUAGE_CODES.values())) == len(LANGUAGES)

    def test_javanese_never_accepts_indonesian(self):
        """The pod's two real wrong-language outputs were Javanese answered
        in Indonesian."""
        from redact.jailbreak.obfuscation.translation import ACCEPTED_LABELS

        assert ACCEPTED_LABELS["Javanese"] == {"jav_Latn"}
        assert {"swh_Latn", "swc_Latn"} <= ACCEPTED_LABELS["Swahili"]
        assert "xho_Latn" in ACCEPTED_LABELS["Zulu"]

    def test_a_table_missing_a_language_is_refused(self):
        import pytest

        from redact.jailbreak.obfuscation.translation import (
            LANGUAGE_CODES,
            _check_tables,
        )

        without_thai = {k: v for k, v in LANGUAGE_CODES.items() if k != "Thai"}
        with pytest.raises(ValueError, match=r"LANGUAGE_CODES.*missing \['Thai'\]"):
            _check_tables({"LANGUAGE_CODES": without_thai})
        with pytest.raises(ValueError, match=r"extra \['Maori'\]"):
            _check_tables({"LANGUAGE_CODES": {**LANGUAGE_CODES, "Maori": "mi"}})

    def test_the_reply_cap_is_about_three_times_the_inputs_tokens(self):
        from redact.jailbreak.obfuscation.translation import reply_cap

        assert [reply_cap("x" * n) for n in (81, 161, 400, 1500, 1505, 9000)] == [
            113, 194, 434, 1532, 1536, 1536]
