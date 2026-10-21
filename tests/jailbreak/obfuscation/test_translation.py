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


#: A translator that is not registered: a chat model.
CHAT = "local-translator"


def _run(fn, prompt, backend):
    """Drive a translation generator on a chat translator to completion
    against a MockBackend."""
    def call(request):
        return backend.generate([request.messages])[0]
    return drive_sync(fn(prompt, translate_model=CHAT), call)


def _names() -> list[str]:
    return [language["name"] for language in LANGUAGES]


class TestLanguages:
    def test_nineteen_languages(self):
        assert len(LANGUAGES) == 19
        assert "Zulu" in _names()
        assert "Swahili" in _names()

    def test_maori_is_not_among_them(self):
        """TranslateGemma invents Maori terms that pass every check."""
        assert "Maori" not in _names()

    def test_they_are_the_rows_of_the_language_table_marked_translation(self):
        """Not Maori, nor the six answer languages of spec 1.2, whose
        translation waits for a measured length ratio and GlotLID labels."""
        from redact.jailbreak.technique import data_rows

        rows = data_rows("languages.json")
        assert [row for row in rows if row["translation"]] == LANGUAGES
        assert [row["name"] for row in rows if not row["translation"]] == [
            "Maori", "Hmong", "Guarani", "Chinese", "Hindi", "Yoruba", "Amharic"]
        assert [fn.__name__ for fn in get_translation_functions()] == [
            f"to_{language['slug']}" for language in LANGUAGES]


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
        assert self._models(translate_model=CHAT) == [CHAT, CHAT]

    def test_a_registered_chat_translator_translates_and_checks(self):
        assert self._models(translate_model="deepseek-v3.2") == [
            "deepseek-v3.2", "deepseek-v3.2"]

    def test_without_one_the_translation_default_translates_as_a_machine(self):
        from redact.jailbreak.obfuscation.translation import (
            _resolve_translate_model,
            is_machine_translation,
        )

        assert _resolve_translate_model() == "translategemma-27b"
        assert is_machine_translation(_resolve_translate_model())
        request = to_swahili("hello", gen_model="gen").send(None)
        assert request.model == "translategemma-27b"
        assert request.messages[0]["content"].startswith("<<<source>>>en")

    def test_without_the_role_the_fallback_is_the_local_translator(self, monkeypatch):
        from redact.jailbreak.obfuscation import translation

        def no_role(role):
            raise KeyError(role)

        monkeypatch.setattr(translation, "default_model_for_role", no_role)
        assert translation._resolve_translate_model() == "translategemma-27b"


class TestMachineTranslationTables:
    """The code, accepted labels and median length ratio of every language,
    keyed by its slug in ``languages.json``."""

    def test_every_translation_language_of_the_table_has_a_code_labels_and_a_ratio(
            self):
        """A row added to ``languages.json`` with ``translation`` true and no
        measurements fails here, as the import does."""
        from redact.jailbreak.obfuscation.translation import (
            ACCEPTED_LABELS,
            LANGUAGE_CODES,
            MEDIAN_RATIO,
        )
        from redact.jailbreak.technique import data_rows

        slugs = [row["slug"] for row in data_rows("languages.json")
                 if row["translation"]]
        for table in (LANGUAGE_CODES, ACCEPTED_LABELS, MEDIAN_RATIO):
            assert list(table) == slugs

    def test_a_language_s_script_is_its_glotlid_labels_script(self):
        from redact.jailbreak.obfuscation.translation import ACCEPTED_LABELS

        for language in LANGUAGES:
            assert {label.split("_")[1] for label in ACCEPTED_LABELS[language["slug"]]
                    } == {language["script"]}
        assert sorted(language["name"] for language in LANGUAGES
                      if language["script"] != "Latn") == [
            "Arabic", "Bengali", "Greek", "Japanese", "Khmer", "Nepali", "Russian",
            "Thai"]

    def test_a_language_s_technique_carries_its_script(self):
        assert (to_thai.script, to_swahili.script) == ("Thai", "Latn")

    def test_codes_are_iso_639_1(self):
        from redact.jailbreak.obfuscation.translation import LANGUAGE_CODES

        assert LANGUAGE_CODES["scots_gaelic"] == "gd"
        assert LANGUAGE_CODES["khmer"] == "km"
        assert all(len(code) == 2 and code.islower()  # noqa: PLR2004
                   for code in LANGUAGE_CODES.values())
        assert len(set(LANGUAGE_CODES.values())) == len(LANGUAGES)

    def test_javanese_never_accepts_indonesian(self):
        """The pod's two real wrong-language outputs were Javanese answered
        in Indonesian."""
        from redact.jailbreak.obfuscation.translation import ACCEPTED_LABELS

        assert ACCEPTED_LABELS["javanese"] == {"jav_Latn"}
        assert {"swh_Latn", "swc_Latn"} <= ACCEPTED_LABELS["swahili"]
        assert "xho_Latn" in ACCEPTED_LABELS["zulu"]

    def test_a_table_missing_a_language_is_refused(self):
        import pytest

        from redact.jailbreak.obfuscation.translation import (
            LANGUAGE_CODES,
            _check_tables,
        )

        without_thai = {k: v for k, v in LANGUAGE_CODES.items() if k != "thai"}
        with pytest.raises(ValueError, match=r"LANGUAGE_CODES.*missing \['thai'\]"):
            _check_tables({"LANGUAGE_CODES": without_thai})
        with pytest.raises(ValueError, match=r"extra \['maori'\]"):
            _check_tables({"LANGUAGE_CODES": {**LANGUAGE_CODES, "maori": "mi"}})

    def test_every_language_has_a_measured_median(self):
        """TranslateGemma on FLORES-200: no 1.0 placeholder is left, and
        Japanese, whose placeholder rejected 634 of 1,012 good translations,
        lies below 0.6."""
        from redact.jailbreak.obfuscation.translation import MEDIAN_RATIO

        assert 1.0 not in MEDIAN_RATIO.values()
        assert MEDIAN_RATIO["japanese"] < 0.6  # noqa: PLR2004

    def test_the_measured_medians(self):
        from redact.jailbreak.obfuscation.translation import (
            LENGTH_RATIO_BAND,
            MEDIAN_RATIO,
        )

        assert MEDIAN_RATIO == {
            "french": 1.27, "japanese": 0.47, "russian": 1.18, "spanish": 1.22,
            "german": 1.24, "arabic": 0.96, "turkish": 1.13, "czech": 1.05,
            "vietnamese": 1.16, "greek": 1.28, "croatian": 1.07, "swahili": 1.16,
            "thai": 1.02, "khmer": 1.09, "nepali": 1.04, "zulu": 1.17,
            "scots_gaelic": 1.31, "bengali": 1.07, "javanese": 1.12,
        }
        assert LENGTH_RATIO_BAND == (0.5, 2.0)

    def test_a_japanese_translation_at_its_median_passes(self, monkeypatch):
        from redact.jailbreak.obfuscation import langid
        from redact.jailbreak.obfuscation.translation import check_translation

        monkeypatch.setattr(langid, "detect", lambda text: "jpn_Jpan")
        assert check_translation("訳" * 47, "x" * 100, "japanese") is None

    def test_the_reply_cap_is_about_three_times_the_inputs_tokens(self):
        from redact.jailbreak.obfuscation.translation import reply_cap

        assert [reply_cap("x" * n) for n in (81, 161, 400, 1500, 1505, 9000)] == [
            113, 194, 434, 1532, 1536, 1536]


class TestMachineTranslationPrompt:
    """jailbreak/translate_mt: the model's wire format, one user message."""

    def test_it_renders_to_one_user_message_and_no_system_prompt(self):
        from redact.llms.prompting import build_messages, load_prompt

        config = load_prompt("jailbreak", "translate_mt")
        assert not {"system_prompt", "few_shot_examples"} & set(config)
        assert config["seed_fields"] == ["source_code", "target_code", "text"]
        text = "a {brace} and\n\ntwo lines"
        assert build_messages(config, source_code="en", target_code="sw",
                              text=text) == [{
            "role": "user", "content": f"<<<source>>>en<<<target>>>sw<<<text>>>{text}",
        }]

    def test_an_overlay_of_it_is_reported(self, tmp_path):
        import json

        from redact.llms.prompting import report_prompt_sources

        folder = tmp_path / "jailbreak" / "translate_mt"
        folder.mkdir(parents=True)
        (folder / "template.json").write_text(
            json.dumps({"template": "{text}"}), encoding="utf-8")
        overridden = report_prompt_sources(tmp_path)["overridden"]
        assert overridden == ["jailbreak/translate_mt"]


class TestBackTranslationJudgePrompt:
    """jailbreak/translate_check_back: the generation model judges the
    back-translation of a machine translation."""

    def test_it_renders_from_the_three_placeholders(self):
        from redact.llms.prompting import build_messages, load_prompt

        config = load_prompt("jailbreak", "translate_check_back")
        assert config["seed_fields"] == ["original", "back_translation", "language"]
        messages = build_messages(config, original="ORIGINAL TEXT",
                                  back_translation="BACK TEXT", language="Swahili")
        assert [m["role"] for m in messages] == ["system", "user"]
        rendered = "\n".join(m["content"] for m in messages)
        for value in ("ORIGINAL TEXT", "BACK TEXT", "Swahili"):
            assert value in rendered
        assert "{" not in rendered

    def test_it_asks_for_the_verdict_first(self):
        """is_accepted reads the reply's first word."""
        from redact.llms.prompting import load_prompt

        config = load_prompt("jailbreak", "translate_check_back")
        text = config["system_prompt"] + config["template"]
        assert "Yes" in text and "No" in text
