"""Tests for translation obfuscation generators with a mock backend.

Translation techniques are now technique generators (they yield LLMRequests).
These tests drive them via ``redact.llm_pipeline.drive_sync``, routing each
yielded request to a MockBackend — the same way ``chain.apply_combination``
runs a single sample.
"""

import pytest

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
    def test_twenty_three_languages(self):
        assert len(LANGUAGES) == 23     # noqa: PLR2004
        assert "Zulu" in _names()
        assert "Swahili" in _names()

    def test_maori_is_not_among_them(self):
        """TranslateGemma invents Maori terms that pass every check."""
        assert "Maori" not in _names()

    def test_they_are_the_rows_of_the_language_table_marked_translation(self):
        """Not Maori, nor Guarani and Yoruba, the two answer languages of spec
        1.2 whose translation measured too poorly (TranslateGemma loops on a
        fifth of Guarani's sentences, and the judge rejects 41% of Yoruba's)."""
        from redact.jailbreak.technique import data_rows

        rows = data_rows("languages.json")
        assert [row for row in rows if row["translation"]] == LANGUAGES
        assert [row["name"] for row in rows if not row["translation"]] == [
            "Maori", "Guarani", "Yoruba"]
        assert [fn.__name__ for fn in get_translation_functions()] == [
            f"to_{language['slug']}" for language in LANGUAGES]


class TestTranslationFunctions:
    def test_all_twenty_three_returned(self):
        funcs = get_translation_functions()
        assert len(funcs) == 23     # noqa: PLR2004

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

    def test_the_stages_check_model_does_not_check_a_translation(self):
        """The stage's check model checks the other techniques; the chat
        translator still checks its own translation."""
        assert self._models(translate_model=CHAT, check_model="check") == [
            CHAT, CHAT]
        assert self._models(translate_model=CHAT,
                            translate_check_model="check") == [CHAT, "check"]

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
            "Amharic", "Arabic", "Bengali", "Chinese", "Greek", "Hindi", "Japanese",
            "Khmer", "Nepali", "Russian", "Thai"]

    def test_a_language_s_technique_carries_its_script(self):
        assert (to_thai.script, to_swahili.script) == ("Thai", "Latn")

    def test_codes_are_iso_639_1_but_hmong_s_which_is_its_name(self):
        from redact.jailbreak.obfuscation.translation import LANGUAGE_CODES

        assert LANGUAGE_CODES["scots_gaelic"] == "gd"
        assert LANGUAGE_CODES["khmer"] == "km"
        assert LANGUAGE_CODES["chinese"] == "zh"
        assert LANGUAGE_CODES["hmong"] == "Hmong"
        assert all(len(code) == 2 and code.islower()  # noqa: PLR2004
                   for slug, code in LANGUAGE_CODES.items() if slug != "hmong")
        assert len(set(LANGUAGE_CODES.values())) == len(LANGUAGES)

    def test_a_hmong_translation_is_asked_for_by_its_name(self):
        """TranslateGemma takes Hmong only as a name: ``hmn`` and ``mww``,
        sent as codes, gave Hawaiian and Marathi."""
        from redact.jailbreak.obfuscation.translation import to_hmong

        request = to_hmong("hello", gen_model="gen").send(None)
        assert request.messages[0]["content"] == (
            "<<<source>>>en<<<target>>>Hmong<<<text>>>hello")

    def test_chinese_labels_are_glotlids_never_the_iso_639_3_macrolanguage(self):
        from redact.jailbreak.obfuscation.translation import ACCEPTED_LABELS

        assert ACCEPTED_LABELS["chinese"] == {"cmn_Hani", "wuu_Hani"}
        assert ACCEPTED_LABELS["hindi"] == {"hin_Deva", "anp_Deva"}
        assert ACCEPTED_LABELS["amharic"] == {"amh_Ethi"}
        assert ACCEPTED_LABELS["hmong"] == {"mww_Latn"}

    def test_the_four_new_languages_dated_spec_1_3_keep_their_answer_languages_1_2(
            self):
        """A row's ``translation_since`` dates its translation technique, so a
        plan kept from spec 1.2 resumes with the growth WARNING."""
        from redact.jailbreak.pool import all_techniques, registry

        by_name = registry(all_techniques())
        for slug in ("hmong", "chinese", "hindi", "amharic"):
            assert by_name[f"to_{slug}"].since == "1.3"
            assert by_name[f"to_ask_answer_in_{slug}"].since == "1.2"
        assert "to_yoruba" not in by_name and "to_guarani" not in by_name

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
            "hmong": 1.31, "chinese": 0.34, "hindi": 1.08, "amharic": 0.76,
        }
        assert LENGTH_RATIO_BAND == (0.5, 2.0)

    def test_a_chinese_translation_at_its_median_passes(self, monkeypatch):
        """Chinese is the shortest by far, 0.34 of its input: the median of
        another language would reject every good translation."""
        from redact.jailbreak.obfuscation import langid
        from redact.jailbreak.obfuscation.translation import (
            ACCEPTED_LABELS, MEDIAN_RATIO, check_translation)

        monkeypatch.setattr(langid, "detect", lambda text: "cmn_Hani")
        assert check_translation("译" * 34, "x" * 100, labels=ACCEPTED_LABELS["chinese"], median_ratio=MEDIAN_RATIO["chinese"]) is None
        assert check_translation(
            "译" * 100, "x" * 100, labels=ACCEPTED_LABELS["chinese"], median_ratio=MEDIAN_RATIO["chinese"]) == "length_ratio=2.94"

    @pytest.mark.parametrize(("translation", "source"), [
        ("问：", "Q:"), ("步骤 1：", "Step 1:"), ("译" * 60, "x" * 19), ("译", "1"),
    ])
    def test_a_piece_under_twenty_characters_is_not_held_to_the_length_band(
            self, monkeypatch, translation, source):
        """``Q:`` is 2.9 times as long in Chinese, a label or a digit keeps its
        characters, and the medians are those of sentences."""
        from redact.jailbreak.obfuscation import langid
        from redact.jailbreak.obfuscation.translation import (
            ACCEPTED_LABELS, MEDIAN_RATIO, check_translation)

        monkeypatch.setattr(langid, "detect", lambda text: "cmn_Hani")
        assert check_translation(translation, source, labels=ACCEPTED_LABELS["chinese"], median_ratio=MEDIAN_RATIO["chinese"]) is None

    def test_a_piece_of_twenty_characters_is_held_to_it(self, monkeypatch):
        from redact.jailbreak.obfuscation import langid
        from redact.jailbreak.obfuscation.translation import (
            ACCEPTED_LABELS, MEDIAN_RATIO, check_translation)

        monkeypatch.setattr(langid, "detect", lambda text: "cmn_Hani")
        assert check_translation("译" * 60, "x" * 20, labels=ACCEPTED_LABELS["chinese"], median_ratio=MEDIAN_RATIO["chinese"]) == "length_ratio=8.82"

    def test_a_short_piece_is_still_checked_for_its_language_and_for_empty(
            self, monkeypatch):
        from redact.jailbreak.obfuscation import langid
        from redact.jailbreak.obfuscation.translation import (
            ACCEPTED_LABELS, MEDIAN_RATIO, check_translation)

        monkeypatch.setattr(langid, "detect", lambda text: "jpn_Jpan")
        assert check_translation("问：", "Q:", labels=ACCEPTED_LABELS["chinese"], median_ratio=MEDIAN_RATIO["chinese"]) == "language=jpn_Jpan"
        assert check_translation(" ", "Q:", labels=ACCEPTED_LABELS["chinese"], median_ratio=MEDIAN_RATIO["chinese"]) == "empty"

    @pytest.mark.parametrize("label", ["und_Bali", "und_Xsux", "und_Latn"])
    def test_an_undetermined_glotlid_label_is_no_verdict_on_the_language(
            self, monkeypatch, label):
        """Two of the three Chinese discards of the measured run were this."""
        from redact.jailbreak.obfuscation import langid
        from redact.jailbreak.obfuscation.translation import (
            ACCEPTED_LABELS, MEDIAN_RATIO, check_translation)

        monkeypatch.setattr(langid, "detect", lambda text: label)
        assert check_translation("译" * 34, "x" * 100, labels=ACCEPTED_LABELS["chinese"], median_ratio=MEDIAN_RATIO["chinese"]) is None
        assert check_translation("译" * 100, "x" * 100, labels=ACCEPTED_LABELS["chinese"], median_ratio=MEDIAN_RATIO["chinese"]) == "length_ratio=2.94"

    def test_a_label_that_only_starts_like_an_undetermined_one_is_a_language(
            self, monkeypatch):
        from redact.jailbreak.obfuscation import langid
        from redact.jailbreak.obfuscation.translation import (
            ACCEPTED_LABELS, MEDIAN_RATIO, check_translation)

        for label in ("ind_Latn", "undd_Hani", "eng_Latn"):
            monkeypatch.setattr(langid, "detect", lambda text, label=label: label)
            assert check_translation(
                "译" * 34, "x" * 100, labels=ACCEPTED_LABELS["chinese"], median_ratio=MEDIAN_RATIO["chinese"]) == f"language={label}"

    def test_a_japanese_translation_at_its_median_passes(self, monkeypatch):
        from redact.jailbreak.obfuscation import langid
        from redact.jailbreak.obfuscation.translation import (
            ACCEPTED_LABELS, MEDIAN_RATIO, check_translation)

        monkeypatch.setattr(langid, "detect", lambda text: "jpn_Jpan")
        assert check_translation("訳" * 47, "x" * 100, labels=ACCEPTED_LABELS["japanese"], median_ratio=MEDIAN_RATIO["japanese"]) is None

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
