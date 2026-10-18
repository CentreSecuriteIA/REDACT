"""The jailbreak stage with a machine-translation model: the checks at its
start, on fakes."""

import json
import logging

import pandas as pd
import pytest

import redact.stages as P
from redact.jailbreak.obfuscation import langid
from redact.jailbreak.obfuscation.translation import check_mt_prompt
from redact.llms.model_config import MODEL_REGISTRY, ModelConfig, VLLMConfig
from redact.stages import generate_jailbreaks

MT = "mt-stage-fake"
TRANSLATION_ONLY = {"adversarial": {"translation": True,
                                    "families": {"allow": ["translation"]}},
                    "natural": None}


@pytest.fixture(autouse=True)
def mt(monkeypatch):
    monkeypatch.setitem(MODEL_REGISTRY, MT, ModelConfig(
        name=MT, roles=["translation", "machine_translation"],
        supports_system_prompt=False, vllm=VLLMConfig(hf_model_id="org/mt")))


def _inputs() -> pd.DataFrame:
    return pd.DataFrame({"prompt": ["Good morning."], "category": ["Cyber"],
                         "entry_type": ["harmful"]})


def _overlay(root, template, system=None):
    folder = root / "jailbreak" / "translate_mt"
    folder.mkdir(parents=True)
    config = {"template": template}
    if system is not None:
        config["system_prompt"] = system
    (folder / "template.json").write_text(json.dumps(config), encoding="utf-8")
    return folder / "template.json"


class TestTheMTPromptCheck:
    def test_the_packaged_prompt_passes(self):
        check_mt_prompt(None)

    def test_an_overlay_with_a_system_prompt_is_refused_naming_its_file(self, tmp_path):
        path = _overlay(tmp_path, "<<<source>>>{source_code}<<<target>>>"
                                  "{target_code}<<<text>>>{text}", system="Be nice.")
        with pytest.raises(ValueError, match="one user message") as raised:
            check_mt_prompt(tmp_path)
        assert str(path) in str(raised.value)

    @pytest.mark.parametrize("template", [
        "Translate {text} from {source_code} to {target_code}",
        "<<<target>>>{target_code}<<<source>>>{source_code}<<<text>>>{text}",
    ])
    def test_an_overlay_without_the_markers_in_order_is_refused(self, tmp_path,
                                                                template):
        _overlay(tmp_path, template)
        with pytest.raises(ValueError, match="<<<source>>>"):
            check_mt_prompt(tmp_path)

    def test_the_stage_stops_before_any_call(self, tmp_path, monkeypatch):
        _overlay(tmp_path / "prompts", "{text}", system="Be nice.")
        monkeypatch.setattr(P.ModelClient, "create",
                            lambda *a, **k: pytest.fail("a client was made"))
        with pytest.raises(ValueError, match="translate_mt"):
            generate_jailbreaks(inputs=_inputs(), output_path=tmp_path / "jb.csv",
                                translation_model=MT, augmentation=TRANSLATION_ONLY,
                                prompt_dir=tmp_path / "prompts", verbose=False)
        assert not (tmp_path / "jb.csv").exists()


class TestTheDetectorWarning:
    def _run(self, tmp_path, monkeypatch, translation_model, augmentation):
        from tests.conftest import make_client

        monkeypatch.setattr(P.ModelClient, "create", lambda m, *a, **k: make_client(
            None, m))
        monkeypatch.setattr(P, "run_augmentation", lambda *a, **k: pd.DataFrame())
        generate_jailbreaks(inputs=_inputs(), output_path=tmp_path / "jb.csv",
                            model="gen-fake", translation_model=translation_model,
                            augmentation=augmentation, verbose=False)

    def _warnings(self, caplog):
        return [r.getMessage() for r in caplog.records
                if r.levelno == logging.WARNING and "langid" in r.getMessage()]

    def test_a_missing_detector_is_warned_about_once(self, tmp_path, monkeypatch,
                                                      caplog):
        monkeypatch.setattr(langid, "available", lambda: False)
        with caplog.at_level(logging.WARNING):
            self._run(tmp_path, monkeypatch, MT, TRANSLATION_ONLY)
        (message,) = self._warnings(caplog)
        assert MT in message and "redact[langid]" in message

    def test_no_warning_with_the_detector(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setattr(langid, "available", lambda: True)
        with caplog.at_level(logging.WARNING):
            self._run(tmp_path, monkeypatch, MT, TRANSLATION_ONLY)
        assert self._warnings(caplog) == []

    def test_nothing_is_asked_of_a_chat_translator_or_a_pool_without_translation(
            self, tmp_path, monkeypatch, caplog):
        monkeypatch.setattr(langid, "available",
                            lambda: pytest.fail("the detector was asked"))
        self._run(tmp_path, monkeypatch, "local-translator", TRANSLATION_ONLY)
        self._run(tmp_path, monkeypatch, MT, {
            "adversarial": {"families": {"allow": ["encode"]}}, "natural": None})
