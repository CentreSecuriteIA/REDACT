"""The jailbreak stage with a machine-translation model: the checks at its
start, on fakes."""

import asyncio
import json
import logging
import re

import pandas as pd
import pytest

import redact.stages as P
from redact.exceptions import PromptTooLongError
from redact.jailbreak.obfuscation import langid
from redact.jailbreak.obfuscation.translation import check_mt_prompt, reply_cap
from redact.llm_pipeline import LLMRequest
from redact.llms.backends import ComputeConfig
from redact.llms.model_config import MODEL_REGISTRY, ModelConfig, VLLMConfig
from redact.stages import generate_jailbreaks
from tests.conftest import MockBackend

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
            "adversarial": {"translation": False, "families": {"allow": ["encode"]}},
            "natural": None})

    def test_nothing_is_asked_for_a_kind_that_matches_no_input(
            self, tmp_path, monkeypatch):
        monkeypatch.setattr(langid, "available",
                            lambda: pytest.fail("the detector was asked"))
        self._run(tmp_path, monkeypatch, MT, {
            "adversarial": {**TRANSLATION_ONLY["adversarial"],
                            "entry_types": ["benign"]}, "natural": None})


# --- translation units on the pool -------------------------------------------

GEN = "mt-stage-gen"


class _Gen(MockBackend):
    """The generation model on the async lane: ``step1 x`` -> ``S1:x``, and
    the judge's verdict ``Yes``. ``served`` is set once it has answered a
    request of a unit that is not a translation."""

    compute_config = ComputeConfig(supports_async_calls=True)

    def __init__(self):
        super().__init__(model=GEN)
        self.served = asyncio.Event()
        self.contents: list[str] = []

    async def agenerate(self, messages, *, max_tokens=None, internals_id=None,
                        **kwargs):
        content = messages[-1]["content"]
        self.contents.append(content)
        if content.startswith("step1 "):
            self.served.set()
            return content.replace("step1 ", "S1:")
        return "Yes"


class _Translator(MockBackend):
    """The machine-translation model on the async lane. Its first request
    waits until the generation model has served another unit: the driver
    must keep the other units going while a translation is in flight."""

    compute_config = ComputeConfig(supports_async_calls=True)

    def __init__(self, gen: _Gen, too_long=()):
        super().__init__(model=MT)
        self.gen, self.too_long = gen, set(too_long)
        self.sent: list[tuple[str, str, str, int]] = []

    async def agenerate(self, messages, *, max_tokens=None, internals_id=None,
                        **kwargs):
        (message,) = messages
        source, target, text = re.fullmatch(
            r"<<<source>>>(\w+)<<<target>>>(\w+)<<<text>>>(.*)", message["content"],
            re.S).groups()
        if not self.sent:
            await asyncio.wait_for(self.gen.served.wait(), timeout=10)
        self.sent.append((source, target, text, max_tokens))
        if text in self.too_long:
            raise PromptTooLongError(f"over the context: {text}")
        return f"sw:{text}" if source == "en" else text.removeprefix("sw:")


def one_step(text, *, gen_model=None, **kwargs):
    """A unit that is not a translation: one call on the generation model."""
    reply = yield LLMRequest(gen_model, [{"role": "user", "content": "step1 " + text}])
    return reply, "one_step"


def _stage(tmp_path, monkeypatch, plan, translator, gen):
    """Run the stage on ``plan``, ``[(prompt, technique name), ...]``, written
    as its manifest; return its rows by prompt."""
    import redact.jailbreak.pool as JPool
    from redact.dataset import Manifest, content_id
    from tests.conftest import make_client

    real = JPool.all_techniques()
    monkeypatch.setattr(JPool, "all_techniques", lambda: [*real, one_step])
    backends = {MT: translator, GEN: gen}
    monkeypatch.setattr(P.ModelClient, "create",
                        lambda m, *a, **k: make_client(backends[m], m))
    monkeypatch.setattr(langid, "available", lambda: True)
    out = tmp_path / "jb.csv"
    Manifest(out.with_name("jb.manifest.jsonl")).write(
        {"sample_id": content_id(p), "iteration": 0, "combination": [name],
         "settings": {}, "status": "planned"} for p, name in plan)
    inputs = pd.DataFrame({"prompt": [p for p, _ in plan], "category": "Cyber",
                           "entry_type": "harmful"})
    generate_jailbreaks(inputs=inputs, output_path=out, model=GEN,
                        translation_model=MT, augmentation=TRANSLATION_ONLY,
                        batch_size=2, verbose=False)
    df = pd.read_csv(out, dtype=str, keep_default_na=False)
    return {r["prompt"]: r for r in df.to_dict("records")}


def test_translation_units_run_on_the_pool_beside_other_units(tmp_path, monkeypatch):
    seen = {}

    def detect(text):
        seen[text] = seen.get(text, 0) + 1
        once = text == "sw:Wrong language once." and seen[text] == 1
        return "ind_Latn" if once else "swh_Latn"

    monkeypatch.setattr(langid, "detect", detect)
    gen = _Gen()
    translator = _Translator(gen, too_long={"Far too long."})
    rows = _stage(tmp_path, monkeypatch, [
        ("Good morning, friend.", "to_swahili"),
        ("other unit", "one_step"),
        ("Wrong language once.", "to_swahili"),
        ("Far too long.", "to_swahili"),
    ], translator, gen)

    assert {p: (r["jailbreak"], r["accepted"]) for p, r in rows.items()} == {
        "Good morning, friend.": ("sw:Good morning, friend.", "True"),
        "other unit": ("S1:other unit", "True"),
        "Wrong language once.": ("sw:Wrong language once.", "True"),
        "Far too long.": ("Far too long.", "False"),
    }
    assert rows["Far too long."]["technique_info"] == (
        "ERROR: over the context: Far too long.")
    # A round: translate, translate back, judge on the generation model.
    by_text = {}
    for source, target, text, cap in translator.sent:
        by_text.setdefault(text.removeprefix("sw:"), []).append((source, target, cap))
    assert by_text["Good morning, friend."] == [
        ("en", "sw", reply_cap("Good morning, friend.")),
        ("sw", "en", reply_cap("sw:Good morning, friend."))]
    # The wrong label was resampled: two forward requests, then one back.
    assert [s for s, _, _ in by_text["Wrong language once."]] == ["en", "en", "sw"]
    # One judge call per translation that passed its checks.
    judged = [c for c in gen.contents if not c.startswith("step1 ")]
    assert len(judged) == 2                     # noqa: PLR2004
