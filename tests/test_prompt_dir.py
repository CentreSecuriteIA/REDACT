"""The prompt overlay reaches every prompt a stage's model calls are built from."""

import json
from pathlib import Path

import pandas as pd
import pytest

from redact import telemetry
from redact.llms import ModelClient
from tests.conftest import MockBackend, make_client


def _overlay(root: Path, name: str, system: str, template: str = "t") -> Path:
    """Write ``root/name/template.json``, whose system prompt is ``system``."""
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "template.json").write_text(
        json.dumps({"system_prompt": system, "template": template}), encoding="utf-8")
    return root


def _systems(backend: MockBackend) -> list[str]:
    """The system prompt of each call ``backend`` got."""
    return [call.get("system_prompt") for call in backend.calls]


def _serve(monkeypatch, backends: dict) -> None:
    """Make ``ModelClient.create`` hand out ``backends`` by model name."""
    monkeypatch.setattr(ModelClient, "create",
                        lambda model, *a, **k: make_client(backends[model], model))


# --- the constitution ---------------------------------------------------------

def test_the_constitution_reads_its_prompt_through_the_overlay(tmp_path, monkeypatch):
    from redact import generate_constitution

    overlay = _overlay(tmp_path / "prompts", "constitution/harmful",
                       "MINE-CONSTITUTION")
    backend = MockBackend("no entry\n[END]")
    _serve(monkeypatch, {"c-model": backend})
    generate_constitution(
        taxonomy={"categories": {"Cyber": "computers"}}, entry_types=["harmful"],
        model="c-model", data_dir=tmp_path, prompt_dir=overlay, verbose=False,
        read_back=False)
    assert backend.calls and set(_systems(backend)) == {"MINE-CONSTITUTION"}


def test_a_run_hands_the_constitution_its_prompt_dir(tmp_path, monkeypatch):
    import redact
    from redact import run_pipeline

    handed = []
    monkeypatch.setattr(redact, "generate_constitution",
                        lambda **k: handed.append(k["prompt_dir"]) or pd.DataFrame())
    recipe = {"dataset_type": "eval", "data_dir": str(tmp_path),
              "stages": ["constitution"], "prompt_dir": str(tmp_path / "prompts")}
    try:
        run_pipeline(recipe, params={}, verbose=False)
    finally:
        telemetry.uninstall()
    assert handed == [str(tmp_path / "prompts")]


# --- the inputs ---------------------------------------------------------------

def test_the_seeded_inputs_read_their_format_and_check_through_the_overlay(
        tmp_path, monkeypatch):
    from redact import generate_inputs

    overlay = tmp_path / "prompts"
    _overlay(overlay, "input/generation/from_constitution/long", "MINE-GEN",
             "about {sample_description}")
    _overlay(overlay, "input/quality_check", "MINE-CHECK")
    numbered = overlay / "format_instructions" / "numbered"
    numbered.mkdir(parents=True)
    (numbered / "template.json").write_text(
        json.dumps({"instruction": " MINE-FORMAT {num_samples}"}), encoding="utf-8")
    gen = MockBackend(["1. one zq sample\n[END]", "1. two zq sample\n[END]"])
    check = MockBackend("Yes")
    _serve(monkeypatch, {"g-model": gen, "c-model": check})
    entry = pd.DataFrame([{"source_category": "Cyber", "sample_description": "d",
                           "constitution_subcategory": "S", "entry_type": "harmful"}])
    generate_inputs(data_dir=tmp_path, constitution_df=entry, samples_per_entry=2,
                    model="g-model", check_model="c-model", prompt_dir=overlay,
                    verbose=False, read_back=False)
    # One sample per call, the entry's two units.
    assert _systems(gen) == ["MINE-GEN MINE-FORMAT 1"] * 2
    assert _systems(check) == ["MINE-CHECK", "MINE-CHECK"]


# --- translation and the benign data ------------------------------------------

def test_a_translation_reads_its_prompts_through_the_overlay(tmp_path):
    from redact.jailbreak.obfuscation.translation import to_swahili

    overlay = tmp_path / "prompts"
    _overlay(overlay, "jailbreak/translate", "MINE-TRANSLATE")
    _overlay(overlay, "jailbreak/translate_retry", "MINE-RETRY")
    _overlay(overlay, "jailbreak/translate_check", "MINE-CHECK")
    gen = to_swahili("hello", translate_model="local-translator", prompt_dir=overlay)
    systems = [gen.send(None).messages[0]["content"]]
    systems.append(gen.send("jambo").messages[0]["content"])
    systems.append(gen.send("No: not Swahili").messages[0]["content"])
    assert systems == ["MINE-TRANSLATE", "MINE-CHECK", "MINE-RETRY"]


def test_a_machine_translation_reads_its_prompts_through_the_overlay(
        tmp_path, monkeypatch):
    from redact.jailbreak.obfuscation import langid
    from redact.jailbreak.obfuscation.translation import to_swahili
    from redact.llms.model_config import MODEL_REGISTRY, ModelConfig, VLLMConfig

    monkeypatch.setitem(MODEL_REGISTRY, "mt-overlay", ModelConfig(
        name="mt-overlay", roles=["translation", "machine_translation"],
        vllm=VLLMConfig(hf_model_id="org/mt")))
    monkeypatch.setattr(langid, "detect", lambda text: None)
    overlay = tmp_path / "prompts"
    _overlay(overlay, "jailbreak/translate_check_back", "MINE-JUDGE",
             template="{original}|{back_translation}|{language}")
    mt = overlay / "jailbreak" / "translate_mt"
    mt.mkdir(parents=True)
    (mt / "template.json").write_text(json.dumps({
        "template": "MINE<<<source>>>{source_code}<<<target>>>{target_code}"
                    "<<<text>>>{text}"}), encoding="utf-8")
    gen = to_swahili("hello", translate_model="mt-overlay", gen_model="gen",
                     prompt_dir=overlay)
    sent = [gen.send(None).messages]
    sent.append(gen.send("jambo").messages)
    sent.append(gen.send("hello").messages)
    assert [m["content"] for m in sent[0]] == [
        "MINE<<<source>>>en<<<target>>>sw<<<text>>>hello"]
    assert [m["content"] for m in sent[1]] == [
        "MINE<<<source>>>sw<<<target>>>en<<<text>>>jambo"]
    assert [m["content"] for m in sent[2]] == ["MINE-JUDGE", "hello|hello|Swahili"]


def test_apply_combination_hands_its_prompt_dir_to_the_benign_data(monkeypatch):
    import redact.jailbreak.chain as chain

    handed = {}

    def benign(**kwargs):
        handed.update(kwargs)
        return {"pairs": []}

    def needs_pairs(text, benign_data=None):
        return text + " with pairs", "needs_pairs"

    needs_pairs.requires_benign = True
    monkeypatch.setattr(chain, "get_or_generate_benign_data", benign)
    chain.apply_combination(needs_pairs, "a prompt", prompt_dir="mine")
    assert handed["prompt_dir"] == "mine"


# --- the conversation prompts -------------------------------------------------

def _conversations(tmp_path: Path) -> Path:
    """One saved conversation under ``tmp_path``; its file."""
    from redact import paths
    from redact.multi_turn import Transcript

    transcript = Transcript()
    transcript.message("attacker", "the goal", role="user")
    transcript.reply("target", "the reply", role="assistant", model="t-model")
    path = paths.conversations_csv(tmp_path)
    path.parent.mkdir(parents=True)
    pd.DataFrame([{"sample_id": "c1",
                   "transcript": json.dumps(transcript.to_records())}]).to_csv(
        path, index=False)
    return path


def test_scoring_reads_the_judge_through_the_overlay(tmp_path):
    from redact.multi_turn import evaluate_conversations

    _conversations(tmp_path)
    overlay = _overlay(tmp_path / "prompts", "conversation/judge", "MINE-JUDGE",
                       "MINE-JUDGE-TEMPLATE {goal}")
    judge = MockBackend("Yes")
    evaluate_conversations(data_dir=tmp_path, judge_model="j-model",
                           resolve=lambda m: make_client(judge, m), prompt_dir=overlay,
                           verbose=False, read_back=False)
    (call,) = judge.calls
    assert call["system_prompt"] == "MINE-JUDGE"
    assert call["messages"][-1]["content"] == "MINE-JUDGE-TEMPLATE the goal"


def test_scoring_attacks_reads_the_success_criterion_through_the_overlay(tmp_path):
    from redact.multiturn_attacks import score_attacks

    _conversations(tmp_path)
    overlay = _overlay(tmp_path / "prompts", "conversation/success_judge",
                       "MINE-SUCCESS")
    judge = MockBackend("Yes")
    score_attacks(data_dir=tmp_path, judge_model="j-model", prompt_dir=overlay,
                  resolve=lambda m: make_client(judge, m), verbose=False,
                  read_back=False)
    assert _systems(judge) == ["MINE-SUCCESS"]


def test_the_pair_attacker_reads_its_prompt_through_the_overlay(tmp_path):
    from redact.multi_turn import Transcript
    from redact.multiturn_attacks import pair_propose

    overlay = _overlay(tmp_path / "prompts", "conversation/pair_attacker", "MINE-PAIR")
    transcript = Transcript()
    transcript.message("attacker", "the goal", role="user")
    request = pair_propose("a-model", prompt_dir=overlay)(transcript, 1).send(None)
    assert request.messages[0]["content"] == "MINE-PAIR"


def test_generate_attacks_hands_its_prompt_dir_to_the_pair_attacker(monkeypatch):
    import redact.multiturn_attacks.pipeline as attack_pipeline
    import redact.multiturn_attacks.setting as setting

    handed = []
    monkeypatch.setattr(setting, "pair_propose",
                        lambda model, prompt_dir=None: handed.append(prompt_dir))
    monkeypatch.setattr(attack_pipeline, "generate_conversations",
                        lambda seeds, factory, **kwargs: pd.DataFrame())
    attack_pipeline.generate_attacks(pd.DataFrame(), "t-model", attack="pair",
                                     attacker_model="a-model", prompt_dir="mine")
    assert handed == ["mine"]


def test_optimize_attack_reads_the_judge_through_the_overlay(tmp_path, monkeypatch):
    import redact.multiturn_attacks.optimize as attack_optimize

    overlay = _overlay(tmp_path / "prompts", "conversation/success_judge",
                       "MINE-SUCCESS")
    handed = {}
    monkeypatch.setattr(attack_optimize, "optimize",
                        lambda *args, **kwargs: handed.update(
                            judge_system=args[4], prompt_dir=kwargs["prompt_dir"]))
    attack_optimize.optimize_attack("the goal", "t-model", [], "j-model",
                                    prompt_dir=overlay)
    assert handed == {"judge_system": "MINE-SUCCESS", "prompt_dir": overlay}


@pytest.mark.parametrize("prompt_dir", [None, "mine"])
def test_the_search_reads_the_judge_through_its_prompt_dir(prompt_dir, monkeypatch):
    import redact.optimization.search as search

    handed = []
    monkeypatch.setattr(search, "judge_prompt",
                        lambda system, prompt_dir=None: handed.append(prompt_dir))
    search.optimize("the goal", "t-model", lambda transcript, depth: [], "j-model",
                    "criterion", depth=1, verbose=False, prompt_dir=prompt_dir,
                    resolve=lambda m: make_client(MockBackend("Yes"), m))
    assert handed == [prompt_dir]
