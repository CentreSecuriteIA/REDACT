"""The files a stage writes, against the files it wrote before a change.

``fixtures/stage_files.json`` holds every file of one stopped and resumed run
of each stage, as the code wrote them before the stages streamed their plans
and units (commit e7106b1); the jailbreak stage's variations files were added
when it gained them, and each kind's variants table (header only: the stub
techniques draw from no template pool) when the stage gained it. Model calls are stubbed and depend only on the
request, so a run writes the same files every time.

A file is compared as data, not bytes, since a CSV's line ending and the
sizes a ledger records differ between platforms: the cells of a CSV as text,
the rows of a manifest, and the lines of a ledger without ``_saved``. The
spec version a jailbreak file records (a row's ``combination_spec_version``,
a plan file's ``spec_version``) is written ``<spec version>`` in the fixture
and read as the spec's version, so a version bump edits no cell; a plan
file's ``rules`` is written ``<rules>`` and read as the fake pool's rules id,
so a change of the spec's rules edits none either.
"""

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from tests.conftest import ParaphraseBackend, make_client, pinned_pool

_FIXTURE = Path(__file__).parent / "fixtures" / "stage_files.json"
_SPEC_VERSION = "<spec version>"
_RULES = "<rules>"

# An id that is given, missing, padded, given twice, NaN and a number.
_IDS = ["given-1", None, " padded ", "given-1", float("nan"), 77]


def _files(folder: Path) -> dict:
    """Every file under ``folder`` as comparable data, by its path from there."""
    found = {}
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        name = path.relative_to(folder).as_posix()
        if path.suffix == ".csv":
            with path.open(encoding="utf-8", newline="") as fh:
                found[name] = list(csv.reader(fh))
        elif path.suffix == ".jsonl":
            lines = path.read_text(encoding="utf-8").splitlines()
            found[name] = [
                {k: v for k, v in json.loads(line).items() if k != "_saved"}
                for line in lines
            ]
        elif path.suffix != ".lock":
            found[name] = path.read_text(encoding="utf-8")
    return found


def _stopped_then_resumed(folder: Path, run) -> dict:
    """The files after ``run(fails=True)`` stopped, and after ``run(fails=False)``."""
    with pytest.raises(RuntimeError, match="engine died"):
        run(fails=True)
    stopped = _files(folder)
    run(fails=False)
    return {"stopped": stopped, "resumed": _files(folder)}


def _dies_on_call(n: int, calls: list):
    """Count a model call in ``calls``; the ``n``-th raises."""
    calls.append(None)
    if len(calls) == n:
        raise RuntimeError("engine died")


def _jailbreaks(folder, monkeypatch) -> dict:
    import redact.jailbreak.augmentation as JA
    import redact.jailbreak.plan as JP
    import redact.jailbreak.pool as JPool
    import redact.stages as P
    from redact import generate_jailbreaks
    from tests.jailbreak.test_pipeline_jailbreaks import (
        GEN,
        _StepBackend,
        broken,
        one_step,
        two_step,
    )

    # What a unit is assigned depends on all that plan_run hands the sampler.
    def assign(sid, pool, *, seed, iteration, used, dealer=None, **settings):
        pick = sum(map(ord, sid)) + seed + iteration + len(used) + len(settings)
        return [pool[pick % len(pool)].__name__]

    monkeypatch.setattr(JP, "assign_combination", assign)
    monkeypatch.setattr(JPool, "all_techniques", lambda: [one_step, two_step, broken])
    # The fake techniques are in no spec.
    monkeypatch.setattr(JA, "_check_names", lambda where, resolved: None)

    inputs = pd.DataFrame({
        "sample_id": _IDS,
        "prompt": ["zq0 first", "zq1 second", "zq2 third", "zq3 fourth, the id of "
                   "the first", "zq4 fifth", "zq5 sixth"],
        "category": ["Cyber", "Cyber", "Fraud", "Fraud", "Cyber", "Fraud"],
        "entry_type": ["harmful", "benign", "harmful", "harmful", "benign", "harmful"],
        "turn": [0, 1, 2, 3, 4, 5],
        "accepted": [True] * 6,
        "note": ["", "N/A", "a, b", "two\nlines", None, 'said "hi"'],
    })

    def run(fails):
        calls: list = []

        class Backend(_StepBackend):
            def generate(self, messages_list, **kwargs):
                if fails:
                    _dies_on_call(3, calls)
                return super().generate(messages_list, **kwargs)

        backend = Backend()
        monkeypatch.setattr(
            P.ModelClient, "create", lambda m: make_client(backend, m))
        # The natural kind draws from the whole (fake) registry. The fixture
        # was written with the request obfuscation off: the fake techniques
        # hold none it could draw, and assign above picks by the number of
        # settings, so the packaged kind's switch would move every pick.
        generate_jailbreaks(
            inputs=inputs, output_path=folder / "jailbreaks.csv",
            variations_path=folder / "variations.csv", model=GEN, seed=5,
            augmentation={
                "adversarial": {"rounds": [{"exact_techniques": 1},
                                           {"max_complexity": 4}],
                                "allow_request_obfuscation": False},
                "natural": {"families": {"allow": None},
                            "techniques": {"allow": ["one_step", "two_step", "broken"],
                                           "deny": []}},
            },
            batch_size=2, auto_generate_benign=False, verbose=False)

    return _stopped_then_resumed(folder, run)


def _outputs(folder, monkeypatch) -> dict:
    import redact.content_moderation.outputs as G
    import redact.stages as P
    from redact import generate_outputs
    from tests.test_pipelines import _TokenBackend

    # The stub's answers are shorter than the length floor: checked anyway.
    monkeypatch.setattr(G, "MIN_RESPONSE_CHARS", 0)
    inputs = pd.DataFrame({
        "sample_id": _IDS,
        "sample": [f"zq{i} question" for i in range(6)],
        "category": ["Cyber", "Cyber", "Fraud", "Fraud", "Cyber", "Fraud"],
        "entry_type": ["harmful", "benign", "harmful", "harmful", "benign", "harmful"],
        "subcategory": ["phishing", None, "", "N/A", "a, b", "cards"],
        "source": ["constitution"] * 6,
    })

    def run(fails):
        calls: list = []

        class Gen(_TokenBackend):
            def generate(self, messages_list, **kwargs):
                if fails:
                    _dies_on_call(3, calls)
                return super().generate(messages_list, **kwargs)

        backends = {
            "gen-model": Gen(lambda t, text: f"answer {t}", "gen-model"),
            # Odd tokens are rejected.
            "check-model": _TokenBackend(
                lambda t, text: f"No: weak {t}" if int(t[2:]) % 2 else "Yes",
                "check-model"),
        }
        monkeypatch.setattr(
            P.ModelClient, "create", lambda m: make_client(backends[m], m))
        generate_outputs(
            inputs=inputs, output_path=folder / "output_responses.csv",
            model="gen-model", check_model="check-model", batch_size=2, verbose=False)

    return _stopped_then_resumed(folder, run)


def _paraphrases(folder, monkeypatch) -> dict:
    import redact.content_moderation.paraphrase as CP
    from redact import generate_paraphrases, residency
    from redact.llms.backends import ComputeConfig
    from redact.llms.model_config import MODEL_REGISTRY, VLLMConfig, register_model

    monkeypatch.setattr(CP, "MIN_WORDS", 0)     # the texts are two words long
    # The pool the fixture pins, whatever other paraphrasers are registered.
    pinned_pool(monkeypatch, "venice-paraphraser", "_para_second")
    monkeypatch.setattr(
        residency, "plan_residency",
        lambda models, copies=None: SimpleNamespace(sequential=False))
    monkeypatch.setattr(residency, "apply_plan", lambda plan, replace=True: None)

    inputs = pd.DataFrame({
        "sample_id": _IDS,
        "sample": ["alpha prompt", "beta prompt", "gamma prompt", "delta prompt",
                   "epsilon prompt", "zeta prompt"],
        "category": ["Cyber", "Cyber", "Fraud", "Fraud", "Cyber", "Fraud"],
        "entry_type": ["harmful", "benign", "harmful", "harmful", "benign", "harmful"],
    })
    outputs = pd.DataFrame({
        "input_id": ["o1", None, "o3", "o4", "o5"],
        "output_response": ["alpha answer", "beta answer", "gamma answer",
                            "delta answer", "epsilon answer"],
        "accepted": [True, True, True, True, False],
        "category": ["Cyber"] * 5,
    })

    def run(fails):
        calls: list = []

        class Backend(ParaphraseBackend):
            """Counts the paraphrasers' generate() calls: the third raises.
            Native batching, so one call carries a whole group of requests."""

            compute_config = ComputeConfig(supports_native_batching=True)

            def generate(self, messages_list, **kwargs):
                if fails and not messages_list[0][-1]["content"].startswith(
                        "ORIGINAL:\n"):
                    _dies_on_call(3, calls)
                return super().generate(messages_list, **kwargs)

        # A "gamma" text comes back unchanged, so its unit is dropped; a text
        # with "delta" is rejected by the check.
        monkeypatch.setattr(CP.ModelClient, "create", lambda m, **kw: make_client(
            Backend(paraphrase=lambda model, text: (
                        text if text.startswith("gamma") else f"{model} says {text}"),
                    check=lambda original, paraphrase: (
                        "meaning changed" if "delta" in paraphrase else "Yes")),
            m))
        generate_paraphrases(
            inputs=inputs, outputs=outputs, paraphraser="distribution",
            target="both", paraphrases_per_sample=2, batch_size=3, seed=9,
            verbose=False,
            inputs_path=folder / "paraphrases_inputs.csv",
            outputs_path=folder / "paraphrases_outputs.csv")

    register_model("_para_second", backend_type="vllm", roles=["paraphraser"],
                   vllm=VLLMConfig(hf_model_id="org/second"))
    try:
        return _stopped_then_resumed(folder, run)
    finally:
        MODEL_REGISTRY.pop("_para_second", None)


def _conversations(folder, monkeypatch) -> dict:
    from redact import generate_conversations
    from tests.conftest import FakeResolver, echo, simple_setting

    # No entry_type column: a row's entry_type is then empty.
    seeds = pd.DataFrame({
        "sample_id": _IDS,
        "seed": [f"question {i}" for i in range(6)],
        "category": ["Cyber", "Cyber", "Fraud", "Fraud", "Cyber", "Fraud"],
    })

    def run(fails):
        calls: list = []

        def reply(model, messages):
            if fails:
                _dies_on_call(7, calls)     # in the third chunk of four
            return echo(model, messages)

        generate_conversations(
            seeds, simple_setting, iterations=2, batch_size=3,
            output_path=folder / "conversations.csv",
            resolve=FakeResolver(reply), verbose=False)

    return _stopped_then_resumed(folder, run)


_STAGES = {
    "jailbreaks": _jailbreaks,
    "outputs": _outputs,
    "paraphrases": _paraphrases,
    "conversations": _conversations,
}


@pytest.mark.parametrize("stage", sorted(_STAGES))
def test_a_stage_writes_the_files_it_wrote_before(stage, tmp_path, monkeypatch):
    from redact.jailbreak import load_spec
    from redact.jailbreak.plan import rules_id
    from tests.jailbreak.test_pipeline_jailbreaks import broken, one_step, two_step

    fixture = _FIXTURE.read_text(encoding="utf-8").replace(
        _SPEC_VERSION, load_spec()["version"]).replace(
        _RULES, rules_id([one_step, two_step, broken]))
    expected = json.loads(fixture)[stage]
    assert _STAGES[stage](tmp_path, monkeypatch) == expected
