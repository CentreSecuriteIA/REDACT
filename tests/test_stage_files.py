"""The files a stage writes, against the files it wrote before a change.

``fixtures/stage_files.json`` holds every file of one stopped and resumed run
of each stage, as the code wrote them before the stages streamed their plans
and units (commit e7106b1). Model calls are stubbed and depend only on the
request, so a run writes the same files every time.

A file is compared as data, not bytes, since a CSV's line ending and the
sizes a ledger records differ between platforms: the cells of a CSV as text,
the rows of a manifest, and the lines of a ledger without ``_saved``.
"""

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from tests.conftest import MockBackend, make_client

_FIXTURE = Path(__file__).parent / "fixtures" / "stage_files.json"

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
    import redact.jailbreak.plan as JP
    import redact.jailbreak.pool as JPool
    import redact.pipelines as P
    from redact import generate_jailbreaks
    from tests.jailbreak.test_pipeline_jailbreaks import (
        GEN,
        _StepBackend,
        broken,
        one_step,
        two_step,
    )

    # What a unit is assigned depends on all that plan_run hands the sampler.
    def assign(sid, pool, *, seed, iteration, used, **settings):
        pick = sum(map(ord, sid)) + seed + iteration + len(used) + len(settings)
        return [pool[pick % len(pool)].__name__]

    monkeypatch.setattr(JP, "assign_combination", assign)
    monkeypatch.setattr(JPool, "all_techniques", lambda: [one_step, two_step, broken])

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
        generate_jailbreaks(
            inputs=inputs, output_path=folder / "jailbreaks.csv", model=GEN, seed=5,
            settings_per_iteration=[{"exact_techniques": 1}, {"max_complexity": 4}],
            batch_size=2, auto_generate_benign=False, verbose=False)

    return _stopped_then_resumed(folder, run)


def _outputs(folder, monkeypatch) -> dict:
    import redact.pipelines as P
    from redact import generate_outputs
    from tests.test_pipelines import _TokenBackend

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
    from redact.llms.model_config import MODEL_REGISTRY, VLLMConfig, register_model

    monkeypatch.setattr(
        CP.ModelClient, "create", lambda m: make_client(MockBackend(), m))
    # A text with "delta" is rejected by the check.
    monkeypatch.setattr(
        CP, "batch_check_samples",
        lambda client, texts, checker, **kw: [
            (False, "meaning changed") if "delta" in t else (True, "") for t in texts])
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

        # A "gamma" text comes back unchanged, so its unit is dropped.
        def paraphrase(client, texts, **kw):
            if fails:
                _dies_on_call(3, calls)
            return [t if t.startswith("gamma") else f"{client.model} says {t}"
                    for t in texts]

        monkeypatch.setattr(CP, "paraphrase_batch", paraphrase)
        generate_paraphrases(
            inputs=inputs, outputs=outputs, paraphraser="distribution",
            paraphrases_per_sample=2, batch_size=3, seed=9, verbose=False,
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
    from tests.test_multi_turn import _echo, _FakeResolver, _simple_setting

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
            return _echo(model, messages)

        generate_conversations(
            seeds, _simple_setting, iterations=2, batch_size=3,
            output_path=folder / "conversations.csv",
            resolve=_FakeResolver(reply), verbose=False)

    return _stopped_then_resumed(folder, run)


_STAGES = {
    "jailbreaks": _jailbreaks,
    "outputs": _outputs,
    "paraphrases": _paraphrases,
    "conversations": _conversations,
}


@pytest.mark.parametrize("stage", sorted(_STAGES))
def test_a_stage_writes_the_files_it_wrote_before(stage, tmp_path, monkeypatch):
    expected = json.loads(_FIXTURE.read_text(encoding="utf-8"))[stage]
    assert _STAGES[stage](tmp_path, monkeypatch) == expected
