"""Resume-ledger tests for constitution-seeded input generation.

Mirrors the ledger tests for the constitution / output / paraphrase stages:
a completed ``(sample_description, entry_type, style)`` unit is acked to
``constitution_inputs.state.jsonl`` and skipped on a resume re-run; ``fresh``
clears it. Fully offline (MockBackend).
"""

import json
import logging
import re

import pandas as pd
import pytest

from redact.content_moderation.generation import (
    InputPipeline,
    _constitution_inputs_ledger,
    _constitution_inputs_manifest,
)
from redact.llms.backends import ComputeConfig
from tests.conftest import MockBackend, make_client

_PROMPT = {"system_prompt": "sys", "template": "Generate items about {sample_description}"}


def _pipeline(tmp_path):
    # Two registry names: one name and setup is one call group, answered by
    # the first request's backend.
    return InputPipeline(
        gen=make_client(MockBackend("1. alpha sample\n2. beta sample"), "m"),
        check=make_client(MockBackend("Yes"), "m-check"),
        dataset_dir=tmp_path,
    )


def _const_df():
    return pd.DataFrame([
        {"source_category": "Violence", "sample_description": "desc one",
         "constitution_subcategory": "Sub", "entry_type": "harmful"},
        {"source_category": "Privacy", "sample_description": "desc two",
         "constitution_subcategory": "Sub", "entry_type": "benign"},
    ])


def test_ledger_written_and_resume_skips(tmp_path):
    pipe = _pipeline(tmp_path)
    r1 = pipe.run_from_constitution(_const_df(), _PROMPT, samples_per_entry=2, verbose=False)
    assert r1.total_entries_processed == 2

    # ledger recorded both entries
    led = _constitution_inputs_ledger(tmp_path)
    assert led.completed() == {("desc one", "harmful", ""), ("desc two", "benign", "")}

    # resume run: every entry already acked -> nothing processed, no gen calls
    pipe2 = _pipeline(tmp_path)
    r2 = pipe2.run_from_constitution(_const_df(), _PROMPT, samples_per_entry=2, verbose=False)
    assert r2.total_entries_processed == 0
    assert pipe2.gen.backend.calls == []


def test_manifest_written_before_generation(tmp_path):
    pipe = _pipeline(tmp_path)
    pipe.run_from_constitution(_const_df(), _PROMPT, samples_per_entry=2,
                               style="long", verbose=False)
    man = _constitution_inputs_manifest(tmp_path)
    rows = man.load()
    # one planned row per constitution entry, keyed like the ledger.
    assert {(r["sample_description"], r["entry_type"], r["style"]) for r in rows} == {
        ("desc one", "harmful", "long"), ("desc two", "benign", "long"),
    }
    assert all(r["status"] == "planned" for r in rows)


def test_fresh_clears_ledger_and_regenerates(tmp_path):
    pipe = _pipeline(tmp_path)
    pipe.run_from_constitution(_const_df(), _PROMPT, samples_per_entry=2, verbose=False)
    assert _constitution_inputs_ledger(tmp_path).completed()

    pipe2 = _pipeline(tmp_path)
    r = pipe2.run_from_constitution(_const_df(), _PROMPT, samples_per_entry=2,
                                    fresh=True, verbose=False)
    assert r.total_entries_processed == 2          # regenerated despite prior ledger
    assert pipe2.gen.backend.calls != []


class _WidthBackend(MockBackend):
    """Native batching, so one ``generate()`` per call; records each call's width."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.widths: list[int] = []

    @property
    def compute_config(self) -> ComputeConfig:
        return ComputeConfig(supports_native_batching=True)

    def generate(self, messages_list, **kwargs):
        self.widths.append(len(messages_list))
        return super().generate(messages_list, **kwargs)


def test_check_calls_are_no_wider_than_batch_size(tmp_path):
    """Checks go out in calls of at most ``batch_size``, like every other call
    of the stage: 4 entries x 2 samples = 8 checks at batch_size=3 fill calls
    of 3 and never more."""
    check = _WidthBackend("Yes")
    pipe = InputPipeline(
        gen=make_client(_TokenBackend(lambda t, n: f"1. first {t}\n2. second {t}"), "m"),
        check=make_client(check, "m-check"),
        dataset_dir=tmp_path,
    )
    pipe.run_from_constitution(_token_df(4), _PROMPT, samples_per_entry=2,
                               batch_size=3, verbose=False)
    assert max(check.widths) == 3 and sum(check.widths) == 8


def test_entry_with_no_extracted_samples_is_acked(tmp_path):
    """An entry that yields nothing writes no row, so the CSV fallback can't
    cover it — un-acked it would be re-generated (and re-paid for) forever."""
    def _pipe():
        return InputPipeline(
            gen=make_client(MockBackend("sorry, nothing here"), "m"),
            check=make_client(MockBackend("Yes"), "m-check"),
            dataset_dir=tmp_path,
        )

    r = _pipe().run_from_constitution(_const_df(), _PROMPT, samples_per_entry=2, verbose=False)
    assert r.skipped_entries == 2 and r.total_entries_processed == 0
    assert _constitution_inputs_ledger(tmp_path).completed() == {
        ("desc one", "harmful", ""), ("desc two", "benign", ""),
    }

    pipe2 = _pipe()
    pipe2.run_from_constitution(_const_df(), _PROMPT, samples_per_entry=2, verbose=False)
    assert pipe2.gen.backend.calls == []     # not re-attempted on resume


def test_resume_falls_back_to_csv_when_ledger_absent(tmp_path):
    # Simulate a pre-ledger run: generate under a named style, then delete the
    # ledger. Resume must still skip via the existing-CSV set (back-compat).
    pipe = _pipeline(tmp_path)
    pipe.run_from_constitution(_const_df(), _PROMPT, samples_per_entry=2,
                               style="long", verbose=False)
    _constitution_inputs_ledger(tmp_path).reset()   # drop the ledger file

    pipe2 = _pipeline(tmp_path)
    r = pipe2.run_from_constitution(_const_df(), _PROMPT, samples_per_entry=2,
                                    style="long", verbose=False)
    assert r.total_entries_processed == 0
    assert pipe2.gen.backend.calls == []


def _one_entry_pipeline(tmp_path, gen_replies):
    gen = MockBackend(gen_replies)
    pipe = InputPipeline(
        gen=make_client(gen, "m"), check=make_client(MockBackend("Yes"), "m-check"),
        dataset_dir=tmp_path,
    )
    return gen, pipe


def test_a_rejected_reply_is_regenerated(tmp_path):
    gen, pipe = _one_entry_pipeline(
        tmp_path, ["sorry, nothing here", "1. alpha sample\n2. beta sample"])
    r = pipe.run_from_constitution(
        _const_df().iloc[:1], _PROMPT, samples_per_entry=2, verbose=False)
    assert len(gen.calls) == 2
    assert r.skipped_entries == 0 and r.total_entries_processed == 1


def test_an_entry_rejected_every_time_is_acked_as_rejected(tmp_path, caplog):
    gen, pipe = _one_entry_pipeline(tmp_path, "sorry, nothing here")
    with caplog.at_level(logging.WARNING):
        r = pipe.run_from_constitution(
            _const_df().iloc[:1], _PROMPT, samples_per_entry=2, max_attempts=2,
            verbose=False)
    assert len(gen.calls) == 2 and r.skipped_entries == 1
    ledger = _constitution_inputs_ledger(tmp_path)
    ack = json.loads(ledger.path.read_text(encoding="utf-8").splitlines()[-1])
    assert (ack["status"], ack["attempts"]) == ("rejected", 2)
    assert "1 entries rejected after 2 attempt(s)" in caplog.text


class _TokenBackend(MockBackend):
    """Native-batching fake: ``reply(token, attempt)`` answers each request.

    ``token`` is the first ``zqN`` in the request, ``attempt`` how many times
    this backend has seen it. Replies depend only on that, never on call
    order, so rows compare across runs whose entries finish in other orders.
    ``batches`` holds each call's tokens, with ``*`` on a repeated one.
    """

    def __init__(self, reply):
        super().__init__()
        self._reply = reply
        self._seen: dict[str, int] = {}
        self.batches: list[list[str]] = []

    @property
    def compute_config(self) -> ComputeConfig:
        return ComputeConfig(supports_native_batching=True)

    def generate(self, messages_list, **kwargs):
        super().generate(messages_list, **kwargs)
        batch, replies = [], []
        for messages in messages_list:
            text = " ".join(m["content"] for m in messages)
            token = re.search(r"zq\d+", text).group()
            attempt = self._seen[token] = self._seen.get(token, 0) + 1
            batch.append(token + "*" * (attempt > 1))
            replies.append(self._reply(token, attempt))
        self.batches.append(batch)
        return replies


def _token_df(n, categories=("Violence", "Privacy")):
    """``n`` entries, ``desc zq0`` ..., taking ``categories`` in turn."""
    return pd.DataFrame([
        {"source_category": categories[i % len(categories)],
         "sample_description": f"desc zq{i}",
         "constitution_subcategory": "Sub", "entry_type": "harmful"}
        for i in range(n)
    ])


def _token_pipeline(tmp_path, gen, check):
    # Two registry names: one name is one call group, answered by one backend.
    return InputPipeline(
        gen=make_client(gen, "tok-gen"), check=make_client(check, "tok-check"),
        dataset_dir=tmp_path,
    )


def _rows(tmp_path):
    frames = [pd.read_csv(p, dtype=str, keep_default_na=False)
              for p in sorted(tmp_path.glob("*/samples.csv"))]
    return pd.concat(frames, ignore_index=True)


def test_pool_run_without_retries_writes_the_same_rows(tmp_path):
    from redact.dataset.io import _hash_text

    gen = _TokenBackend(lambda t, n: f"1. first sample {t}\n2. second sample {t}")
    # Every second sample of an odd entry is rejected.
    check = _TokenBackend(lambda t, n: "No: weak" if int(t[2:]) % 2 and n == 2 else "Yes")
    r = _token_pipeline(tmp_path, gen, check).run_from_constitution(
        _token_df(5), _PROMPT, samples_per_entry=2, batch_size=2, verbose=False)

    def row(i, which, accepted):
        text = f"{which} sample zq{i}"
        category = ("Violence", "Privacy")[i % 2]
        return {
            "sample_id": _hash_text(text), "sample": text, "category": category,
            "turn": "0", "accepted": str(accepted), "source": "constitution",
            "entry_type": "harmful", "subcategory": "Sub",
            "source_sample_description": f"desc zq{i}", "constitution_category": "",
            "source_group_tag": "", "template_style": "",
            "rejection_reason": "" if accepted else "No: weak",
        }

    expected = [row(i, "first", True) for i in range(5)]
    expected += [row(i, "second", i % 2 == 0) for i in range(5)]
    df = _rows(tmp_path)
    assert list(df.columns) == [
        "sample_id", "sample", "category", "subcategory", "entry_type", "turn",
        "accepted", "rejection_reason", "source", "source_sample_description",
        "constitution_category", "source_group_tag", "template_style",
    ]
    assert len(df) == 10
    assert {r["sample_id"]: r for r in df.to_dict("records")} == {
        r["sample_id"]: r for r in expected}
    assert (r.total_prompts_generated, r.total_prompts_accepted) == (10, 8)
    assert _constitution_inputs_ledger(tmp_path).completed() == {
        (f"desc zq{i}", "harmful", "") for i in range(5)}


def test_pool_fills_a_call_with_a_retry_and_a_new_entry(tmp_path):
    # zq1's first reply has no samples, so it is generated again.
    gen = _TokenBackend(
        lambda t, n: "sorry, nothing here" if (t, n) == ("zq1", 1) else f"1. sample {t}")
    check = _TokenBackend(lambda t, n: "Yes")
    r = _token_pipeline(tmp_path, gen, check).run_from_constitution(
        _token_df(6), _PROMPT, samples_per_entry=1, batch_size=2, verbose=False)

    # A chunk loop sends the retry alone ([2, 1, 2, 2]); the pool fills its
    # call with a new entry.
    assert [len(b) for b in gen.batches] == [2, 2, 2, 1]
    assert gen.batches[2] == ["zq1*", "zq4"]
    assert all(len(b) <= 2 for b in check.batches)
    assert r.total_entries_processed == 6 and r.skipped_entries == 0


def test_pool_resume_after_a_failed_append_runs_only_unfinished_entries(
        tmp_path, monkeypatch):
    import redact.content_moderation.generation as G

    real_append = G.append_samples
    appends = []

    def failing_second_append(samples, **kwargs):
        appends.append(len(samples))
        if len(appends) == 2:
            raise OSError("disk full")
        return real_append(samples, **kwargs)

    def run(gen):
        pipe = _token_pipeline(tmp_path, gen, _TokenBackend(lambda t, n: "Yes"))
        return pipe.run_from_constitution(_token_df(70, ("Violence",)), _PROMPT,
                                          samples_per_entry=1, batch_size=4, verbose=False)

    def reply(t, n):
        return f"1. sample {t}"

    monkeypatch.setattr(G, "append_samples", failing_second_append)
    with pytest.raises(OSError, match="disk full"):
        run(_TokenBackend(reply))
    monkeypatch.setattr(G, "append_samples", real_append)
    assert appends == [32, 32] and len(_rows(tmp_path)) == 32

    second = _TokenBackend(reply)
    run(second)
    assert sum(len(b) for b in second.batches) == 70 - 32
    df = _rows(tmp_path)
    assert sorted(df["sample"]) == sorted(f"sample zq{i}" for i in range(70))
def test_csv_rows_cover_an_entry_whose_ack_was_lost(tmp_path):
    """A crash between the append and the ack leaves rows with no ledger
    record; the CSV union still counts the entry done. Also for the default
    style ``""``: its empty cell reads back as NaN, not as ``""``."""
    pipe = _pipeline(tmp_path)
    pipe.run_from_constitution(_const_df(), _PROMPT, samples_per_entry=2, verbose=False)
    _constitution_inputs_ledger(tmp_path).path.unlink()   # the acks never made it to disk

    pipe2 = _pipeline(tmp_path)
    r = pipe2.run_from_constitution(_const_df(), _PROMPT, samples_per_entry=2, verbose=False)
    assert r.total_entries_processed == 0
    assert pipe2.gen.backend.calls == []
