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

from redact.content_moderation.inputs import (
    InputPipeline,
    _constitution_inputs_ledger,
    _constitution_inputs_manifest,
)
from redact.exceptions import PromptTooLongError
from redact.llms.backends import ComputeConfig
from tests.conftest import MockBackend, make_client

_PROMPT = {"system_prompt": "sys", "template": "Generate items about {sample_description}"}


def _pipeline(tmp_path):
    # Two registry names: one name and setup is one call group, answered by
    # the first request's backend. Each of the two entries gets its own
    # texts: a text the other entry wrote is dropped, and an entry left with
    # no row cannot be found in the files once the ledger is gone.
    replies = ["1. alpha sample\n2. beta sample", "1. gamma sample\n2. delta sample"]
    return InputPipeline(
        gen=make_client(MockBackend(replies), "m"),
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


def test_check_calls_carry_the_limit(tmp_path):
    gen, check = MockBackend("1. alpha sample\n2. beta sample"), MockBackend("Yes")
    pipe = InputPipeline(gen=make_client(gen, "m"), check=make_client(check, "m-check"),
                         dataset_dir=tmp_path)
    pipe.run_from_constitution(_const_df(), _PROMPT, samples_per_entry=2, verbose=False,
                               check_max_tokens=32)
    assert check.calls and all(c["max_tokens"] == 32 for c in check.calls)
    assert gen.calls[0]["max_tokens"] == gen.default_max_tokens


def test_an_entry_without_a_source_category_is_asked_and_saved_as_unknown(tmp_path):
    pipe = _pipeline(tmp_path)
    entry = _const_df().drop(columns="source_category").head(1)
    pipe.run_from_constitution(
        entry, {"system_prompt": "sys", "template": "{Category}: {sample_description}"},
        samples_per_entry=2, verbose=False)
    assert pipe.gen.backend.calls[0]["messages"][-1]["content"] == "unknown: desc one"
    assert (tmp_path / "unknown" / "samples.csv").exists()


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
    # ledger. Resume must still skip: the folder is taken over by its rows.
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
    from redact.dataset import content_id

    gen = _TokenBackend(lambda t, n: f"1. first sample {t}\n2. second sample {t}")
    # Every second sample of an odd entry is rejected.
    check = _TokenBackend(lambda t, n: "No: weak" if int(t[2:]) % 2 and n == 2 else "Yes")
    r = _token_pipeline(tmp_path, gen, check).run_from_constitution(
        _token_df(5), _PROMPT, samples_per_entry=2, batch_size=2, verbose=False)

    def row(i, which, accepted):
        text = f"{which} sample zq{i}"
        category = ("Violence", "Privacy")[i % 2]
        return {
            "sample_id": content_id(text), "sample": text, "category": category,
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
    import redact.content_moderation.inputs as G

    real_append = G.append_rows
    appends = []

    def failing_second_append(path, rows):
        appends.append(len(rows))
        if len(appends) == 2:
            raise OSError("disk full")
        return real_append(path, rows)

    def run(gen):
        pipe = _token_pipeline(tmp_path, gen, _TokenBackend(lambda t, n: "Yes"))
        return pipe.run_from_constitution(_token_df(70, ("Violence",)), _PROMPT,
                                          samples_per_entry=1, batch_size=4, verbose=False)

    def reply(t, n):
        return f"1. sample {t}"

    monkeypatch.setattr(G, "append_rows", failing_second_append)
    with pytest.raises(OSError, match="disk full"):
        run(_TokenBackend(reply))
    monkeypatch.setattr(G, "append_rows", real_append)
    assert appends == [32, 32] and len(_rows(tmp_path)) == 32

    second = _TokenBackend(reply)
    run(second)
    assert sum(len(b) for b in second.batches) == 70 - 32
    df = _rows(tmp_path)
    assert sorted(df["sample"]) == sorted(f"sample zq{i}" for i in range(70))
def test_csv_rows_cover_an_entry_whose_ack_was_lost(tmp_path):
    """Rows with no ledger beside them are a folder to take over: its entries
    count as done by their rows. Also for the default style ``""``, which is
    an empty cell in the file."""
    pipe = _pipeline(tmp_path)
    pipe.run_from_constitution(_const_df(), _PROMPT, samples_per_entry=2, verbose=False)
    _constitution_inputs_ledger(tmp_path).path.unlink()   # as before there was a ledger

    pipe2 = _pipeline(tmp_path)
    r = pipe2.run_from_constitution(_const_df(), _PROMPT, samples_per_entry=2, verbose=False)
    assert r.total_entries_processed == 0
    assert pipe2.gen.backend.calls == []


class _LaneGen(MockBackend):
    """Async fake generator: ``1. sample <token>`` for the ``zqN`` in the
    prompt, and a prompt whose token is in ``too_long`` is over the context.

    ``sent`` holds the token of every request.
    """

    compute_config = ComputeConfig(supports_async_calls=True)

    def __init__(self, too_long=()):
        super().__init__()
        self.too_long = set(too_long)
        self.sent: list[str] = []

    async def agenerate(self, messages, *, max_tokens=None, internals_id=None,
                        **kwargs):
        token = re.search(r"zq\d+", " ".join(m["content"] for m in messages)).group()
        self.sent.append(token)
        if token in self.too_long:
            raise PromptTooLongError(f"{token} is over the context")
        return f"1. sample {token}"


def test_an_entry_whose_prompt_is_too_long_is_acked_as_rejected(tmp_path, caplog):
    def run():
        gen = _LaneGen(too_long={"zq1"})
        pipe = _token_pipeline(tmp_path, gen, _TokenBackend(lambda t, n: "Yes"))
        return gen, pipe.run_from_constitution(
            _token_df(3), _PROMPT, samples_per_entry=1, verbose=False)

    with caplog.at_level(logging.WARNING):
        gen, r = run()
    assert sorted(gen.sent) == ["zq0", "zq1", "zq2"]    # zq1 is not generated again
    assert (r.total_entries_processed, r.skipped_entries) == (2, 1)
    assert sorted(_rows(tmp_path)["sample"]) == ["sample zq0", "sample zq2"]
    ledger = _constitution_inputs_ledger(tmp_path)
    # Without the sizes a save records, on its opening and on its last line.
    acks = [{k: v for k, v in json.loads(line).items() if k != "_saved"}
            for line in ledger.path.read_text(encoding="utf-8").splitlines()]
    assert [a for a in acks if a.get("sample_description") == "desc zq1"] == [{
        "sample_description": "desc zq1", "entry_type": "harmful", "style": "",
        "status": "rejected", "attempts": 1}]
    assert ledger.completed() == {(f"desc zq{i}", "harmful", "") for i in range(3)}
    too_long, summary = [record.getMessage() for record in caplog.records]
    assert too_long == ("Entry 'desc zq1' (harmful) rejected, its prompt is too "
                        "long: zq1 is over the context")
    assert summary.startswith("1 entries rejected")

    again, r = run()
    assert again.sent == [] and r.skipped_entries == 0


def test_another_unit_error_still_stops_the_stage(tmp_path, monkeypatch):
    """Only a prompt that is too long is acked as rejected. This entry fails
    building its check, after its samples arrived."""
    import redact.content_moderation.inputs as G

    def no_checker(**kwargs):
        raise KeyError("no checker")

    monkeypatch.setattr(G, "build_quality_checker", no_checker)
    gen = _LaneGen()
    pipe = _token_pipeline(tmp_path, gen, _TokenBackend(lambda t, n: "Yes"))
    with pytest.raises(KeyError, match="no checker"):
        pipe.run_from_constitution(_token_df(1), _PROMPT, samples_per_entry=1,
                                   verbose=False)
    assert gen.sent == ["zq0"]
    assert _constitution_inputs_ledger(tmp_path).completed() == set()
    assert list(tmp_path.glob("*/samples.csv")) == []
