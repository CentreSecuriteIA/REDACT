"""Tests for high-level pipeline functions."""

import json
import logging
import re

import pandas as pd
import pytest

from redact.exceptions import PromptTooLongError
from redact.llms.backends import ComputeConfig
from redact.pipelines import create_taxonomy
from tests.conftest import MockBackend


class TestCreateTaxonomy:
    def test_creates_json(self, tmp_path):
        tax = create_taxonomy(
            "test_tax",
            categories={"CatA": "Description A", "CatB": {"description": "B"}},
            description="Test taxonomy",
            taxonomy_dir=tmp_path,
        )
        path = tmp_path / "test_tax.json"
        assert path.exists()
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["name"] == "test_tax"
        assert "CatA" in loaded["categories"]
        assert loaded["categories"]["CatA"]["description"] == "Description A"

    def test_string_categories_normalized(self, tmp_path):
        tax = create_taxonomy(
            "norm_test",
            categories={"Cat": "just a string"},
            taxonomy_dir=tmp_path,
        )
        assert tax["categories"]["Cat"] == {"description": "just a string"}

    def test_aliases_and_groups(self, tmp_path):
        tax = create_taxonomy(
            "full_test",
            categories={"A": "desc"},
            aliases={"B": "A"},
            groups={"g1": ["A"]},
            taxonomy_dir=tmp_path,
        )
        assert tax["aliases"] == {"B": "A"}
        assert tax["groups"] == {"g1": ["A"]}

    def test_returns_dict(self, tmp_path):
        result = create_taxonomy("ret_test", {"X": "Y"}, taxonomy_dir=tmp_path)
        assert isinstance(result, dict)
        assert result["name"] == "ret_test"


class TestStandaloneInputsDeprecation:
    def test_standalone_generate_inputs_emits_deprecation_warning(self, tmp_path, monkeypatch):
        import redact.pipelines as P
        from redact import generate_inputs
        from tests.conftest import MockBackend, make_client

        monkeypatch.setattr(P.ModelClient, "create", lambda m: make_client(MockBackend(), m))
        # empty taxonomy -> no categories -> warning fires, then a fast empty return
        with pytest.warns(DeprecationWarning, match="constitution-seeded"):
            generate_inputs(
                data_dir=tmp_path, taxonomy={"categories": {}},
                num_categories=0, verbose=False,
            )


class TestGenerateOutputsManifest:
    """generate_outputs writes a plan manifest before the batch loop (offline)."""

    def test_manifest_written_with_one_row_per_input(self, tmp_path, monkeypatch):
        import redact.pipelines as P
        from redact import generate_outputs, paths
        from redact.dataset import Manifest
        from tests.conftest import MockBackend, make_client

        monkeypatch.setattr(P.ModelClient, "create", lambda m: make_client(MockBackend("a plain answer"), m))
        inputs = pd.DataFrame({
            "sample": ["prompt one", "prompt two", "prompt three"],
            "category": ["Cyber"] * 3,
            "entry_type": ["harmful"] * 3,
        })
        generate_outputs(data_dir=tmp_path, inputs=inputs, check_outputs=False, verbose=False)

        out = paths.output_responses_csv(tmp_path)
        rows = Manifest.sidecar(out).load()
        assert len(rows) == 3 and all(r["status"] == "planned" for r in rows)
        # planned input_ids match the output CSV's input_ids
        df = pd.read_csv(out)
        assert {r["input_id"] for r in rows} == set(df["input_id"].astype(str))

    def test_manifest_is_full_plan_even_on_partial_resume(self, tmp_path, monkeypatch):
        # The manifest is the full intended scope, written before the resume
        # filter — so a resume run that only generates the pending units still
        # records every planned unit in the manifest.
        import redact.pipelines as P
        from redact import generate_outputs, paths
        from redact.dataset import Ledger, Manifest, content_id
        from tests.conftest import MockBackend, make_client

        monkeypatch.setattr(P.ModelClient, "create", lambda m: make_client(MockBackend("ans"), m))
        inputs = pd.DataFrame({
            "sample": ["p one", "p two", "p three"],
            "category": ["Cyber"] * 3, "entry_type": ["harmful"] * 3,
        })
        out = paths.output_responses_csv(tmp_path)
        # Pre-seed the ledger as if "p one" was already completed.
        Ledger.sidecar(out, key_fields=("input_id",), casters={"input_id": str}).record(
            [{"input_id": content_id("p one")}]
        )

        generate_outputs(data_dir=tmp_path, inputs=inputs, check_outputs=False,
                         resume=True, verbose=False)

        assert len(Manifest.sidecar(out).load()) == 3   # full plan (all 3 units)
        assert len(pd.read_csv(out)) == 2               # only the 2 pending generated

    def test_the_callers_frame_is_left_as_it_was(self, tmp_path, monkeypatch):
        """The stage adds its id column to a frame of its own."""
        import redact.pipelines as P
        from redact import generate_outputs
        from tests.conftest import MockBackend, make_client

        monkeypatch.setattr(
            P.ModelClient, "create", lambda m: make_client(MockBackend("ans"), m))
        inputs = pd.DataFrame({
            "sample": ["p one", "p two", "p three"],
            "category": ["Cyber"] * 3, "entry_type": ["harmful"] * 3,
        }, index=[7, 3, 5])
        before = inputs.copy()
        df = generate_outputs(data_dir=tmp_path, inputs=inputs, check_outputs=False,
                              verbose=False)
        pd.testing.assert_frame_equal(inputs, before)
        assert list(df["input_prompt"]) == list(inputs["sample"])


def _seed_input_tree(data_dir, seeded: bool):
    """One accepted input row: in the seeded tree of a training run, or in
    the standalone tree. Returns the row's text."""
    from redact import paths
    from redact.dataset import append_samples

    tree = paths.constitution_inputs_dir if seeded else paths.datasets
    append_samples(
        ["zq0 the one input"], category="Cyber", turn=0, source="constitution",
        extra_columns=[{"subcategory": "phishing", "entry_type": "benign"}],
        dataset_dir=tree(data_dir),
    )
    return "zq0 the one input"


class TestGenerateOutputsFindsItsInputs:
    """Called without ``inputs=``, the stage reads the inputs of its run."""

    @pytest.mark.parametrize("seeded", [True, False], ids=["seeded", "standalone"])
    def test_the_inputs_of_the_run_are_found(self, tmp_path, monkeypatch, seeded):
        import redact.pipelines as P
        from redact import generate_outputs
        from redact.dataset import content_id
        from tests.conftest import MockBackend, make_client

        monkeypatch.setattr(
            P.ModelClient, "create", lambda m: make_client(MockBackend("ans"), m))
        text = _seed_input_tree(tmp_path, seeded)
        df = generate_outputs(data_dir=tmp_path, check_outputs=False, verbose=False)
        assert list(df["input_prompt"]) == [text]
        assert list(df["input_id"]) == [content_id(text)]
        assert list(df["entry_type"]) == ["benign"]

    def test_no_inputs_in_either_tree_raises(self, tmp_path):
        from redact import generate_outputs

        with pytest.raises(ValueError, match="No input samples found"):
            generate_outputs(data_dir=tmp_path, check_outputs=False, verbose=False)


class TestGenerateOutputsReadBack:
    def test_without_read_back_the_output_is_written_and_not_read(
            self, tmp_path, monkeypatch):
        import redact.content_moderation.generation as G
        import redact.pipelines as P
        from redact import generate_outputs, paths
        from tests.conftest import MockBackend, make_client

        def read_rows(path):
            raise AssertionError("the output was read back")

        monkeypatch.setattr(
            P.ModelClient, "create", lambda m: make_client(MockBackend("ans"), m))
        monkeypatch.setattr(G, "read_rows", read_rows)
        inputs = pd.DataFrame({
            "sample": ["p one", "p two"], "category": ["Cyber"] * 2,
            "entry_type": ["harmful"] * 2,
        })
        returned = generate_outputs(
            data_dir=tmp_path, inputs=inputs, check_outputs=False, read_back=False)
        assert returned.empty and len(returned.columns) == 0
        written = pd.read_csv(paths.output_responses_csv(tmp_path))
        assert sorted(written["input_prompt"]) == ["p one", "p two"]


class TestGenerateOutputsResume:
    def test_rows_already_in_the_csv_are_not_regenerated(self, tmp_path, monkeypatch):
        """A crash between the CSV append and the ledger ack leaves a row with no
        ack. Without the CSV union, resume regenerates that input and appends a
        *second* row for it — two rows, two sample_ids, no dedup downstream."""
        import redact.pipelines as P
        from redact import generate_outputs, paths
        from redact.dataset import content_id
        from tests.conftest import MockBackend, make_client

        out = paths.output_responses_csv(tmp_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Same column order the stage appends in, since the append is headerless.
        pd.DataFrame([{
            "input_id": content_id("p one"), "input_prompt": "p one",
            "category": "Cyber", "subcategory": "", "entry_type": "harmful",
            "model": "mock-model", "source": "", "sample_id": "prior",
            "output_response": "written before the crash",
            "accepted": True, "rejection_reason": "",
        }]).to_csv(out, index=False)

        backend = MockBackend("ans")
        monkeypatch.setattr(P.ModelClient, "create", lambda m: make_client(backend, m))
        inputs = pd.DataFrame({
            "sample": ["p one", "p two"],
            "category": ["Cyber"] * 2, "entry_type": ["harmful"] * 2,
        })
        df = generate_outputs(data_dir=tmp_path, inputs=inputs, check_outputs=False,
                              resume=True, verbose=False)

        assert len(backend.calls) == 1                      # only "p two" generated
        assert len(df) == 2                                 # prior row kept, one added
        assert list(df["input_id"]).count(content_id("p one")) == 1


class TestGenerateOutputsInternals:
    """Internals capture is a folder-per-input_id side channel — no CSV column,
    ever (see .claude/introspection_backend_plan.md's non-interference principle).
    """

    def test_no_internals_kwarg_for_default_backend(self, tmp_path, monkeypatch):
        import redact.pipelines as P
        from redact import generate_outputs
        from tests.conftest import MockBackend, make_client

        backend = MockBackend("ans")
        monkeypatch.setattr(P.ModelClient, "create", lambda m: make_client(backend, m))
        inputs = pd.DataFrame({
            "sample": ["p one"], "category": ["Cyber"], "entry_type": ["harmful"],
        })
        df = generate_outputs(data_dir=tmp_path, inputs=inputs, check_outputs=False, verbose=False)
        assert "internals_id" not in backend.calls[0]
        assert "internals_id" not in df.columns and "internals_ids" not in df.columns

    def test_gen_internals_id_is_input_id_slash_output(self, tmp_path, monkeypatch):
        import redact.pipelines as P
        from redact import generate_outputs
        from tests.conftest import MockBackend, make_client

        class _InternalsBackend(MockBackend):
            @property
            def compute_config(self) -> ComputeConfig:
                return ComputeConfig(supports_internals=True)

        backend = _InternalsBackend("ans")
        monkeypatch.setattr(P.ModelClient, "create", lambda m: make_client(backend, m))
        inputs = pd.DataFrame({
            "sample": ["p one", "p two"], "category": ["Cyber"] * 2, "entry_type": ["harmful"] * 2,
        })
        df = generate_outputs(data_dir=tmp_path, inputs=inputs, check_outputs=False, verbose=False)

        seen = {c["internals_id"] for c in backend.calls}
        expected = {f'{row["input_id"]}/output' for _, row in df.iterrows()}
        assert seen == expected
        # never a CSV column, regardless of capture
        assert "internals_id" not in df.columns

    def test_check_internals_id_is_input_id_slash_val_out(self, tmp_path, monkeypatch):
        import redact.pipelines as P
        from redact import generate_outputs
        from tests.conftest import MockBackend, make_client

        class _InternalsBackend(MockBackend):
            @property
            def compute_config(self) -> ComputeConfig:
                return ComputeConfig(supports_internals=True)

        gen_backend = _InternalsBackend("the answer")
        check_backend = _InternalsBackend("Yes, fine")
        backends = {"gen-model": gen_backend, "check-model": check_backend}
        monkeypatch.setattr(P.ModelClient, "create", lambda m: make_client(backends[m], m))
        inputs = pd.DataFrame({
            "sample": ["p one"], "category": ["Cyber"], "entry_type": ["harmful"],
        })
        generate_outputs(
            data_dir=tmp_path, inputs=inputs, model="gen-model", check_model="check-model",
            verbose=False,
        )
        assert check_backend.calls[0]["internals_id"].endswith("/val_out")
        assert gen_backend.calls[0]["internals_id"].endswith("/output")
        # same input_id root for both stages
        gen_root = gen_backend.calls[0]["internals_id"].split("/")[0]
        check_root = check_backend.calls[0]["internals_id"].split("/")[0]
        assert gen_root == check_root

    def test_check_internals_not_requested_when_check_backend_lacks_support(self, tmp_path, monkeypatch):
        import redact.pipelines as P
        from redact import generate_outputs
        from tests.conftest import MockBackend, make_client

        class _InternalsBackend(MockBackend):
            @property
            def compute_config(self) -> ComputeConfig:
                return ComputeConfig(supports_internals=True)

        gen_backend = _InternalsBackend("the answer")
        check_backend = MockBackend("Yes, fine")  # no internals support
        backends = {"gen-model": gen_backend, "check-model": check_backend}
        monkeypatch.setattr(P.ModelClient, "create", lambda m: make_client(backends[m], m))
        inputs = pd.DataFrame({
            "sample": ["p one"], "category": ["Cyber"], "entry_type": ["harmful"],
        })
        generate_outputs(
            data_dir=tmp_path, inputs=inputs, model="gen-model", check_model="check-model",
            verbose=False,
        )
        assert "internals_id" not in check_backend.calls[0]
        assert gen_backend.calls[0]["internals_id"].endswith("/output")


class TestGenerateOutputsSampleId:
    """sample_id is this row's own content-hash identity, distinct from input_id
    (which points back to the origin sample) — a library-wide convention, not
    specific to internals logging."""

    def test_sample_id_matches_hash_of_output_response(self, tmp_path, monkeypatch):
        import redact.pipelines as P
        from redact import generate_outputs
        from redact.dataset import content_id
        from tests.conftest import MockBackend, make_client

        monkeypatch.setattr(P.ModelClient, "create", lambda m: make_client(MockBackend("a plain answer"), m))
        inputs = pd.DataFrame({
            "sample": ["p one", "p two"], "category": ["Cyber"] * 2, "entry_type": ["harmful"] * 2,
        })
        df = generate_outputs(data_dir=tmp_path, inputs=inputs, check_outputs=False, verbose=False)

        assert "sample_id" in df.columns
        for _, row in df.iterrows():
            assert row["sample_id"] == content_id(row["output_response"])
            assert row["sample_id"] != row["input_id"]


class TestGenerateOutputsRetry:
    """max_attempts: a rejected response is regenerated with the verdict."""

    def _run(self, tmp_path, monkeypatch, gen_replies, check_replies, **kwargs):
        import redact.pipelines as P
        from redact import generate_outputs
        from tests.conftest import MockBackend, make_client

        gen, check = MockBackend(gen_replies), MockBackend(check_replies)
        backends = {"gen-model": gen, "check-model": check}
        monkeypatch.setattr(P.ModelClient, "create", lambda m: make_client(backends[m], m))
        inputs = pd.DataFrame({
            "sample": ["p one"], "category": ["Cyber"], "entry_type": ["harmful"],
        })
        df = generate_outputs(
            data_dir=tmp_path, inputs=inputs, model="gen-model",
            check_model="check-model", verbose=False, **kwargs,
        )
        return gen, df.iloc[0]

    def test_a_rejected_response_is_regenerated_with_the_verdict(self, tmp_path, monkeypatch):
        gen, row = self._run(tmp_path, monkeypatch, ["first answer", "second answer"],
                             ["No: too vague", "Yes"], max_attempts=2)
        assert len(gen.calls) == 2
        assert "No: too vague" not in gen.calls[0]["messages"][-1]["content"]
        assert "No: too vague" in gen.calls[1]["messages"][-1]["content"]
        assert row["output_response"] == "second answer" and bool(row["accepted"])

    def test_one_attempt_keeps_the_rejected_response(self, tmp_path, monkeypatch):
        gen, row = self._run(tmp_path, monkeypatch, ["first answer"], ["No: too vague"])
        assert len(gen.calls) == 1
        assert not bool(row["accepted"])
        assert row["rejection_reason"] == "No: too vague"

    def test_every_attempt_rejected_keeps_the_last_one(self, tmp_path, monkeypatch):
        gen, row = self._run(tmp_path, monkeypatch, ["first answer", "second answer"],
                             ["No: one", "No: two"], max_attempts=2)
        assert row["output_response"] == "second answer"
        assert not bool(row["accepted"]) and row["rejection_reason"] == "No: two"
        # The feedback is not stacked: the second prompt carries one verdict.
        assert gen.calls[1]["messages"][-1]["content"].count("[FEEDBACK") == 1

    def test_check_calls_carry_the_limit_and_generation_keeps_the_default(self, tmp_path, monkeypatch):
        import redact.pipelines as P
        from redact import generate_outputs
        from tests.conftest import MockBackend, make_client

        gen, check = MockBackend("an answer"), MockBackend("Yes")
        backends = {"gen-model": gen, "check-model": check}
        monkeypatch.setattr(P.ModelClient, "create", lambda m: make_client(backends[m], m))
        inputs = pd.DataFrame({"sample": ["p one"], "category": ["Cyber"], "entry_type": ["harmful"]})
        generate_outputs(data_dir=tmp_path, inputs=inputs, model="gen-model",
                         check_model="check-model", verbose=False, check_max_tokens=64)
        assert check.calls[0]["max_tokens"] == 64
        assert gen.calls[0]["max_tokens"] == gen.default_max_tokens

    def test_a_chunk_is_one_generation_call_and_one_check_call(self, tmp_path, monkeypatch):
        import redact.pipelines as P
        from redact import generate_outputs
        from tests.conftest import make_client
        from tests.llms.test_router import _CountingBackend

        gen = _CountingBackend("an answer", native=True)
        check = _CountingBackend("Yes", native=True)
        backends = {"gen-model": gen, "check-model": check}
        monkeypatch.setattr(P.ModelClient, "create", lambda m: make_client(backends[m], m))
        inputs = pd.DataFrame({
            "sample": ["p one", "p two", "p three"],
            "category": ["Cyber"] * 3, "entry_type": ["harmful"] * 3,
        })
        generate_outputs(data_dir=tmp_path, inputs=inputs, model="gen-model",
                         check_model="check-model", batch_size=3, verbose=False)
        assert (gen.generate_call_count, check.generate_call_count) == (1, 1)


class _TokenBackend(MockBackend):
    """Native-batching fake: ``reply(token, text)`` answers each request, where
    ``token`` is the ``zqN`` in its prompt and ``text`` all its content.

    Replies depend only on the request, never on call order, so rows can be
    compared across runs whose units finish in different orders. ``batches``
    holds one list per ``generate()`` call: the token of each request, with
    ``*`` for a request carrying the checker's feedback.
    """

    def __init__(self, reply, model):
        super().__init__(model=model)
        self._reply = reply
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
            batch.append(token + "*" * ("[FEEDBACK" in text))
            replies.append(self._reply(token, text))
        self.batches.append(batch)
        return replies


class _LaneBackend(MockBackend):
    """Async fake: answers ``answer <token>``, the ``zqN`` in the prompt, and
    rejects a prompt whose token is in ``too_long`` as over the context.

    ``sent`` holds the token of every request.
    """

    compute_config = ComputeConfig(supports_async_calls=True)

    def __init__(self, model, too_long=()):
        super().__init__(model=model)
        self.too_long = set(too_long)
        self.sent: list[str] = []

    async def agenerate(self, messages, *, max_tokens=None, internals_id=None,
                        **kwargs):
        token = re.search(r"zq\d+", " ".join(m["content"] for m in messages)).group()
        self.sent.append(token)
        if token in self.too_long:
            raise PromptTooLongError(f"{token} is over the context")
        return f"answer {token}"


class TestPoolSize:
    """A stage keeps as many units live as its models' async lanes hold."""

    @staticmethod
    def _client(async_calls, max_workers):
        from types import SimpleNamespace

        return SimpleNamespace(
            compute_config=SimpleNamespace(supports_async_calls=async_calls),
            backend=SimpleNamespace(max_workers=max_workers))

    def test_two_calls_worth_without_an_async_lane(self):
        from redact.content_moderation.generation import _pool_size

        assert _pool_size(32, self._client(False, 500), None) == 64

    def test_the_widest_lane_sets_the_pool(self):
        from redact.content_moderation.generation import _pool_size

        assert _pool_size(32, self._client(True, 256), self._client(True, 512)) == 512

    def test_a_lane_narrower_than_two_calls_changes_nothing(self):
        from redact.content_moderation.generation import _pool_size

        assert _pool_size(32, self._client(True, 3), self._client(True, 8)) == 64

    def test_the_outputs_stage_drives_with_it(self, tmp_path, monkeypatch):
        import redact.content_moderation.generation as G
        import redact.pipelines as P
        from redact import generate_outputs
        from tests.conftest import make_client

        backends = {"gen-model": MockBackend("answer"), "check-model": MockBackend("Yes")}
        monkeypatch.setattr(P.ModelClient, "create", lambda m: make_client(backends[m], m))
        seen = {}

        def stop(source, **kwargs):
            seen.update(kwargs)
            raise KeyboardInterrupt

        monkeypatch.setattr(G, "drive_units", stop)
        monkeypatch.setattr(G, "_pool_size", lambda batch_size, *clients: 7 * batch_size)
        inputs = pd.DataFrame({"sample": ["zq0"], "category": ["Cyber"],
                               "entry_type": ["harmful"]})
        with pytest.raises(KeyboardInterrupt):
            generate_outputs(data_dir=tmp_path, inputs=inputs, model="gen-model",
                             check_model="check-model", batch_size=2, verbose=False)
        assert (seen["pool_size"], seen["cap"]) == (14, 2)


class TestGenerateOutputsPool:
    """Outputs run through drive_units: full calls, rows as units finish."""

    def _clients(self, monkeypatch, gen, check):
        import redact.pipelines as P
        from tests.conftest import make_client

        backends = {"gen-model": gen, "check-model": check}
        monkeypatch.setattr(P.ModelClient, "create", lambda m: make_client(backends[m], m))

    def test_a_run_without_retries_writes_the_same_rows(self, tmp_path, monkeypatch):
        from redact import generate_outputs, paths
        from redact.dataset import content_id

        gen = _TokenBackend(lambda t, text: f"answer {t}", "gen-model")
        # Odd tokens rejected; one attempt, so no retries.
        check = _TokenBackend(
            lambda t, text: f"No: weak {t}" if int(t[2:]) % 2 else "Yes", "check-model")
        self._clients(monkeypatch, gen, check)
        tokens = [f"zq{i}" for i in range(6)]
        inputs = pd.DataFrame({
            "sample": tokens, "category": ["Cyber"] * 6, "entry_type": ["harmful"] * 6,
        })
        generate_outputs(data_dir=tmp_path, inputs=inputs, model="gen-model",
                         check_model="check-model", batch_size=2, verbose=False)

        df = pd.read_csv(paths.output_responses_csv(tmp_path), dtype=str,
                         keep_default_na=False)
        assert list(df.columns) == [
            "input_id", "input_prompt", "category", "subcategory", "entry_type",
            "model", "source", "sample_id", "output_response", "accepted",
            "rejection_reason",
        ]
        expected = {
            content_id(t): {
                "input_id": content_id(t), "input_prompt": t, "category": "Cyber",
                "subcategory": "", "entry_type": "harmful", "model": "gen-model",
                "source": "", "sample_id": content_id(f"answer {t}"),
                "output_response": f"answer {t}", "accepted": str(i % 2 == 0),
                "rejection_reason": f"No: weak {t}" if i % 2 else "",
            }
            for i, t in enumerate(tokens)
        }
        assert len(df) == 6
        assert {r["input_id"]: r for r in df.to_dict("records")} == expected

    def test_the_ledger_records_how_much_of_the_output_is_saved(
            self, tmp_path, monkeypatch):
        from redact import generate_outputs, paths
        from redact.dataset import Ledger

        gen = _TokenBackend(lambda t, text: f"answer {t}", "gen-model")
        self._clients(monkeypatch, gen, None)
        inputs = pd.DataFrame({
            "sample": [f"zq{i}" for i in range(3)],
            "category": ["Cyber"] * 3, "entry_type": ["harmful"] * 3,
        })
        generate_outputs(data_dir=tmp_path, inputs=inputs, model="gen-model",
                         check_outputs=False, verbose=False)
        out = paths.output_responses_csv(tmp_path)
        size, _checksum = Ledger.sidecar(out).saved["output_responses.csv"]
        assert size == out.stat().st_size

    def _six_inputs(self):
        return pd.DataFrame({
            "sample": [f"zq{i}" for i in range(6)],
            "category": ["Cyber"] * 6, "entry_type": ["harmful"] * 6,
        })

    def test_a_run_killed_between_rows_and_acks_leaves_each_input_once(
            self, tmp_path, monkeypatch):
        import redact.content_moderation.generation as G
        from redact import generate_outputs, paths
        from redact.dataset import Ledger, content_id
        from tests.conftest import Killed, kill_after_rows, lock_is_free

        inputs = self._six_inputs()

        def run():
            gen = _TokenBackend(lambda t, text: f"answer {t}", "gen-model")
            self._clients(monkeypatch, gen, None)
            generate_outputs(data_dir=tmp_path, inputs=inputs, model="gen-model",
                             check_outputs=False, batch_size=2, verbose=False)
            return gen

        out, prompts = paths.output_responses_csv(tmp_path), list(inputs["sample"])
        state = out.with_name("output_responses.state.jsonl")
        kill_after_rows(monkeypatch, G, on_call=1)
        with pytest.raises(Killed):
            run()
        landed = out.read_bytes()
        assert sorted(pd.read_csv(out)["input_prompt"]) == prompts  # the rows landed
        assert state.read_bytes() == b'{"_saved": {}}\n'            # their acks did not
        assert lock_is_free(state)

        second = run()
        assert sorted(sum(second.batches, [])) == prompts   # every input ran again
        assert sorted(pd.read_csv(out)["input_prompt"]) == prompts
        assert out.with_name("output_responses.csv.cut").read_bytes() == landed
        assert Ledger.sidecar(out, ("input_id",)).completed() == {
            content_id(prompt) for prompt in prompts}
        assert lock_is_free(state)

    def test_a_second_run_is_refused_before_it_writes_its_manifest(
            self, tmp_path, monkeypatch):
        from filelock import FileLock

        from redact import generate_outputs, paths

        gen = _TokenBackend(lambda t, text: f"answer {t}", "gen-model")
        self._clients(monkeypatch, gen, None)
        out = paths.output_responses_csv(tmp_path)
        out.parent.mkdir(parents=True)
        lock = str(out.with_name("output_responses.state.jsonl")) + ".lock"
        with FileLock(lock), pytest.raises(RuntimeError, match="locked"):
            generate_outputs(data_dir=tmp_path, inputs=self._six_inputs(),
                             model="gen-model", check_outputs=False, verbose=False)
        assert gen.batches == []
        assert [p for p in out.parent.iterdir() if p.suffix != ".lock"] == []

    def test_a_retry_shares_a_call_with_a_new_unit(self, tmp_path, monkeypatch):
        from redact import generate_outputs, paths

        gen = _TokenBackend(
            lambda t, text: f"better {t}" if "[FEEDBACK" in text else f"answer {t}",
            "gen-model")
        # zq1's first answer is rejected, everything else accepted.
        check = _TokenBackend(
            lambda t, text: "No: weak" if "answer zq1" in text else "Yes", "check-model")
        self._clients(monkeypatch, gen, check)
        inputs = pd.DataFrame({
            "sample": [f"zq{i}" for i in range(6)],
            "category": ["Cyber"] * 6, "entry_type": ["harmful"] * 6,
        })
        generate_outputs(data_dir=tmp_path, inputs=inputs, model="gen-model",
                         check_model="check-model", batch_size=2, max_attempts=2,
                         verbose=False)

        # A chunk loop sends the retry alone ([2, 1, 2, 2]); the pool fills its
        # call with the next input.
        assert [len(b) for b in gen.batches] == [2, 2, 2, 1]
        assert ["zq1*", "zq4"] in gen.batches
        df = pd.read_csv(paths.output_responses_csv(tmp_path))
        assert len(df) == 6 and df["accepted"].all()
        assert df.set_index("input_prompt").loc["zq1", "output_response"] == "better zq1"

    def test_a_failed_call_keeps_finished_rows_and_resumes_the_rest(self, tmp_path, monkeypatch):
        """A dispatch failure stops the stage; the units already finished are
        flushed and acked on the way out, and only the rest run again."""
        from redact import generate_outputs, paths

        class _DiesOnThirdCall(_TokenBackend):
            def generate(self, messages_list, **kwargs):
                if len(self.batches) == 2:
                    raise RuntimeError("engine died")
                return super().generate(messages_list, **kwargs)

        inputs = pd.DataFrame({
            "sample": [f"zq{i}" for i in range(10)],
            "category": ["Cyber"] * 10, "entry_type": ["harmful"] * 10,
        })
        self._clients(monkeypatch, _DiesOnThirdCall(lambda t, text: f"answer {t}", "gen-model"), None)
        with pytest.raises(RuntimeError, match="engine died"):
            generate_outputs(data_dir=tmp_path, inputs=inputs, model="gen-model",
                             check_outputs=False, batch_size=2, verbose=False)
        out = paths.output_responses_csv(tmp_path)
        ledger = out.with_name("output_responses.state.jsonl")
        assert len(pd.read_csv(out)) == 4                       # two calls of two finished
        opening, *acks = ledger.read_text(encoding="utf-8").splitlines()
        assert json.loads(opening) == {"_saved": {}} and len(acks) == 4

        second = _TokenBackend(lambda t, text: f"answer {t}", "gen-model")
        self._clients(monkeypatch, second, None)
        generate_outputs(data_dir=tmp_path, inputs=inputs, model="gen-model",
                         check_outputs=False, batch_size=2, verbose=False)
        assert sum(len(b) for b in second.batches) == 6
        df = pd.read_csv(out)
        assert sorted(df["input_prompt"]) == sorted(inputs["sample"])

    def test_resume_after_a_failed_append_runs_only_unfinished_units(
            self, tmp_path, monkeypatch):
        from redact import generate_outputs, paths

        inputs = pd.DataFrame({
            "sample": [f"zq{i}" for i in range(70)],
            "category": ["Cyber"] * 70, "entry_type": ["harmful"] * 70,
        })
        real_to_csv = pd.DataFrame.to_csv
        appends = []

        def failing_second_append(self, *args, **kwargs):
            appends.append(len(self))
            if len(appends) == 2:
                raise OSError("disk full")
            return real_to_csv(self, *args, **kwargs)

        first = _TokenBackend(lambda t, text: f"answer {t}", "gen-model")
        self._clients(monkeypatch, first, None)
        monkeypatch.setattr(pd.DataFrame, "to_csv", failing_second_append)
        with pytest.raises(OSError, match="disk full"):
            generate_outputs(data_dir=tmp_path, inputs=inputs, model="gen-model",
                             check_outputs=False, batch_size=4, verbose=False)
        monkeypatch.setattr(pd.DataFrame, "to_csv", real_to_csv)
        out = paths.output_responses_csv(tmp_path)
        assert appends == [32, 32] and len(pd.read_csv(out)) == 32

        second = _TokenBackend(lambda t, text: f"answer {t}", "gen-model")
        self._clients(monkeypatch, second, None)
        generate_outputs(data_dir=tmp_path, inputs=inputs, model="gen-model",
                         check_outputs=False, batch_size=4, verbose=False)

        df = pd.read_csv(out)
        assert sum(len(b) for b in second.batches) == 70 - 32
        assert sorted(df["input_prompt"]) == sorted(inputs["sample"])

    def test_a_prompt_too_long_is_a_rejected_row_and_is_not_sent_again(
            self, tmp_path, monkeypatch, caplog):
        from redact import generate_outputs, paths
        from redact.dataset import Ledger, content_id

        tokens = ["zq0", "zq1", "zq2"]
        inputs = pd.DataFrame({
            "sample": tokens, "category": ["Cyber"] * 3, "entry_type": ["harmful"] * 3,
        })

        def run():
            gen = _LaneBackend("gen-model", too_long={"zq1"})
            self._clients(monkeypatch, gen, None)
            generate_outputs(data_dir=tmp_path, inputs=inputs, model="gen-model",
                             check_outputs=False, verbose=False)
            return gen

        with caplog.at_level(logging.WARNING):
            first = run()
        assert sorted(first.sent) == tokens
        assert [r.getMessage() for r in caplog.records] == [
            f"Input {content_id('zq1')} rejected, its prompt is too long: "
            "zq1 is over the context"]

        out = paths.output_responses_csv(tmp_path)
        df = pd.read_csv(out, dtype=str, keep_default_na=False)
        rows = {r["input_prompt"]: r for r in df.to_dict("records")}
        assert rows["zq1"] == {
            "input_id": content_id("zq1"), "input_prompt": "zq1", "category": "Cyber",
            "subcategory": "", "entry_type": "harmful", "model": "gen-model",
            "source": "", "sample_id": content_id(""), "output_response": "",
            "accepted": "False",
            "rejection_reason": "prompt too long: zq1 is over the context",
        }
        assert len(df) == len(tokens)
        assert {t: (rows[t]["output_response"], rows[t]["accepted"])
                for t in ("zq0", "zq2")} == {
            "zq0": ("answer zq0", "True"), "zq2": ("answer zq2", "True")}
        ledger = Ledger.sidecar(out, key_fields=("input_id",),
                                casters={"input_id": str})
        assert ledger.completed() == {content_id(t) for t in tokens}

        assert run().sent == []
        assert len(pd.read_csv(out)) == len(tokens)

    def test_another_unit_error_still_stops_the_stage(self, tmp_path, monkeypatch):
        """Only a prompt that is too long becomes a row. This unit fails
        building its check, after its response arrived."""
        import redact.content_moderation.generation as G
        from redact import generate_outputs, paths
        from redact.dataset import Ledger

        def no_checker(**kwargs):
            def build(original, text):
                raise KeyError("no checker")
            return build

        monkeypatch.setattr(G, "build_output_quality_checker", no_checker)
        gen = _LaneBackend("gen-model")
        self._clients(monkeypatch, gen, MockBackend("Yes"))
        inputs = pd.DataFrame({
            "sample": ["zq0"], "category": ["Cyber"], "entry_type": ["harmful"],
        })
        with pytest.raises(KeyError, match="no checker"):
            generate_outputs(data_dir=tmp_path, inputs=inputs, model="gen-model",
                             check_model="check-model", verbose=False)
        assert gen.sent == ["zq0"]
        out = paths.output_responses_csv(tmp_path)
        assert not out.exists()
        ledger = Ledger.sidecar(out, key_fields=("input_id",),
                                casters={"input_id": str})
        assert ledger.completed() == set()
