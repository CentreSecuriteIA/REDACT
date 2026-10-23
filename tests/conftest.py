"""Shared test fixtures for the REDACT test suite."""

import multiprocessing
import time
from types import SimpleNamespace

import pandas as pd
import pytest


def pytest_configure(config):
    # vLLM v1 explicitly uses multiprocessing.get_context("fork") by default,
    # ignoring the global start method. Setting this env var before vLLM is
    # imported forces it to use 'spawn', which avoids CUDA re-init failures
    # in forked subprocesses. The library sets the variable itself at engine
    # load and makes no global call; the set_start_method here is the suite's
    # own fallback for anything else that starts a process.
    import os
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

from redact.llms.backends import ComputeConfig, LLMBackend, resolve_setup
from redact.llms.client import ModelClient
from redact.llms.model_config import (
    MODEL_REGISTRY,
    APIConfig,
    get_model_config,
    register_model,
)

@pytest.fixture(autouse=True)
def _api_keys(monkeypatch):
    """Give every registry model a dummy API key, so the suite needs no .env.

    A key that is already set (e.g. from a real .env) is left alone.
    """
    import os

    for config in list(MODEL_REGISTRY.values()):
        if config.api is not None and not os.environ.get(config.api.api_key_env):
            monkeypatch.setenv(config.api.api_key_env, "test-key")


@pytest.fixture(autouse=True)
def _no_planned_shares():
    """A planned share and card belong to one run, as does the way it serves
    its models: a test's must not reach the next."""
    yield
    from redact.llms.backends import capabilities, vllm

    vllm.set_planned_utilization({})
    vllm.set_planned_devices({})
    capabilities.set_run_serving({})


@pytest.fixture(autouse=True)
def _no_busy_cards(monkeypatch):
    """A run reads what the machine's cards hold before it loads a model,
    and a load names its cards as nvidia-smi does: a test's run sees no
    card, also on a machine with a busy one.

    No test asks the real torch either, which on a GPU machine would start
    CUDA in this process (the planner's fallback card size, a transformers
    load's measurement): the probe finds no CUDA, as on a machine without
    one. And the order CUDA counts the cards in, which a plan or a load sets
    unless it is set, is set already: the tests of that setting remove it."""
    from redact.llms.resources import measure

    monkeypatch.setattr(measure, "card_use", lambda: None)
    monkeypatch.setattr(measure, "gpu_cards", lambda visible=False: None)
    monkeypatch.setattr(measure, "_torch", lambda: None)
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")


@pytest.fixture(autouse=True)
def _no_glotlid(monkeypatch):
    """No test fetches GlotLID (1.69 GB) or runs it: the language checks see
    no identifier, as on a machine without the langid extra. Its own tests
    reset both, and a gate test patches ``label_probability``."""
    from redact.jailbreak.obfuscation import langid

    monkeypatch.setattr(langid, "_model", None)
    monkeypatch.setattr(langid, "_unavailable", True)


@pytest.fixture(autouse=True)
def _no_cut_warned():
    """A model's first cut reply in a stage warns once: a test's warning must
    not silence the next test's."""
    yield
    from redact.llms.backends import base

    base._cut_warned.clear()


@pytest.fixture
def no_length_floor(monkeypatch):
    """The outputs stage checks a response of any length: its fake replies
    (``answer zq0``) are shorter than the floor a real response must pass."""
    import redact.content_moderation.outputs as outputs

    monkeypatch.setattr(outputs, "MIN_RESPONSE_CHARS", 0)


def first_cut_warning(model: str, max_tokens: int) -> str:
    """The warning at a model's first reply cut at the token limit in a stage."""
    return (f"{model}: a reply was cut at the token limit (max_tokens={max_tokens}) "
            "and is incomplete; raise max_tokens or the context length "
            "(max_model_len), or ask for shorter replies. Later ones are only "
            "counted: see the stage's end and the telemetry summary.")


@pytest.fixture(autouse=True)
def _no_held_locks():
    """A stage's lock outlives a ``resume_state`` that no ``CommitBuffer``
    followed: a test's must not reach the next."""
    yield
    from redact.dataset import resume

    for lock in resume._locks.values():
        lock.release()
    resume._locks.clear()


# ---------------------------------------------------------------------------
# Mock LLM backend
# ---------------------------------------------------------------------------

class MockBackend(LLMBackend):
    """Configurable mock backend for testing.

    Accepts a list of responses to return sequentially, or a single
    string that is returned every time. Tracks all calls for assertions —
    one entry in ``self.calls`` per *batch item* (not per ``generate()``
    invocation), so existing assertions like ``calls[i]["messages"]`` or
    ``len(calls) == N`` keep meaning "the i-th/N logical samples," regardless
    of whether those samples arrived as one real batch call or several.

    Like every real backend it is a *configured model*: the model name and
    generation defaults are bound at construction, not passed per call.
    """

    def __init__(self, responses: str | list[str] = "mock response", model: str = "mock-model",
                 **identity):
        super().__init__(model, **identity)
        if isinstance(responses, str):
            self._responses = [responses]
            self._cycle = True
        else:
            self._responses = list(responses)
            self._cycle = False
        self._call_index = 0
        self.calls: list[dict] = []

    def generate(
        self,
        messages_list: list[list[dict]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        internals_ids: list[str | None] | None = None,
        **kwargs,
    ) -> list[str]:
        prompts, resolved = self._prepare(messages_list)
        max_tok, temp = self._resolve(max_tokens, temperature)
        if internals_ids is None:
            internals_ids = [None] * len(messages_list)

        # Emit one telemetry event for the whole generate(), exactly as a real
        # backend does — one *transport call*, not one per item. Tests asserting
        # on event granularity depend on this matching production.
        self._record_call(
            n_items=len(messages_list), started=time.perf_counter(),
            in_tok=sum(len(m[-1]["content"]) for m in resolved if m),
            out_tok=0, max_tokens=max_tok, temperature=temp,
        )

        results = []
        for messages, system_prompt, internals_id in zip(resolved, prompts, internals_ids):
            call_record = {
                "messages": messages,
                "model": self.model,
                "max_tokens": max_tok,
                **kwargs,
            }
            if temp is not None:
                call_record["temperature"] = temp
            if system_prompt is not None:
                call_record["system_prompt"] = system_prompt
            if internals_id is not None:
                call_record["internals_id"] = internals_id
            self.calls.append(call_record)

            if self._cycle:
                results.append(self._responses[0])
            elif self._call_index >= len(self._responses):
                results.append(self._responses[-1])
            else:
                results.append(self._responses[self._call_index])
                self._call_index += 1
        return results

    @property
    def backend_name(self) -> str:
        return "mock"


class ParaphraseBackend(MockBackend):
    """A paraphraser and its checker in one fake, answering by prompt kind.

    A request whose user content starts ``ORIGINAL:`` is a check: it gets
    ``check(original, paraphrase)``, both read from the check prompt's two
    blocks. Any other is a paraphrase: it gets ``paraphrase(model,
    sample)``, the sample read from after ``TEXT:\\n`` (the generic prompt) or
    before ``\\n\\n### Rewritten:`` (the humanizer's), else the whole content.
    By default a paraphrase is ``PARA::<sample>`` and every check says Yes.

    ``requests`` (shared when passed) gets one dict per item: ``kind``
    (``paraphrase`` or ``check``), ``model``, ``sample`` or ``original`` and
    ``paraphrase``, ``max_tokens`` and ``temperature`` as the call gave them.
    ``calls`` is ``MockBackend``'s.
    """

    def __init__(self, paraphrase=None, check=None, model: str = "mock-model",
                 requests: list | None = None, **identity):
        super().__init__(model=model, **identity)
        self._paraphrase = paraphrase or (lambda model, sample: f"PARA::{sample}")
        self._check = check or (lambda original, paraphrase: "Yes")
        self.requests = [] if requests is None else requests

    @property
    def rpm(self):
        """Never rate-limited: a fake answers at once, and the shared window
        of a hosted model's rpm would make a test wait on earlier tests'
        calls. ``make_client`` binds a setup's rpm; it is ignored."""
        return None

    @rpm.setter
    def rpm(self, value):
        pass

    def generate(self, messages_list, *, max_tokens=None, temperature=None,
                 internals_ids=None, **kwargs):
        super().generate(messages_list, max_tokens=max_tokens,
                         temperature=temperature, internals_ids=internals_ids,
                         **kwargs)
        return [self._answer(messages, max_tokens, temperature)
                for messages in messages_list]

    def _answer(self, messages, max_tokens, temperature) -> str:
        content = messages[-1]["content"]
        seen = {"model": self.model, "max_tokens": max_tokens,
                "temperature": temperature}
        if content.startswith("ORIGINAL:\n"):
            original, _, rest = content[len("ORIGINAL:\n"):].partition(
                "\n\nPARAPHRASE:\n")
            paraphrase = rest.rsplit("\n\nBegin your reply", 1)[0]
            self.requests.append({"kind": "check", "original": original,
                                  "paraphrase": paraphrase, **seen})
            return self._check(original, paraphrase)
        if content.startswith("TEXT:\n"):
            sample = content[len("TEXT:\n"):]
        elif "\n\n### Rewritten:" in content:
            sample = content.rsplit("\n\n### Rewritten:", 1)[0]
        else:
            sample = content
        self.requests.append({"kind": "paraphrase", "sample": sample, **seen})
        return self._paraphrase(self.model, sample)


def make_client(backend=None, model: str = "mock-model",
                backend_type: str | None = None) -> ModelClient:
    """Wrap a backend in a ModelClient for tests.

    Registers ``model`` if it isn't already, then binds the requested setup's
    identity/budget onto the backend exactly as ``backend_for()`` would — so a
    mock behaves like a real configured backend (defaults, rpm, workers).

    Pass ``backend_type`` to bind a specific setup (e.g. "vllm"), which is what
    decides whether the backend carries an rpm budget at all.
    """
    if model not in MODEL_REGISTRY:
        register_model(
            model,
            backend_type="api",
            api=APIConfig(backend_type="openai", api_key_env="TEST_API_KEY",
                          base_url="https://test.example/v1", rpm=1000),
        )
    config = get_model_config(model)
    setup = resolve_setup(config, backend_type)
    if backend is None:
        backend = MockBackend()
    _bind(backend, config, setup)
    return ModelClient(backend)


def _bind(backend, config, setup: str) -> None:
    """Re-run a backend's identity/budget binding against a registry entry.

    Mocks are constructed before the test picks a model, so this applies what
    ``LLMBackend.__init__`` would have been given by ``from_config()``.
    """
    api = config.api if setup == "api" else None
    LLMBackend.__init__(
        backend,
        rpm=api.rpm if api else None,
        max_workers=api.recommended_max_workers if api else 1,
        **LLMBackend._identity(config),
    )


class VLLMValidationError(ValueError):
    """Stand-in for ``vllm.exceptions.VLLMValidationError``."""

    def __init__(self, message, *, parameter=None, value=None):
        super().__init__(message)
        self.parameter = parameter
        self.value = value


def fake_vllm_modules(engine_cls, SamplingParams=lambda **kw: kw) -> dict[str, object]:
    """The ``sys.modules`` entries ``backends/vllm.py`` imports, with a fake engine.

    vLLM is not installed where the suite runs. Install the entries with
    ``patch.dict("sys.modules", fake_vllm_modules(cls))`` or one
    ``monkeypatch.setitem`` each.

    Args:
        engine_cls: Called with the engine arguments as keywords; what it
            returns is the loaded engine.
        SamplingParams: Stand-in for ``vllm.SamplingParams``. The default
            returns its keywords as a dict.
    """
    return {
        "vllm": SimpleNamespace(SamplingParams=SamplingParams),
        "vllm.engine": SimpleNamespace(),
        "vllm.engine.arg_utils": SimpleNamespace(AsyncEngineArgs=lambda **kw: kw),
        "vllm.exceptions": SimpleNamespace(VLLMValidationError=VLLMValidationError),
        "vllm.sampling_params": SimpleNamespace(
            RequestOutputKind=SimpleNamespace(FINAL_ONLY="FINAL_ONLY"),
            SamplingParams=SamplingParams),
        "vllm.usage": SimpleNamespace(),
        "vllm.usage.usage_lib": SimpleNamespace(
            UsageContext=SimpleNamespace(LLM_CLASS="LLM_CLASS")),
        "vllm.v1": SimpleNamespace(),
        "vllm.v1.engine": SimpleNamespace(),
        "vllm.v1.engine.async_llm": SimpleNamespace(AsyncLLM=SimpleNamespace(
            from_engine_args=lambda args, usage_context: engine_cls(**args))),
    }


def pinned_pool(monkeypatch, *names: str) -> None:
    """Make the ``paraphraser`` role, as the paraphrase stage and the recipe
    read it, give exactly ``names``, whatever roles the registry lists: a
    test of a pool of two does not move when a paraphraser is registered."""
    import redact.content_moderation.paraphrase as CP

    original = CP.get_models_by_role
    monkeypatch.setattr(CP, "get_models_by_role", lambda role: (
        [get_model_config(n) for n in names] if role == "paraphraser"
        else original(role)))


def interrupt_next_run(monkeypatch, when) -> None:
    """Make the next ``aio.run()`` raise ``KeyboardInterrupt`` from its wait.

    As Ctrl-C does to a caller blocked for a result. The coroutine is
    submitted for real; only the wait is interrupted, and only this once.

    Args:
        when: Called from the wait, just before it raises; block in it until
            the coroutine is where the interrupt should find it.
    """
    from redact.llms import aio

    submit = aio.submit

    def interrupted_submit(coro):
        monkeypatch.setattr(aio, "submit", submit)

        def result(timeout=None):
            when()
            raise KeyboardInterrupt

        future = submit(coro)
        future.result = result
        return future

    monkeypatch.setattr(aio, "submit", interrupted_submit)


class Killed(BaseException):
    """Stands for the death of the process, where ``kill_after_rows`` puts it."""


def kill_after_rows(monkeypatch, module, on_call: int) -> None:
    """Kill a stage between the rows of a save and their acks.

    The ``on_call``-th ``append_rows`` call made through ``module`` writes
    its rows and then raises :class:`Killed`; the appends after it are the
    real ones again.
    """
    append, calls = module.append_rows, []

    def dying(path, rows):
        calls.append(path)
        sizes = append(path, rows)
        if len(calls) == on_call:
            monkeypatch.setattr(module, "append_rows", append)
            raise Killed
        return sizes

    monkeypatch.setattr(module, "append_rows", dying)


def lock_is_free(ledger_path) -> bool:
    """Whether another run could take the lock of the stage with this ledger."""
    from filelock import FileLock, Timeout

    lock = FileLock(str(ledger_path) + ".lock", timeout=0)
    try:
        lock.acquire()
    except Timeout:
        return False
    lock.release()
    return True


@pytest.fixture()
def mock_backend():
    """Return a MockBackend that always returns 'mock response'."""
    return MockBackend()


@pytest.fixture()
def mock_client():
    """Return a ModelClient wrapping a MockBackend ('mock response')."""
    return make_client()


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_taxonomy():
    """Minimal taxonomy dict for testing."""
    return {
        "name": "test_taxonomy",
        "description": "Test taxonomy",
        "categories": {
            "Cat A": {
                "description": "Category A description",
                "subcategories": ["Sub1", "Sub2"],
            },
            "Cat B": {
                "description": "Category B description",
                "subcategories": ["Sub3", "Sub4"],
                "group_tag": "group1",
            },
        },
    }


@pytest.fixture()
def sample_prompt_config():
    """Minimal prompt config dict for testing."""
    return {
        "category": "test",
        "pipeline": "test_pipeline",
        "system_prompt": "You are a test assistant.",
        "template": "Generate {num_samples} items about {topic}.",
        "seed_fields": ["num_samples", "topic"],
        "few_shot_examples": [],
        "metadata": {"version": "1.0"},
    }


@pytest.fixture()
def sample_dataframe():
    """Small DataFrame for dataset tests."""
    return pd.DataFrame({
        "sample": ["hello world", "foo bar", "hello world", "baz qux"],
        "category": ["Cat A", "Cat A", "Cat B", "Cat B"],
        "accepted": [True, True, True, False],
        "origin": ["generated", "handcrafted", "generated", "generated"],
    })


@pytest.fixture()
def tmp_dataset_dir(tmp_path):
    """Temporary dataset directory structure."""
    ds_dir = tmp_path / "Datasets"
    ds_dir.mkdir()
    return ds_dir


# A small synthetic constitution, placeholder text only.
CONSTITUTION = pd.DataFrame([
    {"source_category": "Category A", "constitution_category": "Topic A",
     "constitution_subcategory": "Subcategory A",
     "sample_description": "placeholder sample description one",
     "entry_type": "harmful"},
    {"source_category": "Category B", "constitution_category": "Topic B",
     "constitution_subcategory": "Subcategory B",
     "sample_description": "placeholder sample description two",
     "entry_type": "benign"},
])


def save_constitution(data_dir, entries: pd.DataFrame = CONSTITUTION) -> None:
    """Save ``entries`` as the constitution under ``data_dir``, where the
    constitution stage saves it, so a run's inputs stage can start."""
    from redact import paths

    merged = paths.constitution_merged_csv(data_dir)
    merged.parent.mkdir(parents=True, exist_ok=True)
    entries.to_csv(merged, index=False)


def write_inputs(tree, texts, category="Cyber", *, accepted=None,
                 source="generated", extra=None) -> None:
    """Append input rows of ``texts`` to ``category``'s file under ``tree``,
    as the inputs stage writes them.

    ``accepted`` holds one flag per text (all accepted by default); ``extra``
    holds further columns, one dict for every row or a list of one per text.
    """
    from redact import paths
    from redact.dataset import append_rows, sample_row

    accepted = [True] * len(texts) if accepted is None else accepted
    extras = extra if isinstance(extra, list) else [extra or {}] * len(texts)
    path = paths.category_csv(category, root=tree)
    path.parent.mkdir(parents=True, exist_ok=True)
    append_rows(path, [
        sample_row(text, category, turn=0, accepted=flag, source=source, extra=more)
        for text, flag, more in zip(texts, accepted, extras)])


def as_resolver(fake, backends=None):
    """Adapt an old-style fake router to the ``resolve(model) -> client`` seam.

    ``drive_generators`` used to take an object with
    ``batch_generate(model, messages_list, **kw)``; it now resolves a client
    per model and calls ``client.generate(messages_list, **kw)``. Fakes that
    record calls keep working unchanged behind this adapter.

    Pass ``backends={model: backend}`` when a test also asserts on
    ``rename_capture`` — the internals-capture relabel goes to that model's
    backend, the way a real client's would.
    """
    class _Bound:
        def __init__(self, model):
            self.model = model

        def generate(self, messages_list, **kw):
            return fake.batch_generate(self.model, messages_list, **kw)

        def rename_capture(self, old_id, new_id):
            # Route to the model's backend when one is supplied; otherwise a
            # no-op, exactly as on any non-capturing transport.
            target = (backends or {}).get(self.model, fake)
            recorder = getattr(target, "rename_capture", None)
            if recorder is not None:
                recorder(old_id, new_id)

    return _Bound


# ---------------------------------------------------------------------------
# Multi-turn doubles
# ---------------------------------------------------------------------------

class FakeClient:
    """Stands in for a ModelClient: ``reply(model, messages)`` answers each request.

    Its calls are synchronous, as its ``compute_config`` says.
    """

    compute_config = ComputeConfig(supports_async_calls=False)

    def __init__(self, model, reply):
        self.model = model
        self._reply = reply

    def generate(self, messages_list, **kw):
        return [self._reply(self.model, msgs) for msgs in messages_list]


class FakeResolver:
    """``model -> FakeClient`` answering with ``reply``."""

    def __init__(self, reply):
        self._reply = reply

    def __call__(self, model):
        return FakeClient(model, self._reply)


class JudgeResolver:
    """``model -> client`` over one fixed backend."""

    def __init__(self, backend):
        self._backend = backend

    def __call__(self, model):
        return make_client(self._backend, model)


def echo(model, messages):
    """A reply that names the last user message."""
    last_user = [m for m in messages if m["role"] == "user"][-1]["content"]
    return f"ANSWER to: {last_user}"


def simple_setting():
    """A scripted user, then one model reply: one call per conversation."""
    from redact.multi_turn import ModelActor, ScriptedActor, Setting

    return Setting(participants=[ScriptedActor("u", ["f"]), ModelActor("b", "m")],
                   max_turns=2)
