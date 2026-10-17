"""vLLM backend integration tests.

Requires: vllm installed, Linux + CUDA. The GPU-marked classes load a real
checkpoint, so they need enough free VRAM for whichever model they run.

Run all:      pytest tests/test_vllm_backend.py -v
Run non-GPU:  pytest tests/test_vllm_backend.py -v -m "not gpu"

The GPU tests only exercise code paths (batching, ordering, prompt format),
never generation quality, so any small instruct checkpoint works. Override
the default via env vars when the default doesn't fit your GPU or isn't
cached locally::

    REDACT_TEST_VLLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct \
    REDACT_TEST_VLLM_KWARGS='{"max_model_len":512}' \
    pytest tests/test_vllm_backend.py -v
"""

import json
import logging
import os
import subprocess
import sys
import textwrap
import types

import pytest

from redact.exceptions import PromptTooLongError
from redact.llms import ModelClient, aio, observe
from redact.llms.backends import (
    ComputeConfig,
    LLMBackend,
    backend_for,
    clear_transport_caches,
)
from redact.llms.model_config import (
    MODEL_REGISTRY,
    APIConfig,
    ModelConfig,
    VLLMConfig,
    get_model_config,
    register_model,
)
from redact.llm_pipeline import batch_check_samples, drive_units
from tests.conftest import (
    VLLMValidationError,
    fake_vllm_modules,
    first_cut_warning,
    interrupt_next_run,
    make_client,
)
from tests.llm_pipeline.test_drive_async import _Script, _unit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TEST_MODEL_NAME = "test-gemma-2b"

# Which real checkpoint the GPU tests load, and the engine kwargs used to
# fit it on the available card. Both overridable — see the module docstring.
TEST_HF_ID = os.environ.get(
    "REDACT_TEST_VLLM_MODEL", "VibeStudio/Nidum-Gemma-2B-Uncensored"
)
TEST_VLLM_KWARGS: dict = {
    "enforce_eager": True,  # skip the slow CUDA-graph compile step
    **json.loads(os.environ.get("REDACT_TEST_VLLM_KWARGS", "{}")),
}


def _vllm_available() -> bool:
    """Check if vLLM can run: requires Linux + CUDA GPU."""
    import sys
    if sys.platform != "linux":
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


gpu = pytest.mark.skipif(not _vllm_available(), reason="Requires Linux + CUDA GPU (vLLM)")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="class")
def vllm_model():
    """Register a small vLLM model for testing; clean up after."""
    register_model(
                TEST_MODEL_NAME,
                default_max_tokens=100,
                default_temperature=0.7,
                backend_type="vllm",
                vllm=VLLMConfig(hf_model_id=TEST_HF_ID, vllm_kwargs=dict(TEST_VLLM_KWARGS)),
            )
    yield TEST_MODEL_NAME
    MODEL_REGISTRY.pop(TEST_MODEL_NAME, None)
    clear_transport_caches()


@pytest.fixture(autouse=True)
def _clean_cache(request):
    """Ensure transport caches are clean between tests.

    Skipped for GPU tests — loading a vLLM model takes minutes, so the
    GPU class manages its own cache lifetime via the vllm_model fixture.
    """
    yield
    if "TestVLLMBackendGPU" not in request.node.nodeid:
        clear_transport_caches()


# ---------------------------------------------------------------------------
# Non-GPU tests (config and routing logic)
# ---------------------------------------------------------------------------

class TestModelConfigVLLMFields:
    """Verify ModelConfig accepts and stores vLLM-specific fields."""

    def test_fields_present(self):
        config = ModelConfig(
            name="test",
            backend_type="vllm",
            vllm=VLLMConfig(
                hf_model_id="org/model",
                quantization="gptq",
                vllm_kwargs={"max_model_len": 512},
            ),
        )
        assert config.vllm.hf_model_id == "org/model"
        assert config.vllm.quantization == "gptq"
        assert config.vllm.vllm_kwargs == {"max_model_len": 512}

    def test_fields_default_none(self):
        config = ModelConfig(name="test", api=APIConfig(
            backend_type="openai", api_key_env="TEST_API_KEY",
            base_url="https://test.example/v1", rpm=10))
        assert config.vllm is None

    def test_register_model_passes_vllm_fields(self):
        name = "_test_register_vllm"
        try:
            register_model(
                name,
                backend_type="vllm",
                vllm=VLLMConfig(hf_model_id="org/model", quantization="awq", vllm_kwargs={"trust_remote_code": True}),
            )
            config = get_model_config(name)
            assert config.backend_type == "vllm"
            assert config.vllm.hf_model_id == "org/model"
            assert config.vllm.quantization == "awq"
            assert config.vllm.vllm_kwargs == {"trust_remote_code": True}
        finally:
            MODEL_REGISTRY.pop(name, None)

    @pytest.mark.parametrize("copies", [1, 2, None])
    def test_data_parallel_size_is_not_a_setting(self, copies):
        """Whatever the value: the planner decides it from the cards."""
        with pytest.raises(ValueError, match=(
                r"data_parallel_size'\] is not a setting: the planner decides "
                r"it from the cards; set copies in the recipe")):
            VLLMConfig(hf_model_id="org/model",
                       vllm_kwargs={"data_parallel_size": copies})


class TestEngineDownloadDir:
    """The HF cache is shared by both local runtimes — don't fork it.

    Regression guard for a removed HF_HOME -> download_dir bridge. vLLM passes
    download_dir through as ``cache_dir=`` to snapshot_download, and the hub
    cache is ``$HF_HOME/hub``, *not* ``$HF_HOME``. Setting it to HF_HOME
    therefore aimed vLLM one level above the cache transformers uses and
    re-downloaded weights that were already on disk (~47GB for the 24B). It was
    inert unless HF_HOME was set, so it only ever broke the users who set it
    because disk was scarce.
    """

    @staticmethod
    def _load(monkeypatch, vllm_kwargs):
        """Run the load path against a stub engine; return what vLLM was
        handed: the engine arguments, and the usage context under its name."""
        import redact.llms.backends.vllm as vllm_module

        seen = {}

        class _StubLLM:
            def __init__(self, **kw):
                seen.update(kw)

        modules = fake_vllm_modules(_StubLLM)
        async_llm = modules["vllm.v1.engine.async_llm"].AsyncLLM
        build = async_llm.from_engine_args
        async_llm.from_engine_args = (
            lambda args, **context: seen.update(context) or build(args, **context))
        for name, module in modules.items():
            monkeypatch.setitem(sys.modules, name, module)
        monkeypatch.setattr(vllm_module, "_prepare_environment", lambda: None)
        # The load reads the HF config for the default context length.
        monkeypatch.setattr(
            "redact.llms.resources.estimate._load_config", lambda _id: None)
        vllm_module._engines.clear()
        vllm_module._engine_loaded_at.clear()
        try:
            vllm_module._engine("org/model", None, vllm_kwargs)
        finally:
            vllm_module._engines.clear()
            vllm_module._engine_loaded_at.clear()
        return seen

    def test_hf_home_is_not_bridged_into_download_dir(self, monkeypatch):
        monkeypatch.setenv("HF_HOME", "/data/hf")
        seen = self._load(monkeypatch, {})
        assert "download_dir" not in seen, (
            "HF_HOME must reach vLLM through huggingface_hub, which resolves it "
            "to $HF_HOME/hub — injecting it as download_dir points one level too "
            "high and forks the cache."
        )

    def test_explicit_download_dir_still_wins(self, monkeypatch):
        monkeypatch.setenv("HF_HOME", "/data/hf")
        seen = self._load(monkeypatch, {"download_dir": "/mnt/weights"})
        assert seen["download_dir"] == "/mnt/weights"

    def test_caller_kwargs_are_not_mutated(self, monkeypatch):
        """`kwargs = dict(vllm_kwargs)` — the registry's dict is shared."""
        monkeypatch.setenv("HF_HOME", "/data/hf")
        original = {"gpu_memory_utilization": 0.9}
        self._load(monkeypatch, original)
        assert original == {"gpu_memory_utilization": 0.9}


class TestEngineArguments:
    """What a load hands vLLM beyond the registry's kwargs, as ``vllm.LLM`` did."""

    def test_stats_logging_is_off_unless_the_kwargs_turn_it_on(self, monkeypatch):
        load = TestEngineDownloadDir._load
        assert load(monkeypatch, {})["disable_log_stats"] is True
        assert load(monkeypatch, {"disable_log_stats": False})["disable_log_stats"] is False

    def test_the_engine_is_built_in_the_usage_context_of_vllm_llm(self, monkeypatch):
        """vLLM's defaults for running sequences and batched tokens depend on
        the context, and the engine's own has lower ones."""
        seen = TestEngineDownloadDir._load(monkeypatch, {})
        assert seen["usage_context"] == "LLM_CLASS"    # the fake's UsageContext.LLM_CLASS


class TestWorkerStartMethod:
    """vLLM reads its worker start method from an environment variable, so
    Python's process-wide start method is left alone."""

    def test_spawn_is_set_unless_a_method_is_already_chosen(self, monkeypatch):
        import multiprocessing

        import redact.llms.backends.vllm as vllm_module

        monkeypatch.setattr(
            multiprocessing, "set_start_method",
            lambda *a, **k: pytest.fail("changed the global start method"))
        monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "fork")
        vllm_module._prepare_environment()
        assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "fork"
        monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD")
        vllm_module._prepare_environment()
        assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"


class TestExitHook:
    """A process that ends with an engine loaded shuts it down first, or
    multiprocessing's exit hook waits for the engine's child process forever.
    A load in progress is waited for, at most ``EXIT_WAIT_S``."""

    # The scripts import tests.conftest, so they run from the repository root.
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def test_exit_shuts_down_a_loaded_engine(self):
        code = textwrap.dedent("""
            import multiprocessing, sys, time
            import redact.llms.backends.vllm as vllm_module
            import redact.llms.resources.estimate as estimate
            from tests.conftest import fake_vllm_modules

            class _LLM:
                def __init__(self, **kw):
                    self.proc = multiprocessing.get_context("spawn").Process(
                        target=time.sleep, args=(60,))
                    self.proc.start()

                def shutdown(self):
                    self.proc.terminate()
                    self.proc.join()
                    print("shut down", flush=True)

            sys.modules.update(fake_vllm_modules(_LLM))
            vllm_module._prepare_environment = lambda: None
            estimate._load_config = lambda _id: None
            vllm_module._engine("org/x", None, {})
        """)
        result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                                text=True, timeout=60, cwd=self.ROOT)
        assert result.returncode == 0, result.stderr
        assert "shut down" in result.stdout

    def test_exit_during_a_load_waits_for_it_then_shuts_down(self):
        """The hook is registered when a load starts. A script that exits while
        a background load is building the engine blocks on the engine lock until
        the load is done, then shuts the engine down. Registered after the load
        instead, the hook never exists and multiprocessing's exit hook joins the
        engine's child process for as long as it lives."""
        code = textwrap.dedent("""
            import multiprocessing, sys, threading, time
            import redact.llms.backends.vllm as vllm_module
            import redact.llms.resources.estimate as estimate
            from tests.conftest import fake_vllm_modules

            started = threading.Event()

            class _LLM:
                def __init__(self, **kw):
                    self.proc = multiprocessing.get_context("spawn").Process(
                        target=time.sleep, args=(60,))
                    self.proc.start()        # the engine's child process exists...
                    started.set()
                    time.sleep(1)            # ...while the load goes on

                def shutdown(self):
                    self.proc.terminate()
                    self.proc.join()
                    print("shut down", flush=True)

            sys.modules.update(fake_vllm_modules(_LLM))
            vllm_module._prepare_environment = lambda: None
            estimate._load_config = lambda _id: None
            threading.Thread(target=vllm_module._engine, args=("org/x", None, {}),
                             daemon=True).start()
            started.wait()                   # exit mid-load, as a script does after preload()
        """)
        result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                                text=True, timeout=30, cwd=self.ROOT)
        assert result.returncode == 0, result.stderr
        assert "shut down" in result.stdout


    #: Seconds within which a script whose load hangs has exited.
    EXIT_BOUND_S = 30

    # A script whose fake engine starts a child process, prints its pid and
    # then does what the script's {WAIT} says; the exit waits 1 s for it.
    HUNG = textwrap.dedent("""
        import logging, multiprocessing, sys, threading, time
        import redact.llms.backends.vllm as vllm_module
        import redact.llms.resources.estimate as estimate
        from tests.conftest import fake_vllm_modules

        logging.basicConfig(level=logging.INFO)
        started = threading.Event()

        class _LLM:
            def __init__(self, **kw):
                proc = multiprocessing.get_context("spawn").Process(
                    target=time.sleep, args=(60,), name="EngineCore")
                proc.start()
                print(proc.pid, flush=True)
                started.set()
                {WAIT}

        sys.modules.update(fake_vllm_modules(_LLM))
        vllm_module._prepare_environment = lambda: None
        estimate._load_config = lambda _id: None
        vllm_module.EXIT_WAIT_S, vllm_module._GRACE_S = 1.0, 0.5
        threading.Thread(target=vllm_module._engine, args=("org/x", None, {{}}),
                         daemon=True).start()
        started.wait()
    """)

    def _run_hung(self, wait):
        psutil = pytest.importorskip("psutil")
        import time

        code = self.HUNG.format(WAIT=wait)
        started = time.monotonic()
        result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                                text=True, timeout=60, cwd=self.ROOT)
        took = time.monotonic() - started
        assert result.returncode == 0, result.stderr
        assert took < self.EXIT_BOUND_S, took
        assert "The load of org/x has not returned" in result.stderr
        # Never os.kill(pid, 0) to check: on Windows that ends the process.
        assert not psutil.pid_exists(int(result.stdout.split()[0]))
        return result

    def test_a_load_that_fails_once_its_process_is_ended_lets_the_exit_finish(self):
        """vLLM's start-up wait fails as soon as an engine process exits:
        the load lets go of the lock and the exit releases as usual."""
        result = self._run_hung(
            "while proc.is_alive(): time.sleep(0.05)\n"
            "        raise RuntimeError('Engine core initialization failed.')")
        assert "still held" not in result.stderr

    def test_a_load_that_never_returns_does_not_keep_the_process_up(self):
        result = self._run_hung("threading.Event().wait()")
        assert "still held" in result.stderr


class _Proc:
    """A ``multiprocessing`` handle of a child process. ``stubborn`` ones
    outlive SIGTERM; ``immortal`` ones outlive SIGKILL too."""

    def __init__(self, pid, name="EngineCore", done=None, stubborn=False,
                 immortal=False):
        self.pid, self.name = pid, name
        self.done = done if done is not None else []
        self.alive, self.stubborn, self.immortal = True, stubborn, immortal

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.done.append(("terminate", self.pid))
        self.alive = self.alive and self.stubborn

    def kill(self):
        self.done.append(("kill", self.pid))
        self.alive = self.alive and self.immortal

    def join(self, timeout=None):
        self.done.append(("join", self.pid))


def _fake_psutil(children, done, stubborn=(), immortal=()):
    """A stand-in for psutil: ``children[pid]`` are the pids of a process's
    descendants, which outlive SIGTERM when in ``stubborn`` and SIGKILL too
    when in ``immortal``."""
    alive = {pid for kids in children.values() for pid in kids}

    class Process:
        def __init__(self, pid):
            self.pid = pid

        def children(self, recursive=False):
            assert recursive
            return [Process(pid) for pid in children.get(self.pid, ())]

        def terminate(self):
            done.append(("terminate", self.pid))
            if self.pid not in stubborn:
                alive.discard(self.pid)

        def kill(self):
            done.append(("kill", self.pid))
            if self.pid not in immortal:
                alive.discard(self.pid)

    def wait_procs(procs, timeout):
        procs = list(procs)
        done.append(("wait", sorted(p.pid for p in procs)))
        return ([p for p in procs if p.pid not in alive],
                [p for p in procs if p.pid in alive])

    return types.SimpleNamespace(Process=Process, wait_procs=wait_procs, Error=OSError)


class TestEngineProcesses:
    """A build records the processes it started, and those are what is
    ended when an engine has to be stopped: never another child of this
    process."""

    @pytest.fixture()
    def children(self, monkeypatch):
        """``multiprocessing.active_children()`` as a list a test fills. psutil
        is a fake that knows no children: the pids here are made up."""
        import multiprocessing

        import redact.llms.backends.vllm as vllm_module

        listed = []
        monkeypatch.setattr(multiprocessing, "active_children",
                            lambda: [p for p in listed if p.is_alive()])
        monkeypatch.setitem(sys.modules, "psutil", _fake_psutil({}, []))
        monkeypatch.setattr(vllm_module, "_engine_procs", {})
        monkeypatch.setattr(vllm_module, "_stray", [])
        monkeypatch.setattr(vllm_module, "_END_WAIT_S", 0.01)
        return vllm_module, listed

    @staticmethod
    def _build(monkeypatch, listed, started, fail=False):
        """Load ``org/p`` with a fake engine whose build starts ``started``."""
        import redact.llms.backends.vllm as vllm_module

        class _LLM:
            scheduler_config = types.SimpleNamespace(max_num_seqs=8)

            def __init__(self, **kw):
                listed.extend(started)
                if fail:
                    raise RuntimeError("engine core died")

        for name, module in fake_vllm_modules(_LLM).items():
            monkeypatch.setitem(sys.modules, name, module)
        monkeypatch.setattr(vllm_module, "_prepare_environment", lambda: None)
        monkeypatch.setattr(
            "redact.llms.resources.estimate._load_config", lambda _id: None)
        return vllm_module._engine("org/p", None, {})

    def test_a_build_records_the_processes_it_started(self, children, monkeypatch):
        vllm_module, listed = children
        listed.append(_Proc(1, name="Other"))
        core, coordinator = _Proc(2), _Proc(3, name="VLLM_DP_Coordinator")
        self._build(monkeypatch, listed, [core, coordinator])
        key = vllm_module._engine_key("org/p", None, {})
        assert vllm_module._engine_procs == {key: [core, coordinator]}
        assert vllm_module._stray == []

    def test_a_failed_build_ends_its_processes(self, children, monkeypatch):
        """One that outlives SIGKILL is kept, to be tried again on the way out."""
        vllm_module, listed = children
        ended, kept = _Proc(2), _Proc(3, stubborn=True, immortal=True)
        with pytest.raises(RuntimeError, match="engine core died"):
            self._build(monkeypatch, listed, [ended, kept], fail=True)
        assert ("terminate", 2) in ended.done and not ended.alive
        assert vllm_module._engine_procs == {} and vllm_module._stray == [kept]

    def test_terminate_then_kill_what_outlives_it(self, children, monkeypatch):
        vllm_module, _ = children
        done = []
        monkeypatch.setitem(sys.modules, "psutil",
                            _fake_psutil({2: [21, 22]}, done, stubborn={22}))
        core, other = _Proc(2, done=done, stubborn=True), _Proc(5, done=done)
        assert vllm_module.terminate_engine_processes([core, other]) == []
        # The children are listed before their engine is signalled, then each
        # gets SIGTERM, and SIGKILL if it is still alive after the wait.
        assert done == [
            ("terminate", 2), ("terminate", 5), ("terminate", 21), ("terminate", 22),
            ("join", 2), ("join", 5), ("wait", [21, 22]),
            ("kill", 2), ("kill", 22),
            ("join", 2), ("wait", [22]),
        ]

    def test_what_outlives_sigkill_is_returned_and_logged(
            self, children, monkeypatch, caplog):
        vllm_module, _ = children
        done = []
        monkeypatch.setitem(sys.modules, "psutil", _fake_psutil(
            {2: [21]}, done, stubborn={21}, immortal={21}))
        core = _Proc(2, done=done, stubborn=True, immortal=True)
        with caplog.at_level(logging.WARNING, logger=vllm_module.logger.name):
            assert vllm_module.terminate_engine_processes([core]) == [2, 21]
        assert "kill -9 2 21" in caplog.text

    def test_quiet_logs_nothing_and_returns_what_lived_on(
            self, children, monkeypatch, caplog):
        """For a signal handler, where a logging call may re-enter a write."""
        vllm_module, _ = children
        monkeypatch.setitem(sys.modules, "psutil", _fake_psutil(
            {2: [21]}, [], stubborn={21}, immortal={21}))
        core = _Proc(2, stubborn=True, immortal=True)
        with caplog.at_level(logging.DEBUG):
            assert vllm_module.terminate_engine_processes([core], quiet=True) == [2, 21]
            monkeypatch.setitem(sys.modules, "psutil", None)
            assert vllm_module.terminate_engine_processes([_Proc(3)], quiet=True) == [3]
        assert caplog.records == []

    def test_a_process_that_ended_already_is_left_alone(self, children, monkeypatch):
        vllm_module, _ = children
        monkeypatch.setitem(sys.modules, "psutil", None)    # not even imported
        ended = _Proc(2)
        ended.alive = False
        assert vllm_module.terminate_engine_processes([ended]) == []

    def test_without_psutil_nothing_is_ended_and_all_is_returned(
            self, children, monkeypatch, caplog):
        vllm_module, _ = children
        monkeypatch.setitem(sys.modules, "psutil", None)
        done = []
        with caplog.at_level(logging.ERROR, logger=vllm_module.logger.name):
            assert vllm_module.terminate_engine_processes([_Proc(2, done=done)]) == [2]
        assert done == [] and "psutil" in caplog.text

    def test_all_ends_every_engine_process_and_no_other_child(
            self, children, monkeypatch):
        """For the forced exit of a second signal: the processes of every
        build, and every child named as vLLM names its engine processes."""
        vllm_module, listed = children
        done = []
        monkeypatch.setitem(sys.modules, "psutil", _fake_psutil({}, done))
        loaded, stray = _Proc(2, name="x", done=done), _Proc(3, name="y", done=done)
        unrecorded = [_Proc(4, "EngineCore_DP1", done),
                      _Proc(6, "VLLM_DP_Coordinator", done)]
        other = _Proc(5, name="SyncManager-1", done=done)
        listed.extend([loaded, stray, *unrecorded, other])
        vllm_module._engine_procs[("org/p", None, "[]")] = [loaded]
        vllm_module._stray.append(stray)
        vllm_module.terminate_engine_processes(all=True)
        assert {pid for what, pid in done if what == "terminate"} == {2, 3, 4, 6}
        assert other.alive

    def test_a_real_engine_process_and_its_child_are_ended(self):
        """Processes of this machine: a child of this process, and the
        process it started, which the child's end does not end."""
        import multiprocessing
        import time

        psutil = pytest.importorskip("psutil")
        import redact.llms.backends.vllm as vllm_module

        engine = multiprocessing.get_context("spawn").Process(
            target=subprocess.call,
            args=([sys.executable, "-c", "import time; time.sleep(60)"],),
            name="EngineCore")
        engine.start()
        try:
            deadline = time.monotonic() + 30
            while not psutil.Process(engine.pid).children():
                assert time.monotonic() < deadline, "the worker never started"
                time.sleep(0.1)
            (worker,) = psutil.Process(engine.pid).children()
            assert vllm_module.terminate_engine_processes([engine]) == []
            assert not engine.is_alive()
            # Never os.kill(pid, 0) to check: on Windows that ends the process.
            assert not worker.is_running()
        finally:
            if engine.is_alive():
                engine.kill()


class TestReleaseNote:
    NOTE = ("[vllm] shutting down 1 engine(s). vLLM may log 'AsyncLLM "
            "output_handler failed' with an EngineDeadError here: that is "
            "this shutdown, not a failed call.")

    def test_a_release_says_that_vllm_may_log_an_error(self, monkeypatch, caplog):
        import redact.llms.backends.vllm as vllm_module

        class _Engine:
            def shutdown(self):
                pass

        monkeypatch.setattr(vllm_module, "_engines", {("org/x", None, ()): _Engine()})
        with caplog.at_level(logging.INFO, logger="redact.llms.backends.vllm"):
            vllm_module.VLLMBackend.clear_cache()
            vllm_module.VLLMBackend.clear_cache()      # nothing loaded: no line
        assert [r.getMessage() for r in caplog.records] == [self.NOTE]


def _ask(name, system=None):
    """A chat whose request the fake engine names ``name``."""
    chat = [{"role": "user", "content": name}]
    return [{"role": "system", "content": system}, *chat] if system else chat


def _reply(text, token_ids=(3, 4), finish_reason="stop"):
    """One vLLM CompletionOutput as the backend reads it."""
    return types.SimpleNamespace(
        text=text, token_ids=list(token_ids), finish_reason=finish_reason)


class _FakeTokenizer:
    """A chat template that makes one token id of each message."""

    def __init__(self):
        self.messages = []       # token id -> the message it stands for
        self.options = []        # (add_generation_prompt, tokenize) of each call
        self.as_mapping = False  # True: return the ids inside a mapping

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        self.options.append((add_generation_prompt, tokenize))
        ids = list(range(len(self.messages), len(self.messages) + len(messages)))
        self.messages.extend(messages)
        return {"input_ids": ids} if self.as_mapping else ids


class _FakeAsyncLLM:
    """A vLLM ``AsyncLLM`` whose requests follow a :class:`_Script`.

    A request is named by the content of its last message. Once the script
    lets it end it answers with ``replies[name]``, or with no output when the
    name has no reply.

    Attributes:
        sent: name -> what the engine was handed for that request, with
            ``chat`` the messages its token ids stand for.
    """

    scheduler_config = types.SimpleNamespace(max_num_seqs=48)

    def __init__(self, replies, script=None):
        self.replies = replies
        self.script = script or _Script()
        self.tokenizer = _FakeTokenizer()
        self.sent = {}

    def get_tokenizer(self):
        return self.tokenizer

    async def generate(self, prompt, params, request_id):
        ids = prompt["prompt_token_ids"]
        chat = [self.tokenizer.messages[i] for i in ids]
        name = chat[-1]["content"]
        self.sent[name] = types.SimpleNamespace(
            chat=chat, prompt=prompt, params=params, request_id=request_id)
        await self.script.play(name, None, None)
        if name in self.replies:
            yield types.SimpleNamespace(prompt_token_ids=ids, outputs=[self.replies[name]])

    def shutdown(self):
        """So clear_cache() at teardown finds a hook and stays quiet."""


def _vllm_backend(monkeypatch, replies, script=None, **settings):
    """A VLLMBackend on a fake engine; returns ``(backend, engine)``."""
    import redact.llms.backends.vllm as vllm_module
    from redact.llms.backends import VLLMBackend

    for name, module in fake_vllm_modules(_FakeAsyncLLM).items():
        monkeypatch.setitem(sys.modules, name, module)
    # _engine() returns a cached engine before it imports vllm.
    engine = _FakeAsyncLLM(replies, script)
    vllm_module._engines[vllm_module._engine_key("org/x", None, {})] = engine
    return VLLMBackend("m", "org/x", **{"default_max_tokens": 8, **settings}), engine


@pytest.fixture()
def events():
    seen = []
    observe.set_emitter(seen.append)
    yield seen
    observe.set_emitter(None)


# SamplingParams of a request with no overrides, as the fake module builds them.
_DEFAULT_PARAMS = {"max_tokens": 8, "temperature": 0.7, "output_kind": "FINAL_ONLY",
                   "top_p": 0.85}


class TestVLLMAgenerate:
    """agenerate() is one request to the engine, awaited on the shared loop."""

    def test_the_prompt_is_the_chat_template_of_the_engines_tokenizer(self, monkeypatch):
        backend, engine = _vllm_backend(monkeypatch, {"hi": _reply("ok")})
        messages = [{"role": "system", "content": "A"}, {"role": "system", "content": "B"},
                    {"role": "user", "content": "hi"}]
        assert aio.run(backend.agenerate(messages)) == "ok"
        assert engine.sent["hi"].chat == _ask("hi", system="A\n\nB")
        assert engine.sent["hi"].prompt == {"prompt_token_ids": [0, 1]}
        assert engine.tokenizer.options == [(True, True)]

    def test_a_model_without_a_system_role_gets_it_folded_into_the_user_message(
            self, monkeypatch):
        backend, engine = _vllm_backend(
            monkeypatch, {"Sys.\n\nhi": _reply("ok")}, supports_system_prompt=False)
        assert aio.run(backend.agenerate(_ask("hi", system="Sys."))) == "ok"
        (sent,) = engine.sent.values()
        assert sent.chat == _ask("Sys.\n\nhi")
        assert sent.prompt == {"prompt_token_ids": [0]}

    def test_a_tokenizer_that_returns_a_mapping_is_read_for_its_ids(self, monkeypatch):
        backend, engine = _vllm_backend(monkeypatch, {"hi": _reply("ok")})
        engine.tokenizer.as_mapping = True
        assert aio.run(backend.agenerate(_ask("hi", system="Sys."))) == "ok"
        assert engine.sent["hi"].prompt == {"prompt_token_ids": [0, 1]}

    def test_sampling_params_are_the_models_defaults_under_the_calls_overrides(
            self, monkeypatch):
        backend, engine = _vllm_backend(
            monkeypatch, {"plain": _reply("ok"), "tuned": _reply("ok")},
            sampling={"top_k": 20})
        aio.run(backend.agenerate(_ask("plain")))
        aio.run(backend.agenerate(_ask("tuned"), max_tokens=50, temperature=0.0, top_p=0.5))
        assert engine.sent["plain"].params == {**_DEFAULT_PARAMS, "top_k": 20}
        assert engine.sent["tuned"].params == {
            "max_tokens": 50, "temperature": 0.0, "output_kind": "FINAL_ONLY",
            "top_p": 0.5, "top_k": 20}

    def test_the_reply_loses_its_whitespace_and_the_tokenizer_artifact(self, monkeypatch):
        backend, _ = _vllm_backend(monkeypatch, {"hi": _reply(" |> hello \n")})
        assert aio.run(backend.agenerate(_ask("hi"))) == "hello"

    def test_a_request_records_one_call_event(self, monkeypatch, events):
        backend, _ = _vllm_backend(
            monkeypatch, {"hi": _reply("ok", token_ids=[5, 6, 7], finish_reason="length")})
        aio.run(backend.agenerate(_ask("hi", system="Sys."), max_tokens=3))
        (event,) = events
        del event["ms"]
        assert event == {
            "ev": "call", "stage": None, "model": "m", "backend": "vllm", "n_items": 1,
            "in_tok": 2, "out_tok": 3, "out_tok_max": 3, "max_tokens": 3,
            "temperature": 0.7, "finish_reason": "length", "cut": True}

    def test_a_failed_request_is_recorded_then_re_raised(self, monkeypatch, events):
        backend, _ = _vllm_backend(monkeypatch, {}, _Script(["hi"], fail={"hi"}))
        with pytest.raises(RuntimeError, match="hi boom"):
            aio.run(backend.agenerate(_ask("hi")))
        (event,) = events
        assert (event["error"], event["n_items"], event["max_tokens"]) == (
            "RuntimeError: hi boom", 1, 8)

    def test_a_request_the_engine_sends_no_output_for_fails(self, monkeypatch, events):
        backend, engine = _vllm_backend(monkeypatch, {})
        with pytest.raises(RuntimeError, match="sent no output for request") as raised:
            aio.run(backend.agenerate(_ask("hi")))
        assert engine.sent["hi"].request_id in str(raised.value)
        assert [event["error"] for event in events] == [f"RuntimeError: {raised.value}"]

    def test_a_request_the_engine_aborted_fails_with_no_reply(self, monkeypatch, events):
        """An engine released with a request in flight ends it with the text it
        had so far. Part of a reply must never pass as the reply."""
        backend, engine = _vllm_backend(
            monkeypatch, {"hi": _reply("partial", finish_reason="abort")})
        with pytest.raises(RuntimeError) as raised:
            aio.run(backend.agenerate(_ask("hi")))
        assert str(raised.value) == (
            f"The vLLM engine of 'm' aborted request {engine.sent['hi'].request_id}: "
            "the engine was released or shut down while the request was in flight.")
        (event,) = events
        del event["ms"]
        assert event == {
            "ev": "call", "stage": None, "model": "m", "backend": "vllm", "n_items": 1,
            "in_tok": None, "out_tok": None, "out_tok_max": None, "max_tokens": 8,
            "temperature": 0.7, "error": f"RuntimeError: {raised.value}"}

    def test_a_prompt_longer_than_the_context_raises_prompt_too_long(
            self, monkeypatch, events):
        """The engine fails that one request with a validation error on
        ``input_tokens``."""
        error = VLLMValidationError(
            "This model's maximum context length is 8192 tokens. However, you "
            "requested 0 output tokens and your prompt contains at least 10036 "
            "input tokens", parameter="input_tokens", value=10036)
        backend, _ = _vllm_backend(monkeypatch, {}, _Script(["hi"], fail={"hi": error}))
        with pytest.raises(PromptTooLongError) as raised:
            aio.run(backend.agenerate(_ask("hi")))
        assert str(raised.value) == str(error) and raised.value.__cause__ is error
        assert [event["error"] for event in events] == [f"VLLMValidationError: {error}"]

    def test_a_validation_error_on_another_parameter_stays_as_it_is(
            self, monkeypatch, events):
        error = VLLMValidationError(
            "max_tokens must be at least 1, got 0.", parameter="max_tokens", value=0)
        backend, _ = _vllm_backend(monkeypatch, {}, _Script(["hi"], fail={"hi": error}))
        with pytest.raises(VLLMValidationError) as raised:
            aio.run(backend.agenerate(_ask("hi")))
        assert raised.value is error
        assert [event["error"] for event in events] == [f"VLLMValidationError: {error}"]

    def test_internals_id_is_ignored(self, monkeypatch):
        backend, engine = _vllm_backend(monkeypatch, {"hi": _reply("ok")})
        assert aio.run(backend.agenerate(_ask("hi"), internals_id="root/0")) == "ok"
        assert engine.sent["hi"].params == _DEFAULT_PARAMS


class TestVLLMGenerateBatch:
    """generate() sends each item as its own request and blocks for them all."""

    def test_replies_are_in_input_order_whatever_order_they_end_in(self, monkeypatch):
        script = _Script(["c", "b", "a"])
        backend, _ = _vllm_backend(monkeypatch, {n: _reply(n.upper()) for n in "abc"}, script)
        assert backend.generate([_ask(n) for n in "abc"]) == ["A", "B", "C"]
        assert script.log == [("start", "a"), ("start", "b"), ("start", "c"),
                              ("end", "c"), ("end", "b"), ("end", "a")]

    def test_a_batch_is_one_call_to_the_shared_loop(self, monkeypatch):
        import redact.llms.backends.vllm as vllm_module

        backend, _ = _vllm_backend(monkeypatch, {n: _reply(n.upper()) for n in "abc"})
        runs = []
        monkeypatch.setattr(vllm_module, "aio", types.SimpleNamespace(
            run=lambda coro: runs.append(coro) or aio.run(coro)))
        assert backend.generate([_ask(n) for n in "abc"]) == ["A", "B", "C"]
        assert len(runs) == 1

    def test_every_request_carries_the_calls_overrides_and_its_own_id(self, monkeypatch):
        backend, engine = _vllm_backend(monkeypatch, {"a": _reply("A"), "b": _reply("B")})
        backend.generate([_ask("a"), _ask("b")], max_tokens=50, top_k=5,
                         internals_ids=[None, None])
        params = {**_DEFAULT_PARAMS, "max_tokens": 50, "top_k": 5}
        assert [engine.sent[n].params for n in "ab"] == [params, params]
        assert engine.sent["a"].request_id != engine.sent["b"].request_id

    def test_a_failed_request_raises_and_cancels_the_others(self, monkeypatch, events):
        """The failing request is the last to start, so the other two are in
        flight when it fails, and they have stopped when the batch ends. The
        log is read on the loop at that moment: read by the caller, it would
        also hold what the loop did before the caller woke."""
        import redact.llms.backends.vllm as vllm_module

        script = _Script(["bad"], fail={"bad"})
        backend, _ = _vllm_backend(monkeypatch, {}, script)
        logged = []

        async def then_read_the_log(batch):
            try:
                return await batch
            finally:
                logged.extend(script.log)

        monkeypatch.setattr(vllm_module, "aio", types.SimpleNamespace(
            run=lambda batch: aio.run(then_read_the_log(batch))))
        with pytest.raises(RuntimeError, match="bad boom"):
            backend.generate([_ask("a"), _ask("b"), _ask("bad")])
        assert logged[:4] == [("start", "a"), ("start", "b"), ("start", "bad"),
                              ("end", "bad")]
        assert sorted(logged[4:]) == [("cancel", "a"), ("cancel", "b")]
        assert [event.get("error") for event in events] == ["RuntimeError: bad boom"]

    def test_an_interrupted_call_cancels_its_requests(self, monkeypatch):
        """Ctrl-C reaches the caller blocked in generate(). The engine must not
        go on generating a batch that nobody waits for."""
        script = _Script([])                    # no request ends by itself
        backend, _ = _vllm_backend(monkeypatch, {}, script)
        # The interrupt finds both requests in flight.
        interrupt_next_run(monkeypatch, when=lambda: script.wait_until("start", "b"))
        with pytest.raises(KeyboardInterrupt):
            backend.generate([_ask("a"), _ask("b")])
        script.wait_until("cancel", "a")
        script.wait_until("cancel", "b")
        assert script.log[:2] == [("start", "a"), ("start", "b")]
        assert sorted(script.log[2:]) == [("cancel", "a"), ("cancel", "b")]

    def test_a_request_the_engine_aborted_fails_the_batch(self, monkeypatch, events):
        backend, _ = _vllm_backend(monkeypatch, {
            "a": _reply("A"), "b": _reply("partial", finish_reason="abort")})
        with pytest.raises(RuntimeError, match="'m' aborted request"):
            backend.generate([_ask("a"), _ask("b")])
        assert [(event.get("finish_reason"), "error" in event) for event in events] == [
            ("stop", False), (None, True)]

    def test_an_empty_batch_returns_without_the_loop(self, monkeypatch):
        import redact.llms.backends.vllm as vllm_module

        backend, _ = _vllm_backend(monkeypatch, {})
        monkeypatch.setattr(vllm_module, "aio", None)
        assert backend.generate([]) == []


class TestVLLMCutReplies:
    """A reply vLLM ended with "length" is cut: its call event says so, and
    the model's first one warns."""

    CUT = first_cut_warning("m", 8)

    def test_only_the_first_cut_reply_warns(self, monkeypatch, caplog, events):
        backend, _ = _vllm_backend(monkeypatch, {
            "a": _reply("A", finish_reason="length"),
            "b": _reply("B"),
            "c": _reply("C", finish_reason="length"),
        })
        with caplog.at_level(logging.WARNING, logger="redact.llms.backends"):
            assert backend.generate([_ask(n) for n in "abc"]) == ["A", "B", "C"]
            assert aio.run(backend.agenerate(_ask("a"))) == "A"
        assert [r.getMessage() for r in caplog.records] == [self.CUT]
        assert [e.get("cut") for e in events] == [True, None, True, True]

    def test_a_cut_reply_warns_from_agenerate(self, monkeypatch, caplog):
        backend, _ = _vllm_backend(monkeypatch, {"a": _reply("A", finish_reason="length")})
        with caplog.at_level(logging.WARNING, logger="redact.llms.backends"):
            assert aio.run(backend.agenerate(_ask("a"))) == "A"
        assert [r.getMessage() for r in caplog.records] == [self.CUT]

    @pytest.mark.parametrize("reason", ["stop", None])
    def test_complete_replies_do_not_warn(self, monkeypatch, caplog, reason):
        backend, _ = _vllm_backend(monkeypatch, {
            n: _reply(n.upper(), finish_reason=reason) for n in "ab"})
        with caplog.at_level(logging.WARNING, logger="redact.llms.backends"):
            assert backend.generate([_ask("a"), _ask("b")]) == ["A", "B"]
            assert aio.run(backend.agenerate(_ask("a"))) == "A"
        assert caplog.records == []


class TestVLLMCallEvent:
    def test_each_request_of_a_batch_records_its_own_event(self, monkeypatch, events):
        backend, _ = _vllm_backend(monkeypatch, {
            "a": _reply("A", token_ids=[1, 2, 3], finish_reason="length"),
            "b": _reply("B", token_ids=[1]),
        })
        backend.generate([_ask("a", system="Sys."), _ask("b")])
        assert [(e["n_items"], e["in_tok"], e["out_tok"], e["out_tok_max"], e["finish_reason"])
                for e in events] == [(1, 2, 3, 3, "length"), (1, 1, 1, 1, "stop")]


class TestVLLMFlags:
    """What sends a vLLM model's requests down drive_units' async lane."""

    def test_the_class_declares_native_batching_parallel_and_async_calls(self):
        from redact.llms.backends import VLLMBackend

        assert VLLMBackend.compute_config == ComputeConfig(
            supports_native_batching=True, supports_parallel_calls=True,
            supports_internals=False, supports_async_calls=True)

    def test_max_workers_is_what_the_engine_runs_at_once(self, monkeypatch):
        from redact.llms.backends import VLLMBackend

        backend, engine = _vllm_backend(monkeypatch, {})
        assert backend.max_workers == engine.scheduler_config.max_num_seqs == 48
        assert VLLMBackend("m", "org/x", max_workers=4).max_workers == 4

    def test_max_workers_counts_every_copy(self, monkeypatch):
        """Each copy of an engine runs ``max_num_seqs`` sequences of its own."""
        from redact.llms.backends import VLLMBackend

        _, engine = _vllm_backend(monkeypatch, {})
        engine.vllm_config = types.SimpleNamespace(
            parallel_config=types.SimpleNamespace(data_parallel_size=2))
        assert VLLMBackend("m", "org/x").max_workers == 96
        assert VLLMBackend("m", "org/x", max_workers=4).max_workers == 4

    def test_the_client_hands_generate_the_whole_list(self, monkeypatch):
        backend, _ = _vllm_backend(monkeypatch, {})
        assert ModelClient(backend)._caller is None

    def test_drive_units_has_both_requests_in_flight_before_either_ends(self, monkeypatch):
        """One unit per prompt. The first request cannot end before the second
        has, so a driver that waited for it would never send the second."""
        script = _Script(["b1", "a1"])
        backend, _ = _vllm_backend(
            monkeypatch, {"a1": _reply("A"), "b1": _reply("B")}, script)
        client = ModelClient(backend)
        results = {}
        n = drive_units(iter([(k, _unit(client, k)) for k in "ab"]),
                        pool_size=2, on_done=results.__setitem__)
        assert script.log == [("start", "a1"), ("start", "b1"), ("end", "b1"), ("end", "a1")]
        assert n == 2 and results == {"b": ["B"], "a": ["A"]}


class MockBackend(LLMBackend):
    """Minimal backend that returns preset responses for testing."""

    # Behave like vLLM: the client hands a native-batching backend the whole
    # chunk in one generate() call, so `_calls` counts engine passes.
    compute_config = ComputeConfig(supports_native_batching=True)

    def __init__(self, responses: list[str], model: str = "mock-vllm-model"):
        super().__init__(model)
        self._responses = list(responses)
        self._calls: list[list[list[dict]]] = []  # record generate() batch calls

    def generate(self, messages_list: list[list[dict]], **kwargs) -> list[str]:
        self._calls.append(messages_list)
        return [self._responses.pop(0) for _ in messages_list]

    @property
    def backend_name(self) -> str:
        return "mock"


def _checker(original: str, sample: str) -> list[dict]:
    return [{"role": "user", "content": f"Is this acceptable? {sample}"}]


# ---------------------------------------------------------------------------
# batch_check_samples — no GPU required
# ---------------------------------------------------------------------------

class TestBatchCheckSamples:
    """Unit tests for batch_check_samples() using a mock backend."""

    def test_empty_samples_returns_empty(self):
        backend = MockBackend([])
        result = batch_check_samples(make_client(backend), [], _checker)
        assert result == []
        assert backend._calls == []  # no engine call made

    def test_accepted_on_yes_prefix(self):
        backend = MockBackend(["Yes, this is fine."])
        result = batch_check_samples(make_client(backend), ["sample"], _checker)
        assert result == [(True, "")]

    def test_accepted_on_all_prefixes(self):
        responses = ["yes ok", "ok got it", "Accept.", "Pass — looks good"]
        backend = MockBackend(responses)
        result = batch_check_samples(make_client(backend), ["a", "b", "c", "d"], _checker)
        assert all(accepted for accepted, _ in result)
        assert all(reasoning == "" for _, reasoning in result)

    def test_rejected_returns_full_response(self):
        response = "No, this contains harmful content."
        backend = MockBackend([response])
        result = batch_check_samples(make_client(backend), ["bad sample"], _checker)
        assert result == [(False, response)]

    def test_case_insensitive_acceptance(self):
        backend = MockBackend(["YES THIS IS FINE", "OK ACCEPTED"])
        result = batch_check_samples(make_client(backend), ["a", "b"], _checker)
        assert result == [(True, ""), (True, "")]

    def test_order_preserved(self):
        responses = ["yes", "No bad", "ok", "Reject", "pass"]
        backend = MockBackend(responses)
        samples = ["s1", "s2", "s3", "s4", "s5"]
        result = batch_check_samples(make_client(backend), samples, _checker)
        assert result[0] == (True, "")
        assert result[1] == (False, "No bad")
        assert result[2] == (True, "")
        assert result[3] == (False, "Reject")
        assert result[4] == (True, "")

    def test_single_batch_when_samples_fit(self):
        backend = MockBackend(["yes", "no", "ok"])
        batch_check_samples(make_client(backend), ["a", "b", "c"], _checker, batch_size=32)
        assert len(backend._calls) == 1
        assert len(backend._calls[0]) == 3  # all 3 in one batch_generate call

    def test_chunking_into_multiple_batches(self):
        responses = ["yes"] * 5
        backend = MockBackend(responses)
        batch_check_samples(make_client(backend), ["s"] * 5, _checker, batch_size=2)
        # ceil(5/2) = 3 calls: chunks of [2, 2, 1]
        assert len(backend._calls) == 3
        assert len(backend._calls[0]) == 2
        assert len(backend._calls[1]) == 2
        assert len(backend._calls[2]) == 1

    def test_results_length_matches_samples(self):
        backend = MockBackend(["yes"] * 7)
        result = batch_check_samples(make_client(backend), ["s"] * 7, _checker, batch_size=3)
        assert len(result) == 7

    def test_build_check_messages_called_with_correct_sample(self):
        seen = []

        def tracking_checker(original: str, sample: str) -> list[dict]:
            seen.append(sample)
            return [{"role": "user", "content": sample}]

        backend = MockBackend(["yes", "no"])
        batch_check_samples(make_client(backend), ["apple", "banana"], tracking_checker)
        assert seen == ["apple", "banana"]


class TestBackendForRouting:
    """Verify backend_for() routing logic without needing a GPU."""

    def test_missing_vllm_setup_raises(self):
        name = "_test_no_vllm_setup"
        try:
            register_model(name, api=APIConfig(backend_type="openai", api_key_env="TEST_API_KEY", base_url="https://test.example/v1", rpm=10))
            with pytest.raises(ValueError, match="no vllm setup"):
                backend_for(get_model_config(name), "vllm")
        finally:
            MODEL_REGISTRY.pop(name, None)

    def test_venice_registry_entry_unchanged(self):
        config = get_model_config("venice-uncensored")
        assert config.backend_type == "api"            # prefers its endpoint
        assert config.api.backend_type == "openai"     # via the openai transport

    def test_venice_entry_also_carries_the_local_setup(self):
        # No separate "-vllm" row: the same entry describes both transports.
        config = get_model_config("venice-uncensored")
        assert config.vllm.hf_model_id == "dphn/Dolphin-Mistral-24B-Venice-Edition"


# ---------------------------------------------------------------------------
# GPU tests (require vllm + CUDA)
# ---------------------------------------------------------------------------

@gpu
class TestVLLMBackendGPU:
    """Integration tests that load a real model on GPU."""

    @pytest.fixture(scope="class")
    def backend(self, vllm_model):
        """Load the model once so the engine cache is populated."""
        from redact.llms.backends import VLLMBackend
        b = backend_for(get_model_config(vllm_model))
        assert isinstance(b, VLLMBackend)
        return b

    def test_backend_for_routes_vllm(self, vllm_model, backend):
        from redact.llms.backends import VLLMBackend
        assert isinstance(backend_for(get_model_config(vllm_model)), VLLMBackend)

    def test_engine_is_cached_across_backends(self, vllm_model, backend):
        # Backends are per-model and cheap; the engine underneath is not, so
        # a second build must reuse the loaded one rather than reload weights.
        b2 = backend_for(get_model_config(vllm_model))
        assert b2._llm is backend._llm

    def test_generate(self, backend):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello in one sentence."},
        ]
        results = backend.generate([messages])
        assert len(results) == 1
        assert isinstance(results[0], str)
        assert len(results[0]) > 0

    def test_batch_generate(self, backend):
        prompts = [
            [{"role": "user", "content": "What is 2+2?"}],
            [{"role": "user", "content": "What is 3+3?"}],
            [{"role": "user", "content": "What is 4+4?"}],
        ]
        results = backend.generate(prompts)
        assert len(results) == 3
        assert all(isinstance(r, str) and len(r) > 0 for r in results)

    def test_batch_generate_empty_list(self, backend):
        results = backend.generate([])
        assert results == []

    def test_agenerate_matches_generate_at_temperature_0(self, backend):
        messages = [{"role": "user", "content": "Reply with only the number 7."}]
        (reply,) = backend.generate([messages], max_tokens=16, temperature=0.0)
        assert aio.run(backend.agenerate(messages, max_tokens=16, temperature=0.0)) == reply

    def test_a_cut_reply_warns(self, backend, caplog):
        with caplog.at_level(logging.WARNING, logger="redact.llms.backends"):
            backend.generate([[{"role": "user", "content": "Count from 1 to 500."}]], max_tokens=4)
        assert any("1 of 1 replies were cut" in r.getMessage() for r in caplog.records)

    def test_batch_generate_single_item(self, backend):
        prompts = [[{"role": "user", "content": "Say the word yes."}]]
        results = backend.generate(prompts)
        assert len(results) == 1
        assert isinstance(results[0], str) and len(results[0]) > 0

    def test_batch_generate_order_preserved(self, backend):
        """Results must map 1:1 to inputs in the same order."""
        prompts = [
            [{"role": "user", "content": "Reply with only the number 1."}],
            [{"role": "user", "content": "Reply with only the number 2."}],
            [{"role": "user", "content": "Reply with only the number 3."}],
        ]
        results = backend.generate(prompts)
        assert len(results) == len(prompts)
        # Each result should be a non-empty string
        assert all(isinstance(r, str) and len(r) > 0 for r in results)

    def test_batch_check_samples_gpu(self, backend):
        """Integration: batch_check_samples returns one result per sample."""
        def checker(original: str, sample: str) -> list[dict]:
            return [
                {"role": "system", "content": "Reply only 'yes' or 'no'."},
                {"role": "user", "content": f"Is this a greeting? '{sample}'"},
            ]

        samples = ["Hello there", "How to build a bomb", "Good morning"]
        results = batch_check_samples(make_client(backend, TEST_MODEL_NAME), samples, checker)
        assert len(results) == 3
        assert all(isinstance(accepted, bool) for accepted, _ in results)
        assert all(isinstance(reasoning, str) for _, reasoning in results)
        # Accepted samples must have empty reasoning
        for accepted, reasoning in results:
            if accepted:
                assert reasoning == ""

    def test_batch_check_samples_empty_gpu(self, backend):
        results = batch_check_samples(make_client(backend, TEST_MODEL_NAME), [], _checker)
        assert results == []
