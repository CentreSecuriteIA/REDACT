"""``scripts/run.py``'s signal handlers, run on stubs."""

import importlib.util
import logging
import os
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest


class _Exited(BaseException):
    """Stands for the end of the process."""


@pytest.fixture
def second_signal(monkeypatch):
    """The script's handler for a second Ctrl-C, after a first one, on a stub
    run. Returns ``(handle, done)``: ``done`` gets ``("end", kwargs)`` for
    each call to ``terminate_engine_processes`` and ``("exit", code)`` for
    ``os._exit``, which raises :class:`_Exited`. ``handle.left`` is what the
    stub ``terminate_engine_processes`` returns."""
    spec = importlib.util.spec_from_file_location(
        "run_script", Path(__file__).parents[1] / "scripts" / "run.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    done, handlers = [], {}

    def hard_exit(code):
        done.append(("exit", code))
        raise _Exited

    def terminate(**kwargs):
        done.append(("end", kwargs))
        return handle.left

    monkeypatch.setattr(script, "run_pipeline", lambda recipe, **kw: {})
    monkeypatch.setattr(script, "os", SimpleNamespace(write=os.write, _exit=hard_exit))
    monkeypatch.setattr(script, "terminate_engine_processes", terminate)
    monkeypatch.setattr(logging, "basicConfig", lambda **kw: None)
    monkeypatch.setattr(signal, "signal",
                        lambda signum, fn: handlers.update({signum: fn}))
    script.main(["recipe.json"])
    with pytest.raises(KeyboardInterrupt):
        handlers[signal.SIGINT](signal.SIGINT, None)

    def handle():
        handlers[signal.SIGINT](signal.SIGINT, None)

    handle.left = []
    return handle, done


def test_the_forced_exit_ends_the_engine_processes_without_logging(second_signal):
    """The handler may interrupt a write to stderr: a logging call there
    could raise and skip the exit, so it ends the processes quietly."""
    handle, done = second_signal
    with pytest.raises(_Exited):
        handle()
    assert done == [("end", {"all": True, "quiet": True}),
                    ("exit", 128 + signal.SIGINT)]


def test_the_forced_exit_names_what_outlived_it(second_signal, capfd):
    handle, done = second_signal
    handle.left = [21, 22]
    capfd.readouterr()
    with pytest.raises(_Exited):
        handle()
    assert "kill -9 21 22" in capfd.readouterr().err
    assert done[-1] == ("exit", 128 + signal.SIGINT)


@pytest.mark.parametrize(("flags", "root", "own"), [
    ([], logging.INFO, logging.NOTSET),
    (["--quiet"], logging.WARNING, logging.NOTSET),
    (["--debug"], logging.INFO, logging.DEBUG),
])
def test_debug_shows_redacts_detail_only(monkeypatch, flags, root, own):
    """--debug sets the redact logger to DEBUG, so httpx, httpcore and
    filelock lines do not bury redact's own; the root stays at INFO."""
    spec = importlib.util.spec_from_file_location(
        "run_script", Path(__file__).parents[1] / "scripts" / "run.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    configured = []
    redact_logger = logging.getLogger("redact")
    monkeypatch.setattr(redact_logger, "level", redact_logger.level)
    monkeypatch.setattr(script, "run_pipeline", lambda recipe, **kw: {})
    monkeypatch.setattr(script, "_exit_through_release", lambda: None)
    monkeypatch.setattr(logging, "basicConfig", lambda **kw: configured.append(kw))
    assert script.main(["recipe.json", *flags]) == 0
    assert [kw["level"] for kw in configured] == [root]
    assert redact_logger.level == own
