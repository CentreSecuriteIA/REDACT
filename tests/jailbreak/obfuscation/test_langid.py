"""GlotLID language identification, on stubs: no model is fetched."""

import logging
import sys
import threading
from types import SimpleNamespace

import pytest

from redact.jailbreak.obfuscation import langid


@pytest.fixture(autouse=True)
def _unloaded(monkeypatch):
    """Each test starts with no model loaded and nothing failed."""
    monkeypatch.setattr(langid, "_model", None)
    monkeypatch.setattr(langid, "_unavailable", False)


class _Model:
    """Stands in for a fastText model: one label for every line."""

    def __init__(self, label="swh_Latn"):
        self.label = label
        self.lines: list[str] = []

    def predict(self, text, k=1, threshold=0.0):
        if "\n" in text:
            raise ValueError("predict processes one line at a time (remove '\\n')")
        self.lines.append(text)
        return (f"__label__{self.label}",), (0.97,)


def _installed(monkeypatch, model=None, *, fail=None):
    """Stub ``fasttext`` and ``huggingface_hub``; return the downloads made."""
    downloads = []

    def hf_hub_download(repo_id, filename):
        downloads.append((repo_id, filename))
        if fail is not None:
            raise fail
        return f"/cache/{repo_id}/{filename}"

    monkeypatch.setitem(sys.modules, "fasttext", SimpleNamespace(
        load_model=lambda path: model or _Model()))
    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        SimpleNamespace(hf_hub_download=hf_hub_download))
    return downloads


def test_detect_gives_the_top_label_of_the_text_on_one_line(monkeypatch):
    model = _Model("swh_Latn")
    _installed(monkeypatch, model)
    assert langid.detect("habari\n\nyako  leo") == "swh_Latn"
    assert model.lines == ["habari yako leo"]
    assert langid.available()


def test_without_the_package_nothing_is_detected(monkeypatch):
    monkeypatch.setitem(sys.modules, "fasttext", None)      # import raises
    assert not langid.available()
    assert langid.detect("habari") is None


def test_text_without_a_word_has_no_label(monkeypatch):
    model = _Model()
    _installed(monkeypatch, model)
    assert langid.detect(" \n ") is None and model.lines == []


def test_the_model_is_fetched_once_under_a_lock(monkeypatch):
    downloads = _installed(monkeypatch)
    barrier = threading.Barrier(4)

    def detect():
        barrier.wait()
        langid.detect("habari")

    threads = [threading.Thread(target=detect) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert downloads == [(langid.GLOTLID_REPO, langid.GLOTLID_FILE)]
    assert (langid.GLOTLID_REPO, langid.GLOTLID_FILE) == ("cis-lmu/glotlid",
                                                          "model_v3.bin")


def test_a_model_that_cannot_be_had_warns_once_and_skips(monkeypatch, caplog):
    downloads = _installed(monkeypatch, fail=OSError("offline"))
    with caplog.at_level(logging.WARNING, logger=langid.__name__):
        assert langid.detect("habari") is None
        assert not langid.available()
    assert len(downloads) == 1
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "offline" in warnings[0].getMessage()
