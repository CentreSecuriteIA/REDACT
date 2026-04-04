"""Shared test fixtures for the REDACT test suite."""

import multiprocessing
import pytest
import pandas as pd


def pytest_configure(config):
    # vLLM v1 explicitly uses multiprocessing.get_context("fork") by default,
    # ignoring the global start method. Setting this env var before vLLM is
    # imported forces it to use 'spawn', which avoids CUDA re-init failures
    # in forked subprocesses. The global set_start_method call is kept as a
    # belt-and-suspenders fallback.
    import os
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

from redact.llms.base import LLMBackend


# ---------------------------------------------------------------------------
# Mock LLM backend
# ---------------------------------------------------------------------------

class MockBackend(LLMBackend):
    """Configurable mock backend for testing.

    Accepts a list of responses to return sequentially, or a single
    string that is returned every time. Tracks all calls for assertions.
    """

    def __init__(self, responses: str | list[str] = "mock response"):
        if isinstance(responses, str):
            self._responses = [responses]
            self._cycle = True
        else:
            self._responses = list(responses)
            self._cycle = False
        self._call_index = 0
        self.calls: list[dict] = []

    def generate(self, messages: list[dict], model: str, **kwargs) -> str:
        self.calls.append({"messages": messages, "model": model, **kwargs})
        if self._cycle:
            return self._responses[0]
        if self._call_index >= len(self._responses):
            return self._responses[-1]
        resp = self._responses[self._call_index]
        self._call_index += 1
        return resp

    @property
    def backend_name(self) -> str:
        return "mock"


@pytest.fixture()
def mock_backend():
    """Return a MockBackend that always returns 'mock response'."""
    return MockBackend()


@pytest.fixture()
def mock_backend_factory():
    """Return a factory for creating MockBackend with custom responses."""
    return MockBackend


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
        "aliases": {
            "Alias A": "Cat A",
            "Alt B": "Cat B",
        },
        "groups": {
            "group1": ["Cat B"],
            "group2": ["Cat A", "Cat B"],
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
