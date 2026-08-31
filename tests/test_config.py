"""Tests for package-level config, seed, and output directory."""

import random

import pytest


class TestGetOutputDir:
    def test_from_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("REDACT_OUTPUT_DIR", str(tmp_path))
        from redact import get_output_dir
        assert get_output_dir() == tmp_path.resolve()

    def test_fallback_no_env(self, monkeypatch):
        monkeypatch.delenv("REDACT_OUTPUT_DIR", raising=False)
        from redact import get_output_dir
        # Should return something without crashing
        result = get_output_dir()
        assert result.is_absolute()


class TestSetSeed:
    def test_deterministic(self):
        from redact import set_seed
        set_seed(42)
        a = random.random()
        set_seed(42)
        b = random.random()
        assert a == b

    def test_different_seeds_differ(self):
        from redact import set_seed
        set_seed(1)
        a = random.random()
        set_seed(2)
        b = random.random()
        assert a != b


class TestConfig:
    def test_get_returns_env(self, monkeypatch):
        monkeypatch.setenv("TEST_VAR_XYZ", "hello")
        from redact import Config
        assert Config.get("TEST_VAR_XYZ") == "hello"

    def test_get_default(self, monkeypatch):
        monkeypatch.delenv("TEST_VAR_XYZ", raising=False)
        from redact import Config
        assert Config.get("TEST_VAR_XYZ", "fallback") == "fallback"

    def test_validate_missing_raises(self, monkeypatch):
        monkeypatch.delenv("VENICE_API_KEY", raising=False)
        from redact import Config
        with pytest.raises(ValueError, match="Missing required"):
            Config.validate(require=["VENICE_API_KEY"])

    def test_validate_present_passes(self, monkeypatch):
        monkeypatch.setenv("VENICE_API_KEY", "test-key")
        from redact import Config
        Config.validate(require=["VENICE_API_KEY"])  # should not raise

    def test_is_configured_true(self, monkeypatch):
        monkeypatch.setenv("KEY_A", "val")
        monkeypatch.setenv("KEY_B", "val")
        from redact import Config
        assert Config.is_configured("KEY_A", "KEY_B")

    def test_is_configured_false(self, monkeypatch):
        monkeypatch.delenv("KEY_A", raising=False)
        monkeypatch.setenv("KEY_B", "val")
        from redact import Config
        assert not Config.is_configured("KEY_A", "KEY_B")
