"""Tests for the package root: the output directory and the seed."""

import os
import random
import subprocess
import sys
from pathlib import Path

import redact


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

    def test_importing_the_package_seeds_nothing(self):
        """The global RNG is the application's: import calls no ``random.seed``."""
        src = str(Path(redact.__file__).parents[1])
        code = ("import random; calls = []; "
                "random.seed = lambda *a, **k: calls.append(a); "
                "import redact; print(calls)")
        env = {**os.environ, "PYTHONPATH": src}
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, env=env, check=True)
        assert out.stdout.strip() == "[]"


REPO = Path(__file__).resolve().parents[1]


class TestProjectFiles:
    def test_the_project_urls_name_the_repository(self):
        import tomllib
        urls = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))[
            "project"]["urls"]
        assert urls["Homepage"] == "https://github.com/CentreSecuriteIA/REDACT"
        assert urls["Issues"] == "https://github.com/CentreSecuriteIA/REDACT/issues"
