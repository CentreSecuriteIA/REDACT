"""Tests for benign sample generation and loading."""

import pandas as pd
import pytest

from redact.jailbreak.manipulation.benign import (
    BENIGN_CATEGORIES,
    process_category,
    load_benign_data,
)
from tests.conftest import MockBackend


class TestBenignCategories:
    def test_count(self):
        assert len(BENIGN_CATEGORIES) == 58

    def test_tuple_structure(self):
        for main, sub in BENIGN_CATEGORIES:
            assert isinstance(main, str)
            assert isinstance(sub, str)

    def test_domains(self):
        domains = set(cat[0] for cat in BENIGN_CATEGORIES)
        assert "Home & Daily Life" in domains
        assert "Health & Wellness" in domains
        assert "Technology & Tips" in domains


class TestProcessCategory:
    def test_returns_rows(self):
        qa_text = (
            "**Prompt 1:**\n"
            "**Question:** What is cooking?\n"
            "**Answer:** Cooking is preparing food.\n\n"
            "**Prompt 2:**\n"
            "**Question:** How to bake?\n"
            "**Answer:** Use an oven."
        )
        backend = MockBackend(qa_text)
        rows = process_category(
            ("Home & Daily Life", "Cooking & Baking"),
            backend, "model",
        )
        assert len(rows) > 0
        assert all("prompt" in r for r in rows)
        assert all("answer" in r for r in rows)

    def test_short_and_long_types(self):
        qa_text = (
            "**Prompt 1:**\n"
            "**Question:** Q?\n"
            "**Answer:** A."
        )
        backend = MockBackend(qa_text)
        rows = process_category(
            ("Test", "Sub"), backend, "model",
        )
        types = set(r["answer_type"] for r in rows)
        assert "short" in types
        assert "long" in types


class TestLoadBenignData:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_benign_data(path=tmp_path / "nonexistent.csv")

    def test_loads_csv(self, tmp_path):
        csv_path = tmp_path / "benign.csv"
        df = pd.DataFrame({
            "main_category": ["A", "A"],
            "sub_category": ["S1", "S1"],
            "prompt": ["q1", "q2"],
            "answer": ["a1", "a2"],
            "answer_type": ["short", "long"],
        })
        df.to_csv(csv_path, index=False)

        data = load_benign_data(path=csv_path)
        assert "short" in data
        assert "long" in data
        assert "all_subcategories" in data
        assert "S1" in data["all_subcategories"]
