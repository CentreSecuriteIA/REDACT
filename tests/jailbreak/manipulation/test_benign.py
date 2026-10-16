"""Tests for benign sample generation and loading."""

import pandas as pd
import pytest

from redact.dataset import load_taxonomy
from redact.jailbreak.manipulation.benign import (
    BENIGN_CATEGORIES,
    benign_categories,
    load_benign_data,
    process_category,
)
from tests.conftest import MockBackend, make_client


class TestBenignTaxonomy:
    def test_the_packaged_file_holds_the_categories_in_code(self):
        assert benign_categories(load_taxonomy("benign")) == BENIGN_CATEGORIES

    def test_subcategories_as_a_list_a_dict_or_none(self):
        taxonomy = {"categories": {
            "A": {"description": "", "subcategories": ["a1", "a2"]},
            "B": {"description": "", "subcategories": {"b1": "", "b2": ""}},
            "C": "a category given as its description",
            "D": {"description": "", "subcategories": ["d1", "d1"]},
        }}
        assert benign_categories(taxonomy) == [
            ("A", "a1"), ("A", "a2"), ("B", "b1"), ("B", "b2"), ("C", "C"),
            ("D", "d1")]


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
    def test_wrong_pair_count_rejects_the_reply(self):
        backend = MockBackend("**Question:** Q?\n**Answer:** A.")
        rows = process_category(("Test", "Sub"), make_client(backend), num_samples=2)
        assert rows == []

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
            make_client(backend),
            num_samples=2,
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
            ("Test", "Sub"), make_client(backend), num_samples=1,
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


def test_answer_styles_come_from_the_prompt_file():
    from redact.llms.prompting import load_prompt

    config = load_prompt("jailbreak", "benign_generation")
    assert set(config["answer_styles"]) == {"short", "long"}
