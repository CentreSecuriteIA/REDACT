"""Tests for jailbreak distribution re-exports."""

import pandas as pd

from redact.jailbreak.distribution import (
    balanced_counts,
    deterministic_balanced_assign,
    merge_groups,
    split_by_technique_functions,
)


class TestReExports:
    def test_balanced_counts_sums(self):
        dists = [0, 0, 0]
        new_counts, updated = balanced_counts(10, dists)
        assert sum(new_counts) == 10
        assert max(new_counts) - min(new_counts) <= 1

    def test_merge_groups_is_normalize_origin(self):
        df = pd.DataFrame({"origin": ["generated", "handcrafted", "unknown"]})
        result = merge_groups(df)
        assert list(result["origin"]) == ["generated", "handcrafted", "dataset"]

    def test_deterministic_reproducible(self):
        df = pd.DataFrame({
            "category": ["A"] * 10 + ["B"] * 10,
            "origin": ["generated"] * 20,
            "text": [f"t{i}" for i in range(20)],
        })
        r1 = deterministic_balanced_assign(df, 3)
        r2 = deterministic_balanced_assign(df, 3)
        for a, b in zip(r1, r2):
            assert list(a["text"]) == list(b["text"])

    def test_split_by_technique_functions_exists(self):
        assert callable(split_by_technique_functions)
