"""The variants table: which template variant each written row drew."""

import json

import pandas as pd
import pytest

from redact.jailbreak.directives import PoolRef, read_pools
from redact.jailbreak.pool import all_techniques
from redact.jailbreak.variants import COLUMNS, variants_of, write_variants
from redact.stages import generate_jailbreaks


def _technique(name: str, ref: PoolRef | None = None):
    """A stand-in technique named ``name`` drawing from ``ref``."""
    def fn(prompt, *, rng=None):
        return prompt, ""
    fn.__name__ = name
    if ref is not None:
        fn.variant_pool = ref
    return fn


_REGISTRY = {fn.__name__: fn for fn in (
    _technique("to_frame", PoolRef("t.json", "frame")),
    _technique("to_ask", PoolRef("t.json", "ask")),
    _technique("to_cont", PoolRef("t.json", "cont", (), info_key="style")),
    _technique("to_wrap", PoolRef("t.json", "wrap", info_key=None)),
    _technique("to_code"),
)}


class TestVariantsOf:
    def test_variants_go_to_the_techniques_that_write_them_in_order(self):
        assert variants_of("to_frame+to_ask",
                           "directive=frame;variant=one;type=ask;variant=two",
                           _REGISTRY) == [("to_frame", "one"), ("to_ask", "two")]

    def test_a_technique_names_its_variant_under_its_own_key(self):
        assert variants_of("to_frame+to_cont", "directive=frame;variant=one;style=s",
                           _REGISTRY) == [("to_frame", "one"), ("to_cont", "s")]

    def test_a_technique_that_names_no_variant_draws_none(self):
        assert variants_of("to_code+to_wrap+to_ask",
                           "shift=3;policy_id=p;type=ask;variant=two",
                           _REGISTRY) == [("to_ask", "two")]

    def test_a_noop_step_wrote_no_variant(self):
        assert variants_of("to_frame+to_ask", "noop=to_frame;type=ask;variant=two",
                           _REGISTRY) == [("to_ask", "two")]

    @pytest.mark.parametrize("info", [
        "DISCARDED; technique=to_ask; reason=empty text", "ERROR: boom"])
    def test_a_rejected_or_error_row_drew_none(self, info):
        assert variants_of("to_frame+to_ask", info, _REGISTRY) == []


def test_the_table_counts_each_variant_on_of_the_pool_and_any_other_drawn(tmp_path):
    """``to_old`` is out of the pool (a kept plan's), ``gamma`` is off, and
    ``to_wrap`` names no variant. An absolute file is read where it is."""
    (tmp_path / "t.json").write_text(json.dumps({"d": [
        {"name": "alpha", "template": "alpha {prompt}"},
        {"name": "beta", "template": "beta {prompt}"},
        {"name": "gamma", "template": "gamma {prompt}", "weight": 0},
    ]}), encoding="utf-8")
    file = str(tmp_path / "t.json")
    ref = PoolRef(file, "d", info_key="task")
    pool = [_technique("to_a", ref), _technique("to_b", ref),
            _technique("to_wrap", PoolRef(file, "d", info_key=None))]
    registry = {fn.__name__: fn for fn in [*pool, _technique("to_old", ref)]}
    pd.DataFrame({
        "prompt": ["p"] * 5,
        "technique": ["to_a", "to_a", "to_b", "to_b", "to_old"],
        "technique_info": ["task=alpha", "task=alpha", "task=beta",
                           "DISCARDED; technique=to_b; reason=empty text",
                           "task=alpha"],
    }).to_csv(tmp_path / "jb.csv", index=False)

    out = write_variants(tmp_path / "jb.csv", tmp_path / "jb.manifest.jsonl",
                         pool, registry)

    assert out == tmp_path / "jb.variants.csv"
    assert pd.read_csv(out).values.tolist() == [
        ["to_a", "alpha", 2], ["to_a", "beta", 0],
        ["to_b", "alpha", 0], ["to_b", "beta", 1],
        ["to_old", "alpha", 1],
    ]


def _inputs(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame({"prompt": [f"prompt number {i} about a topic"
                                    for i in range(n)],
                         "category": ["Cyber"] * n, "entry_type": ["benign"] * n})


def test_a_kind_counts_the_variants_its_rows_drew(tmp_path):
    """Natural variations by the indirect techniques, whose infos name their
    variant: one count per row, against the variants of each pool."""
    generate_jailbreaks(
        inputs=_inputs(), output_path=tmp_path / "jb.csv", seed=3,
        augmentation={"adversarial": None,
                      "natural": {"families": {"allow": ["indirect"]},
                                  "techniques": {"allow": None}}},
        auto_generate_benign=False, verbose=False)

    table = pd.read_csv(tmp_path / "variations.variants.csv")
    rows = pd.read_csv(tmp_path / "variations.csv")
    assert list(table.columns) == COLUMNS
    assert table["rows"].sum() == rows["accepted"].sum() == 5
    indirect = {fn.__name__: fn for fn in all_techniques()
                if "indirect" in getattr(fn, "families", ())}
    for name, fn in indirect.items():
        ref = fn.variant_pool
        names = [v["name"] for v in read_pools(ref.file)[ref.directive]]
        assert table.loc[table["technique"] == name, "variant"].tolist() == names


def test_a_fresh_run_removes_the_last_table_before_it_plans(tmp_path, monkeypatch):
    import redact.jailbreak.pipeline as JPipe

    stale = tmp_path / "jb.variants.csv"
    stale.write_text("technique,variant,rows\nto_x,old,3\n", encoding="utf-8")

    def stop(source, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(JPipe, "drive_units", stop)
    with pytest.raises(KeyboardInterrupt):
        generate_jailbreaks(
            inputs=_inputs(), output_path=tmp_path / "jb.csv", seed=3, resume=False,
            augmentation={"adversarial": {"pure_only": True}, "natural": None},
            auto_generate_benign=False, verbose=False)
    assert not stale.exists()


def test_a_kind_skipped_on_a_fresh_run_clears_its_table(tmp_path):
    stale = tmp_path / "jb.variants.csv"
    stale.write_text("technique,variant,rows\n", encoding="utf-8")
    generate_jailbreaks(
        inputs=_inputs(), output_path=tmp_path / "jb.csv", seed=3, resume=False,
        augmentation={"adversarial": {"entry_types": ["harmful"]}, "natural": None},
        auto_generate_benign=False, verbose=False)
    assert not stale.exists()
