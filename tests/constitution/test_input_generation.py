"""Tests for constitution-to-input generation pipeline."""

import pandas as pd
import pytest

from tests.conftest import MockBackend
from redact.constitution.input_generation import (
    ConstitutionInputResult,
    ConstitutionInputPipeline,
    get_available_styles,
)


class TestConstitutionInputResult:
    def test_acceptance_rate(self):
        result = ConstitutionInputResult(
            total_prompts_generated=10,
            total_prompts_accepted=7,
            total_prompts_rejected=3,
        )
        assert result.acceptance_rate == 0.7

    def test_zero_generated(self):
        result = ConstitutionInputResult()
        assert result.acceptance_rate == 0.0

    def test_fields(self):
        result = ConstitutionInputResult()
        assert result.total_entries_processed == 0
        assert result.skipped_entries == 0


class TestGetAvailableStyles:
    def test_returns_list(self):
        styles = get_available_styles()
        assert isinstance(styles, list)

    def test_nonexistent_dir(self, tmp_path):
        styles = get_available_styles(prompt_dir=tmp_path / "nope")
        assert styles == []


class TestConstitutionInputPipelineLoading:
    def _write_constitution_csv(self, dir_path, entries):
        dir_path.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(entries)
        df.to_csv(dir_path / "merged.csv", index=False)

    def test_load_constitution_from_merged(self, tmp_path):
        const_dir = tmp_path / "constitution"
        self._write_constitution_csv(const_dir, [
            {
                "constitution_category": "Cat",
                "constitution_subcategory": "Sub",
                "sample_description": "sample",
                "entry_type": "harmful",
                "source_category": "Violence",
                "source_group_tag": "general",
                "model": "test",
            },
        ])
        backend = MockBackend()
        pipeline = ConstitutionInputPipeline(
            gen_backend=backend, gen_model="m",
            constitution_dir=const_dir,
            output_dir=tmp_path / "output",
        )
        df = pipeline._load_constitution()
        assert len(df) == 1
        assert df.iloc[0]["source_category"] == "Violence"

    def test_load_constitution_filters(self, tmp_path):
        const_dir = tmp_path / "constitution"
        self._write_constitution_csv(const_dir, [
            {"constitution_category": "C1", "constitution_subcategory": "S",
             "sample_description": "s1", "entry_type": "harmful",
             "source_category": "Violence", "source_group_tag": "g", "model": "m"},
            {"constitution_category": "C2", "constitution_subcategory": "S",
             "sample_description": "s2", "entry_type": "benign",
             "source_category": "Privacy", "source_group_tag": "g", "model": "m"},
        ])
        backend = MockBackend()
        pipeline = ConstitutionInputPipeline(
            gen_backend=backend, gen_model="m",
            constitution_dir=const_dir,
            output_dir=tmp_path / "output",
        )

        df_harmful = pipeline._load_constitution(entry_types=["harmful"])
        assert len(df_harmful) == 1

        df_privacy = pipeline._load_constitution(source_categories=["Privacy"])
        assert len(df_privacy) == 1
        assert df_privacy.iloc[0]["entry_type"] == "benign"

    def test_load_constitution_missing_raises(self, tmp_path):
        backend = MockBackend()
        pipeline = ConstitutionInputPipeline(
            gen_backend=backend, gen_model="m",
            constitution_dir=tmp_path / "empty",
            output_dir=tmp_path / "output",
        )
        with pytest.raises(FileNotFoundError):
            pipeline._load_constitution()


class TestConstitutionInputPipelineGeneration:
    def test_generate_for_entry_no_checker(self, tmp_path):
        backend = MockBackend("1. Full prompt one\n2. Full prompt two\n3. Full prompt three")
        pipeline = ConstitutionInputPipeline(
            gen_backend=backend, gen_model="m",
            constitution_dir=tmp_path,
            output_dir=tmp_path / "output",
        )
        entry = pd.Series({
            "source_category": "Violence",
            "sample_description": "Instructions for harm",
            "constitution_subcategory": "Weapons",
            "entry_type": "harmful",
        })
        results = pipeline.generate_for_entry(
            entry=entry,
            prompt_config={"system_prompt": "Test", "template": "Generate {num_samples} prompts about {sample_description}"},
            samples_per_entry=3,
            build_check_messages=None,
        )
        assert len(results) > 0
        assert all(r.accepted is True for r in results)

    def test_generate_for_entry_with_checker(self, tmp_path):
        gen_backend = MockBackend("1. Sample one\n2. Sample two")
        check_backend = MockBackend(["Yes", "No bad"])
        pipeline = ConstitutionInputPipeline(
            gen_backend=gen_backend, gen_model="m",
            check_backend=check_backend, check_model="m",
            constitution_dir=tmp_path,
            output_dir=tmp_path / "output",
        )
        entry = pd.Series({
            "source_category": "Test",
            "sample_description": "desc",
            "constitution_subcategory": "sub",
            "entry_type": "harmful",
        })
        results = pipeline.generate_for_entry(
            entry=entry,
            prompt_config={"system_prompt": "Test", "template": "Generate {num_samples} items about {sample_description}"},
            samples_per_entry=2,
            build_check_messages=lambda s: [{"role": "user", "content": f"check {s}"}],
        )
        accepted = [r for r in results if r.accepted]
        rejected = [r for r in results if not r.accepted]
        assert len(accepted) == 1
        assert len(rejected) == 1

    def test_save_results(self, tmp_path):
        backend = MockBackend("1. Generated prompt")
        output_dir = tmp_path / "output"
        pipeline = ConstitutionInputPipeline(
            gen_backend=backend, gen_model="m",
            constitution_dir=tmp_path,
            output_dir=output_dir,
        )
        from redact.content_moderation.generation import SampleResult
        entry = pd.Series({
            "source_category": "Violence",
            "constitution_category": "Cat",
            "constitution_subcategory": "Sub",
            "sample_description": "desc",
            "entry_type": "harmful",
            "source_group_tag": "general",
        })
        results = [SampleResult(text="generated text", accepted=True, reasoning="", turn=0)]
        pipeline._save_results(entry, results, "long")
        # Check CSV was written
        cat_dir = output_dir / "Violence"
        assert cat_dir.exists()
        csv_files = list(cat_dir.glob("*.csv"))
        assert len(csv_files) == 1
