"""Tests for content moderation InputPipeline with mock backend."""

import pytest

from tests.conftest import MockBackend
from redact.content_moderation.generation import (
    SampleResult,
    TurnResult,
    CategoryResult,
    InputPipeline,
)


class TestRunCategoryDeprecation:
    def test_run_category_emits_deprecation_warning(self, tmp_path):
        pipeline = InputPipeline(
            gen_backend=MockBackend("1. a\n2. b"), gen_model="m",
            check_backend=MockBackend("Yes"), check_model="m",
            dataset_dir=tmp_path,
        )
        with pytest.warns(DeprecationWarning, match="constitution-seeded"):
            pipeline.run_category(
                category="Cyber",
                prompt_config={"system_prompt": "s", "template": "gen {Category}"},
                build_check_messages=lambda s: [{"role": "user", "content": s}],
                num_turns=0, save=False,   # 0 turns: warning fires, no generation
            )


class TestDataclasses:
    def test_sample_result(self):
        sr = SampleResult(text="hello", accepted=True, reasoning="", turn=0)
        assert sr.accepted is True

    def test_turn_result_acceptance_rate(self):
        tr = TurnResult(
            turn_index=0, raw_output="", extracted_count=10,
            accepted_count=8, rejected_count=2,
        )
        assert tr.acceptance_rate == 0.8

    def test_turn_result_zero_extracted(self):
        tr = TurnResult(
            turn_index=0, raw_output="", extracted_count=0,
            accepted_count=0, rejected_count=0,
        )
        assert tr.acceptance_rate == 0.0

    def test_category_result_totals(self):
        cr = CategoryResult(category="test")
        cr.turns.append(TurnResult(0, "", 5, 3, 2))
        cr.turns.append(TurnResult(1, "", 5, 4, 1))
        assert cr.total_extracted == 10
        assert cr.total_accepted == 7
        assert cr.total_rejected == 3
        assert cr.overall_acceptance_rate == 0.7

    def test_category_result_empty(self):
        cr = CategoryResult(category="empty")
        assert cr.total_extracted == 0
        assert cr.overall_acceptance_rate == 0.0


class TestInputPipelineGenerateBatch:
    def test_generate_batch(self, tmp_path):
        backend = MockBackend("1. First sample\n2. Second sample\n3. Third sample")
        pipeline = InputPipeline(
            gen_backend=backend, gen_model="m",
            check_backend=backend, check_model="m",
            dataset_dir=tmp_path,
        )
        config = {"system_prompt": "Test", "template": "Generate {num_samples} items"}
        raw, extracted = pipeline.generate_batch(config, samples_per_request=3)
        assert len(extracted) >= 2


class TestInputPipelineCheckSamples:
    def test_check_samples(self, tmp_path):
        check_backend = MockBackend(["Yes", "No bad quality"])
        pipeline = InputPipeline(
            gen_backend=MockBackend(), gen_model="m",
            check_backend=check_backend, check_model="m",
            dataset_dir=tmp_path,
        )
        results = pipeline.check_samples(
            ["sample1", "sample2"],
            build_check_messages=lambda s: [{"role": "user", "content": f"check {s}"}],
            turn_index=0,
        )
        assert len(results) == 2
        assert results[0].accepted is True
        assert results[1].accepted is False


class TestInputPipelineRunTurn:
    def test_run_turn(self, tmp_path):
        gen_backend = MockBackend("1. Good sample\n2. Another sample")
        check_backend = MockBackend("Yes")
        ds_dir = tmp_path / "Datasets"
        ds_dir.mkdir()

        pipeline = InputPipeline(
            gen_backend=gen_backend, gen_model="m",
            check_backend=check_backend, check_model="m",
            dataset_dir=ds_dir,
        )
        config = {"system_prompt": "Test", "template": "Generate {num_samples} items"}
        result = pipeline.run_turn(
            prompt_config=config,
            build_check_messages=lambda s: [{"role": "user", "content": s}],
            turn_index=0,
            samples_per_request=2,
            category="test_cat",
        )
        assert isinstance(result, TurnResult)
        assert result.accepted_count >= 0
