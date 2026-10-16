"""Tests for content moderation InputPipeline with mock backend."""

from redact.content_moderation.inputs import InputPipeline
from redact.content_moderation.results import (
    CategoryResult,
    SampleResult,
    TurnResult,
)
from tests.conftest import MockBackend, make_client


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
            gen=make_client(backend, "m"),
            check=make_client(backend, "m"),
            dataset_dir=tmp_path,
        )
        config = {"system_prompt": "Test", "template": "Generate {num_samples} items"}
        raw, extracted = pipeline.generate_batch(config, samples_per_request=3)
        assert len(extracted) >= 2


class TestInputPipelineCheckSamples:
    def test_check_samples(self, tmp_path):
        check_backend = MockBackend(["Yes", "No bad quality"])
        pipeline = InputPipeline(
            gen=make_client(MockBackend(), "m"),
            check=make_client(check_backend, "m"),
            dataset_dir=tmp_path,
        )
        results = pipeline.check_samples(
            ["sample1", "sample2"],
            build_check_messages=lambda o, s: [{"role": "user", "content": f"check {s}"}],
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
            gen=make_client(gen_backend, "m"),
            check=make_client(check_backend, "m"),
            dataset_dir=ds_dir,
        )
        config = {"system_prompt": "Test", "template": "Generate {num_samples} items"}
        result = pipeline.run_turn(
            prompt_config=config,
            build_check_messages=lambda o, s: [{"role": "user", "content": s}],
            turn_index=0,
            samples_per_request=2,
            category="test_cat",
        )
        assert isinstance(result, TurnResult)
        assert result.accepted_count >= 0


class TestRunStandaloneInputs:
    def test_a_fresh_run_clears_its_categories_then_runs_the_turn_loop(
            self, tmp_path, monkeypatch):
        from redact.content_moderation import run_standalone_inputs
        from redact.dataset import append_samples

        append_samples(["old cyber"], category="Cyber", turn=0, dataset_dir=tmp_path)
        append_samples(["old fraud"], category="Fraud", turn=0, dataset_dir=tmp_path)
        ran = []

        def run_standalone(self, categories, **kwargs):
            ran.append((self.gen, self.check, self.dataset_dir, categories,
                        (tmp_path / "Cyber" / "samples.csv").exists(), kwargs))
            return ["the results"]

        monkeypatch.setattr(InputPipeline, "run_standalone", run_standalone)
        gen = make_client(MockBackend(), "gen-model")
        check = make_client(MockBackend(), "check-model")
        returned = run_standalone_inputs(
            [("Cyber", {})], gen, check, tmp_path,
            samples_per_category=4, samples_per_request=2, use_metaprompt=True,
            seeds_name="unused", num_seeds=3, resume=False, verbose=False,
        )
        assert returned == ["the results"]
        (got_gen, got_check, folder, categories, cyber_left, kwargs), = ran
        assert (got_gen, got_check, folder) == (gen, check, tmp_path)
        assert categories == [("Cyber", {})] and not cyber_left
        assert (tmp_path / "Fraud" / "samples.csv").exists()
        forwarded = ("samples_per_category", "samples_per_request", "num_seeds")
        assert {name: kwargs[name] for name in forwarded} == {
            "samples_per_category": 4, "samples_per_request": 2, "num_seeds": 3}
        assert kwargs["seeds_db"] is None
