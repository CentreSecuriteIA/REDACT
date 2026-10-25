"""Tests for benign sample generation and loading."""

import json

import pandas as pd
import pytest

from redact.dataset import Ledger, Manifest, create_taxonomy, load_taxonomy
from redact.jailbreak.manipulation import benign as B
from redact.jailbreak.manipulation.benign import (
    benign_categories,
    get_or_generate_benign_data,
    load_benign_data,
)
from redact.llms.backends import ComputeConfig
from redact.llms.client import ModelClient
from tests.conftest import (
    Killed,
    MockBackend,
    kill_after_rows,
    lock_is_free,
    make_client,
)

# The pairs a unit asks for.
_PAIRS = 8
_KEY = ["main_category", "sub_category", "answer_type"]


def _reply(n: int = _PAIRS) -> str:
    return "\n\n".join(
        f"**Prompt {i}:**\n**Question:** Question {i}?\n**Answer:** Answer {i}."
        for i in range(1, n + 1))


def _yes() -> ModelClient:
    """A check model that accepts every pair."""
    return make_client(MockBackend("Yes", model="mock-check"), "mock-check")


def _generate(client, out, **kwargs) -> dict:
    """The benign data, quietly, with :func:`_yes` as the check model."""
    return get_or_generate_benign_data(client, out, verbose=False, check=_yes(),
                                       **kwargs)


def _taxonomy(subcategories: int) -> dict:
    return {"name": "test", "categories": {"Main": {
        "description": "", "subcategories": [f"s{i}" for i in range(subcategories)]}}}


def _ledger(out) -> Ledger:
    return Ledger.sidecar(out, _KEY)


class _WidthBackend(MockBackend):
    """Native batching, so one ``generate()`` per call; records each call's width."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.widths: list[int] = []

    @property
    def compute_config(self) -> ComputeConfig:
        return ComputeConfig(supports_native_batching=True)

    def generate(self, messages_list, **kwargs):
        self.widths.append(len(messages_list))
        return super().generate(messages_list, **kwargs)


class TestBenignTaxonomy:
    def test_the_packaged_file_gives_named_subcategories_each_once(self):
        pairs = benign_categories(load_taxonomy("benign"))
        assert pairs and len(set(pairs)) == len(pairs)
        assert all(main.strip() and sub.strip() for main, sub in pairs)

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


class TestGeneration:
    def test_every_unit_is_live_at_once_and_a_call_carries_many(
            self, tmp_path, monkeypatch):
        seen, drive_units = {}, B.drive_units

        def recording(source, **kwargs):
            seen.update(kwargs)
            return drive_units(source, **kwargs)

        monkeypatch.setattr(B, "drive_units", recording)
        backend, subcategories, batch_size = _WidthBackend(_reply()), 5, 16
        data = get_or_generate_benign_data(
            make_client(backend), tmp_path / "benign.csv", verbose=False, check=_yes(),
            taxonomy=_taxonomy(subcategories), batch_size=batch_size)
        units = 2 * subcategories           # a short and a long unit each
        assert seen["pool_size"] >= units and seen["cap"] == batch_size
        assert backend.widths == [units]    # all in one call
        assert len(data["short"]) == len(data["long"]) == subcategories * _PAIRS

    def test_a_taxonomy_file_gives_a_unit_per_subcategory_and_answer_type(
            self, tmp_path):
        tax_dir, out = tmp_path / "taxonomy", tmp_path / "benign.csv"
        create_taxonomy("mine", {
            "Cooking": {"description": "", "subcategories": ["Bread", "Soup"]},
            "Travel": {"description": "", "subcategories": ["Trains"]},
        }, taxonomy_dir=tax_dir, verbose=False)
        backend = MockBackend(_reply())
        data = get_or_generate_benign_data(
            make_client(backend), out, verbose=False, check=_yes(), taxonomy="mine",
            taxonomy_dir=tax_dir)

        units = [(main, sub, kind) for main, sub in
                 [("Cooking", "Bread"), ("Cooking", "Soup"), ("Travel", "Trains")]
                 for kind in ("short", "long")]
        planned = [tuple(row[k] for k in _KEY) for row in Manifest.sidecar(out).load()]
        assert planned == units
        assert len(backend.calls) == len(units)
        df = pd.read_csv(out)
        assert list(df.columns) == [
            "main_category", "sub_category", "prompt", "answer", "answer_type"]
        assert set(df.groupby(_KEY).size().items()) == {(u, _PAIRS) for u in units}
        assert sorted(data["all_subcategories"]) == ["Bread", "Soup", "Trains"]
        assert _ledger(out).completed() == set(units)

    def test_a_finished_run_makes_no_call(self, tmp_path):
        out, backend, units = tmp_path / "benign.csv", MockBackend(_reply()), 4
        for _ in range(2):
            _generate(make_client(backend), out, taxonomy=_taxonomy(units // 2))
        assert len(backend.calls) == units
        assert len(pd.read_csv(out)) == units * _PAIRS
        assert lock_is_free(_ledger(out).path)

    def test_a_run_killed_between_rows_and_acks_leaves_each_unit_once(
            self, tmp_path, monkeypatch):
        """40 units save as 32, then 8; the kill lands between the second
        save's rows and its acks."""
        out, backend = tmp_path / "benign.csv", MockBackend(_reply())
        units, saved = 40, 32               # CommitBuffer saves every 32 units
        kill_after_rows(monkeypatch, B, on_call=2)
        with pytest.raises(Killed):
            _generate(make_client(backend), out, taxonomy=_taxonomy(units // 2))
        landed = out.read_bytes()
        assert len(pd.read_csv(out)) == units * _PAIRS      # every row landed
        assert len(_ledger(out).completed()) == saved       # the last 8 acks did not
        assert lock_is_free(_ledger(out).path)

        _generate(make_client(backend), out, taxonomy=_taxonomy(units // 2))
        assert len(backend.calls) == units + (units - saved)    # only those 8 again
        df = pd.read_csv(out)
        assert len(df) == units * _PAIRS
        assert (df.groupby(_KEY).size() == _PAIRS).all()
        assert landed.endswith(out.with_name("benign.csv.cut").read_bytes())
        assert len(_ledger(out).completed()) == units
        assert lock_is_free(_ledger(out).path)

    def test_an_old_cache_is_read_and_only_its_missing_units_generated(self, tmp_path):
        out = tmp_path / "benign_samples.csv"
        old = pd.DataFrame([
            {"main_category": "Main", "sub_category": sub, "prompt": f"q {sub} {kind}",
             "answer": "a", "answer_type": kind}
            for sub, kind in [("s0", "short"), ("s0", "long"), ("s1", "short")]])
        old.to_csv(out, index=False)

        # Without a client the cache is read as it is.
        data = _generate(None, out)
        assert data["all_subcategories"] == ["s0", "s1"] and len(data["long"]) == 1

        backend = MockBackend(_reply())
        data = _generate(make_client(backend), out, taxonomy=_taxonomy(2))
        assert len(backend.calls) == 1                       # s1's long pairs
        df = pd.read_csv(out)
        pd.testing.assert_frame_equal(df.iloc[:len(old)], old)
        assert len(data["by_subcat_long"]["s1"]) == _PAIRS
        assert _ledger(out).completed() == {
            ("Main", sub, kind) for sub in ("s0", "s1") for kind in ("short", "long")}

    def test_a_unit_rejected_every_time_is_acked_with_no_rows(self, tmp_path):
        out = tmp_path / "benign.csv"
        backend = MockBackend(_reply(_PAIRS - 1))
        for _ in range(2):
            with pytest.raises(ValueError, match="rejected"):
                _generate(make_client(backend), out, taxonomy=_taxonomy(1))
            # Two units, three attempts each, and none again on resume.
            assert len(backend.calls) == 2 * 3
        assert not out.exists()
        acks = [json.loads(line) for line in _ledger(out).path.read_text().splitlines()]
        assert [ack.get("status") for ack in acks if "answer_type" in ack] == [
            "rejected", "rejected"]

    def test_a_reply_with_the_wrong_count_is_asked_again(self, tmp_path):
        out = tmp_path / "benign.csv"
        replies = [_reply(), _reply(1), _reply(1), _reply()]
        backend = _WidthBackend(replies)
        data = _generate(make_client(backend), out, taxonomy=_taxonomy(1))
        assert len(backend.calls) == len(replies)
        assert len(data["short"]) == len(data["long"]) == _PAIRS

    def test_the_prompt_overlay_reaches_the_prompts(self, tmp_path, monkeypatch):
        overlays, load = [], B.load_prompt
        monkeypatch.setattr(B, "load_prompt", lambda pipeline, name, prompt_dir: (
            overlays.append((name, prompt_dir)) or load(pipeline, name)))
        _generate(make_client(MockBackend(_reply())), tmp_path / "benign.csv",
                  taxonomy=_taxonomy(1), prompt_dir=tmp_path / "mine")
        assert overlays == [("benign_generation", tmp_path / "mine"),
                            ("benign_check", tmp_path / "mine")]

    def test_a_taxonomy_without_a_category_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="no category"):
            _generate(make_client(MockBackend(_reply())), tmp_path / "benign.csv",
                      taxonomy={"name": "empty", "categories": {}})

    def test_no_cache_and_no_client_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="no client"):
            _generate(None, tmp_path / "benign.csv")


class _Fake(MockBackend):
    """A backend that answers each request with ``reply(messages, temperature)``
    and records it."""

    def __init__(self, reply, model: str):
        super().__init__(model=model)
        self.reply, self.requests = reply, []

    def generate(self, messages_list, *, temperature=None, **kwargs):
        replies = []
        for messages in messages_list:
            self.requests.append({"messages": messages, "temperature": temperature})
            replies.append(self.reply(messages, temperature))
        return replies


#: The units of :func:`_run`'s one subcategory: a short and a long one.
_UNITS = 2


def _one_pair_prompts() -> tuple[str, ...]:
    """What the generation prompt says when it asks for one pair (the
    taxonomy of :func:`_taxonomy`, its first subcategory), for each answer
    type."""
    from redact.llms.prompting import build_messages, load_prompt

    config = load_prompt("jailbreak", "benign_generation")
    return tuple(build_messages(config, sub_category="s0", main_category="Main",
                                num_samples="1", answer_style=style)[-1]["content"]
                 for style in config["answer_styles"].values())


def _asks_one(messages) -> bool:
    return messages[-1]["content"].startswith(_one_pair_prompts())


def _again(n: int = 1) -> str:
    """A reply asked for one pair: ``n`` pairs, each the same."""
    return "\n\n".join("**Question:** Again?\n**Answer:** Again." for _ in range(n))


def _generator(again=_again) -> _Fake:
    """The generation model: the unit's eight pairs, or one when asked for one."""
    return _Fake(lambda messages, _t: again() if _asks_one(messages) else _reply(),
                 "mock-gen")


def _checker(rejects) -> _Fake:
    """A check model that rejects a pair whose check prompt holds any of
    ``rejects``, with a verdict naming it."""
    def verdict(messages, _temperature):
        content = str(messages)
        hit = next((text for text in rejects if text in content), None)
        return "Yes" if hit is None else f"No, {hit} is not benign."
    return _Fake(verdict, "mock-check")


def _run(tmp_path, gen: _Fake, check: _Fake | None, units: int = 1):
    out = tmp_path / "benign.csv"
    data = get_or_generate_benign_data(
        make_client(gen, gen.model), out, verbose=False,
        check=make_client(check, check.model) if check else None,
        taxonomy=_taxonomy(units))
    return out, data


def _acks(out) -> list[dict]:
    return [ack for ack in (json.loads(line)
                            for line in _ledger(out).path.read_text().splitlines())
            if "answer_type" in ack]


class TestCheck:
    def test_each_pair_is_read_by_the_check_model_at_the_check_temperature(
            self, tmp_path):
        from redact.llm_pipeline import CHECK_TEMPERATURE

        gen, check = _generator(), _checker([])
        out, data = _run(tmp_path, gen, check)
        assert len(gen.requests) == _UNITS           # one reply per unit
        assert len(check.requests) == 2 * _PAIRS     # each pair once
        assert {r["temperature"] for r in check.requests} == {CHECK_TEMPERATURE}
        shown = [str(r["messages"]) for r in check.requests]
        for i in range(1, _PAIRS + 1):
            assert sum(f"Question {i}?" in s and f"Answer {i}." in s
                       for s in shown) == _UNITS     # the short and the long unit
        assert all("Main / s0" in s for s in shown)
        assert len(data["short"]) == len(data["long"]) == _PAIRS
        assert [ack.get("rejected_pairs") for ack in _acks(out)] == [None, None]

    def test_without_a_check_model_the_generation_model_checks(self, tmp_path):
        from redact.llm_pipeline import CHECK_TEMPERATURE

        gen = _Fake(lambda messages, temperature: (
            "Yes" if temperature == CHECK_TEMPERATURE else _reply()), "mock-gen")
        _out, data = _run(tmp_path, gen, None)
        temperatures = [r["temperature"] for r in gen.requests]
        assert temperatures.count(CHECK_TEMPERATURE) == 2 * _PAIRS
        assert len(data["short"]) == _PAIRS

    def test_a_rejected_pair_is_asked_for_alone_again_with_the_verdict(self, tmp_path):
        from redact.llm_pipeline import CHECK_TEMPERATURE

        gen, check = _generator(), _checker(["Question 3?"])
        out, data = _run(tmp_path, gen, check)
        again = [r for r in gen.requests if _asks_one(r["messages"])]
        assert len(again) == _UNITS                  # pair 3 of each unit
        assert all("No, Question 3? is not benign." in r["messages"][-1]["content"]
                   for r in again)
        # The new pair was checked and accepted, and takes the old one's place;
        # its check, as the first, at the check temperature.
        rechecks = [r for r in check.requests if "Again?" in str(r["messages"])]
        assert len(rechecks) == _UNITS
        assert {r["temperature"] for r in rechecks} == {CHECK_TEMPERATURE}
        assert {r["temperature"] for r in check.requests} == {CHECK_TEMPERATURE}
        questions = [pair["prompt"] for pair in data["short"]]
        assert "Question 3?" not in questions and "Again?" in questions
        assert len(questions) == _PAIRS
        assert [ack.get("rejected_pairs") for ack in _acks(out)] == [None, None]

    def test_a_pair_rejected_every_time_is_left_out_and_counted(self, tmp_path):
        """Three generations of the pair in all: the reply's, then two alone,
        each with the last verdict."""
        gen, check = _generator(), _checker(["Question 3?", "Again?"])
        out, data = _run(tmp_path, gen, check)
        again = [r for r in gen.requests if _asks_one(r["messages"])]
        assert len(again) == 2 * 2
        assert "No, Again? is not benign." in again[-1]["messages"][-1]["content"]
        assert len(data["short"]) == len(data["long"]) == _PAIRS - 1
        assert {pair["prompt"] for pair in data["short"]} & {"Question 3?",
                                                               "Again?"} == set()
        acks = _acks(out)
        assert [ack.get("rejected_pairs") for ack in acks] == [1, 1]
        assert [ack.get("status") for ack in acks] == [None, None]

    def test_a_unit_whose_every_pair_is_rejected_is_acked_rejected(self, tmp_path):
        gen, check = _generator(), _checker(["Question", "Again?"])
        with pytest.raises(ValueError, match="No benign pairs.*every pair rejected"):
            _run(tmp_path, gen, check)
        out = tmp_path / "benign.csv"
        assert not out.exists()
        assert [(ack["status"], ack["rejected_pairs"]) for ack in _acks(out)] == [
            ("rejected", _PAIRS), ("rejected", _PAIRS)]
        # Acked, so a resume asks for nothing.
        before = len(gen.requests)
        with pytest.raises(ValueError, match="No benign pairs"):
            _run(tmp_path, gen, check)
        assert len(gen.requests) == before

    def test_a_reply_that_is_not_one_pair_is_asked_again_without_a_check(
            self, tmp_path):
        from redact.content_moderation.feedback import NOT_ONE_PAIR

        def reply(messages, _temperature):
            """Two pairs when first asked for one, one pair once told."""
            if not _asks_one(messages):
                return _reply()
            return _again(1 if NOT_ONE_PAIR in messages[-1]["content"] else 2)

        gen, check = _Fake(reply, "mock-gen"), _checker(["Question 3?"])
        _out, data = _run(tmp_path, gen, check)
        again = [r for r in gen.requests if _asks_one(r["messages"])]
        assert len(again) == 2 * 2
        assert sum(NOT_ONE_PAIR in r["messages"][-1]["content"]
                   for r in again) == _UNITS
        # The two-pair reply got no check call; the one-pair reply did.
        assert sum("Again?" in str(r["messages"]) for r in check.requests) == _UNITS
        assert "Again?" in [pair["prompt"] for pair in data["short"]]


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
