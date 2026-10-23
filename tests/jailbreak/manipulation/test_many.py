"""The many-shot variants of fsh and dap: MANY_PAIRS short pairs from a pool
across the cache's subcategories."""

import hashlib
import inspect
import random
import re

import pandas as pd
import pytest

from redact.jailbreak.manipulation.benign import load_benign_data
from redact.jailbreak.manipulation.dap import (
    to_dap_random_long,
    to_dap_random_many,
    to_dap_random_short,
    to_dap_selected_long,
    to_dap_selected_many,
    to_dap_selected_short,
)
from redact.jailbreak.manipulation.fsh import (
    to_fsh_random_long,
    to_fsh_random_many,
    to_fsh_random_short,
    to_fsh_selected_long,
    to_fsh_selected_many,
    to_fsh_selected_short,
)
from redact.jailbreak.manipulation.pairs import MANY_PAIRS, _many_pairs
from redact.llm_pipeline import drive_sync

# Five subcategories of eight pairs in two main categories: 40 short pairs.
_LAYOUT = {"Main A": {"Sub 1": 8, "Sub 2": 8, "Sub 3": 8},
           "Main B": {"Sub 4": 8, "Sub 5": 8}}
_SMALL = {"Main A": {"Sub 1": 8, "Sub 2": 8}}

_MANY = [to_fsh_random_many, to_fsh_selected_many,
         to_dap_random_many, to_dap_selected_many]


def _cache(tmp_path, layout: dict) -> dict:
    """A benign cache of numbered placeholder pairs, short and long, as
    ``load_benign_data`` reads it."""
    rows, n = [], 0
    for answer_type in ("short", "long"):
        for main, subs in layout.items():
            for sub, count in subs.items():
                for _ in range(count):
                    n += 1
                    rows.append({"main_category": main, "sub_category": sub,
                                 "prompt": f"question {n}", "answer": f"answer {n}",
                                 "answer_type": answer_type})
    path = tmp_path / "benign_samples.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return load_benign_data(path)


def _call(fn, data: dict, rng: random.Random, reply: str = "Sub 1"):
    """``fn`` on the request, a model picking ``reply`` when it asks one."""
    if inspect.isgeneratorfunction(fn):
        gen = fn("the request", benign_data=data, gen_model="m", rng=rng)
        return drive_sync(gen, lambda request: reply)
    return fn("the request", data, rng=rng)


def _questions(text: str) -> list[str]:
    """The questions of a prompt's ``Qn:`` lines, in order."""
    return re.findall(r"^Q\d+: (.*)$", text, re.M)


@pytest.fixture
def data(tmp_path):
    return _cache(tmp_path, _LAYOUT)


def test_many_pairs_are_many_pairs_each_once(data):
    pairs = _many_pairs(data, random.Random(0))
    assert len(pairs) == MANY_PAIRS
    assert len({pair["prompt"] for pair in pairs}) == MANY_PAIRS


def test_the_chosen_subcategory_comes_first_then_its_main_category_then_the_rest(data):
    pairs = _many_pairs(data, random.Random(0), "Sub 1")
    subcats = [pair["sub_category"] for pair in pairs]
    assert subcats[:8] == ["Sub 1"] * 8
    assert set(subcats[8:24]) == {"Sub 2", "Sub 3"}
    assert set(subcats[24:]) == {"Sub 4", "Sub 5"}


def test_a_cache_of_fewer_pairs_gives_them_all(tmp_path):
    data = _cache(tmp_path, _SMALL)
    pairs = _many_pairs(data, random.Random(0), "Sub 2")
    assert sorted(pair["prompt"] for pair in pairs) == sorted(
        pair["prompt"] for pair in data["short"])


def test_a_question_the_cache_holds_twice_is_shown_once(data):
    data["short"].append(dict(data["short"][0]))
    pairs = _many_pairs(data, random.Random(0))
    assert len({pair["prompt"] for pair in pairs}) == len(pairs) == MANY_PAIRS


@pytest.mark.parametrize("fn", _MANY, ids=lambda fn: fn.__name__)
def test_a_many_shot_variant_shows_many_distinct_pairs_and_the_request_once(fn, data):
    text, info = _call(fn, data, random.Random(1))
    questions = _questions(text)
    assert len(questions) == MANY_PAIRS + 1
    assert len(set(questions)) == MANY_PAIRS + 1
    assert text.count("the request") == 1
    assert info.startswith(f"num_shots={MANY_PAIRS};")


@pytest.mark.parametrize("fn", _MANY, ids=lambda fn: fn.__name__)
def test_a_seed_writes_the_same_many_shot_prompt(fn, data):
    assert _call(fn, data, random.Random(3)) == _call(fn, data, random.Random(3))


@pytest.mark.parametrize("fn", [to_fsh_random_many, to_fsh_selected_many])
def test_fsh_many_asks_the_request_after_every_pair(fn, data):
    text, _ = _call(fn, data, random.Random(2))
    assert _questions(text)[-1] == "the request"
    assert text.endswith(f"A{MANY_PAIRS + 1}:")


@pytest.mark.parametrize("fn", [to_dap_random_many, to_dap_selected_many])
@pytest.mark.parametrize("seed", range(5))
def test_dap_many_asks_the_request_among_the_pairs(fn, seed, data):
    text, info = _call(fn, data, random.Random(seed))
    position = _questions(text).index("the request") + 1
    assert 1 < position <= MANY_PAIRS  # never first, never last
    assert f"harmful_idx={position};" in info


def test_random_many_draws_from_every_subcategory(data):
    _, info = to_fsh_random_many("the request", data, rng=random.Random(0))
    assert info == f"num_shots={MANY_PAIRS};sub_category=all"


def test_selected_many_shows_the_chosen_main_category_whole(data):
    text, info = _call(to_fsh_selected_many, data, random.Random(0), reply="Sub 1")
    main_a = {pair["prompt"] for pair in data["short"]
              if pair["main_category"] == "Main A"}
    assert main_a <= set(_questions(text))
    assert info == f"num_shots={MANY_PAIRS};sub_category=Sub 1;selection=selected"


def test_selected_many_falls_back_to_a_random_subcategory(data):
    _, info = _call(to_dap_selected_many, data, random.Random(0), reply="no such one")
    assert info.endswith(";selection=fallback_random")


@pytest.mark.parametrize("fn", _MANY, ids=lambda fn: fn.__name__)
def test_the_info_says_when_the_cache_held_fewer_pairs(fn, tmp_path):
    data = _cache(tmp_path, _SMALL)
    cached = len(data["short"])     # 16, fewer than MANY_PAIRS
    text, info = _call(fn, data, random.Random(0))
    assert len(_questions(text)) == cached + 1
    assert info.startswith(f"num_shots={cached};wanted={MANY_PAIRS};")


# What the eight short and long variants write on the placeholder cache, for
# six seeds and a model reply that names a subcategory and one that does not,
# pinned before the many-shot variants were added beside them.
_PINNED_EIGHT = "63d036a5ecd97832ac92c7b10725b270231c5c3b2ef0abf5bb87163b165cc08a"


def test_the_short_and_long_variants_write_and_draw_as_before(data):
    sha = hashlib.sha256()
    for fn in (to_fsh_random_short, to_fsh_random_long, to_fsh_selected_short,
               to_fsh_selected_long, to_dap_random_short, to_dap_random_long,
               to_dap_selected_short, to_dap_selected_long):
        for seed in range(6):
            rng = random.Random(seed)
            for reply in ("Sub 2", "no such subcategory"):
                text, info = _call(fn, data, rng, reply)
                sha.update(f"{text}\x1f{info}\x1e".encode())
            sha.update(repr(rng.getstate()).encode())
    assert sha.hexdigest() == _PINNED_EIGHT
