"""The payload-split techniques: the prompt cut into two to four lettered
parts, which a variant of the pool asks the reader to join.

The packaged variants are the owner's text; these tests fill the pool with
stand-ins of their own and assert on the parts, the letters, the counts and
the verdicts, never on the packaged wording.
"""

import json
import random
import re
import string

import pytest

import redact.jailbreak.obfuscation.payload_split as PS
from redact.jailbreak.directives import check_pools, read_pools
from redact.jailbreak.obfuscation import get_all_obfuscation_functions
from redact.jailbreak.technique import DISCARDED
from redact.llm_pipeline import drive_sync

PROMPT = "the cat sat on the mat today"
#: A prompt with runs of blanks and a newline, which the parts do not keep.
LOOSE = "the cat   sat on\nthe  mat today"
PROMPTS = [PROMPT, LOOSE, "one two", "a b c", "one two three four five six seven eight"]


class _Count(random.Random):
    """A seeded rng whose count draw (``randint``) is the one a test names."""

    def __init__(self, count: int):
        super().__init__(0)
        self.count = count

    def randint(self, a, b):
        return self.count


@pytest.fixture
def stand_ins(monkeypatch):
    """Two variants of the tests' own: the letters, then the parts."""
    monkeypatch.setattr(PS, "_TEMPLATES", {"reassembly": [
        {"name": "one", "template": "<one {names}>\n{parts}"},
        {"name": "two", "template": "<two {names}>\n{parts}"}]})


def _parts_of(text: str) -> list[str]:
    """The quoted parts of a stand-in rendering, in order, after checking
    each is on a line of its own, named A, B, C ..."""
    lines = text.split("\n")[1:]
    assert [line[:3] for line in lines] == [
        f"{name}: " for name in string.ascii_uppercase[:len(lines)]]
    assert all(line.startswith('"', 3) and line.endswith('"') for line in lines)
    return [line[4:-1] for line in lines]


def _drive(gen, replies: list[str]):
    """Answer ``gen``'s requests with ``replies`` in turn; return
    ``(requests, result)``."""
    requests, answers = [], iter(replies)

    def answer(request):
        requests.append(request)
        return next(answers)

    return requests, drive_sync(gen, answer)


def _numbered(parts: list[str]) -> str:
    return "\n".join(f"{i}. {part}" for i, part in enumerate(parts, start=1))


def _into(count: int, text: str = PROMPT) -> list[str]:
    """``text`` in ``count`` runs of words, the last one the longest: a cut
    of the test's own, not the technique's."""
    words = text.split()
    size = len(words) // count
    cuts = [i * size for i in range(count)] + [len(words)]
    return [" ".join(words[a:b]) for a, b in zip(cuts, cuts[1:])]


def _semantic(prompt: str, rng: random.Random, replies: list[str]):
    """``to_payload_split_semantic`` driven with ``replies``."""
    return _drive(PS.to_payload_split_semantic(prompt, gen_model="m", rng=rng), replies)


# --- what the spec says of them ------------------------------------------------

@pytest.mark.parametrize(("fn", "llm", "calls"), [
    (PS.to_payload_split, False, None),
    (PS.to_payload_split_semantic, True, [1, 3]),
], ids=lambda value: getattr(value, "__name__", None))
def test_the_spec_tags_an_attack_only_obfuscation_of_spec_1_3(fn, llm, calls):
    assert fn.families == ["payload_split"]
    assert (fn.layer, fn.within_layer_order, fn.complexity) == ("obfuscation", 65, 2)
    assert (fn.requires_llm, fn.requires_benign, fn.calls, fn.check) == (
        llm, False, calls, None)
    assert (fn.kinds, fn.since) == (["adversarial"], "1.3")


def test_the_obfuscation_layer_registers_both():
    names = [fn.__name__ for fn in get_all_obfuscation_functions()]
    assert {"to_payload_split", "to_payload_split_semantic"} <= set(names)


def test_the_eval_type_draws_neither():
    from redact.jailbreak.augmentation import resolve
    from redact.jailbreak.pool import technique_pool
    from redact.recipe import load_dataset_type

    section = load_dataset_type("eval")["augmentations"]
    names = {fn.__name__ for fn in technique_pool(resolve(section, "adversarial"))}
    assert not names & {"to_payload_split", "to_payload_split_semantic"}


# --- the pool they draw from ---------------------------------------------------

@pytest.mark.parametrize("fn", PS.get_payload_split_functions(),
                         ids=lambda fn: fn.__name__)
def test_both_draw_the_reassembly_pool_whose_templates_hold_no_request_slot(fn):
    ref = fn.variant_pool
    assert (ref.file, ref.directive, ref.slots, ref.request_slot, ref.info_key) == (
        "obfuscation/payload_split_templates.json", "reassembly",
        ("parts", "names"), False, "variant")
    check_pools([ref])


def test_the_packaged_pool_has_8_variants_that_each_name_both_slots():
    variants = read_pools(PS._FILE)["reassembly"]
    assert len(variants) == 8       # noqa: PLR2004
    for variant in variants:
        slots = {field for _, field, _, _ in string.Formatter().parse(
            variant["template"]) if field}
        assert slots == {"parts", "names"}, variant["name"]


def test_the_packaged_pool_gives_each_part_a_line_of_its_own_in_order():
    for seed in range(len(PS._TEMPLATES["reassembly"]) * 3):
        text, info = PS.to_payload_split(PROMPT, rng=random.Random(seed))
        count = int(re.fullmatch(r"variant=\w+;parts=(\d)", info)[1])
        lines = [line for line in text.split("\n")
                 if re.match(r'[A-D]: "', line)]
        assert [line[:1] for line in lines] == list(string.ascii_uppercase[:count])
        assert " ".join(line[4:-1] for line in lines) == PROMPT


# --- to_payload_split: the cut at word boundaries ------------------------------

@pytest.mark.parametrize("prompt", PROMPTS)
@pytest.mark.parametrize("seed", range(12))
def test_the_parts_joined_in_order_give_the_prompt_back_apart_from_whitespace(
        stand_ins, prompt, seed):
    text, _ = PS.to_payload_split(prompt, rng=random.Random(seed))
    assert " ".join(_parts_of(text)) == " ".join(prompt.split())


def test_the_part_count_is_two_to_four_and_every_count_is_drawn(stand_ins):
    counts = {len(_parts_of(PS.to_payload_split(PROMPT, rng=random.Random(seed))[0]))
              for seed in range(60)}
    assert counts == {2, 3, 4}


@pytest.mark.parametrize("seed", range(12))
def test_the_parts_have_nearly_equal_word_counts(stand_ins, seed):
    text, _ = PS.to_payload_split(
        "one two three four five six seven eight nine", rng=random.Random(seed))
    sizes = [len(part.split()) for part in _parts_of(text)]
    assert max(sizes) - min(sizes) <= 1


QUOTED = 'say "hello" to her, then \\n and back\\slash and "done"'


@pytest.mark.parametrize("count", [2, 3, 4])
def test_a_quote_or_a_backslash_in_the_prompt_leaves_each_part_one_balanced_string(
        stand_ins, count):
    """Each part reads as a JSON string whatever it holds, so the quotes round
    it are the part's own and the parts still give the prompt back."""
    text, _ = PS.to_payload_split(QUOTED, rng=_Count(count))
    lines = text.split("\n")[1:]
    parts = [json.loads(line[3:]) for line in lines]
    assert [line[:3] for line in lines] == [
        f"{name}: " for name in string.ascii_uppercase[:count]]
    assert " ".join(parts) == " ".join(QUOTED.split())


def test_the_semantic_cut_escapes_the_quotes_of_its_parts_too(stand_ins):
    text, _ = _semantic(QUOTED, _Count(2), [_numbered(_into(2, QUOTED))])[1]
    parts = [json.loads(line[3:]) for line in text.split("\n")[1:]]
    assert " ".join(parts) == " ".join(QUOTED.split())


def test_a_part_without_a_quote_or_a_backslash_is_written_as_it_was(stand_ins):
    text, _ = PS.to_payload_split("café au lait", rng=_Count(2))
    assert text.split("\n")[1:] == ['A: "café au"', 'B: "lait"']


def test_the_names_are_listed_in_the_order_of_the_parts(stand_ins):
    for count in (2, 3, 4):
        text, _ = PS.to_payload_split(PROMPT, rng=_Count(count))
        assert text.split("\n")[0] in {
            f"<{variant} {', '.join(string.ascii_uppercase[:count])}>"
            for variant in ("one", "two")}


def test_the_info_names_the_variant_and_the_count(stand_ins):
    seen = set()
    for seed in range(30):
        text, info = PS.to_payload_split(PROMPT, rng=random.Random(seed))
        variant, count = re.fullmatch(r"variant=(\w+);parts=(\d)", info).groups()
        assert text.startswith(f"<{variant} ")
        assert int(count) == len(_parts_of(text))
        seen.add(variant)
    assert seen == {"one", "two"}


@pytest.mark.parametrize("words", [2, 3])
def test_a_short_prompt_is_cut_into_no_more_parts_than_it_has_words(stand_ins, words):
    prompt = " ".join(PROMPT.split()[:words])
    for seed in range(20):
        parts = _parts_of(PS.to_payload_split(prompt, rng=random.Random(seed))[0])
        assert len(parts) <= words
        assert " ".join(parts) == prompt


@pytest.mark.parametrize("prompt", ["", "   ", "single"])
def test_a_prompt_of_fewer_than_two_words_is_discarded_unchanged(stand_ins, prompt):
    text, info = PS.to_payload_split(prompt, rng=random.Random(0))
    assert text == prompt
    assert info == PS.TOO_FEW_WORDS
    assert info.startswith(DISCARDED)


def test_a_replayed_seed_gives_the_same_text_and_other_seeds_give_others(stand_ins):
    first = PS.to_payload_split(PROMPT, rng=random.Random(5))
    assert PS.to_payload_split(PROMPT, rng=random.Random(5)) == first
    assert len({PS.to_payload_split(PROMPT, rng=random.Random(seed))
                for seed in range(20)}) > 1


# --- to_payload_split_semantic: the cut by a model -----------------------------

@pytest.mark.parametrize("count", [2, 3, 4])
def test_the_model_s_parts_are_named_in_order_and_kept_whole(stand_ins, count):
    requests, (text, info) = _semantic(
        PROMPT, _Count(count), [_numbered(_into(count))])
    assert _parts_of(text) == _into(count)
    assert re.fullmatch(rf"variant=\w+;parts={count}", info)
    assert len(requests) == 1


def test_the_request_goes_to_the_generation_model_with_the_prompt_in_it(stand_ins):
    requests, _ = _semantic(PROMPT, _Count(2), [_numbered(_into(2))])
    (request,) = requests
    assert request.model == "m"
    assert any(PROMPT in message["content"] for message in request.messages)


def test_a_reply_of_another_count_is_asked_again_until_it_has_the_drawn_count(
        stand_ins):
    replies = [_numbered(_into(n)) for n in (2, 3, 4)]
    for seed in range(20):
        requests, (text, info) = _semantic(PROMPT, random.Random(seed), replies)
        count = int(info.rpartition("parts=")[2])
        assert 2 <= count <= 4          # noqa: PLR2004
        assert len(requests) == count - 1
        assert _parts_of(text) == _into(count)


def test_parts_that_give_the_prompt_back_apart_from_whitespace_are_kept(stand_ins):
    """A cut inside a word and a doubled blank are the model's: only the
    letters and their order are compared."""
    parts = ["the cat  sat o", "n the mat today"]
    _, (text, info) = _semantic(PROMPT, _Count(2), [_numbered(parts)])
    assert _parts_of(text) == parts
    assert not info.startswith(DISCARDED)


@pytest.mark.parametrize("parts", [
    ["the cat sat", "on the rug today"],        # a word changed
    ["the cat", "on the mat today"],            # a word dropped
    ["the cat sat on", "the mat today too"],    # a word added
    ["on the mat today", "the cat sat"],        # out of order
], ids=["changed", "dropped", "added", "reordered"])
def test_parts_that_differ_from_the_prompt_discard_it(stand_ins, parts):
    requests, (text, info) = _semantic(PROMPT, _Count(2), [_numbered(parts)])
    assert text == PROMPT
    assert info == PS.PARTS_DIFFER
    assert info.startswith(DISCARDED)
    assert len(requests) == 1


@pytest.mark.parametrize("reply", [
    "the cat sat on the mat today",                      # no list
    _numbered(_into(3)),                                 # the wrong count
    "1. the cat sat\n3. on the mat today",               # out of sequence
], ids=["not a list", "wrong count", "out of sequence"])
def test_a_reply_that_is_never_the_asked_list_discards_the_prompt_after_3_replies(
        stand_ins, reply):
    requests, (text, info) = _semantic(PROMPT, _Count(2), [reply] * 3)
    assert text == PROMPT
    assert info == PS.MALFORMED_PARTS
    assert info.startswith(DISCARDED)
    assert len(requests) == 3       # noqa: PLR2004


def test_a_prompt_of_fewer_than_two_words_is_discarded_with_no_request(stand_ins):
    requests, (text, info) = _semantic("single", random.Random(0), [])
    assert (requests, text, info) == ([], "single", PS.TOO_FEW_WORDS)


def test_a_two_word_prompt_is_asked_for_two_parts_at_most(stand_ins):
    reply = _numbered(["one", "two"])
    for seed in range(20):
        requests, (text, info) = _semantic("one two", random.Random(seed), [reply])
        assert _parts_of(text) == ["one", "two"]
        assert info.endswith("parts=2")
        assert len(requests) == 1


def test_a_replayed_seed_gives_the_same_semantic_text(stand_ins):
    replies = [_numbered(_into(n)) for n in (2, 3, 4)]
    first = _semantic(PROMPT, random.Random(9), replies)[1]
    assert _semantic(PROMPT, random.Random(9), replies)[1] == first
