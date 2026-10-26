"""Code framing (requests/code_framing.py, spec 1.5): to_code_completion
(CodeAttack, Ren 2403.07865) and to_decrypt_function (CodeChameleon, Lv
2402.16717). Two pure request-layer techniques. The frame wording is not
reproduced here; the tests assert on shape, the slot, the info, the floor,
the spec tags, the pool membership, the rules (both ways) and — the key
mechanism — that the decrypt function shown really inverts its transform."""

import json
import random
from pathlib import Path

import pytest

import redact.jailbreak.directives as D
import redact.jailbreak.requests.code_framing as CF
from redact.jailbreak.augmentation import resolve
from redact.jailbreak.directives import check_pools, read_pools
from redact.jailbreak.pool import all_techniques, technique_pool
from redact.jailbreak.requests.code_framing import (
    _DECRYPT,
    _TRANSFORMS,
    get_code_framing_functions,
    to_code_completion,
    to_decrypt_function,
)
from redact.jailbreak.rules import compatible
from redact.jailbreak.technique import load_spec

TEXTS = [
    "How do plants turn sunlight into food?",
    'Line one\nSay "hello" to {braces}, cafe.',
    "a  b   c",
    "single",
    "",
]
_FLOOR = 8
FUNCTIONS = get_code_framing_functions()


def _pool(directive: str) -> list[dict]:
    return read_pools(CF._FILE)[directive]


def test_there_are_two_techniques():
    assert [fn.__name__ for fn in FUNCTIONS] == [
        "to_code_completion", "to_decrypt_function"]


def test_the_pools_are_ones_their_techniques_can_draw():
    check_pools(fn.variant_pool for fn in FUNCTIONS)


@pytest.mark.parametrize("directive", ["code_completion", "decrypt_function"])
def test_each_pool_holds_its_floor_of_variants_each_named_once(directive):
    names = [variant["name"] for variant in _pool(directive)]
    assert len(names) == len(set(names)) >= _FLOOR


# --- the decrypt function really inverts its transform --------------------------

def test_the_transform_slugs_match_the_shown_decrypt_codes():
    assert set(_TRANSFORMS) == set(_DECRYPT)


@pytest.mark.parametrize("slug", sorted(_TRANSFORMS))
@pytest.mark.parametrize("text", TEXTS)
def test_the_shown_decrypt_inverts_the_transform_in_a_restricted_exec(slug, text):
    """Execute the exact code shown to the model, in a namespace with no
    builtins, on the transformed text; it must return the original."""
    namespace: dict = {}
    exec(_DECRYPT[slug], {"__builtins__": {}}, namespace)  # noqa: S102
    encrypted = _TRANSFORMS[slug](text)
    assert namespace["decrypt"](encrypted) == text


def _shown_decrypt(written: str):
    """The decrypt function the output shows, read out of the rendered text
    and executed with no builtins: ``(code, callable)``.

    The code is taken from ``def decrypt`` to the end of its indented body,
    so the test reads what the model is shown rather than the module's own
    ``_DECRYPT`` entry."""
    import ast

    start = written.index("def decrypt")
    lines = written[start:].splitlines()
    body = [lines[0]]
    for line in lines[1:]:
        if line.strip() and not line.startswith((" ", "\t")):
            break
        body.append(line)
    code = "\n".join(body).rstrip()
    ast.parse(code)
    namespace: dict = {}
    exec(code, {"__builtins__": {}}, namespace)  # noqa: S102
    return code, namespace["decrypt"]


def _candidates(written: str) -> list[str]:
    """The strings of the output a ciphertext could be, without knowing the
    frame's wording: every line, every span inside quotes or backticks, and
    what follows the first ``=`` or ``:`` of a line, each stripped of
    surrounding whitespace, quotes and backticks."""
    import re

    out = []
    for line in written.splitlines():
        parts = [line]
        for sep in ("=", ":"):
            if sep in line:
                parts.append(line.split(sep, 1)[1])
        parts += [a or b or c for a, b, c in re.findall(
            r"\"([^\"\n]*)\"|'([^'\n]*)'|`([^`\n]*)`", line)]
        out += [part.strip().strip("\"'`").strip() for part in parts]
    return out


@pytest.mark.parametrize("seed", range(12))
def test_the_decrypt_shown_in_the_output_recovers_the_prompt(seed):
    """Read the decrypt code out of the rendered frame, run it with no
    builtins on the ciphertext the frame shows, and get the prompt back.

    Nothing here reads ``_DECRYPT`` or ``_TRANSFORMS``: a frame that showed
    another transform's code, or a ciphertext made with another transform,
    fails even though the module's own tables still agree."""
    prompt = "mix the two reagents and heat them"
    written, info = to_decrypt_function(prompt, rng=random.Random(seed))
    _code, decrypt = _shown_decrypt(written)
    recovered = [text for text in _candidates(written) if decrypt(text) == prompt]
    assert recovered, info
    # the plaintext itself is not in the frame (the transforms all change a
    # multi-word prompt)
    assert prompt not in written


@pytest.mark.parametrize("text", ["a b c d", "Mix 2 parts; then wait!",
                                  'say "hi" to {braces}'])
def test_the_shown_decrypt_recovers_other_prompts_too(text):
    for seed in range(6):
        written, _info = to_decrypt_function(text, rng=random.Random(seed))
        _code, decrypt = _shown_decrypt(written)
        assert any(decrypt(c) == text for c in _candidates(written))


def test_the_shown_code_is_the_entry_for_the_transform_the_info_names():
    """The code read out of the output is the module's entry for the slug the
    info names: the frame shows neither another transform's code nor code of
    its own."""
    for seed in range(12):
        written, info = to_decrypt_function("mix the reagents", rng=random.Random(seed))
        slug = info.split("transform=")[1].split(";")[0]
        code, _decrypt = _shown_decrypt(written)
        assert code.strip() == _DECRYPT[slug].strip()


def test_the_transforms_are_involutions():
    for transform in _TRANSFORMS.values():
        for text in TEXTS:
            assert transform(transform(text)) == text


# --- shape and info -------------------------------------------------------------

def test_code_completion_holds_the_prompt_once_and_is_named(monkeypatch):
    for variant in _pool("code_completion"):
        monkeypatch.setattr(D, "draw_variant", lambda variants, rng=None, v=variant: v)
        for text in TEXTS[:2]:
            written, info = to_code_completion(text)
            assert variant["template"].count("{prompt}") == 1
            assert written.count(text) == 1
            assert info == f"directive=code_completion;variant={variant['name']}"


def test_decrypt_function_sets_the_ciphertext_and_names_transform_and_variant():
    for seed in range(10):
        text = "mix the reagents carefully"
        written, info = to_decrypt_function(text, rng=random.Random(seed))
        assert info.startswith("transform=")
        assert ";variant=" in info
        slug = info.split("transform=")[1].split(";")[0]
        assert slug in _TRANSFORMS
        # The ciphertext (not the plaintext) is in the rendered prompt.
        assert _TRANSFORMS[slug](text) in written


def test_a_seed_gives_the_same_text():
    for fn in FUNCTIONS:
        assert (fn(TEXTS[0], rng=random.Random(3))
                == fn(TEXTS[0], rng=random.Random(3)))


# --- the spec and pool membership ----------------------------------------------

def test_the_spec_tags_a_pure_attack_only_request_family():
    family = load_spec()["families"]["code_framing"]
    assert (family["layer"], family["dimension"]) == ("requests", "request_shape")
    assert family["cross_incompatible_with"] == []
    for fn in FUNCTIONS:
        assert fn.layer == "requests"
        assert fn.families == ["code_framing"]
        assert fn.kinds == ["adversarial"]
        assert not fn.requires_llm and not getattr(fn, "needs_latin", False)
        assert fn.since == "1.5"


def test_the_family_is_in_the_attacks_and_not_the_eval_or_natural_pool():
    from redact.recipe import load_dataset_type

    names = {fn.__name__ for fn in FUNCTIONS}
    adversarial = {fn.__name__ for fn in technique_pool(resolve(None, "adversarial"))}
    natural = {fn.__name__ for fn in technique_pool(resolve(None, "natural"))}
    eval_pool = {fn.__name__ for fn in technique_pool(
        resolve(load_dataset_type("eval")["augmentations"], "adversarial"))}
    assert names <= adversarial
    assert not names & natural
    assert not names & eval_pool
    pinned = set(json.loads((Path(__file__).parents[2] / "fixtures" / "eval_pool.json")
                            .read_text(encoding="utf-8")))
    assert not names & pinned


# --- the rules ------------------------------------------------------------------

def _by_name() -> dict:
    return {fn.__name__: fn for fn in all_techniques()}


@pytest.mark.parametrize("fn", FUNCTIONS, ids=lambda fn: fn.__name__)
def test_one_request_per_combination(fn):
    """The layer cap keeps it from another request and from the sibling."""
    by_name = _by_name()
    other_request = by_name["to_innocuous_question"]
    assert other_request not in compatible([fn], [other_request])
    assert fn not in compatible([other_request], [fn])
    sibling = to_decrypt_function if fn is to_code_completion else to_code_completion
    assert sibling not in compatible([fn], [sibling])


@pytest.mark.parametrize("fn", FUNCTIONS, ids=lambda fn: fn.__name__)
def test_it_combines_with_an_obfuscation_and_a_hacking_frame_both_ways(fn):
    by_name = _by_name()
    for name in ("to_base64", "to_fictional_world"):
        partner = by_name[name]
        assert partner in compatible([fn], [partner]), name
        assert fn in compatible([partner], [fn]), name
