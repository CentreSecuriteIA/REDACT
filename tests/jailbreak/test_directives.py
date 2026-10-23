"""Variant pools: how a template file is read, checked and drawn from, and a
variant applied to a prompt.

The pools here are small synthetic ones written to ``tmp_path``.
"""

import json
import random
from collections import Counter
from pathlib import Path

import pytest

import redact.jailbreak  # noqa: F401  (tags the techniques)
from redact.jailbreak.directives import (
    PoolRef,
    apply_template_directive,
    check_pools,
    draw_variant,
    read_pools,
    uses_pool,
)
from redact.jailbreak.hacking.framing import get_framing_functions
from redact.jailbreak.obfuscation.structural import get_structural_functions
from redact.jailbreak.pool import all_techniques
from redact.jailbreak.requests import get_all_request_functions

_TEMPLATES = {"ask": [{"name": "plain", "template": "{prompt} (in {language})"}]}

_ALPHA = {"name": "alpha", "template": "alpha {prompt}"}
_BETA = {"name": "beta", "template": "beta {prompt}"}
_GAMMA = {"name": "gamma", "template": "gamma {prompt}"}

_FIXTURES = Path(__file__).parents[1] / "fixtures"
_EVAL_POOL = json.loads((_FIXTURES / "eval_pool.json").read_text(encoding="utf-8"))
_TEMPLATE_TECHNIQUES = {fn.__name__: fn for fn in (
    get_framing_functions() + get_structural_functions() + get_all_request_functions())}


def _folder(tmp_path: Path, data: dict) -> Path:
    """``tmp_path`` holding ``data`` as the template file ``t.json``."""
    (tmp_path / "t.json").write_text(json.dumps(data), encoding="utf-8")
    return tmp_path


def test_extra_fields_fill_the_template_beside_the_prompt():
    assert apply_template_directive(
        _TEMPLATES, "ask", "hi", info_key="language", extra={"language": "Welsh"},
    ) == ("hi (in Welsh)", "language=ask;variant=plain")


def test_the_prompt_is_not_an_extra_field():
    text, _ = apply_template_directive(
        _TEMPLATES, "ask", "hi", extra={"prompt": "no", "language": "Welsh"})
    assert text == "hi (in Welsh)"


class TestReadPools:
    def test_a_directive_file_reads_as_its_pools_less_comments(self, tmp_path):
        folder = _folder(tmp_path, {"_comment": "x", "d": [_ALPHA, _BETA]})
        assert read_pools("t.json", folder) == {"d": [_ALPHA, _BETA]}

    def test_a_single_template_entry_is_a_pool_of_one_named_after_it(self, tmp_path):
        folder = _folder(tmp_path, {"to_x": {"template": "x {prompt}", "ids": [1, 2]}})
        assert read_pools("t.json", folder) == {
            "to_x": [{"name": "to_x", "template": "x {prompt}"}]}


class TestCheckPools:
    def test_the_packaged_pools_pass(self):
        check_pools(fn.variant_pool for fn in all_techniques()
                    if hasattr(fn, "variant_pool"))

    def test_a_declared_slot_may_be_named(self, tmp_path):
        folder = _folder(tmp_path, {"d": [
            {"name": "alpha", "template": "alpha {prompt} in {language}"}]})
        check_pools([PoolRef("t.json", "d", ("prompt", "language"))], folder)

    def test_a_prefix_and_suffix_are_not_formatted_so_they_hold_no_slot(self, tmp_path):
        folder = _folder(tmp_path, {"d": [
            {"name": "alpha", "prefix": "alpha {", "suffix": "} beta"}]})
        check_pools([PoolRef("t.json", "d", ())], folder)

    @pytest.mark.parametrize(("variants", "message"), [
        ([], "not a non-empty list of variants"),
        ([{"name": "", "template": "alpha {prompt}"}], "a variant has no name"),
        ([_ALPHA, {"name": "alpha", "template": "beta {prompt}"}],
         "variant 'alpha': the name is taken"),
        ([{"name": "alpha", "template": "alpha"}],
         "variant 'alpha': the template has no {prompt} slot"),
        ([{"name": "alpha", "template": "alpha {prompt} in {language}"}],
         "variant 'alpha': the template names ['language']"),
        ([{"name": "alpha", "template": "alpha {prompt"}],
         "variant 'alpha': the template does not parse"),
        ([{**_ALPHA, "weight": -1}], "variant 'alpha': weight must be a number"),
        ([{**_ALPHA, "weight": "2"}], "variant 'alpha': weight must be a number"),
        ([{**_ALPHA, "weight": True}], "variant 'alpha': weight must be a number"),
        ([{**_ALPHA, "weight": 0}, {**_BETA, "weight": 0}],
         "every variant has weight 0"),
        ([{"name": "alpha", "prefix": "alpha "}],
         "variant 'alpha': a variant has a template, or a prefix and a suffix"),
        ([{**_ALPHA, "wieght": 2}], "variant 'alpha': unknown keys ['wieght']"),
        ([{"name": "alpha", "template": "alpha {prompt} and {prompt}"}],
         "variant 'alpha': the template has the {prompt} slot 2 times"),
        ([{"name": "alpha", "template": "alpha {prompt!r}"}],
         "variant 'alpha': the slot {prompt} has a conversion or a format spec (!r)"),
        ([{"name": "alpha", "template": "alpha {prompt:>9}"}],
         "variant 'alpha': the slot {prompt} has a conversion or a format spec (:>9)"),
        ([{"name": "alpha", "template": "alpha {prompt:{width}}"}],
         "variant 'alpha': the slot {prompt} has a conversion or a format spec"),
        ([{"name": "alpha", "template": "alpha {prompt!s:^5}"}],
         "has a conversion or a format spec (!s:^5)"),
        ([{"name": "a;variant=b", "template": "alpha {prompt}"}],
         "variant 'a;variant=b': the name holds a character other than"),
        ([{"name": "has space", "template": "alpha {prompt}"}],
         "variant 'has space': the name holds a character other than"),
        ([{"name": "caf\u00e9", "template": "alpha {prompt}"}],
         "the name holds a character other than"),
    ])
    def test_a_bad_pool_raises_naming_the_file_directive_and_variant(
            self, tmp_path, variants, message):
        folder = _folder(tmp_path, {"d": variants})
        with pytest.raises(ValueError) as raised:
            check_pools([PoolRef("t.json", "d")], folder)
        assert "configs/jailbreak/t.json, directive 'd'" in str(raised.value)
        assert message in str(raised.value)

    @pytest.mark.parametrize("name", ["alpha", "Alpha-2", "to_x", "hint_1", "2"])
    def test_a_name_of_letters_digits_underscores_and_hyphens_passes(
            self, tmp_path, name):
        folder = _folder(tmp_path, {"d": [{"name": name, "template": "{prompt}"}]})
        check_pools([PoolRef("t.json", "d")], folder)

    @pytest.mark.parametrize("template", [
        "{prompt}", "{{prompt}} and {prompt}", "a {prompt}\n\nb", "{prompt} {{}}"])
    def test_a_slot_written_alone_once_passes(self, tmp_path, template):
        folder = _folder(tmp_path, {"d": [{"name": "alpha", "template": template}]})
        check_pools([PoolRef("t.json", "d")], folder)

    def test_a_declared_slot_with_a_format_spec_raises(self, tmp_path):
        folder = _folder(tmp_path, {"d": [
            {"name": "alpha", "template": "{prompt} {language:>3}"}]})
        with pytest.raises(ValueError, match=r"the slot \{language\} has a conversion"):
            check_pools([PoolRef("t.json", "d", ("prompt", "language"))], folder)

    def test_a_slot_other_than_the_request_may_be_named_twice(self, tmp_path):
        folder = _folder(tmp_path, {"d": [
            {"name": "alpha", "template": "{prompt} {language} {language}"}]})
        check_pools([PoolRef("t.json", "d", ("prompt", "language"))], folder)

    def test_a_directive_the_file_lacks_raises(self, tmp_path):
        folder = _folder(tmp_path, {"d": [_ALPHA]})
        with pytest.raises(ValueError, match="directive 'e': the file has no such"):
            check_pools([PoolRef("t.json", "e")], folder)

    def test_a_pool_without_a_request_slot_needs_no_prompt(self, tmp_path):
        """Its variants are set into another template, which holds the request."""
        folder = _folder(tmp_path, {"d": [
            {"name": "alpha", "template": "alpha {persona}"},
            {"name": "beta", "template": "beta"}]})
        check_pools([PoolRef("t.json", "d", ("persona",), request_slot=False)], folder)

    def test_a_pool_without_a_request_slot_refuses_the_prompt(self, tmp_path):
        folder = _folder(tmp_path, {"d": [
            {"name": "alpha", "template": "alpha {persona} {prompt}"}]})
        with pytest.raises(ValueError, match=r"variant 'alpha': the template names "
                           r"\['prompt'\], which its technique does not fill"):
            check_pools([PoolRef("t.json", "d", ("persona",), request_slot=False)],
                        folder)

    def test_a_pool_has_a_request_slot_unless_it_says_otherwise(self, tmp_path):
        folder = _folder(tmp_path, {"d": [
            {"name": "alpha", "template": "alpha {persona}"}]})
        with pytest.raises(ValueError, match="the template has no {prompt} slot"):
            check_pools([PoolRef("t.json", "d", ("prompt", "persona"))], folder)

    def test_uses_pool_marks_a_pool_without_a_request_slot(self):
        @uses_pool("t.json", "d", ("persona",), request_slot=False)
        def to_x(prompt, *, rng=None):
            return prompt, ""

        assert to_x.variant_pool == PoolRef(
            "t.json", "d", ("persona",), "variant", request_slot=False)
        assert uses_pool("t.json", "d")(lambda p: p).variant_pool.request_slot is True


class TestDrawVariant:
    def test_an_unweighted_pool_draws_what_rng_choice_draws(self):
        pool = [_ALPHA, _BETA, _GAMMA]
        for seed in range(50):
            ours, theirs = random.Random(seed), random.Random(seed)
            assert draw_variant(pool, ours) is theirs.choice(pool)
            assert ours.getstate() == theirs.getstate()

    def test_a_pool_of_one_makes_no_draw(self):
        rng = random.Random(1)
        state = rng.getstate()
        assert draw_variant([_ALPHA], rng) is _ALPHA
        assert draw_variant([_ALPHA, {**_BETA, "weight": 0}], rng) is _ALPHA
        assert rng.getstate() == state

    def test_a_variant_of_weight_0_is_never_drawn(self):
        pool = [_ALPHA, {**_BETA, "weight": 0}, _GAMMA]
        drawn = Counter(draw_variant(pool, random.Random(seed))["name"]
                        for seed in range(500))
        assert set(drawn) == {"alpha", "gamma"}

    def test_a_weighted_pool_draws_with_rng_choices(self):
        pool = [{**_ALPHA, "weight": 3}, _BETA]
        for seed in range(50):
            expected = random.Random(seed).choices(pool, weights=[3, 1])[0]
            assert draw_variant(pool, random.Random(seed)) is expected

    def test_a_weighted_pool_draws_each_variant_by_its_weight(self):
        pool = [{**_ALPHA, "weight": 3}, _BETA]
        rng = random.Random(0)
        drawn = Counter(draw_variant(pool, rng)["name"] for _ in range(4000))
        assert 0.72 < drawn["alpha"] / 4000 < 0.78


def test_every_template_technique_names_its_pool_but_to_json():
    """``to_json`` writes its wrapper in code until the owner's wrapper step."""
    unmarked = [name for name, fn in _TEMPLATE_TECHNIQUES.items()
                if not hasattr(fn, "variant_pool")]
    assert unmarked == ["to_json"]


@pytest.mark.parametrize("name", sorted(
    name for name, fn in _TEMPLATE_TECHNIQUES.items()
    if hasattr(fn, "variant_pool") and fn.variant_pool.slots in (("prompt",), ())))
def test_a_technique_writes_a_variant_of_the_pool_it_names(name):
    """Its ``uses_pool`` names the directive its body applies: what it
    writes is one of that pool's variants. Compared as a flag, so a failure
    shows no template text."""
    fn = _TEMPLATE_TECHNIQUES[name]
    ref = fn.variant_pool
    probe = "a neutral probe"
    texts = {variant["template"].format_map({"prompt": probe}) if "template" in variant
             else variant["prefix"] + probe + variant["suffix"]
             for variant in read_pools(ref.file)[ref.directive]}
    for seed in range(8):
        kwargs = {"rng": random.Random(seed)} if name != "to_markdown" else {}
        found = fn(probe, **kwargs)[0] in texts
        assert found, f"{name}: its text is not a variant of {ref.name}"


def test_no_eval_pool_technique_has_a_weighted_variant():
    """A weight of 0 takes a variant out of the eval text, and any weight
    draws the whole pool by ``rng.choices``, so every seed may draw another
    variant: the eval pool's variants carry none until the owner's eval
    review."""
    weighted = sorted(
        name for name in _EVAL_POOL
        if (ref := getattr(_TEMPLATE_TECHNIQUES.get(name), "variant_pool", None))
        and any("weight" in variant
                for variant in read_pools(ref.file)[ref.directive]))
    assert weighted == []
