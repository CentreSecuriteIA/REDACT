"""The order a combination applies its techniques in.

A fresh plan sorts each draw with ``in_layer_order``: by layer, then, in the
obfuscation layer, by each family's ``within_layer_order``, the whole-text
rewrites before the word-level steps. A kept plan replays the order its
manifest recorded, whatever the spec says now.
"""

import pandas as pd

import redact.jailbreak  # noqa: F401  (tags the techniques)
from redact.jailbreak.chain import in_layer_order
from redact.jailbreak.plan import combination_of
from redact.jailbreak.pool import all_techniques, registry
from tests.conftest import MockBackend
from tests.jailbreak.test_pipeline_jailbreaks import GEN, _llm_run

BY_NAME = registry(all_techniques())

#: One technique of each obfuscation family, by family.
ONE_PER_FAMILY = {
    "translation": "to_thai", "typos": "to_rewrite_with_typos_low",
    "casual_writing": "to_lowercase", "sensitive_words": "to_sensitive_words_split",
    "tokenbreak": "to_tokenbreak_split", "suffixes": "to_adversarial_suffix_fragments",
    "encode": "to_base64", "structural": "to_json", "ascii_art": "to_ascii_art",
}


def _families_in_order(names) -> list[str]:
    ordered = in_layer_order(BY_NAME[name] for name in names)
    return [fn.families[0] for fn in ordered]


def test_the_whole_text_rewrites_apply_before_the_word_level_steps():
    assert _families_in_order(reversed(ONE_PER_FAMILY.values())) == [
        "translation", "typos", "casual_writing", "sensitive_words", "tokenbreak",
        "suffixes", "encode", "structural", "ascii_art"]


def test_the_families_eval_draws_keep_their_order():
    """Translation, then encode, structural, ascii_art, as before the
    renumbering, so an eval combination applies in the same order."""
    eval_families = ["ascii_art", "structural", "encode", "translation"]
    names = [ONE_PER_FAMILY[family] for family in eval_families]
    assert _families_in_order(names) == [
        "translation", "encode", "structural", "ascii_art"]


def test_a_word_encoding_takes_the_order_of_its_first_family():
    """The ``to_sensitive_words_encode_*`` are in ``sensitive_words`` and
    ``encode``; ``sensitive_words`` comes first in their spec entries."""
    word_encodings = [fn for name, fn in BY_NAME.items()
                      if name.startswith("to_sensitive_words_encode_")]
    assert len(word_encodings) == 10     # noqa: PLR2004
    assert {fn.within_layer_order for fn in word_encodings} == {
        BY_NAME["to_sensitive_words_split"].within_layer_order}


def test_a_kept_plan_names_its_techniques_in_the_order_it_recorded():
    """A plan made before the renumbering applied sensitive words first."""
    recorded = ["to_sensitive_words_variables", "to_uppercase"]
    combined = combination_of(recorded, BY_NAME)
    assert [fn.__name__ for fn in combined.techniques] == recorded
    assert [fn.__name__ for fn in in_layer_order(combined.techniques)] == [
        "to_uppercase", "to_sensitive_words_variables"]


def test_a_kept_plan_runs_its_techniques_in_the_order_it_recorded(
        tmp_path, monkeypatch):
    """Sensitive words, then upper case: the definitions clause the first
    step adds is upper-cased too, which the fresh order would not do."""
    chain = (BY_NAME["to_sensitive_words_variables"], BY_NAME["to_uppercase"])
    out = _llm_run(tmp_path, monkeypatch, [("a word here", chain)],
                   MockBackend("word", model=GEN), techniques=list(chain))
    (row,) = pd.read_csv(out).to_dict("records")
    assert row["technique"] == "to_sensitive_words_variables+to_uppercase"
    assert row["jailbreak"] == "A VAR_1 HERE\n\nWHERE VAR_1 = WORD"
