"""``scripts/literature_smoke.py``: the reason codes ``_reason`` reports from a
technique's ``info`` (synthetic DISCARDED strings, no model or prompt text).

Every code-side gate reason of the spec 1.5 literature families maps to a fixed
label, so the smoke run's counts never carry a model's words; anything else in
``feedback=`` is a check verdict and collapses to ``check:<word>``.
"""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location(
        "literature_smoke",
        Path(__file__).parents[1] / "scripts" / "literature_smoke.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_accepted_info_is_accepted(script):
    assert script._reason("form=math problem") == "accepted"
    assert script._reason("pivot=french") == "accepted"
    assert script._reason("variety=Singlish") == "accepted"
    assert script._reason("") == "accepted"


@pytest.mark.parametrize(("info", "expected"), [
    # semantic_reencoding: each form's code gate.
    ("DISCARDED; reason=reencode failed; form=math problem; "
     "feedback=no math notation", "no_math_notation"),
    ("DISCARDED; reason=reencode failed; form=poem; feedback=too few lines=2",
     "too_few_lines"),
    ("DISCARDED; reason=reencode failed; form=poem; feedback=not verse",
     "not_verse"),
    ("DISCARDED; reason=reencode failed; form=acrostic paragraph; "
     "feedback=acrostic mismatch", "acrostic_mismatch"),
    ("DISCARDED; reason=reencode failed; form=acrostic paragraph; "
     "feedback=acrostic count=3 want=4", "acrostic_count"),
    ("DISCARDED; reason=reencode failed; form=information overload; "
     "feedback=length_ratio=1.10", "length_ratio"),
    ("DISCARDED; reason=reencode failed; form=math problem; feedback=unchanged",
     "unchanged"),
    ("DISCARDED; reason=reencode failed; form=poem; feedback=empty", "empty"),
    # translationese: the two legs and the judge.
    ("DISCARDED; language=translationese; pivot=french; piece=1/1; "
     "feedback=forward:length_ratio=2.50", "forward_leg"),
    ("DISCARDED; language=translationese; pivot=german; piece=2/3; "
     "feedback=back:language=deu_Latn", "back_leg"),
    ("DISCARDED; language=translationese; pivot=french; piece=all; "
     "feedback=judge=No the meaning drifted", "check_judge"),
    ("DISCARDED; language=translationese; pivot=spanish; "
     "feedback=to pivot: language=Spanish; feedback=No", "leg_to_pivot"),
    # sociolect: its gate, then a verdict.
    ("DISCARDED; reason=sociolect failed; variety=Singlish; feedback=unchanged",
     "unchanged"),
    ("DISCARDED; reason=sociolect failed; variety=Indian English; "
     "feedback=length_ratio=0.30", "length_ratio"),
])
def test_a_code_side_gate_reason_maps_to_its_fixed_label(script, info, expected):
    assert script._reason(info) == expected


@pytest.mark.parametrize(("verdict", "expected"), [
    ("No, it drops a detail.", "check:no"),
    ("no - the request is gone", "check:no"),
    ("Rejected: the meaning changed", "check:rejected"),
    # A verdict that starts with no recognised word is reported without its text.
    ("Mostly fine, although the second clause", "check:other"),
])
def test_a_check_verdict_is_reported_without_its_text(script, verdict, expected):
    info = f"DISCARDED; reason=reencode failed; form=poem; feedback={verdict}"
    assert script._reason(info) == expected


def test_a_chain_level_rejection_falls_back_to_its_fixed_reason(script):
    assert script._reason(
        "DISCARDED; technique=to_mathprompt; reason=empty text") == "empty_text"


def test_no_feedback_text_reaches_the_counts(script):
    """Every label the script can print is a fixed string of its own table or
    a one-word verdict, so a model's wording is never counted or logged."""
    labels = {label for _, label in script._GATE_LABELS}
    assert "check:other" not in labels
    for prefix, label in script._GATE_LABELS:
        assert " " not in label and "=" not in label, (prefix, label)


def test_the_three_families_are_offered(script):
    assert sorted(script._FAMILIES) == ["semantic_reencoding", "sociolect",
                                        "translationese"]
    for build in script._FAMILIES.values():
        assert build(), "each family registers at least one technique"


def test_the_sentences_are_benign_and_plain(script):
    assert len(script.SENTENCES) >= 20     # noqa: PLR2004
    assert all(s.endswith(("?", ".")) for s in script.SENTENCES)
