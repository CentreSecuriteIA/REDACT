"""The translation technique on a machine-translation model: fake replies, a
stubbed language identifier, no model and no detector loaded."""

import re

import pytest

from redact.jailbreak.obfuscation import langid
from redact.jailbreak.obfuscation import translation as T
from redact.jailbreak.obfuscation.translation import (
    LANGUAGE_CODES,
    LANGUAGES,
    MAX_PIECE_CHARS,
    join_pieces,
    reply_cap,
    split_text,
    to_swahili,
)
from redact.llms.model_config import MODEL_REGISTRY, ModelConfig, VLLMConfig

MT = "mt-fake"
GEN = "gen-fake"
_WIRE = re.compile(r"<<<source>>>(\w+)<<<target>>>(\w+)<<<text>>>(.*)", re.S)


@pytest.fixture(autouse=True)
def mt(monkeypatch):
    """A registered machine-translation model, and a language identifier that
    reads every text as ``label["value"]``."""
    monkeypatch.setitem(MODEL_REGISTRY, MT, ModelConfig(
        name=MT, roles=["translation", "machine_translation"],
        supports_system_prompt=False, vllm=VLLMConfig(hf_model_id="org/mt")))
    label = {"value": "swh_Latn"}
    monkeypatch.setattr(langid, "detect", lambda text: label["value"])
    return label


def _wire(request) -> tuple[str, str, str]:
    """``(source, target, text)`` of a translator request."""
    (message,) = request.messages
    assert message["role"] == "user"
    return _WIRE.fullmatch(message["content"]).groups()


class _Fake:
    """Answers the translator by prefixing (forward) or unprefixing (back) the
    text, and the judge with ``verdicts`` in turn, the last repeated."""

    def __init__(self, verdicts=("Yes",), forward=None):
        self.verdicts = list(verdicts)
        self.forward = forward or (lambda text, n: f"sw:{text}")
        self.forwards: dict[str, int] = {}

    def __call__(self, request) -> str:
        if request.model != MT:
            return self.verdicts.pop(0) if len(self.verdicts) > 1 else self.verdicts[0]
        source, _target, text = _wire(request)
        if source != "en":
            return text.removeprefix("sw:")
        n = self.forwards[text] = self.forwards.get(text, 0) + 1
        return self.forward(text, n)


def _drive(gen, answer, limit=100):
    """Drive ``gen``, answering each request; return ``(requests, result)``."""
    requests, reply = [], None
    try:
        while len(requests) < limit:
            request = gen.send(reply)
            requests.append(request)
            reply = answer(request)
    except StopIteration as stop:
        return requests, stop.value
    raise AssertionError("the technique made more requests than expected")


def _run(text, answer, **kwargs):
    kwargs = {"translate_model": MT, "gen_model": GEN, **kwargs}
    return _drive(to_swahili(text, **kwargs), answer)


class TestOneRound:
    def test_a_passing_round_translates_translates_back_and_judges(self):
        requests, result = _run("Good morning, friend.", _Fake())
        assert [r.model for r in requests] == [MT, MT, GEN]
        assert result == ("sw:Good morning, friend.", "Swahili")
        assert [_wire(r)[:2] for r in requests[:2]] == [("en", "sw"), ("sw", "en")]
        assert _wire(requests[1])[2] == "sw:Good morning, friend."
        assert [r.max_tokens for r in requests] == [
            reply_cap("Good morning, friend."), reply_cap("sw:Good morning, friend."),
            T.JUDGE_MAX_TOKENS]
        assert T.JUDGE_MAX_TOKENS == 128      # noqa: PLR2004

    def test_the_judge_reads_the_original_against_the_back_translation(self):
        requests, _ = _run("Good morning, friend.", _Fake())
        judge = "\n".join(m["content"] for m in requests[2].messages)
        assert "Good morning, friend." in judge and "Swahili" in judge

    def test_check_model_names_the_judge(self):
        requests, _ = _run("Good morning.", _Fake(), check_model="judge-fake")
        assert [r.model for r in requests] == [MT, MT, "judge-fake"]

    def test_without_a_generation_model_the_uncensored_gen_default_judges(self):
        from redact.llms.model_config import default_model_for_role

        requests, _ = _run("Good morning.", _Fake(), gen_model=None)
        assert requests[2].model == default_model_for_role("uncensored_gen")

    @pytest.mark.parametrize("language", LANGUAGES)
    def test_every_language_goes_out_under_its_code(self, language):
        fn = getattr(T, "to_" + language.lower().replace(" ", "_"))
        request = fn("Good morning.", translate_model=MT, gen_model=GEN).send(None)
        assert _wire(request)[:2] == ("en", LANGUAGE_CODES[language])


class TestTheChecks:
    def test_a_wrong_language_is_resampled_without_feedback(self, mt):
        mt["value"] = "ind_Latn"
        requests, (text, info) = _run("Good morning.", _Fake())
        assert [r.model for r in requests] == [MT] * 4
        assert len({_wire(r) for r in requests}) == 1      # the same request
        assert info == ("DISCARDED; language=Swahili; piece=1/1; "
                        "feedback=language=ind_Latn")

    def test_a_sibling_label_is_accepted(self, mt):
        mt["value"] = "swc_Latn"
        _, (_, info) = _run("Good morning.", _Fake())
        assert info == "Swahili"

    def test_a_reply_three_times_too_long_is_a_length_outlier(self):
        def long(text, n):
            return text * 3 if n == 1 else f"sw:{text}"

        requests, (text, info) = _run("Good morning, friend.", _Fake(forward=long))
        assert [r.model for r in requests] == [MT, MT, MT, GEN]
        assert info == "Swahili" and text == "sw:Good morning, friend."

    def test_an_empty_reply_is_rejected(self):
        _, (_, info) = _run("Good morning.", _Fake(forward=lambda text, n: " "))
        assert info.endswith("piece=1/1; feedback=empty")

    def test_without_the_detector_the_other_checks_still_run(self, monkeypatch):
        monkeypatch.setattr(langid, "detect", lambda text: None)
        requests, (_, info) = _run("Good morning.", _Fake())
        assert info == "Swahili" and len(requests) == 3      # noqa: PLR2004
        _, (_, info) = _run("Good morning.", _Fake(forward=lambda text, n: text * 4))
        assert "feedback=length_ratio=" in info

    def test_a_copied_text_is_left_to_the_chain_as_a_noop(self):
        _, (text, info) = _run("OK", _Fake(forward=lambda text, n: text))
        assert (text, info) == ("OK", "Swahili")

    def test_four_judge_rejections_discard_after_twelve_requests(self):
        requests, (_, info) = _run("Good morning.", _Fake(verdicts=["No: refused."]))
        assert [r.model for r in requests] == [MT, MT, GEN] * 4
        assert info == ("DISCARDED; language=Swahili; piece=all; "
                        "feedback=judge=No: refused.")

    def test_a_judge_rejection_resamples_then_accepts(self):
        requests, (_, info) = _run("Good morning.", _Fake(verdicts=["No", "Yes"]))
        assert [r.model for r in requests] == [MT, MT, GEN] * 2
        assert info == "Swahili"


THREE = "First paragraph here.\n\nSecond one, a little longer.\n \nThird."


class TestParagraphs:
    def test_a_text_is_translated_one_piece_at_a_time_in_order(self):
        requests, (text, info) = _run(THREE, _Fake())
        forward = [_wire(r)[2] for r in requests if _wire_or_none(r, "en")]
        assert forward == ["First paragraph here.", "Second one, a little longer.",
                           "Third."]
        assert [r.model for r in requests] == [MT] * 6 + [GEN]
        assert text == ("sw:First paragraph here.\n\nsw:Second one, a little "
                        "longer.\n \nsw:Third.")
        assert info == "Swahili"
        assert [r.max_tokens for r in requests[:3]] == [
            reply_cap(p) for p in ("First paragraph here.",
                                   "Second one, a little longer.", "Third.")]

    def test_one_judge_call_reads_the_joined_back_translation(self):
        requests, _ = _run(THREE, _Fake())
        judge = "\n".join(m["content"] for m in requests[-1].messages)
        assert THREE in judge                  # original and back-translation alike
        assert judge.count(THREE) == 2  # noqa: PLR2004

    def test_a_failing_piece_is_resampled_alone(self):
        def second_fails_once(text, n):
            return text * 5 if text.startswith("Second") and n == 1 else f"sw:{text}"

        requests, (text, info) = _run(THREE, _Fake(forward=second_fails_once))
        sent = [(_wire(r)[0], _wire(r)[2]) if r.model == MT else "judge"
                for r in requests]
        assert sent[:4] == [("en", "First paragraph here."),
                            ("en", "Second one, a little longer."),
                            ("en", "Third."),
                            ("en", "Second one, a little longer.")]
        assert len(sent) == 4 + 3 + 1                         # three back, one judge
        assert info == "Swahili"

    def test_a_piece_that_keeps_failing_is_named(self):
        def second_always_fails(text, n):
            return text * 5 if text.startswith("Second") else f"sw:{text}"

        requests, (_, info) = _run(THREE, _Fake(forward=second_always_fails))
        # All three pieces, then the second one alone in each later round.
        assert len(requests) == 3 + 3      # noqa: PLR2004
        assert info.startswith("DISCARDED; language=Swahili; piece=2/3; "
                               "feedback=length_ratio=")

    def test_a_judge_rejection_resamples_every_piece(self):
        requests, (_, info) = _run(THREE, _Fake(verdicts=["No", "Yes"]))
        assert [r.model for r in requests] == ([MT] * 6 + [GEN]) * 2
        assert info == "Swahili"

    def test_a_short_text_is_one_piece_sent_unchanged(self):
        requests, _ = _run("Just one line.", _Fake())
        assert _wire(requests[0])[2] == "Just one line."
        assert len(requests) == 3      # noqa: PLR2004


def _wire_or_none(request, source):
    return request.model == MT and _wire(request)[0] == source


class TestSplitText:
    @pytest.mark.parametrize("text", [
        "one", "  padded  ", THREE, "\n\nlead and trail\n\n", "a\nb\n\n\n\nc", "",
        " \n ",
    ])
    def test_the_pieces_and_gaps_give_the_text_back(self, text):
        pieces, gaps = split_text(text)
        assert join_pieces(pieces, gaps) == text
        assert len(gaps) == len(pieces) + 1
        assert all(p == p.strip() and p for p in pieces)

    def test_a_text_without_a_word_is_no_piece(self):
        assert split_text(" \n\n ") == ([], [" \n\n "])

    def test_a_single_newline_stays_inside_a_paragraph(self):
        assert split_text("line one\nline two")[0] == ["line one\nline two"]

    def test_a_long_paragraph_is_cut_at_sentence_ends(self):
        sentence = "This sentence is about forty characters. "
        paragraph = (sentence * 70).strip()             # about 2,900 characters
        pieces, gaps = split_text(paragraph)
        assert len(pieces) > 1
        assert all(len(p) <= MAX_PIECE_CHARS for p in pieces)
        assert all(p.endswith(".") for p in pieces)
        assert set(gaps[1:-1]) == {" "}
        assert join_pieces(pieces, gaps) == paragraph

    def test_a_long_sentence_without_an_end_stays_whole(self):
        paragraph = "word " * 400 + "end."
        assert split_text(paragraph.strip())[0] == [paragraph.strip()]

    def test_the_threshold_is_about_twelve_hundred_characters(self):
        assert MAX_PIECE_CHARS == 1200      # noqa: PLR2004
        exact = "x" * (MAX_PIECE_CHARS - 1) + "."
        assert split_text(exact)[0] == [exact]
