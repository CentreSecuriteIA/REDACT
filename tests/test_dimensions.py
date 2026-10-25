"""The dimensions table, the tag each augmentation family and each generation
axis carries, and the check that no axis varies a family's dimension."""

import json
import re

import pytest

import redact.jailbreak.pool as JPool
from redact import dimensions, paths
from redact.content_moderation import axes as A
from redact.jailbreak.technique import load_spec
from redact.recipe import augmentations_of
from redact.recipe import check_dimensions as _check_dimensions

_FAMILY_TAGS = {
    "language": ["translation", "answer_language"],
    "surface_noise": ["typos", "casual_writing", "noise", "suffixes"],
    "encoding": ["encode", "sensitive_words", "wording", "tokenbreak", "ascii_art",
                 "payload_split", "word_art"],
    "container": ["structural", "neutral_container"],
    "role_framing": ["persona", "framing", "cognitive", "impersonation",
                     "benign_role", "persona_template"],
    "request_shape": ["indirect", "distractor_prefix", "distractor_suffix",
                      "continuation", "asking", "temporal", "smalltalk_prefix",
                      "smalltalk_suffix", "guardrail_address", "persuasion",
                      "tense_rewrite", "model_distractors"],
    "answer_shape": ["answer_format"],
    "example_priming": ["fsh", "dap", "fake_turns"],
}

_AXIS_TAGS = {
    "authorship": ["inputs.writer"],
    "request_intent": ["inputs.framing", "inputs.directness"],
    "length": ["inputs.length"],
    "register": ["inputs.register"],
    "focus": ["inputs.focus"],
    "domain": ["inputs.domain"],
    "answer_length": ["outputs.length"],
    "answer_layout": ["outputs.format"],
    "answer_tone": ["outputs.register"],
}


@pytest.fixture
def table_at(tmp_path, monkeypatch):
    """Point the table at a file of the test's own."""
    monkeypatch.setattr(paths, "configs_dir", lambda: tmp_path)
    return lambda entries: (tmp_path / "dimensions.json").write_text(
        json.dumps(entries), encoding="utf-8")


class TestTheTable:
    def test_every_tag_has_a_side_and_a_definition(self):
        for entry in dimensions.table().values():
            assert entry["side"] in dimensions.SIDES
            assert entry["definition"].endswith(".")

    def test_the_augmentation_tags_are_the_eight_of_the_plan(self):
        assert dimensions.tags_of("augmentation") == list(_FAMILY_TAGS)

    def test_the_generation_tags_are_the_nine_of_the_plan(self):
        assert dimensions.tags_of("generation") == list(_AXIS_TAGS)

    @pytest.mark.parametrize(("entry", "message"), [
        ({"side": "both", "definition": "A thing."}, "side must be one of"),
        ({"side": "generation", "definition": " "}, "must be a sentence"),
        ({"side": "generation"}, "nothing else"),
        ({"side": "generation", "definition": "A thing.", "x": 1}, "nothing else"),
        ({"side": "augmentation", "definition": "A thing.", "same_as": []},
         "nothing else"),
        ({"side": "generation", "definition": "A thing.", "same_as": ["tag"]},
         "must each list augmentation tags"),
        ({"side": "generation", "definition": "A thing.", "same_as": "noise"},
         "must each list augmentation tags"),
        ({"side": "generation", "definition": "A thing.", "same_as": ["noise"],
          "undecided": ["noise"]}, "none in both"),
    ])
    def test_a_malformed_tag_raises(self, table_at, entry, message):
        table_at({"_comment": "skipped", "tag": entry,
                  "noise": {"side": "augmentation", "definition": "A thing."}})
        with pytest.raises(ValueError, match=message):
            dimensions.table()


class TestTheFamilies:
    def test_each_family_carries_the_tag_of_the_plan(self):
        assert {name: family["dimension"]
                for name, family in load_spec()["families"].items()} == {
            family: tag for tag, families in _FAMILY_TAGS.items()
            for family in families}

    def test_the_spec_passes_the_check(self):
        JPool._check_dimensions()

    @pytest.mark.parametrize(("tag", "named"), [
        (None, "fsh must carry"),
        ("example", "fsh ('example') must carry"),
        ("authorship", "fsh ('authorship') must carry"),
    ])
    def test_a_family_without_an_augmentation_tag_raises(self, monkeypatch, tag, named):
        spec = load_spec()
        families = {**spec["families"], "fsh": {**spec["families"]["fsh"]}}
        families["fsh"].pop("dimension")
        if tag is not None:
            families["fsh"]["dimension"] = tag
        monkeypatch.setattr(JPool, "load_spec", lambda: {**spec, "families": families})
        with pytest.raises(ValueError, match=rf"families: {re.escape(named)}"):
            JPool._check_dimensions()


class TestTheAxes:
    def test_each_axis_carries_the_tag_of_the_plan(self):
        assert {f"{stage}.{name}": entry["dimension"] for stage in A.STAGES
                for name, entry in A.resolve(stage).items()} == {
            axis: tag for tag, axes in _AXIS_TAGS.items() for axis in axes}

    @pytest.mark.parametrize(("tag", "named"), [
        (None, "length must carry"),
        ("encoding", "length ('encoding') must carry"),
    ])
    @pytest.mark.parametrize("axis", [
        {"label": "Length",
         "values": {"short": {"label": "s", "hint": "", "weight": 1}}},
        {"label": "Length", "from": "subcategory"},
    ], ids=["values", "from"])
    def test_an_axis_without_a_generation_tag_raises(self, axis, tag, named):
        section = {"length": {**axis, **({"dimension": tag} if tag else {})}}
        with pytest.raises(ValueError, match=rf"x: {re.escape(named)}"):
            A._checked("x", section)


# The families of request_shape, container and example_priming in each kind,
# the dimensions the focus axis is undecided against.
_FOCUS_FAMILIES = [*_FAMILY_TAGS["request_shape"], *_FAMILY_TAGS["container"],
                   *_FAMILY_TAGS["example_priming"]]
_MIXED = {"inputs": {"axes": {"focus": {"mixed": 0.25}}}}


def check_dimensions(params: dict, augmentations: dict) -> None:
    """The check of a training run whose params file is ``params``."""
    _check_dimensions(params, augmentations, dataset_type="training")


def _without_focus_families() -> dict:
    deny = {"families": {"deny": _FOCUS_FAMILIES}}
    return augmentations_of({"augmentations": {"adversarial": deny, "natural": deny}})


class TestTheCheck:
    def test_the_shipped_config_passes(self):
        check_dimensions({}, augmentations_of({}))

    def test_the_shipped_config_passes_with_translation_on(self):
        on = {"translation": True}
        check_dimensions({}, augmentations_of(
            {"augmentations": {"adversarial": on, "natural": on}}))

    def test_an_axis_that_varies_a_family_s_dimension_raises(self):
        with pytest.raises(ValueError) as raised:
            check_dimensions(_MIXED, augmentations_of({}))
        message = str(raised.value)
        assert "the axis inputs.focus (focus) and the request_shape families" in message
        assert "adversarial: asking, continuation" in message
        # The natural kind draws neither continuation, temporal nor
        # structural; the attacks draw no small talk.
        assert ("natural: asking, indirect, smalltalk_prefix, smalltalk_suffix)"
                in message)
        assert ("the container families (adversarial: neutral_container, "
                "structural; natural: neutral_container).") in message
        assert ("the example_priming families (adversarial: dap, fake_turns, fsh)"
                in message)
        assert message.count("undecided") == len(("request_shape", "container",
                                                   "example_priming"))
        assert 'params inputs.axes, as {"focus": null}' in message
        assert "augmentations.<kind>.families.deny" in message

    def test_denying_the_families_lets_the_axis_vary(self):
        check_dimensions(_MIXED, _without_focus_families())

    def test_a_kind_switched_off_draws_nothing(self):
        kinds = _without_focus_families()
        kinds["adversarial"] = None
        check_dimensions(_MIXED, kinds)

    def test_an_axis_left_out_varies_nothing(self):
        check_dimensions({"inputs": {"axes": {"focus": None}}}, augmentations_of({}))
        check_dimensions({"inputs": {"axes": False}}, augmentations_of({}))

    def test_a_wrong_axes_entry_raises_as_the_stage_would(self):
        with pytest.raises(ValueError, match="no axis 'fokus'"):
            check_dimensions({"inputs": {"axes": {"fokus": None}}},
                             augmentations_of({}))

    def test_a_same_as_pair_raises_without_the_undecided_note(self, table_at):
        table_at({"noise": {"side": "augmentation", "definition": "A thing."},
                  "tone": {"side": "generation", "definition": "A thing.",
                           "same_as": ["noise"]}})
        with pytest.raises(ValueError) as raised:
            dimensions.check({"inputs.register": "tone"},
                             {"natural": {"typos": "noise"}})
        assert ("the axis inputs.register (tone) and the noise families "
                "(natural: typos).") in str(raised.value)
        assert "undecided" not in str(raised.value)

    def test_authorship_and_role_framing_are_ruled_distinct(self):
        """The owner's ruling: who writes the message (the writer axis) is not
        a role or scenario the message sets up, so the axis varies beside the
        persona, framing, cognitive, persona_template and impersonation
        families."""
        assert dimensions.relation("authorship", "role_framing") is None
        assert A.varied(A.resolve("inputs"))["writer"] == "authorship"
        kinds = augmentations_of({})
        drawn = {fn.families[0] for kind in kinds.values()
                 for fn in JPool.technique_pool(kind)}
        assert set(_FAMILY_TAGS["role_framing"]) <= drawn
        check_dimensions({}, augmentations_of({}))

    @pytest.mark.parametrize("name", ["eval_example.json", "training_example.json"])
    def test_the_example_recipes_and_their_types_pass(self, name):
        """With their params files, and with their dataset types alone: the
        natural kind draws the benign roles (role_framing) beside the writer
        axis, which the ruling above keeps apart."""
        from redact.recipe import load_params, load_recipe

        recipe = load_recipe(paths.configs_dir() / "runs" / name)
        params = load_params(recipe["params_file"], base_dir=recipe["_base_dir"])
        dataset_type = recipe["dataset_type"]
        _check_dimensions(params, augmentations_of(recipe, params["jailbreaks"]),
                          dataset_type=dataset_type)
        _check_dimensions({}, load_recipe({"dataset_type": dataset_type})[
            "augmentations"], dataset_type=dataset_type)

    def test_one_tag_on_both_sides_is_one_dimension(self, table_at):
        table_at({"noise": {"side": "generation", "definition": "A thing."}})
        assert dimensions.relation("noise", "noise") == "same"


class TestWhatVaries:
    def test_an_axis_dealt_one_value_varies_nothing(self):
        assert "focus" not in A.varied(A.resolve("inputs", {"focus": {"mixed": 0}}))
        mixed = A.resolve("inputs", _MIXED["inputs"]["axes"])
        assert A.varied(mixed)["focus"] == "focus"

    def test_an_axis_with_from_varies(self):
        assert A.varied(A.resolve("inputs"))["domain"] == "domain"

    def test_no_axes_vary_nothing(self):
        assert A.varied(None) == {}
