"""The generation axes: the packaged table, its overlay, the deal, the
rendered specs, the columns and the coverage table."""

import csv
from collections import Counter

import pytest

from redact.content_moderation import axes as A

_GROUP = 40         # units of one group in a deal
_PER_UNIT = 3       # specs per unit
_GIVEN = 0.5        # a weight a params file gives
_LENGTHS = ("fragment", "one_liner", "short", "long", "very_long")


def _units(n: int, group=("Cyber", "harmful")) -> list[tuple[tuple[str, str], str]]:
    return [(group, f"unit {i}") for i in range(n)]


def _spec_counts(specs: list[list[dict]], axis: str) -> Counter:
    return Counter(spec[axis] for unit in specs for spec in unit)


class TestThePackagedTable:
    @pytest.mark.parametrize("stage", A.STAGES)
    def test_each_stage_has_axes_with_a_value_to_deal(self, stage):
        axes = A.resolve(stage)
        dealt = [name for name, entry in axes.items() if "values" in entry]
        assert dealt
        for name in dealt:
            values = axes[name]["values"].values()
            assert any(value["weight"] > 0 for value in values), name
            assert all(value["label"] and value["hint"] for value in values), name

    def test_the_input_axes_are_those_of_the_plan(self):
        assert list(A.resolve("inputs")) == [
            "length", "register", "writer", "framing", "directness", "focus",
            "domain"]

    def test_the_writer_is_anonymous_or_one_of_the_roles(self):
        writer = A.resolve("inputs")["writer"]["values"]
        assert writer["anonymous"]["weight"] == pytest.approx(0.30)
        roles = [v for name, v in writer.items() if name != "anonymous"]
        assert sum(v["weight"] for v in roles) == pytest.approx(0.70)

    def test_the_response_axes_are_length_format_and_register(self):
        assert list(A.resolve("outputs")) == ["length", "format", "register"]


class TestTheOverlay:
    def test_none_and_true_are_the_packaged_axes(self):
        packaged = A.resolve("inputs")
        assert A.resolve("inputs", None) == A.resolve("inputs", True) == packaged

    def test_false_is_no_axes(self):
        assert A.resolve("inputs", False) is None

    def test_a_weight_given_replaces_the_packaged_one(self):
        axes = A.resolve("inputs", {"length": {"very_long": _GIVEN}})
        assert axes["length"]["values"]["very_long"]["weight"] == _GIVEN
        assert axes["length"]["values"]["fragment"]["weight"] == pytest.approx(0.15)

    def test_an_axis_given_as_null_is_left_out(self):
        assert "writer" not in A.resolve("inputs", {"writer": None})

    def test_comment_keys_are_skipped(self):
        assert A.resolve("inputs", {"_note": "x", "length": {"_why": "y"}}) == \
            A.resolve("inputs")

    def test_no_axis_left_to_deal_is_no_axes(self):
        given = dict.fromkeys(("length", "format", "register"))
        assert A.resolve("outputs", given) is None

    @pytest.mark.parametrize(("given", "message"), [
        ({"lenght": {}}, "no axis 'lenght'"),
        ({"length": {"tiny": 1}}, "no value 'tiny'"),
        ({"length": {"short": -1}}, "a number of at least 0"),
        ({"length": {"short": True}}, "a number of at least 0"),
        ({"length": {"short": "1"}}, "a number of at least 0"),
        ({"length": dict.fromkeys(_LENGTHS, 0)}, "every weight"),
        ({"length": ["short"]}, "an object of value: weight"),
        ({"domain": {"x": 1}}, "takes no weights"),
        (["length"], "an object of axis"),
        ("on", "an object of axis"),
    ])
    def test_a_wrong_name_or_weight_raises(self, given, message):
        with pytest.raises(ValueError, match=message):
            A.resolve("inputs", given)

    def test_an_unknown_stage_raises(self):
        with pytest.raises(ValueError, match="no generation axes"):
            A.resolve("jailbreaks")


class TestTheDeal:
    def test_each_value_gets_its_share_of_a_group(self):
        axes = A.resolve("inputs", {"length": {"fragment": 1, "one_liner": 1,
                                                "short": 2, "long": 0, "very_long": 0}})
        counts = _spec_counts(A.deal(axes, _units(_GROUP), _PER_UNIT, salt="t"),
                              "length")
        total = _GROUP * _PER_UNIT
        assert counts == {"fragment": total // 4, "one_liner": total // 4,
                          "short": total // 2}

    def test_the_deal_is_the_same_whatever_the_order_of_the_units(self):
        axes = A.resolve("inputs")
        units = _units(_GROUP)
        dealt = dict(zip(units, A.deal(axes, units, _PER_UNIT, salt="t")))
        backwards = list(reversed(units))
        again = A.deal(axes, backwards, _PER_UNIT, salt="t")
        assert dict(zip(backwards, again)) == dealt

    def test_groups_are_dealt_apart(self):
        axes = A.resolve("inputs")
        one = A.deal(axes, _units(_GROUP), _PER_UNIT, salt="t")
        mixed = _units(_GROUP) + _units(_GROUP, ("Fraud", "benign"))
        assert A.deal(axes, mixed, _PER_UNIT, salt="t")[:_GROUP] == one

    def test_another_salt_deals_another_plan(self):
        axes = A.resolve("inputs")
        assert A.deal(axes, _units(_GROUP), 1, salt="a") != \
            A.deal(axes, _units(_GROUP), 1, salt="b")

    def test_a_value_is_never_dealt_with_one_it_names_in_not_with(self):
        axes = A.resolve("inputs")
        for unit in A.deal(axes, _units(4 * _GROUP), _PER_UNIT, salt="t"):
            for spec in unit:
                if spec["focus"] == "mixed":
                    assert spec["length"] not in ("fragment", "one_liner")
                if spec["framing"] == "advice":
                    assert spec["length"] != "fragment"

    def test_a_value_of_weight_zero_is_never_dealt(self):
        axes = A.resolve("outputs", {"format": {"code": 0}})
        counts = _spec_counts(A.deal(axes, _units(_GROUP), 1, salt="t"), "format")
        assert "code" not in counts

    def test_a_spec_names_every_dealt_axis_only(self):
        axes = A.resolve("inputs")
        (spec,), = A.deal(axes, _units(1), 1, salt="t")
        assert list(spec) == ["length", "register", "writer", "framing",
                              "directness", "focus"]


class TestRendering:
    _AXES = {
        "length": {"label": "Length", "values": {
            "short": {"label": "short", "hint": "2 to 4 sentences.", "weight": 1}}},
        "tone": {"label": "Tone", "values": {
            "dry": {"label": "dry", "hint": "", "weight": 1}}},
        "domain": {"label": "Domain", "from": "subcategory"},
    }

    def test_one_spec_is_a_line_per_axis(self):
        text = A.render_spec(self._AXES, {"length": "short", "tone": "dry"},
                             {"subcategory": "Phishing"})
        assert text == ("Length: short — 2 to 4 sentences.\nTone: dry\n"
                        "Domain: Phishing")

    def test_an_empty_field_has_no_line(self):
        text = A.render_spec(self._AXES, {"length": "short", "tone": "dry"},
                             {"subcategory": ""})
        assert text == "Length: short — 2 to 4 sentences.\nTone: dry"

    def test_the_specs_of_a_call_are_numbered(self):
        spec = {"length": "short", "tone": "dry"}
        text = A.render_specs(self._AXES, [spec, spec], {"subcategory": ""})
        assert text == (
            "1. Length: short — 2 to 4 sentences.\n   Tone: dry\n"
            "2. Length: short — 2 to 4 sentences.\n   Tone: dry")

    def test_no_axes_render_nothing(self):
        assert A.render_specs(None, [{}, {}], {}) == ""
        assert A.render_spec(None, {}, {}) == ""

    def test_columns_are_the_dealt_axes_with_the_stage_prefix(self):
        spec = {"length": "short", "tone": "dry"}
        assert A.columns(self._AXES, spec, prefix="response_") == {
            "response_length": "short", "response_tone": "dry"}
        assert A.columns(None, {}, prefix="") == {}


class TestCoverage:
    def test_every_value_of_every_group_has_a_row(self, tmp_path):
        axes = {"length": {"label": "Length", "values": {
            "short": {"label": "s", "hint": "", "weight": 1},
            "long": {"label": "l", "hint": "", "weight": 1}}},
            "domain": {"label": "Domain", "from": "subcategory"}}
        cyber, fraud = ("Cyber", "harmful"), ("Fraud", "benign")
        planned = [(cyber, {"length": "short"}), (cyber, {"length": "short"}),
                   (fraud, {"length": "long"})]
        accepted = [(cyber, {"length": "short"})]
        path = A.write_coverage(tmp_path / "x.coverage.csv", axes, planned, accepted)

        with path.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert rows == [
            {"axis": "length", "value": "short", "category": "Cyber",
             "entry_type": "harmful", "planned": "2", "accepted": "1"},
            {"axis": "length", "value": "long", "category": "Cyber",
             "entry_type": "harmful", "planned": "0", "accepted": "0"},
            {"axis": "length", "value": "short", "category": "Fraud",
             "entry_type": "benign", "planned": "0", "accepted": "0"},
            {"axis": "length", "value": "long", "category": "Fraud",
             "entry_type": "benign", "planned": "1", "accepted": "0"},
        ]

    def test_a_value_the_table_no_longer_has_keeps_its_counts(self, tmp_path):
        axes = {"length": {"label": "Length", "values": {
            "short": {"label": "s", "hint": "", "weight": 1}}}}
        group = ("Cyber", "harmful")
        path = A.write_coverage(tmp_path / "x.coverage.csv", axes, [],
                                [(group, {"length": "gone"})])
        with path.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert rows[-1] == {"axis": "length", "value": "gone", "category": "Cyber",
                            "entry_type": "harmful", "planned": "0", "accepted": "1"}
