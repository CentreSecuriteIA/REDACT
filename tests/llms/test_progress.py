"""Tests for the shared ProgressReporter."""

from redact.llms.progress import ProgressReporter


class TestProgressReporter:
    def test_announce_prints_start_line(self, capsys):
        ProgressReporter("gen chunk 1/2", 10)
        out = capsys.readouterr().out
        assert "gen chunk 1/2: 0/10 ..." in out

    def test_announce_disabled(self, capsys):
        ProgressReporter("x", 10, announce=False)
        assert capsys.readouterr().out == ""

    def test_no_announce_for_empty_total(self, capsys):
        ProgressReporter("x", 0)
        assert capsys.readouterr().out == ""

    def test_throttled_to_about_five_ticks(self, capsys):
        # total=20 -> every=4 -> ticks at 4,8,12,16,20 = 5 ticks.
        r = ProgressReporter("lbl", 20, announce=False)
        for i in range(20):
            r.on_complete(i, "resp")
        ticks = [ln for ln in capsys.readouterr().out.splitlines() if "done" in ln]
        assert len(ticks) == 5
        assert ticks[0] == "      lbl: 4/20 done"
        assert ticks[-1] == "      lbl: 20/20 done"

    def test_final_tick_always_emitted(self, capsys):
        # total=7 -> every=1, but verify the last completion always prints.
        r = ProgressReporter("lbl", 7, announce=False)
        for i in range(7):
            r.on_complete(i, "resp")
        out = capsys.readouterr().out
        assert "      lbl: 7/7 done" in out

    def test_small_total_every_is_one(self, capsys):
        # total=3 -> every=max(1, 3//5)=1 -> a tick per completion.
        r = ProgressReporter("lbl", 3, announce=False)
        for i in range(3):
            r.on_complete(i, "resp")
        ticks = [ln for ln in capsys.readouterr().out.splitlines() if "done" in ln]
        assert ticks == [
            "      lbl: 1/3 done",
            "      lbl: 2/3 done",
            "      lbl: 3/3 done",
        ]

    def test_custom_indent_and_every(self, capsys):
        r = ProgressReporter("lbl", 10, every=5, indent="  ", announce=False)
        for i in range(10):
            r.on_complete(i, "resp")
        ticks = [ln for ln in capsys.readouterr().out.splitlines() if "done" in ln]
        assert ticks == ["  lbl: 5/10 done", "  lbl: 10/10 done"]
