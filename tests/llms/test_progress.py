"""Tests for the shared ProgressReporter."""

import logging

from redact.llms.progress import ProgressReporter


class TestProgressReporter:
    def test_announce_prints_start_line(self, caplog):
        with caplog.at_level(logging.INFO, logger="redact.llms.progress"):
            ProgressReporter("gen chunk 1/2", 10)
        assert "gen chunk 1/2: 0/10 ..." in caplog.text

    def test_announce_disabled(self, caplog):
        with caplog.at_level(logging.INFO, logger="redact.llms.progress"):
            ProgressReporter("x", 10, announce=False)
        assert caplog.text == ""

    def test_no_announce_for_empty_total(self, caplog):
        with caplog.at_level(logging.INFO, logger="redact.llms.progress"):
            ProgressReporter("x", 0)
        assert caplog.text == ""

    def test_throttled_to_about_five_ticks(self, caplog):
        # total=20 -> every=4 -> ticks at 4,8,12,16,20 = 5 ticks.
        with caplog.at_level(logging.INFO, logger="redact.llms.progress"):
            r = ProgressReporter("lbl", 20, announce=False)
            for i in range(20):
                r.on_complete(i, "resp")
        ticks = [rec.message for rec in caplog.records if "done" in rec.message]
        assert len(ticks) == 5
        assert ticks[0] == "      lbl: 4/20 done"
        assert ticks[-1] == "      lbl: 20/20 done"

    def test_final_tick_always_emitted(self, caplog):
        # total=7 -> every=1, but verify the last completion always logs.
        with caplog.at_level(logging.INFO, logger="redact.llms.progress"):
            r = ProgressReporter("lbl", 7, announce=False)
            for i in range(7):
                r.on_complete(i, "resp")
        assert "      lbl: 7/7 done" in caplog.text

    def test_small_total_every_is_one(self, caplog):
        # total=3 -> every=max(1, 3//5)=1 -> a tick per completion.
        with caplog.at_level(logging.INFO, logger="redact.llms.progress"):
            r = ProgressReporter("lbl", 3, announce=False)
            for i in range(3):
                r.on_complete(i, "resp")
        ticks = [rec.message for rec in caplog.records if "done" in rec.message]
        assert ticks == [
            "      lbl: 1/3 done",
            "      lbl: 2/3 done",
            "      lbl: 3/3 done",
        ]

    def test_custom_indent_and_every(self, caplog):
        with caplog.at_level(logging.INFO, logger="redact.llms.progress"):
            r = ProgressReporter("lbl", 10, every=5, indent="  ", announce=False)
            for i in range(10):
                r.on_complete(i, "resp")
        ticks = [rec.message for rec in caplog.records if "done" in rec.message]
        assert ticks == ["  lbl: 5/10 done", "  lbl: 10/10 done"]
