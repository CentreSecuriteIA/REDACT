"""Progress logging for batched LLM calls.

``ModelClient.generate()`` builds a :class:`ProgressReporter` when it is given
``progress="label"``. The reporter logs throttled plain-text ticks at INFO.
"""

import logging
import threading

logger = logging.getLogger(__name__)


class ProgressReporter:
    """Log throttled completion ticks for a batch of LLM calls.

    Lines look like::

        <indent><label>: <done>/<total> done

    Args:
        label: Label shown on every line (e.g. "gen chunk 1/2").
        total: Expected number of completions.
        every: Log a tick every ``every`` completions, plus the final one.
            Defaults to ``max(1, total // 5)``, about five ticks per batch.
        indent: Leading whitespace for every line.
        announce: Log a "0/total ..." line at construction when
            ``total > 0``, so a slow batch shows something before its first
            completion.
    """

    def __init__(
        self,
        label: str,
        total: int,
        *,
        every: int | None = None,
        indent: str = "      ",
        announce: bool = True,
    ):
        self.label = label
        self.total = total
        self.every = every if every is not None else max(1, total // 5)
        self.indent = indent
        self._done = 0
        self._lock = threading.Lock()

        if announce and total > 0:
            logger.info("%s%s: 0/%d ...", indent, label, total)

    def on_complete(self, index: int, result: str) -> None:
        """Count one completion and log a tick when due.

        ``index`` and ``result`` match ``BatchCaller``'s ``on_complete``
        signature and are not used.
        """
        with self._lock:
            self._done += 1
            done = self._done
        if done % self.every == 0 or done == self.total:
            logger.info("%s%s: %d/%d done", self.indent, self.label, done, self.total)
