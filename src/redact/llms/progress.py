"""Shared progress reporting for batched LLM generation.

A single :class:`ProgressReporter` is the one mechanism every generation
pipeline uses to surface live progress during a long batched call. It plugs
into ``router.py``'s ``_dispatch_batch()``'s ``on_complete(index, result)``
hook (the chokepoint all batched generation flows through), so output, input
(constitution-seeded), and jailbreak generation all emit progress in the same
format — controlled by each pipeline's existing ``verbose`` flag.

The reporter logs throttled plain-text ticks at INFO (no tqdm dependency),
matching the level pipeline progress narration uses elsewhere.
"""

import logging
import threading

logger = logging.getLogger(__name__)


class ProgressReporter:
    """Log throttled completion ticks for a batch of LLM calls.

    You won't usually construct this directly — the real call site
    (``_dispatch_batch()`` in ``router.py``) builds one automatically
    whenever a caller passes ``progress="label"``, chaining its
    :meth:`on_complete` with any caller-supplied ``on_complete`` callback so
    both fire. Construct it yourself only if you're driving completions
    outside that path.

    Output lines look like::

        <indent><label>: <done>/<total> done

    Attributes:
        label: Context label shown on every line (e.g. "gen chunk 1/2").
        total: Expected number of completions.
        every: Log a tick every ``every`` completions (plus the final one).
            Defaults to ``max(1, total // 5)`` → roughly five ticks, so large
            batches don't flood the output.
        indent: Leading whitespace for every logged line.
        announce: If True (default) and ``total > 0``, log a "0/total ..."
            line at construction time, so a slow batch shows something
            before its first completion (which may be minutes away on a
            large LLM call). Pass False to suppress that opening line.
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

        # A "started" line so a slow batch shows something before the first
        # completion (which may be minutes away on a large LLM call).
        if announce and total > 0:
            logger.info("%s%s: 0/%d ...", indent, label, total)

    def on_complete(self, index: int, result: str) -> None:
        """Record one completion and log a tick when due.

        Safe to pass directly as ``BatchCaller``'s ``on_complete`` callback.
        ``index`` / ``result`` are accepted to match that signature but only
        the running count is used.
        """
        with self._lock:
            self._done += 1
            done = self._done
        if done % self.every == 0 or done == self.total:
            logger.info("%s%s: %d/%d done", self.indent, self.label, done, self.total)
