"""Shared progress reporting for batched LLM generation.

A single :class:`ProgressReporter` is the one mechanism every generation
pipeline uses to surface live progress during a long batched call. It plugs
into :meth:`BatchCaller.batch_generate`'s ``on_complete(index, result)`` hook
(the chokepoint all batched generation flows through), so output, input
(constitution-seeded), and jailbreak generation all emit progress in the same
format — controlled by each pipeline's existing ``verbose`` flag.

The reporter prints throttled plain-text ticks (no tqdm dependency) to stdout,
matching the print style of the pipeline summaries.
"""

import threading


class ProgressReporter:
    """Print throttled completion ticks for a batch of LLM calls.

    Construct one per batched call with a human-readable ``label`` and the
    expected ``total`` number of completions, then pass :meth:`on_complete`
    as the ``on_complete`` callback to :meth:`BatchCaller.batch_generate`.

    Output lines look like::

        <indent><label>: <done>/<total> done

    Attributes:
        label: Context label shown on every line (e.g. "gen chunk 1/2").
        total: Expected number of completions.
        every: Print a tick every ``every`` completions (plus the final one).
            Defaults to ``max(1, total // 5)`` → roughly five ticks, so large
            batches don't flood the output.
        indent: Leading whitespace for every printed line.
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
            print(f"{indent}{label}: 0/{total} ...")

    def on_complete(self, index: int, result: str) -> None:
        """Record one completion and print a tick when due.

        Safe to pass directly as ``BatchCaller``'s ``on_complete`` callback.
        ``index`` / ``result`` are accepted to match that signature but only
        the running count is used.
        """
        with self._lock:
            self._done += 1
            done = self._done
        if done % self.every == 0 or done == self.total:
            print(f"{self.indent}{self.label}: {done}/{self.total} done")
