"""Shared utilities for jailbreak technique composition.

Ported from reference utils.py combine_techniques (lines 24-49).
Retry wrappers (with_retries, with_feedback_retries) live in LLMs/wrappers.py.
"""

from typing import Callable


def combine_techniques(*techniques: Callable) -> Callable:
    """Chain multiple technique functions sequentially.

    Each technique receives the output of the previous one.
    Additional_info strings are joined with ';'.

    Works with both pure functions (str -> (str, str)) and
    LLM functions that accept **kwargs. Extra kwargs are passed
    through to every technique in the chain, so pure functions
    ignore them and LLM functions pick up backend/model/rate_limiter.

    Args:
        *techniques: Functions with signature (str, **kwargs) -> (str, str)

    Returns:
        Combined function: (str, **kwargs) -> (str, str)
    """

    def combined(text: str, **kwargs) -> tuple[str, str]:
        result = text
        info_parts: list[str] = []
        for technique in techniques:
            result, info = technique(result, **kwargs)
            if info:
                info_parts.append(info)
        return result, ";".join(info_parts)

    combined.__name__ = "+".join(t.__name__ for t in techniques)
    return combined
