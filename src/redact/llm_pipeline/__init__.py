"""Helpers between the stages and ``llms/``: drive generators that yield
model calls, and read replies to decide the next one.

The stages import this; it imports only ``redact.llms``.
"""

from .check import batch_check_samples, check_sample, is_accepted
from .drive import drive_generators, drive_sync
from .request import LLMRequest

__all__ = [
    "LLMRequest",
    "batch_check_samples",
    "check_sample",
    "drive_generators",
    "drive_sync",
    "is_accepted",
]
