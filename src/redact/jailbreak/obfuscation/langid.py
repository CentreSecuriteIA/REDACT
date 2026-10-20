"""Language identification with GlotLID, for the checks of a machine
translation and of a paraphrase.

The paraphrase check (``content_moderation/paraphrase.py``'s language gate)
reads :func:`label_probability` from here too. The model is fetched into the
HuggingFace cache at its first use (or ahead, by
``scripts/download_models.py``). Without the ``langid`` extra, or when the
model cannot be had, :func:`detect` and :func:`label_probability` give
``None`` and the check is skipped.
"""

import logging
import threading

logger = logging.getLogger(__name__)

#: GlotLID v3, which knows every translation language (1.69 GB).
GLOTLID_REPO = "cis-lmu/glotlid"
GLOTLID_FILE = "model_v3.bin"

_LABEL_PREFIX = "__label__"

_lock = threading.Lock()
_model = None
_unavailable = False


def _load():
    """The GlotLID model, loaded once; ``None`` when it cannot be had."""
    global _model, _unavailable
    if _model is None and not _unavailable:
        with _lock:
            if _model is None and not _unavailable:
                try:
                    import fasttext
                    from huggingface_hub import hf_hub_download
                except ImportError:
                    _unavailable = True
                    return None
                try:
                    _model = fasttext.load_model(
                        hf_hub_download(GLOTLID_REPO, GLOTLID_FILE))
                except Exception as exc:  # noqa: BLE001 - a check, not a stop
                    logger.warning(
                        "GlotLID (%s, %s) could not be loaded, so the language "
                        "checks (machine translation, paraphrase) are skipped: "
                        "%s: %s",
                        GLOTLID_REPO, GLOTLID_FILE, type(exc).__name__, exc)
                    _unavailable = True
    return _model


def available() -> bool:
    """Whether :func:`detect` identifies languages: the ``langid`` extra is
    installed and the model loads, fetched at the first call."""
    return _load() is not None


def detect(text: str) -> str | None:
    """GlotLID's most likely label for ``text`` (``"swh_Latn"``), read as one
    line; ``None`` without the detector or for a text with no word."""
    model = _load()
    line = " ".join(text.split())
    if model is None or not line:
        return None
    labels, _ = model.predict(line)
    return labels[0].removeprefix(_LABEL_PREFIX) if labels else None


def label_probability(text: str) -> tuple[str, float] | None:
    """GlotLID's most likely label for ``text`` and its probability
    (``("swh_Latn", 0.97)``), read as one line as :func:`detect` reads it;
    ``None`` without the detector or for a text with no word."""
    model = _load()
    line = " ".join(text.split())
    if model is None or not line:
        return None
    labels, probs = model.predict(line)
    if not labels:
        return None
    return labels[0].removeprefix(_LABEL_PREFIX), float(probs[0])
