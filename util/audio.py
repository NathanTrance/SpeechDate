"""Audio I/O helpers.

Placeholders only: implementations land together with the standardize step.
"""

from typing import Optional


def load_audio(path: str, sample_rate: Optional[int] = None):
    """Load an audio file. Returns raw samples (and sample rate).

    Implemented with the standardize step.
    """
    raise NotImplementedError("load_audio: implemented with the standardize step")


def save_audio(samples, path: str, sample_rate: int) -> str:
    """Save audio samples to disk. Returns the output path.

    Implemented with the standardize step.
    """
    raise NotImplementedError("save_audio: implemented with the standardize step")
