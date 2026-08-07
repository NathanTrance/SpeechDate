"""Pipeline steps.

Each step can run standalone (python -m step.<name>) or as part of the
pipeline orchestrated by script/run_pipeline.py.
"""

from step.base import BaseStep, add_common_args, main_cli
from step.standardize import StandardizeStep
from step.sidon import SidonStep
from step.diarize import DiarizeStep
from step.asr import AsrStep
from step.punctuation import PunctuationStep
from step.caption import CaptionStep

PIPELINE_ORDER = ["standardize", "sidon", "diarize", "asr", "punctuation", "caption"]

STEPS = {
    step_cls.name: step_cls
    for step_cls in (
        StandardizeStep,
        SidonStep,
        DiarizeStep,
        AsrStep,
        PunctuationStep,
        CaptionStep,
    )
}

__all__ = [
    "BaseStep",
    "add_common_args",
    "main_cli",
    "PIPELINE_ORDER",
    "STEPS",
]
