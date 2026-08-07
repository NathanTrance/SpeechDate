"""Shared scaffolding for pipeline steps."""

import argparse
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

from util.config import default_config_path, load_config
from util.logging import setup_logger


class BaseStep(ABC):
    """Base class for every pipeline step.

    A step is a self-contained processing unit:
    - it can be run standalone via its CLI (python -m step.<name>)
    - it can be chained in the pipeline, where the previous step's
      output path is passed in as the next step's input.
    """

    name: str = "base"

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def run(self, input_path: str, output_path: str, **kwargs) -> str:
        """Execute the step on input_path, writing results to output_path.

        Returns the output path so it can feed the next step in the pipeline.
        """


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add the CLI arguments shared by every standalone step."""
    parser.add_argument(
        "--input", "-i", required=True,
        help="Input path (audio file, dataset, or a previous step's output)",
    )
    parser.add_argument("--output", "-o", required=True, help="Output path")
    parser.add_argument(
        "--config", "-c", default=None,
        help="Path to a config YAML (defaults to config/<step>.yaml)",
    )


def main_cli(step_cls) -> None:
    """Standard standalone entrypoint for a step class."""
    parser = argparse.ArgumentParser(description=f"{step_cls.name} step")
    add_common_args(parser)
    args = parser.parse_args()

    logger = setup_logger(step_cls.name)
    config_path = Path(args.config) if args.config else default_config_path(step_cls.name)
    config = load_config(config_path)

    step = step_cls(config)
    logger.info("Running %s step with config: %s", step_cls.name, config_path)
    output = step.run(args.input, args.output)
    logger.info("%s step finished: %s", step_cls.name, output)
