"""Pipeline orchestrator.

Chains steps in order, feeding each step's output into the next step's input.
Steps are reused from the step package, so a step behaves identically when run
standalone or inside the pipeline.

Usage (from the repo root):
    python -m script.run_pipeline -i <input> -o <output_dir>
    python -m script.run_pipeline -i <input> -o <output_dir> --steps standardize asr
"""

import argparse
from pathlib import Path
from typing import List, Optional

from step import PIPELINE_ORDER, STEPS
from util.config import default_config_path, load_config
from util.logging import setup_logger


def run_pipeline(
    input_path: str,
    output_dir: str,
    steps: Optional[List[str]] = None,
    config_dir: Optional[str] = None,
) -> str:
    """Run the given steps (default: all) on input_path, writing to output_dir.

    Returns the final output path.
    """
    logger = setup_logger("pipeline")
    steps = steps or PIPELINE_ORDER
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    current_input = input_path
    for name in steps:
        if name not in STEPS:
            raise ValueError(f"Unknown step: {name}. Available: {list(STEPS)}")
        config_path = (
            Path(config_dir) / f"{name}.yaml"
            if config_dir
            else default_config_path(name)
        )
        config = load_config(config_path)

        step_cls = STEPS[name]
        step = step_cls(config)
        current_output = str(output_dir / f"{name}_out")

        logger.info("Running step '%s' (%s -> %s)", name, current_input, current_output)
        current_input = step.run(current_input, current_output)

    logger.info("Pipeline finished. Final output: %s", current_input)
    return current_input


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the speech data processing pipeline")
    parser.add_argument("--input", "-i", required=True, help="Input path (audio file or dataset)")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument(
        "--steps", nargs="*", default=None,
        help=f"Subset of steps to run (default: all: {PIPELINE_ORDER})",
    )
    parser.add_argument(
        "--config-dir", default=None,
        help="Directory containing <step>.yaml configs (defaults to config/)",
    )
    args = parser.parse_args()
    run_pipeline(args.input, args.output, args.steps, args.config_dir)


if __name__ == "__main__":
    main()
