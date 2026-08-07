"""Standardize step: convert audio to a canonical format."""

from step.base import BaseStep, main_cli


class StandardizeStep(BaseStep):
    name = "standardize"

    def run(self, input_path: str, output_path: str, **kwargs) -> str:
        raise NotImplementedError("standardize step: not implemented yet")


def main():
    main_cli(StandardizeStep)


if __name__ == "__main__":
    main()
