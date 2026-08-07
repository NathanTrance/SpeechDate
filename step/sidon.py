"""Sidon step: enhance audio quality."""

from step.base import BaseStep, main_cli


class SidonStep(BaseStep):
    name = "sidon"

    def run(self, input_path: str, output_path: str, **kwargs) -> str:
        raise NotImplementedError("sidon step: not implemented yet")


def main():
    main_cli(SidonStep)


if __name__ == "__main__":
    main()
