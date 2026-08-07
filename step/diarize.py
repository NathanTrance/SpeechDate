"""Diarization step: split audio into speaker segments."""

from step.base import BaseStep, main_cli


class DiarizeStep(BaseStep):
    name = "diarize"

    def run(self, input_path: str, output_path: str, **kwargs) -> str:
        raise NotImplementedError("diarize step: not implemented yet")


def main():
    main_cli(DiarizeStep)


if __name__ == "__main__":
    main()
