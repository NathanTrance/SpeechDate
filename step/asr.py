"""ASR step: transcribe speech to text."""

from step.base import BaseStep, main_cli


class AsrStep(BaseStep):
    name = "asr"

    def run(self, input_path: str, output_path: str, **kwargs) -> str:
        raise NotImplementedError("asr step: not implemented yet")


def main():
    main_cli(AsrStep)


if __name__ == "__main__":
    main()
