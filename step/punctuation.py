"""Punctuation step: restore punctuation with the omni model via the hosted server.

Takes a directory of audio files and optionally the ASR step's output jsonl
(records with "id", "transcript", "alignment"). Writes a jsonl file where each
line is:
    {"id": "<relative path without extension>", "transcript": "<punctuated transcript>"}

- With an ASR jsonl: the raw transcript (and alignment info, if present) is fed
  to the omni model together with the audio so it can restore punctuation.
- Standalone (no ASR jsonl): the omni model is used directly on the audio to
  produce a punctuated transcript.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

from script.call_omni import caption_audio
from step.base import BaseStep, add_common_args
from util.config import default_config_path, load_config
from util.logging import setup_logger

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def _normalize_id(value: str) -> str:
    """Strip a known audio suffix from an id so ids match across steps."""
    for suffix in AUDIO_SUFFIXES:
        if value.lower().endswith(suffix):
            return value[: -len(suffix)]
    return value


def load_asr_jsonl(path: Optional[str]) -> Dict[str, dict]:
    """Load the ASR step's jsonl into {normalized id: record}."""
    if not path:
        return {}
    records: Dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records[_normalize_id(str(record.get("id", "")))] = record
    return records


def build_punctuation_prompt(record: Optional[dict], system_prompt: str, standalone_prompt: str) -> str:
    """Build the prompt for the omni model, using the ASR transcript if available."""
    if record and record.get("transcript"):
        prompt = system_prompt.format(transcript=record["transcript"])
        alignment = record.get("alignment")
        if alignment:
            prompt += "\n\nWord-level alignment information:\n" + str(alignment)
        return prompt
    return standalone_prompt


class PunctuationStep(BaseStep):
    name = "punctuation"

    def run(self, input_path: str, output_path: str, **kwargs) -> str:
        cfg = self.config.get("step", self.config)
        server_url = kwargs.get("server_url") or cfg.get("server_url", "http://localhost:8000")
        asr_jsonl = kwargs.get("asr_jsonl") or cfg.get("asr_jsonl") or ""
        system_prompt = kwargs.get("system_prompt") or cfg.get("system_prompt") or ""
        standalone_prompt = (
            kwargs.get("standalone_system_prompt")
            or cfg.get("standalone_system_prompt")
            or system_prompt
        )
        max_new_tokens = kwargs.get("max_new_tokens") or cfg.get("max_new_tokens", 2560)

        logger = setup_logger(self.name)
        input_dir = Path(input_path)
        if not input_dir.is_dir():
            raise ValueError(f"Input must be a directory of audio files: {input_path}")

        asr_records = load_asr_jsonl(asr_jsonl)
        if asr_records:
            logger.info("Loaded %d records from ASR jsonl", len(asr_records))
        else:
            logger.info("No ASR jsonl provided (or empty) - using standalone punctuation mode")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        audio_files = sorted(
            f for f in input_dir.rglob("*") if f.suffix.lower() in AUDIO_EXTENSIONS
        )
        logger.info("Punctuating %d audio files from %s", len(audio_files), input_dir)

        with open(output_path, "w", encoding="utf-8") as out:
            for idx, audio_file in enumerate(audio_files, 1):
                fid = str(audio_file.relative_to(input_dir).with_suffix("")).replace("\\", "/")
                record = asr_records.get(fid) or asr_records.get(_normalize_id(fid))
                prompt = build_punctuation_prompt(record, system_prompt, standalone_prompt)
                logger.info("[%d/%d] Punctuating: %s", idx, len(audio_files), fid)
                text = caption_audio(
                    server_url, str(audio_file), prompt, max_new_tokens
                )
                out.write(json.dumps({"id": fid, "transcript": text}, ensure_ascii=False) + "\n")
                out.flush()

        logger.info("Wrote punctuated transcripts to %s", output_path)
        return str(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="punctuation step")
    add_common_args(parser)
    parser.add_argument("--asr-jsonl", default=None, help="Optional ASR step output jsonl (id/transcript/alignment)")
    parser.add_argument("--system-prompt", "-s", default=None, help="Optional system prompt (overrides config)")
    parser.add_argument("--server-url", default=None, help="Omni server URL (overrides config)")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else default_config_path("punctuation")
    config = load_config(config_path)

    step = PunctuationStep(config)
    step.run(
        args.input,
        args.output,
        asr_jsonl=args.asr_jsonl,
        system_prompt=args.system_prompt,
        server_url=args.server_url,
    )


if __name__ == "__main__":
    main()
