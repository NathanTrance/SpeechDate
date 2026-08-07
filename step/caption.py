"""Captioning step: generate captions with omni models via the hosted server.

Takes a directory of audio files and writes a jsonl file where each line is:
    {"id": "<relative path without extension>", "caption": "<generated caption>"}

When an LLM is configured (config/caption.yaml -> llm), the free-form caption
is additionally transformed into structured JSON metadata via the LLM and
stored under "structured_caption".
"""

import argparse
import json
from pathlib import Path

from script.call_llm import chat_completion, extract_json
from script.call_omni import caption_audio
from step.base import BaseStep, add_common_args
from util.config import default_config_path, load_config
from util.logging import setup_logger

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


class CaptionStep(BaseStep):
    name = "caption"

    def run(self, input_path: str, output_path: str, **kwargs) -> str:
        cfg = self.config.get("step", self.config)
        server_url = kwargs.get("server_url") or cfg.get("server_url", "http://localhost:8000")
        system_prompt = kwargs.get("system_prompt") or cfg.get("system_prompt") or ""
        max_new_tokens = kwargs.get("max_new_tokens") or cfg.get("max_new_tokens", 2560)

        llm_cfg = cfg.get("llm") or {}
        llm_url = kwargs.get("llm_server_url") or llm_cfg.get("server_url") or ""

        logger = setup_logger(self.name)
        input_dir = Path(input_path)
        if not input_dir.is_dir():
            raise ValueError(f"Input must be a directory of audio files: {input_path}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        audio_files = sorted(
            f for f in input_dir.rglob("*") if f.suffix.lower() in AUDIO_EXTENSIONS
        )
        logger.info("Captioning %d audio files from %s", len(audio_files), input_dir)
        if llm_url:
            logger.info("LLM structuring enabled: %s (%s)", llm_url, llm_cfg.get("model", ""))

        with open(output_path, "w", encoding="utf-8") as out:
            for idx, audio_file in enumerate(audio_files, 1):
                fid = str(audio_file.relative_to(input_dir).with_suffix("")).replace("\\", "/")
                logger.info("[%d/%d] Captioning: %s", idx, len(audio_files), fid)
                caption = caption_audio(
                    server_url, str(audio_file), system_prompt, max_new_tokens
                )
                record = {"id": fid, "caption": caption}
                if llm_url:
                    record["structured_caption"] = self._structure_caption(caption, llm_cfg, llm_url, logger)
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()

        logger.info("Wrote captions to %s", output_path)
        return str(output_path)

    def _structure_caption(self, caption: str, llm_cfg: dict, llm_url: str, logger) -> dict:
        """Use the LLM to turn a free-form caption into structured JSON metadata."""
        model = llm_cfg.get("model") or "Gemma-4-31B-it"
        system_prompt = llm_cfg.get("system_prompt") or ""
        max_tokens = llm_cfg.get("max_tokens", 1024)

        raw = chat_completion(llm_url, model, system_prompt, caption, max_tokens=max_tokens)
        if not raw:
            logger.warning("LLM returned an empty response for caption; storing raw")
            return {"raw": ""}
        try:
            return extract_json(raw)
        except Exception as exc:
            logger.warning("Failed to parse structured caption as JSON (%s); storing raw text", exc)
            return {"raw": raw}


def main() -> None:
    parser = argparse.ArgumentParser(description="caption step")
    add_common_args(parser)
    parser.add_argument("--system-prompt", "-s", default=None, help="Optional system prompt (overrides config)")
    parser.add_argument("--server-url", default=None, help="Omni server URL (overrides config)")
    parser.add_argument("--llm-server-url", default=None, help="LLM (vLLM) URL, enables structured captions (overrides config)")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else default_config_path("caption")
    config = load_config(config_path)

    step = CaptionStep(config)
    step.run(
        args.input,
        args.output,
        system_prompt=args.system_prompt,
        server_url=args.server_url,
        llm_server_url=args.llm_server_url,
    )


if __name__ == "__main__":
    main()
