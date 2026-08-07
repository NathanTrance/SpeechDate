"""Call the hosted omni captioning server.

Takes an audio file (path) plus an optional system prompt, uploads it to the
server, and prints (or saves) the generated text.

Usage:
    python -m script.call_omni --url http://localhost:8000 --audio path/to/file.wav
    python -m script.call_omni --audio file.wav --system-prompt "Describe this audio." --output out.txt
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import requests

from util.logging import setup_logger

logger = setup_logger("call_omni")


def caption_audio(
    server_url: str,
    audio_path: str,
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 2560,
    timeout: Optional[int] = 600,
) -> str:
    """Send one audio file to the server and return the generated text."""
    url = server_url.rstrip("/") + "/caption"
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    with open(audio_path, "rb") as f:
        files = {"file": (audio_path.name, f, "application/octet-stream")}
        data = {
            "system_prompt": system_prompt or "",
            "max_new_tokens": str(max_new_tokens),
        }
        resp = requests.post(url, files=files, data=data, timeout=timeout)

    resp.raise_for_status()
    return resp.json()["text"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Call the hosted omni captioning server")
    parser.add_argument("--url", "-u", default="http://localhost:8000", help="Server base URL")
    parser.add_argument("--audio", "-a", required=True, help="Path to the audio file")
    parser.add_argument("--system-prompt", "-s", default=None, help="Optional system prompt")
    parser.add_argument("--max-new-tokens", type=int, default=2560, help="Max generated tokens")
    parser.add_argument("--output", "-o", default=None, help="Optional file to write the text to")
    args = parser.parse_args()

    text = caption_audio(args.url, args.audio, args.system_prompt, args.max_new_tokens)
    logger.info("Generated text (%d chars)", len(text))

    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
        logger.info("Saved to %s", args.output)
    else:
        print(json.dumps({"text": text}, ensure_ascii=False))


if __name__ == "__main__":
    main()
