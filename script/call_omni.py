"""Call the vLLM-served omni model via its OpenAI-compatible API.

The client references local audio paths (no upload), so the vLLM server must
be started with --allowed-local-media-path covering the audio files (the
serve_omni wrapper defaults it to "/").

Usage:
    python -m script.call_omni --audio path/to/file.wav
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
    max_tokens: int = 2560,
    temperature: float = 0.1,
    top_p: float = 0.95,
    timeout: Optional[int] = 600,
    model: str = "Qwen3-Omni-30B-A3B-Instruct",
) -> str:
    """Caption one audio file via the omni server. Returns the generated text."""
    url = server_url.rstrip("/") + "/v1/chat/completions"

    if audio_path.startswith(("http://", "https://", "data:", "file:")):
        audio_url = audio_path
    else:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        audio_url = path.resolve().as_uri()  # file:///... (vLLM requires a URL scheme)

    content = [{"type": "audio_url", "audio_url": {"url": audio_url}}]
    if system_prompt:
        content.append({"type": "text", "text": system_prompt})

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"] or ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Call the vLLM-served omni model")
    parser.add_argument("--url", "-u", default="http://localhost:8000", help="vLLM base URL")
    parser.add_argument("--audio", "-a", required=True, help="Path to the audio file (or an http(s) URL)")
    parser.add_argument("--system-prompt", "-s", default=None, help="Optional system prompt")
    parser.add_argument("--model", "-m", default="Qwen3-Omni-30B-A3B-Instruct", help="Served model name")
    parser.add_argument("--max-tokens", type=int, default=2560, help="Max generated tokens")
    parser.add_argument("--output", "-o", default=None, help="Optional file to write the text to")
    args = parser.parse_args()

    text = caption_audio(args.url, args.audio, args.system_prompt, args.max_tokens, model=args.model)
    logger.info("Generated text (%d chars)", len(text))

    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
        logger.info("Saved to %s", args.output)
    else:
        print(json.dumps({"text": text}, ensure_ascii=False))


if __name__ == "__main__":
    main()
