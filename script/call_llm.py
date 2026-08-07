"""Call a vLLM-served LLM via its OpenAI-compatible API.

Used to turn free-form captions into structured JSON metadata
(see the caption step's "structured_caption").

Usage:
    python -m script.call_llm --system-prompt "Extract JSON." --user-prompt "The speaker is..."
    python -m script.call_llm -s "Extract JSON." -p "The speaker is..." --output out.txt
"""

import argparse
import json
from typing import Optional

import requests

from util.logging import setup_logger

logger = setup_logger("call_llm")


def chat_completion(
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.1,
    timeout: Optional[int] = 300,
) -> str:
    """Call /v1/chat/completions and return the assistant message text."""
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"] or ""


def extract_json(text: str) -> dict:
    """Parse a JSON object from LLM output, tolerating markdown fences and prose."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.find("\n")
        if first_newline != -1:
            cleaned = cleaned[first_newline + 1:]
        cleaned = cleaned.strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start:end + 1]
    return json.loads(cleaned)


def main() -> None:
    parser = argparse.ArgumentParser(description="Call a vLLM-served LLM")
    parser.add_argument("--url", "-u", default="http://localhost:8037", help="vLLM base URL")
    parser.add_argument("--model", "-m", default="Gemma-4-31B-it", help="Served model name")
    parser.add_argument("--system-prompt", "-s", required=True)
    parser.add_argument("--user-prompt", "-p", required=True)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--output", "-o", default=None, help="Optional file to write the response to")
    args = parser.parse_args()

    text = chat_completion(args.url, args.model, args.system_prompt, args.user_prompt, args.max_tokens)
    logger.info("LLM response (%d chars)", len(text))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        logger.info("Saved to %s", args.output)
    else:
        print(text)


if __name__ == "__main__":
    main()
