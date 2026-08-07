"""Host the Qwen3-Omni Thinker model as an HTTP service.

Serves the captioning model so steps (and other scripts) can call it over
HTTP instead of loading the model in every process.

Usage (on the machine with the model + GPU):
    python -m script.serve_omni --model-id /path/to/Qwen3-Omni-30B-A3B-Instruct

Endpoints:
    GET  /health      -> {"status": "ok", "model_loaded": true/false}
    POST /caption     -> multipart form: file (audio) + optional system_prompt
"""

import argparse
import threading
import tempfile
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile

try:
    from transformers import Qwen3OmniMoeProcessor, Qwen3OmniMoeThinkerForConditionalGeneration
except ImportError:
    raise ImportError(
        "transformers with Qwen3-Omni support is required. Install a version "
        "compatible with your Qwen3-Omni checkpoint."
    )

try:
    from qwen_omni_utils import process_mm_info
except ImportError:
    raise ImportError(
        "qwen_omni_utils is required. Install it from the Qwen3-Omni repository "
        "(https://github.com/QwenLM/Qwen3-Omni)."
    )

from util.logging import setup_logger

logger = setup_logger("serve_omni")

DEFAULT_CAPTION_PROMPT = """
You are an advanced audio captioning system specialized in Vietnamese speech analysis. Your task is to listen to the input audio and generate a comprehensive, natural-sounding paragraph that captions the speaker's vocal characteristics.

Your description must seamlessly integrate the following 8 dimensions:
- Perceived gender (e.g., a female voice...)
- Pitch characteristics (e.g., high-pitched, deep, resonant)
- Estimated age group (e.g., young adult, elderly)
- Regional Vietnamese accent/dialect (e.g., Southern/Saigon, Northern/Hanoi, Central)
- Emotion (e.g., cheerful, distressed, calm)
- Tone of delivery (e.g., authoritative, gentle, hurried)
- Perceived personality traits (e.g., confident, warm, introverted)
- Speech clarity and articulation (e.g., clearly articulated, muffled by background noise)

Guidelines:
1. Write the caption in English.
2. Synthesize these observations into a fluid, descriptive paragraph (2-4 sentences).
3. Do not include any introductory or concluding remarks (e.g., do not say "Here is the caption:"). Output only the narrative description.
"""

app = FastAPI(title="Omni Captioning Server")

_model_lock = threading.Lock()
_configured_model_id: str = ""
_loaded: dict = {"processor": None, "model": None, "model_id": None}


def initialize_model(model_id: str):
    """Load the Qwen3-Omni Thinker model for text-only output."""
    logger.info("Loading Thinker model: %s...", model_id)
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_id)
    model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto",
        attn_implementation="flash_attention_2",
    )
    model.eval()
    return processor, model


def get_model():
    """Lazily load (once) and return the (processor, model) pair."""
    with _model_lock:
        if _loaded["model"] is None:
            if not _configured_model_id:
                raise RuntimeError("No model id configured. Start the server with --model-id.")
            _loaded["processor"], _loaded["model"] = initialize_model(_configured_model_id)
            _loaded["model_id"] = _configured_model_id
        return _loaded["processor"], _loaded["model"]


def generate_response(
    audio_path: Path,
    processor,
    model,
    system_prompt: str,
    max_new_tokens: int = 2560,
) -> str:
    """Run the Thinker on one audio file and return the generated text."""
    conversations = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": str(audio_path)},
                {"type": "text", "text": system_prompt},
            ],
        }
    ]

    text_prompt = processor.apply_chat_template(
        conversations, add_generation_prompt=True, tokenize=False
    )
    audios, images, videos = process_mm_info(conversations, use_audio_in_video=False)
    inputs = processor(
        text=text_prompt,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
    )
    inputs = {
        k: v.to(device=model.device, dtype=model.dtype) if v.is_floating_point() else v.to(model.device)
        for k, v in inputs.items()
    }

    stop_tokens = [processor.tokenizer.eos_token_id]
    im_end_token = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_token is not None:
        stop_tokens.append(im_end_token)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.1,
            top_p=0.95,
            eos_token_id=stop_tokens,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _loaded["model"] is not None}


@app.post("/caption")
async def caption(
    file: UploadFile = File(...),
    system_prompt: str = Form(""),
    max_new_tokens: int = Form(2560),
):
    """Caption one audio file. Returns {"text": "<generated text>"}."""
    processor, model = get_model()
    prompt = system_prompt.strip() or DEFAULT_CAPTION_PROMPT

    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        logger.info("Captioning: %s", file.filename)
        text = generate_response(tmp_path, processor, model, prompt, max_new_tokens)
        logger.info("Done: %s (%d chars)", file.filename, len(text))
        return {"text": text}
    finally:
        tmp_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Host the Qwen3-Omni Thinker as an HTTP server")
    parser.add_argument("--model-id", "-m", required=True, help="Model id or local checkpoint path")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", "-p", type=int, default=8000, help="Bind port (default: 8000)")
    args = parser.parse_args()

    global _configured_model_id
    _configured_model_id = args.model_id

    # Trigger the (slow) model load before accepting traffic.
    get_model()
    logger.info("Model loaded. Serving on http://%s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
