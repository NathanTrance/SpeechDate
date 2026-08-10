# SpeechDate

Speech data processing pipeline: format standardization, audio enhancement, diarization, ASR, punctuation restoration, and captioning with omni models.

Heavy models are hosted once as HTTP servers and called by the steps, so each model loads in a single process:
- the omni captioning model (Qwen3-Omni Thinker) generates captions from audio
- an LLM (vLLM, e.g. Gemma-4-31B-it) transforms free-form captions into structured JSON metadata

## Setup

```bash
pip install -r requirements.txt
```

Additional requirements on the machines that host the models:
- Omni server: `vllm` with Qwen3-Omni support (the `qwen3_omni` branch, see below), a GPU
- LLM server: `vllm` (install per your environment), a GPU

Install the Qwen3-Omni-capable vLLM (see `ref/Qwen3-Omni-30B-A3B-Instruct_README.md` -> "vLLM Usage"):

```bash
git clone -b qwen3_omni https://github.com/wangxiongts/vllm.git
cd vllm && pip install -e . --no-build-isolation
pip install git+https://github.com/huggingface/transformers
pip install qwen-omni-utils -U
```

## 1. Host the omni model (vLLM)

On the machine with the model and GPU:

```bash
python -m script.serve_omni -m /path/to/Qwen3-Omni-30B-A3B-Instruct
# background with logs:
nohup python -m script.serve_omni -m /path/to/Qwen3-Omni-30B-A3B-Instruct > logs/vllm-omni.log 2>&1 &
# options: --port 8000 (default), --tensor-parallel-size 4 (multi-GPU), --max-num-seqs 64 (default)
```

The wrapper is preconfigured with the flags from the model README (`--dtype bfloat16 --max-model-len 32768 --allowed-local-media-path /`) and sets `VLLM_USE_V1=0` (engine v1 is not supported yet). It also sets `--served-model-name` to the model's basename (e.g. `Qwen3-Omni-30B-A3B-Instruct`) so it matches the clients' default `model` field. The client references local audio paths (no upload), so the audio files must live under the allowed media path (default `/`). Pass `--limit-mm-per-prompt '{"audio": 3}'` to allow more audio per prompt (uses more VRAM).

Endpoints (OpenAI-compatible):

| Endpoint | Description |
| --- | --- |
| `GET /v1/models` | lists the served model |
| `POST /v1/chat/completions` | user content: `[{"type": "audio_url", "audio_url": {"url": "<local path>"}}, {"type": "text", "text": "..."}]` |

A transformers-based fallback server (slow, RTF ~1 on a B200) is available as `script/serve_omni_transformers.py` if vLLM is unavailable.

## 2. Call the omni server directly

```bash
python -m script.call_omni --audio path/to/file.wav
python -m script.call_omni --audio file.wav --system-prompt "Describe this audio." --output out.txt
# options: --url http://localhost:8000 (default), --max-tokens 2560 (default)
```

The client sends the local audio path as `audio_url` (or an http(s) URL if given); the model name in the request defaults to `Qwen3-Omni-30B-A3B-Instruct`. If no `--system-prompt` is given, the prompt is whatever the steps/you supply — there is no server-side default prompt anymore.

## 3. Host the LLM (vLLM)

The LLM (e.g. Gemma-4-31B-it) turns free-form captions into structured JSON metadata. Serve it with vLLM on the GPU machine:

```bash
python -m script.serve_llm -m /home/voice/data/models/gemma-4-31B-it
# background with logs (matches the reference setup):
nohup python -m script.serve_llm -m /home/voice/data/models/gemma-4-31B-it > logs/vllm-gemma4-31b-it.log 2>&1 &
# options: --port 8037 (default), --served-model-name Gemma-4-31B-it (default: model basename), --gpus 0 (default)
```

The wrapper is preconfigured with the project's flags (`--dtype bfloat16 --enable-prefix-caching --trust-remote-code --max-num-batched-tokens 131072 --async-scheduling --max-num-seqs 512 --max-model-len auto --gpu-memory-utilization 0.9`); pass `--no-*` flags to disable, or `--extra-flags "..."` to add more.

## 4. Call the LLM directly

```bash
python -m script.call_llm --system-prompt "Extract JSON." --user-prompt "The speaker is..."
# options: --url http://localhost:8037 (default), --model Gemma-4-31B-it (default), --output out.txt
```

The client uses vLLM's OpenAI-compatible endpoint (`/v1/chat/completions`).

## 5. Caption step

Generates a caption for every audio file in a directory and writes a jsonl file.

```bash
python -m step.caption -i <wav_folder> -o captions.jsonl
# options: --system-prompt "..." (optional), --server-url http://localhost:8000 (default), --llm-server-url (optional, enables structured captions)
```

When an LLM is configured (`llm.server_url` in `config/caption.yaml`), each caption is also transformed into structured JSON metadata via the LLM and stored under `structured_caption`. Output format (one JSON object per line):

```json
{"id": "sub/audio1", "caption": "The speaker is a young adult male with a ...", "structured_caption": {"gender": "male", "pitch": "deep", "age": "young adult", "accent": "Southern/Saigon", "emotion": "neutral", "tone": "conversational", "personality": "confident", "clarity": "crystal clear"}}
```

If the LLM response cannot be parsed as JSON, `structured_caption` is stored as `{"raw": "<unparsed text>"}` so no data is lost. `id` is the file's path relative to the input folder, without the extension. Prompts live in `config/caption.yaml`: `system_prompt` (captioning) and `llm.system_prompt` (JSON extraction, based on `ref/Omni_LLM_test-Copy1.ipynb`).

## 6. Punctuation step

Restores punctuation for every audio file in a directory and writes a jsonl file.

With an ASR jsonl (recommended in the pipeline — the raw transcript plus alignment info is fed to the model together with the audio):

```bash
python -m step.punctuation -i <wav_folder> -o punctuated.jsonl --asr-jsonl asr_out.jsonl
```

Standalone (no ASR output; the omni model transcribes + punctuates from audio alone):

```bash
python -m step.punctuation -i <wav_folder> -o punctuated.jsonl
# options: --system-prompt "..." (optional), --server-url http://localhost:8000 (default)
```

ASR jsonl input format (as produced by the ASR step):

```json
{"id": "sub/audio1", "transcript": "do you want to go to the market", "alignment": {"word": [0.0, 0.4], ...}}
```

`alignment` is optional; if present it is passed to the model as alignment information. Output format (one JSON object per line):

```json
{"id": "sub/audio1", "transcript": "Do you want to go to the market?"}
```

The prompts live in `config/punctuation.yaml`: `system_prompt` (used when an ASR transcript is available, `{transcript}` is substituted) and `standalone_system_prompt` (used otherwise).

Ids in the ASR jsonl must match the audio file ids (relative path without extension). Audio suffixes (`.wav`, `.mp3`, ...) on ids are tolerated.

## 7. Running in the pipeline

```bash
python -m script.run_pipeline -i <input> -o <output_dir>
# or a subset:
python -m script.run_pipeline -i <input> -o <output_dir> --steps standardize asr
```

Each step can also be invoked standalone, so steps can be combined in any order. Steps that call the omni server (`caption`, `punctuation`) require a running server (see section 1); the caption step also needs the LLM server (see section 3) for `structured_caption`.
