"""Serve the Qwen3-Omni Thinker model with vLLM.

vLLM is the recommended deployment for Qwen3-Omni (see
ref/Qwen3-Omni-30B-A3B-Instruct_README.md) and is far faster than the
transformers-based server (script/serve_omni_transformers.py). This wrapper
preconfigures `vllm serve` with the project's flags.

Requirements on the GPU machine:
- vLLM build with Qwen3-Omni support (qwen3_omni branch, see the README's
  "vLLM Usage -> Installation" section). The vLLM engine v1 is not supported
  yet, so VLLM_USE_V1=0 is set automatically.
- The audio files must live under the allowed media path (default "/") since
  the client references local paths instead of uploading.

Usage (on the machine with the model + GPU):
    python -m script.serve_omni -m /path/to/Qwen3-Omni-30B-A3B-Instruct
    # background with logs:
    nohup python -m script.serve_omni -m /path/to/Qwen3-Omni-30B-A3B-Instruct > logs/vllm-omni.log 2>&1 &

Endpoints (OpenAI-compatible):
    GET  /v1/models
    POST /v1/chat/completions  -> user content: [{"type": "audio_url", "audio_url": {"url": "<local path>"}}, {"type": "text", "text": "..."}]
"""

import argparse
import os
import shlex
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Qwen3-Omni with vLLM")
    parser.add_argument("--model", "-m", required=True, help="Model id or local checkpoint path")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", "-p", type=int, default=8000, help="Bind port")
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", default=32768)
    parser.add_argument(
        "--limit-mm-per-prompt", default=None,
        help='Per-prompt multimodal data limit, e.g. \'{"audio": 3}\' (more audio = more VRAM)',
    )
    parser.add_argument(
        "--allowed-local-media-path", default="/",
        help="Root directory that local audio_url paths must live under",
    )
    parser.add_argument("--max-num-seqs", type=int, default=64, help="Sequences processed in parallel per step")
    parser.add_argument("--extra-flags", default="", help="Additional vLLM serve flags (quoted)")
    args = parser.parse_args()

    cmd = ["vllm", "serve", args.model]
    cmd += ["--host", args.host, "--port", str(args.port)]
    cmd += ["--dtype", args.dtype]
    cmd += ["--max-model-len", str(args.max_model_len)]
    cmd += ["--gpu-memory-utilization", str(args.gpu_memory_utilization)]
    cmd += ["--tensor-parallel-size", str(args.tensor_parallel_size)]
    cmd += ["--allowed-local-media-path", args.allowed_local_media_path]
    cmd += ["--max-num-seqs", str(args.max_num_seqs)]
    if args.limit_mm_per_prompt:
        cmd += ["--limit-mm-per-prompt", args.limit_mm_per_prompt]
    if args.extra_flags:
        cmd += shlex.split(args.extra_flags)

    env = os.environ.copy()
    # vLLM engine v1 does not support Qwen3-Omni yet.
    env["VLLM_USE_V1"] = "0"

    print("Running: " + " ".join(cmd))
    sys.exit(subprocess.call(cmd, env=env))


if __name__ == "__main__":
    main()
