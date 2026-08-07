"""Serve the LLM used to structure captions (vLLM).

Thin wrapper around `vllm serve` preconfigured for this project's setup
(Gemma-4-31B-it in the reference command). Run it on the GPU machine.

Usage:
    python -m script.serve_llm -m /home/voice/data/models/gemma-4-31B-it
    # background with logs (as in the reference setup):
    nohup python -m script.serve_llm -m /home/voice/data/models/gemma-4-31B-it > logs/vllm-llm.log 2>&1 &
"""

import argparse
import os
import shlex
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the LLM with vLLM")
    parser.add_argument("--model", "-m", required=True, help="Model path or id")
    parser.add_argument("--served-model-name", "-n", default=None, help="Model name exposed by the API")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", "-p", type=int, default=8037, help="Bind port")
    parser.add_argument("--gpus", default="0", help="CUDA_VISIBLE_DEVICES value (e.g. 0)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-num-batched-tokens", type=int, default=131072)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--no-prefix-caching", action="store_true", help="Disable --enable-prefix-caching")
    parser.add_argument("--no-trust-remote-code", action="store_true", help="Disable --trust-remote-code")
    parser.add_argument("--no-async-scheduling", action="store_true", help="Disable --async-scheduling")
    parser.add_argument("--extra-flags", default="", help="Additional vLLM serve flags (quoted)")
    args = parser.parse_args()

    cmd = ["vllm", "serve", args.model]
    if args.served_model_name:
        cmd += ["--served-model-name", args.served_model_name]
    cmd += ["--host", args.host, "--port", str(args.port)]
    cmd += ["--gpu-memory-utilization", str(args.gpu_memory_utilization)]
    cmd += ["--dtype", args.dtype]
    cmd += ["--max-model-len", "auto"]
    if not args.no_prefix_caching:
        cmd += ["--enable-prefix-caching"]
    if not args.no_trust_remote_code:
        cmd += ["--trust-remote-code"]
    cmd += ["--max-num-batched-tokens", str(args.max_num_batched_tokens)]
    if not args.no_async_scheduling:
        cmd += ["--async-scheduling"]
    cmd += ["--max-num-seqs", str(args.max_num_seqs)]
    if args.extra_flags:
        cmd += shlex.split(args.extra_flags)

    env = os.environ.copy()
    if args.gpus:
        env["CUDA_VISIBLE_DEVICES"] = args.gpus

    print("Running: " + " ".join(cmd))
    sys.exit(subprocess.call(cmd, env=env))


if __name__ == "__main__":
    main()
