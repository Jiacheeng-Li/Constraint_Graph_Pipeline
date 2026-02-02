#!/usr/bin/env python
"""
Run pipeline_runner on a slice of oasst2_min200_div0.1.json and log per-sample output.

Usage example (4-way sharding over the dataset):

CUDA_VISIBLE_DEVICES=0 python scripts/run_filtered_batch_oasst2.py \
  --num-shards 4 --shard-id 0 --skip-successful \
  --model-path /data/MODELS/Qwen3-32B-N \
  --endpoint http://127.0.0.1:9001/v1/chat/completions \
  > experiments/oasst2/logs/run_filtered_batch_shard0.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python scripts/run_filtered_batch_oasst2.py \
  --num-shards 4 --shard-id 1 --skip-successful \
  --model-path /data/MODELS/Qwen3-32B-N \
  --endpoint http://127.0.0.1:9001/v1/chat/completions \
  > experiments/oasst2/logs/run_filtered_batch_shard1.log 2>&1 &

... (and similarly for shard-id 2, 3)
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys
from typing import List, Dict, Any


def load_data(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_text(path: str, content: str) -> None:
    pathlib.Path(path).write_text(content, encoding="utf-8")


def log_indicates_success(log_path: pathlib.Path) -> bool:
    """Heuristically decide whether a previous run succeeded."""
    if not log_path.exists():
        return False
    text = log_path.read_text(errors="ignore")
    if "===== PIPELINE DONE" not in text:
        return False
    lowered = text.lower()
    if "traceback" in lowered:
        return False
    if "failed" in lowered:
        return False
    if "llm_error" in lowered:
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Batch runner for oasst2_min200_div0.1.json slices (with optional sharding)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="experiments/oasst2/oasst2_min200_div0.1.json",
        help="Input JSON file containing filtered oasst2 samples",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="experiments/oasst2",
        help="Data directory passed to pipeline_runner",
    )
    parser.add_argument(
        "--tmp-dir",
        type=str,
        default="experiments/oasst2/tmp_seed",
        help="Temporary directory to store instruction/answer text files",
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="experiments/oasst2/logs",
        help="Directory to store per-sample pipeline logs",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/data/MODELS/Qwen3-32B-N",
        help="Local model path for vLLM server",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://127.0.0.1:9001/v1/chat/completions",
        help="OpenAI-compatible endpoint served by vLLM",
    )
    parser.add_argument(
        "--sample-prefix",
        type=str,
        default="oasst2_seed_",
        help="Prefix for sample_id (log file names follow this)",
    )
    parser.add_argument("--start", type=int, default=0, help="Start index (used when num-shards=1)")
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Maximum number of samples to run (used when num-shards=1; -1 means no limit)",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of shards to split the dataset into",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="Index of this shard in [0, num-shards)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip samples whose log file already exists",
    )
    parser.add_argument(
        "--skip-successful",
        action="store_true",
        help="Skip samples that already have a successful log; rerun failed/incomplete logs",
    )
    args = parser.parse_args()

    data = load_data(args.input)
    total = len(data)

    # Determine [start, end) range either via sharding or explicit start/limit.
    if args.num_shards > 1:
        if args.shard_id < 0 or args.shard_id >= args.num_shards:
            raise ValueError(f"shard-id must be in [0, num-shards), got shard-id={args.shard_id}, num-shards={args.num_shards}")
        shard_size = (total + args.num_shards - 1) // args.num_shards
        start = args.shard_id * shard_size
        end = min(total, start + shard_size)
    else:
        start = args.start
        end = total if args.limit < 0 else min(total, args.start + args.limit)

    if start >= total:
        print(f"Start index {start} is >= total samples {total}; nothing to run.")
        return

    subset = data[start:end]

    tmp_dir = pathlib.Path(args.tmp_dir)
    logs_dir = pathlib.Path(args.logs_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    if args.num_shards > 1:
        print(
            f"Total samples: {total}; running shard {args.shard_id+1}/{args.num_shards} "
            f"range [{start}, {end}) ({len(subset)} items)"
        )
    else:
        print(f"Total samples: {total}; running [{start}, {end}) ({len(subset)} items)")

    for item in subset:
        sample_id = f"{args.sample_prefix}{item.get('id')}"
        log_file = logs_dir / f"{sample_id}.log"
        if args.skip_successful and log_indicates_success(log_file):
            print(f"[skip successful log] {sample_id}")
            continue
        if args.skip_existing and log_file.exists():
            print(f"[skip existing] {sample_id}")
            continue

        instr = (item.get("messages") or [{}])[0].get("content", "")
        ans = (item.get("messages") or [{}, {}])[1].get("content", "")
        instr_file = tmp_dir / f"{sample_id}.instruction.txt"
        ans_file = tmp_dir / f"{sample_id}.answer.txt"
        write_text(instr_file, instr)
        write_text(ans_file, ans)

        cmd = [
            sys.executable,
            "-m",
            "src.pipeline_runner",
            "--sample-id",
            sample_id,
            "--instruction-file",
            str(instr_file),
            "--answer-file",
            str(ans_file),
            "--data-dir",
            args.data_dir,
        ]

        env = os.environ.copy()
        env["USE_LOCAL_VLLM"] = "1"
        env["LOCAL_OPENAI_ENDPOINT"] = args.endpoint
        env["LOCAL_MODEL_PATH"] = args.model_path
        env["PIPELINE_ENABLE_STEP8"] = "1"

        print(f"[run] {sample_id}")
        with open(log_file, "w", encoding="utf-8") as lf:
            proc = subprocess.run(cmd, cwd=os.getcwd(), env=env, stdout=lf, stderr=lf)
        status = "ok" if proc.returncode == 0 else f"fail({proc.returncode})"
        print(f"[done] {sample_id} -> {status} log={log_file}")


if __name__ == "__main__":
    main()
