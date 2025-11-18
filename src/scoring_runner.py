

"""
Scoring Runner - Execute eval_protocol on a candidate answer

Purpose
- Grade a target model's output against the verifier specs emitted in Step 7.
- Complements pipeline_runner: generation happens there, evaluation happens here.

CLI expectations
- --sample-id: must match the ID used when running pipeline_runner.
- --data-dir: base directory (defaults to data/) that now contains reports/<sample_id>.eval.json.
- --candidate-file (optional): override path to the model's answer; otherwise use reports/<sample_id>.candidate.txt.

Workflow
1. Load eval_protocol + metadata from reports/<sample_id>.eval.json.
2. Read the candidate answer text.
3. Call verifier.evaluate.run_evaluation to score every constraint and branch.
4. Persist reports/<sample_id>.score.json with summary statistics and branch selection info.
5. Print a concise summary to stdout for quick inspection.

Notes
- This script never queries an LLM; it operates purely on the stored protocol + answer.
- Ensure you have already saved the candidate output before invoking scoring_runner.
"""

import os
import argparse
from datetime import datetime, timezone
from typing import Dict, Any

from .verifier.evaluate import run_evaluation
from .utils.export_utils import write_json


# ------------------------------------------------------------
# Basic file IO helpers
# ------------------------------------------------------------

def _read_file(path: str) -> str:
    """Read UTF-8 file. Raise if missing/empty, because scoring requires real content."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    if not data.strip():
        raise ValueError(f"File is empty: {path}")
    return data


def _load_eval_protocol(eval_json_path: str) -> Dict[str, Any]:
    """
    Read <sample_id>.eval.json and extract eval_protocol.
    Expected shape of the JSON file (produced in pipeline_runner.run_pipeline_once):
        {
          "sample_id": ...,
          "timestamp_utc": ...,
          "eval_protocol": { ... },
          "meta": { ... }
        }
    """
    if not os.path.exists(eval_json_path):
        raise FileNotFoundError(f"Eval spec not found: {eval_json_path}")

    import json
    with open(eval_json_path, "r", encoding="utf-8") as f:
        blob = json.load(f)

    eval_protocol = blob.get("eval_protocol")
    if not eval_protocol:
        raise ValueError(f"No eval_protocol found in {eval_json_path}")
    return eval_protocol


# ------------------------------------------------------------
# Core scoring pipeline for one sample
# ------------------------------------------------------------

def run_scoring_once(sample_id: str,
                      data_dir: str = "data",
                      candidate_file: str = None) -> Dict[str, Any]:
    """
    Perform evaluation for a single sample_id.

    Steps:
        1. Resolve file paths under data_dir
        2. Load eval_protocol and candidate answer
        3. run_evaluation -> score_report
        4. Write score_report to data/reports/<sample_id>.score.json

    Return dict with useful paths and summary.
    """

    reports_dir = os.path.join(data_dir, "reports")

    # Where to read eval protocol
    eval_json_path = os.path.join(reports_dir, f"{sample_id}.eval.json")

    # Where to read the candidate answer (either override or default)
    if candidate_file is None:
        candidate_file = os.path.join(reports_dir, f"{sample_id}.candidate.txt")

    # Where to write final score report
    score_json_path = os.path.join(reports_dir, f"{sample_id}.score.json")

    # 1. load eval_protocol
    eval_protocol = _load_eval_protocol(eval_json_path)

    # 2. load candidate answer
    candidate_answer = _read_file(candidate_file)

    # 3. run evaluation
    score_core = run_evaluation(eval_protocol=eval_protocol,
                                candidate_answer=candidate_answer)

    # 4. augment with metadata and persist
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    score_record = {
        "sample_id": sample_id,
        "timestamp_utc": timestamp_utc,
        "candidate_file": candidate_file,
        "summary": score_core.get("summary", {}),
        "branch_choice": score_core.get("branch_choice", {}),
        "per_constraint": score_core.get("per_constraint", []),
    }

    write_json(score_json_path, score_record)

    return {
        "sample_id": sample_id,
        "score_json_path": score_json_path,
        "summary": score_core.get("summary", {}),
        "branch_choice": score_core.get("branch_choice", {}),
    }


# ------------------------------------------------------------
# CLI entry
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Score a candidate answer against the eval protocol for a given sample_id.",
    )
    parser.add_argument(
        "--sample-id",
        required=True,
        help="Identifier for this sample (same one used in pipeline_runner)",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Base data directory (expects subdir: reports/)",
    )
    parser.add_argument(
        "--candidate-file",
        default=None,
        help=(
            "Optional explicit path to candidate answer. "
            "If omitted, will use <data-dir>/reports/<sample_id>.candidate.txt"
        ),
    )

    args = parser.parse_args()

    result = run_scoring_once(
        sample_id=args.sample_id,
        data_dir=args.data_dir,
        candidate_file=args.candidate_file,
    )

    # Print concise summary to stdout
    print("===== SCORING DONE =====")
    print(f"sample_id                  : {result['sample_id']}")
    print("--- artifacts ---")
    print(f"score_json_path            : {result['score_json_path']}")
    print("--- summary ---")
    summ = result.get("summary", {})
    print(f"global_pass_rate           : {summ.get('global_pass_rate')}")
    print(f"block_pass_rate            : {summ.get('block_pass_rate')}")
    print(f"branch_pass_rate           : {summ.get('branch_pass_rate')}")
    print(f"overall_pass_rate          : {summ.get('overall_pass_rate')}")

    # branch choice info (which branch the model seems to follow)
    if result.get("branch_choice"):
        print("--- branch_choice ---")
        for sid, info in result["branch_choice"].items():
            print(f"{sid} -> chosen={info.get('chosen')}, "
                  f"score_real={info.get('score_branch_real')}, "
                  f"score_alt={info.get('score_branch_alt')}")


if __name__ == "__main__":
    main()
