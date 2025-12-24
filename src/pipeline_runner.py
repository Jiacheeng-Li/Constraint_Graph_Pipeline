

"""
Pipeline Runner - Steps 1->8 Orchestrator

Purpose
- Glue the stage modules together for one (instruction, exemplar answer) pair and persist every artifact used later for prompting, graph debugging, or scoring.

Artifacts produced
- data/graphs/<sample_id>.graph.json     - serialized ConstraintGraph snapshot.
- data/graphs/<sample_id>.graph.mmd      - Mermaid visualization of the graph.
- data/instructions/<sample_id>.machine.txt - raw machine prompt from Step 7 (pre-polish).
- data/instructions/<sample_id>.prompt.txt - final machine prompt (Step 8 output or Step 7 fallback).
- data/reports/<sample_id>.eval.json     - eval protocol + meta for scoring.
- data/reports/<sample_id>.bundle.json   - full Step 7 bundle for audits.

Contracts
- Calls each step in order, passing the expected inputs/outputs documented in the step modules.
- Tracks LLM_CALL_EVENTS for observability and records which steps actually hit the API.
- Respects CLI flag `--skip-step8-polish` and env `PIPELINE_ENABLE_STEP8` to disable the polish pass.

Usage
```bash
python -m src.pipeline_runner \
    --sample-id sample_0001 \
    --instruction-file data/raw_examples/example_003_instruction.txt \
    --answer-file data/raw_examples/example_003_answer.txt
```
"""

import os
import argparse
from typing import Dict, Any
from datetime import datetime, timezone

from .utils.export_utils import write_json, write_text, save_graph_outputs
from .utils.deepseek_client import LLM_CALL_EVENTS

# Step modules
from .step1_seed_task import extract_seed_task
from .step2_segmentation import segment_response
from .step3_global_constraints import extract_global_constraints
from .step4_back_translation import extract_block_constraints
from .step5_selection_augment import generate_selection_branches
from .step6_graph_assembly import assemble_constraint_graph
from .step7_instruction_synthesis import synthesize_instruction_bundle
from .step8_prompt_refinement import refine_instruction_prompt


# ------------------------------------------------------------
# Helpers for file IO
# ------------------------------------------------------------

def _read_file(path: str) -> str:
    """Read a UTF-8 text file safely; return empty string if not found."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def _record_llm_status(status_list, step_name: str, start_index: int) -> None:
    new_events = LLM_CALL_EVENTS[start_index:]
    if not new_events:
        status = "no-llm"
    else:
        status = "success" if all(event.get("success") for event in new_events) else "failed"
    status_list.append({
        "step": step_name,
        "status": status,
        "calls": len(new_events),
    })



# ------------------------------------------------------------
# Core pipeline for one sample
# ------------------------------------------------------------

def run_pipeline_once(sample_id: str,
                       original_instruction: str,
                       model_answer: str,
                       base_data_dir: str = "data",
                       *,
                       enable_step8_polish: bool = True) -> Dict[str, Any]:
    """
    Run Steps 1 -> 8 on a single (instruction, answer) pair.

    Returns a dict with useful artifacts and file paths.
    Also writes:
        data/graphs/<sample_id>.graph.json        # Step6 graph snapshot (machine readable)
        data/graphs/<sample_id>.graph.mmd         # Step6 Mermaid visualization
        data/instructions/<sample_id>.prompt.txt  # Step8 polished machine_prompt
        data/reports/<sample_id>.eval.json        # Step7 eval_protocol (+ meta)
        data/reports/<sample_id>.bundle.json      # Full bundle (debug/forensics)
    """

    # Derive output dirs from base_data_dir according to the project layout
    graphs_dir = os.path.join(base_data_dir, "graphs")          # constraint graphs + mermaid
    instructions_dir = os.path.join(base_data_dir, "instructions")  # final machine prompts
    reports_dir = os.path.join(base_data_dir, "reports")        # eval protocol / bundle

    ts_utc = datetime.now(timezone.utc).isoformat()
    LLM_CALL_EVENTS.clear()

    llm_step_statuses = []

    # Step 1: extract seed task (core imperative task statement)
    idx_llm = len(LLM_CALL_EVENTS)
    seed_task = extract_seed_task(instruction_text=original_instruction)
    _record_llm_status(llm_step_statuses, "Step1 seed_task", idx_llm)

    # Step 2: segment the answer into ordered blocks
    idx_llm = len(LLM_CALL_EVENTS)
    segmentation = segment_response(model_answer)
    _record_llm_status(llm_step_statuses, "Step2 segmentation", idx_llm)

    # Step 3: global constraints that should apply to entire answer
    #    We now pass segmentation so the LLM can see structural outline
    #    but is STILL required (in step3 module) to ground every rule in
    #    the actual answer text, not in imagination.
    idx_llm = len(LLM_CALL_EVENTS)
    global_nodes = extract_global_constraints(
        response_text=model_answer,
        segmentation=segmentation,
    )
    _record_llm_status(llm_step_statuses, "Step3 global_constraints", idx_llm)

    # Step 4: local constraints per block (back-translation)
    idx_llm = len(LLM_CALL_EVENTS)
    step4_out = extract_block_constraints(
        segmentation=segmentation,
        seed_task=seed_task,
    )
    _record_llm_status(llm_step_statuses, "Step4 block_constraints", idx_llm)

    # Step 5: generate conditional branches / selections
    idx_llm = len(LLM_CALL_EVENTS)
    step5_out = generate_selection_branches(
        segmentation=segmentation,
        seed_task=seed_task,
        step4_output=step4_out,
    )
    _record_llm_status(llm_step_statuses, "Step5 selection_augment", idx_llm)

    # Step 6: assemble final constraint graph and save .graph.json / .graph.mmd
    graph = assemble_constraint_graph(
        seed_task=seed_task,
        segmentation=segmentation,
        global_constraints=global_nodes,
        step5_output=step5_out,
    )

    saved_paths = save_graph_outputs(
        graph,
        sample_id=sample_id,
        base_dir=graphs_dir,
    )

    total_constraints = len(graph.global_constraints) + sum(
        len(bcs.constraints) for bcs in graph.block_constraint_sets
    )
    selection_count = len(graph.selections)

    # Step 7: synthesize final instruction bundle (prompt + eval protocol)
    bundle = synthesize_instruction_bundle(graph)

    # Extract machine_prompt (to be used as the eval prompt for the target model)
    machine_prompt_raw = bundle.get("machine_prompt", "")
    raw_prompt_path = os.path.join(instructions_dir, f"{sample_id}.machine.txt")
    write_text(raw_prompt_path, machine_prompt_raw)

    idx_llm = len(LLM_CALL_EVENTS)
    polish_result = refine_instruction_prompt(
        machine_prompt=machine_prompt_raw,
        seed_task=seed_task,
        enable=enable_step8_polish,
    )
    _record_llm_status(llm_step_statuses, "Step8 prompt_refinement", idx_llm)
    machine_prompt = polish_result.get("text", machine_prompt_raw)
    bundle["machine_prompt_original"] = machine_prompt_raw
    bundle["machine_prompt"] = machine_prompt
    bundle["step8_polish"] = {k: v for k, v in polish_result.items() if k != "text"}
    prompt_length = len(machine_prompt or "")

    # Extract eval_protocol (verifier-oriented scoring spec)
    eval_protocol = bundle.get("eval_protocol", {})

    # 1) Write graph snapshot + Mermaid (already handled by save_graph_outputs)
    #    saved_paths["graph_json"], saved_paths["mermaid_mmd"]

    # 2) Write machine_prompt to data/instructions/<sample_id>.prompt.txt
    prompt_path = os.path.join(instructions_dir, f"{sample_id}.prompt.txt")
    write_text(prompt_path, machine_prompt)

    # 3) Write eval_protocol (+ meta + timestamp) to data/reports/<sample_id>.eval.json
    eval_record = {
        "sample_id": sample_id,
        "timestamp_utc": ts_utc,
        "eval_protocol": eval_protocol,
        "meta": bundle.get("meta", {}),
    }
    eval_path = os.path.join(reports_dir, f"{sample_id}.eval.json")
    write_json(eval_path, eval_record)

    # 4) Write the entire bundle (debug use) to data/reports/<sample_id>.bundle.json
    bundle_debug = {
        "sample_id": sample_id,
        "timestamp_utc": ts_utc,
        **bundle,
    }
    bundle_path = os.path.join(reports_dir, f"{sample_id}.bundle.json")
    write_json(bundle_path, bundle_debug)

    # return summary
    return {
        "seed_task": seed_task,
        "segmentation": segmentation,
        "global_constraints_count": len(global_nodes),
        "graph_paths": saved_paths,                # graph.json + mermaid.mmd
        "prompt_path": prompt_path,                # instructions/<id>.prompt.txt
        "prompt_path_machine": raw_prompt_path,    # instructions/<id>.machine.txt
        "eval_path": eval_path,                    # reports/<id>.eval.json
        "bundle_path": bundle_path,                # reports/<id>.bundle.json
        "prompt_length": prompt_length,            # char length of final prompt
        "constraint_total_count": total_constraints,
        "selection_count": selection_count,
        "bundle": bundle,
        "polish_result": polish_result,
        "llm_statuses": llm_step_statuses,
    }


# ------------------------------------------------------------
# CLI entry
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run the full pipeline (Steps 1-7) for one sample.",
    )
    parser.add_argument(
        "--sample-id",
        required=True,
        help="Identifier for this sample (used to name output files)",
    )
    parser.add_argument(
        "--instruction-file",
        required=True,
        help="Path to a text file containing the original user instruction / task request",
    )
    parser.add_argument(
        "--answer-file",
        required=True,
        help="Path to a text file containing the model's answer that we want to analyze",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Base data directory (expects subdirs: graphs/, instructions/, reports/)",
    )
    parser.add_argument(
        "--skip-step8-polish",
        action="store_true",
        help="Disable the Step 8 LLM-based polish pass (default enabled, can also set PIPELINE_ENABLE_STEP8=0).",
    )

    args = parser.parse_args()

    original_instruction = _read_file(args.instruction_file)
    model_answer = _read_file(args.answer_file)

    if not original_instruction.strip():
        raise ValueError("instruction-file is empty or missing")
    if not model_answer.strip():
        raise ValueError("answer-file is empty or missing")

    env_flag = os.getenv("PIPELINE_ENABLE_STEP8", "1").lower()
    step8_enabled_default = env_flag not in {"0", "false", "no"}
    enable_step8 = step8_enabled_default and not args.skip_step8_polish

    result = run_pipeline_once(
        sample_id=args.sample_id,
        original_instruction=original_instruction,
        model_answer=model_answer,
        base_data_dir=args.data_dir,
        enable_step8_polish=enable_step8,
    )

    # Print a short human summary to stdout
    print("===== PIPELINE DONE =====")
    print(f"sample_id                  : {args.sample_id}")
    print(f"seed_task                  : {result['seed_task']}")
    print(f"blocks                     : {len(result['segmentation'].get('blocks', []))}")
    print(f"global_constraints         : {result['global_constraints_count']}")
    print(f"total_constraints          : {result['constraint_total_count']}")
    print(f"conditional_branches       : {result['selection_count']}")
    print(f"prompt_length_chars        : {result['prompt_length']}")
    print("--- artifacts ---")
    print(f"graph_json_path            : {result['graph_paths']['graph_json']}")
    print(f"graph_mermaid_path         : {result['graph_paths']['mermaid_mmd']}")
    print(f"prompt_path_raw (step7)    : {result['prompt_path_machine']}")
    print(f"prompt_path (to eval LLM)  : {result['prompt_path']}")
    print(f"eval_protocol_path         : {result['eval_path']}")
    print(f"bundle_debug_path          : {result['bundle_path']}")
    polish_info = result.get("polish_result") or {}
    print(f"step8_polish_used          : {polish_info.get('used_llm', False)} ({polish_info.get('reason', 'n/a')})")
    print("--- LLM call status by step ---")
    for entry in result.get("llm_statuses", []):
        print(f"{entry['step']:<32}: {entry['status']} (calls={entry['calls']})")


if __name__ == "__main__":
    main()
