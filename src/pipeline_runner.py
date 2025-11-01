

"""
pipeline_runner.py

Top-level pipeline runner.
This script wires Steps 1 -> 7 together for ONE (instruction, model_answer) pair.
It produces:
    - data/graphs/<sample_id>.graph.json        (Step 6 structured constraint graph snapshot)
    - data/graphs/<sample_id>.graph.mmd         (Step 6 Mermaid visualization of the constraint graph)
    - data/instructions/<sample_id>.prompt.txt  (Step 7 machine_prompt: the final complex eval instruction)
    - data/reports/<sample_id>.eval.json        (Step 7 eval_protocol + meta for scoring)
    - data/reports/<sample_id>.bundle.json      (Full Step 7 bundle, including graph snapshot, for audit/forensics)

Assumptions / contracts between steps:
    Step1 (step1_seed_task.extract_seed_task)
        input: original_instruction (str)
        output: seed_task (str)

    Step2 (step2_segmentation.segment_response)
        input: model_answer (str)
        output: segmentation dict => {"blocks": [{block_id, intent, text_span, order_index}, ...], "order": [...]}

    Step3 (step3_global_constraints.extract_global_constraints)
        input: model_answer (str), segmentation (from step2)
        output: List[ConstraintNode]  (global scope nodes, each has verifier_spec)

    Step4 (step4_back_translation.extract_block_constraints)
        input: segmentation (from step2), seed_task (from step1)
        output: {
            "block_constraints": {block_id: [ConstraintNode,...]},
            "block_logic": {block_id: "AND" | "sub-chain"}
        }

    Step5 (step5_selection_augment.generate_selection_branches)
        input:
            seed_task (str)
            segmentation (from step2)
            step4_output (dict with block_constraints + block_logic)
        output:
            {
                "block_constraints": {block_id: [...]},      # may be augmented with new nodes
                "block_logic": {block_id: logic},
                "selections": [SelectionNode,...]
            }

    Step6 (step6_graph_assembly.assemble_constraint_graph)
        input: seed_task, segmentation, global_constraints (step3), step5_output
        output: ConstraintGraph object
        plus we can call save_graph_outputs(...) to write .graph.json and .graph.mmd

    Step7 (step7_instruction_synthesis.synthesize_instruction_bundle)
        input: graph (ConstraintGraph)
        output: {
            "machine_prompt": str,
            "eval_protocol": dict,
            "graph_serialized": dict,
            "meta": dict,
        }
        We persist this bundle as <sample_id>.bundle.json for convenience.

IMPORTANT:
- We assume any LLM calls / verifiers required by step1..5 have already been implemented
  using DeepSeek etc. in those modules.
- Runner itself does not talk to LLM. It only orchestrates.

USAGE (conceptually):
    python pipeline_runner.py \
        --sample-id sample_0001 \
        --instruction-file data/raw/instruction.txt \
        --answer-file data/raw/answer.txt

This runner is intentionally simple: one sample in, one batch of artifacts out.
You can wrap this later for multi-sample generation.
"""

import os
import argparse
from typing import Dict, Any
from datetime import datetime, timezone

from .utils.export_utils import write_json, write_text, save_graph_outputs

# Step modules
from .step1_seed_task import extract_seed_task
from .step2_segmentation import segment_response
from .step3_global_constraints import extract_global_constraints
from .step4_back_translation import extract_block_constraints
from .step5_selection_augment import generate_selection_branches
from .step6_graph_assembly import assemble_constraint_graph
from .step7_instruction_synthesis import synthesize_instruction_bundle


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



# ------------------------------------------------------------
# Core pipeline for one sample
# ------------------------------------------------------------

def run_pipeline_once(sample_id: str,
                       original_instruction: str,
                       model_answer: str,
                       base_data_dir: str = "data") -> Dict[str, Any]:
    """
    Run Steps 1 -> 7 on a single (instruction, answer) pair.

    Returns a dict with useful artifacts and file paths.
    Also writes:
        data/graphs/<sample_id>.graph.json        # Step6 graph snapshot (machine readable)
        data/graphs/<sample_id>.graph.mmd         # Step6 Mermaid visualization
        data/instructions/<sample_id>.prompt.txt  # Step7 machine_prompt (final eval prompt)
        data/reports/<sample_id>.eval.json        # Step7 eval_protocol (+ meta)
        data/reports/<sample_id>.bundle.json      # Full Step7 bundle (debug/forensics)
    """

    # Derive output dirs from base_data_dir according to the project layout
    graphs_dir = os.path.join(base_data_dir, "graphs")          # constraint graphs + mermaid
    instructions_dir = os.path.join(base_data_dir, "instructions")  # final machine prompts
    reports_dir = os.path.join(base_data_dir, "reports")        # eval protocol / bundle

    ts_utc = datetime.now(timezone.utc).isoformat()

    # Step 1: extract seed task (core imperative task statement)
    seed_task = extract_seed_task(instruction_text=original_instruction)

    # Step 2: segment the answer into ordered blocks
    segmentation = segment_response(model_answer)

    # Step 3: global constraints that should apply to entire answer
    #    We now pass segmentation so the LLM can see structural outline
    #    but is STILL required (in step3 module) to ground every rule in
    #    the actual answer text, not in imagination.
    global_nodes = extract_global_constraints(
        response_text=model_answer,
        segmentation=segmentation,
    )

    # Step 4: local constraints per block (back-translation)
    step4_out = extract_block_constraints(
        segmentation=segmentation,
        seed_task=seed_task,
    )

    # Step 5: generate conditional branches / selections
    step5_out = generate_selection_branches(
        segmentation=segmentation,
        seed_task=seed_task,
        step4_output=step4_out,
    )

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

    # Step 7: synthesize final instruction bundle (prompt + eval protocol)
    bundle = synthesize_instruction_bundle(graph)

    # Extract machine_prompt (to be used as the eval prompt for the target model)
    machine_prompt = bundle.get("machine_prompt", "")

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
        "eval_path": eval_path,                    # reports/<id>.eval.json
        "bundle_path": bundle_path,                # reports/<id>.bundle.json
        "bundle": bundle,
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

    args = parser.parse_args()

    original_instruction = _read_file(args.instruction_file)
    model_answer = _read_file(args.answer_file)

    if not original_instruction.strip():
        raise ValueError("instruction-file is empty or missing")
    if not model_answer.strip():
        raise ValueError("answer-file is empty or missing")

    result = run_pipeline_once(
        sample_id=args.sample_id,
        original_instruction=original_instruction,
        model_answer=model_answer,
        base_data_dir=args.data_dir,
    )

    # Print a short human summary to stdout
    print("===== PIPELINE DONE =====")
    print(f"sample_id                  : {args.sample_id}")
    print(f"seed_task                  : {result['seed_task']}")
    print(f"blocks                     : {len(result['segmentation'].get('blocks', []))}")
    print(f"global_constraints         : {result['global_constraints_count']}")
    print("--- artifacts ---")
    print(f"graph_json_path            : {result['graph_paths']['graph_json']}")
    print(f"graph_mermaid_path         : {result['graph_paths']['mermaid_mmd']}")
    print(f"prompt_path (to eval LLM)  : {result['prompt_path']}")
    print(f"eval_protocol_path         : {result['eval_path']}")
    print(f"bundle_debug_path          : {result['bundle_path']}")


if __name__ == "__main__":
    main()
