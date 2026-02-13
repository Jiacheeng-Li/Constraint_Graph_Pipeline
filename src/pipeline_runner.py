

"""
Pipeline Runner - Steps 1->8 Orchestrator

Purpose
- Glue the stage modules together for one (instruction, exemplar answer) pair and persist every artifact used later for prompting, graph debugging, or scoring.
- Optionally runs the graph augmenter to emit curriculum/multi-turn variants.
- Optionally runs Step 6.5 + 7.5 to render template prompts from augmented graphs.
- Optionally runs Step 8.5 to produce diversified natural-language prompt variants.

Artifacts produced
- data/graphs/<sample_id>.graph.json     - serialized ConstraintGraph snapshot.
- data/graphs/<sample_id>.graph.mmd      - Mermaid visualization of the graph.
- data/instructions/<sample_id>.machine.txt - raw machine prompt from Step 7 (pre-polish).
- data/instructions/<sample_id>.prompt.txt - final machine prompt (Step 8 output or Step 7 fallback).
- data/reports/<sample_id>.eval.json     - eval protocol + meta for scoring.
- data/reports/<sample_id>.bundle.json   - full Step 7 bundle for audits.
- data/reports/<sample_id>.evidence.json - normalized evidence metadata (when provided).

Contracts
- Calls each step in order, passing the expected inputs/outputs documented in the step modules.
- Tracks LLM_CALL_EVENTS for observability and records which steps actually hit the API.
- Respects CLI flag `--skip-step8-polish` and env `PIPELINE_ENABLE_STEP8` to disable the polish pass.
- Step 7.5 is off by default; enable via CLI or env `PIPELINE_ENABLE_STEP7_5`.
- Step 8.5 is off by default; enable via CLI or env `PIPELINE_ENABLE_STEP8_5`.
- Augmentation is off by default; enable via CLI `--enable-augment` or env `PIPELINE_ENABLE_AUGMENT`.
- Augment + diversity is off by default; enable via CLI `--enable-augment-diversity` or env `PIPELINE_ENABLE_AUGMENT_DIVERSITY`.

Usage
```bash
python -m src.pipeline_runner \
    --sample-id sample_0001 \
    --instruction-file data/raw_examples/example_003_instruction.txt \
    --answer-file data/raw_examples/example_003_answer.txt
```
"""

import os
import re
import json
import random
import argparse
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

from .utils.export_utils import write_json, write_text, save_graph_outputs
from .utils.deepseek_client import LLM_CALL_EVENTS

# Step modules
from .step1_seed_task import extract_seed_task
from .step2_segmentation import segment_response
from .step3_global_constraints import extract_global_constraints
from .step3_5_knowledge_constraints import extract_knowledge_constraints
from .step4_back_translation import extract_block_constraints
from .step5_selection_augment import generate_selection_branches
from .step6_graph_assembly import assemble_constraint_graph
from .step7_instruction_synthesis import synthesize_instruction_bundle
from .step8_prompt_refinement import refine_instruction_prompt
from .step7_5_prompt_renderer import render_prompt_variant
from .step6_5_graph_augment import augment_graphs_only
from .graph_augmenter import augment_graph, _graph_from_serialized
from .step8_5_prompt_diversification import diversify_instruction_prompt
from .graph_schema import ConstraintNode


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


def _read_json(path: str) -> Dict[str, Any]:
    """Read a UTF-8 JSON file safely; return empty dict if missing/invalid."""
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _extract_evidence_ref_map(evidence: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Normalize evidence.json into a {ref_id: payload} map.
    Supported shapes:
      1) {"references": {"b0": {...}, ...}, ...}
      2) {"b0": {...}, "b1": {...}, ...}
    """
    if not isinstance(evidence, dict) or not evidence:
        return {}

    refs = evidence.get("references")
    if isinstance(refs, dict):
        return {str(k): v for k, v in refs.items() if isinstance(v, dict)}

    if all(isinstance(v, dict) for v in evidence.values()):
        if any(("title" in v or "doi" in v or "pdfUrl" in v) for v in evidence.values()):
            return {str(k): v for k, v in evidence.items() if isinstance(v, dict)}
    return {}


def _build_evidence_citation_constraints(
    evidence: Dict[str, Any],
    answer_text: str,
    start_index: int,
) -> list[ConstraintNode]:
    """
    Build explicit citation-driven global constraints from evidence metadata.
    """
    ref_map = _extract_evidence_ref_map(evidence)
    if not ref_map:
        return []

    ref_ids = sorted(str(rid) for rid in ref_map.keys())
    ref_count = len(ref_ids)
    if ref_count <= 0:
        return []

    min_unique = max(1, min(ref_count, max(3, min(12, int(round(ref_count * 0.12))))))
    min_markers = max(min_unique, min(30, max(4, int(round(ref_count * 0.2)))))

    has_ref_id = bool(re.search(r"\[[A-Za-z][A-Za-z0-9_-]*\d+\]", answer_text or ""))
    has_numeric = bool(re.search(r"\[\s*\d{1,4}(?:\s*[-,;]\s*\d{1,4})*\s*\]", answer_text or ""))
    has_author_year = bool(
        re.search(r"\([A-Z][A-Za-z.\-]+(?:\s+et\s+al\.)?,\s*\d{4}[a-z]?\)", answer_text or "")
    )

    inferred_style = None
    if has_ref_id:
        inferred_style = "ref_id"
    elif has_numeric and not has_author_year:
        inferred_style = "numeric"
    elif has_author_year and not has_numeric:
        inferred_style = "author_year"

    nodes: list[ConstraintNode] = []
    next_idx = start_index

    nodes.append(
        ConstraintNode(
            cid=f"G{next_idx}",
            desc=(
                f"Support core claims with inline citations and include at least {min_markers} "
                "citation markers across the full response."
            ),
            scope="global",
            verifier_spec={"check": "min_citation_markers", "args": {"min_count": min_markers}},
            trace_to=None,
            derived_from="evidence",
        )
    )
    next_idx += 1

    nodes.append(
        ConstraintNode(
            cid=f"G{next_idx}",
            desc=f"Use at least {min_unique} distinct cited sources in the response.",
            scope="global",
            verifier_spec={"check": "min_distinct_citations", "args": {"min_unique": min_unique}},
            trace_to=None,
            derived_from="evidence",
        )
    )
    next_idx += 1

    if inferred_style:
        nodes.append(
            ConstraintNode(
                cid=f"G{next_idx}",
                desc=(
                    "Keep a consistent citation style throughout the response "
                    f"(detected style: {inferred_style})."
                ),
                scope="global",
                verifier_spec={"check": "citation_style", "args": {"style": inferred_style}},
                trace_to=None,
                derived_from="evidence",
            )
        )
        next_idx += 1

    if has_ref_id:
        nodes.append(
            ConstraintNode(
                cid=f"G{next_idx}",
                desc=(
                    "When using ref-id citations, only cite ref_ids present in the evidence set "
                    "(for example [b12])."
                ),
                scope="global",
                verifier_spec={
                    "check": "citation_refs_from_allowed_set",
                    "args": {
                        "allowed_ref_ids": ref_ids,
                        "min_unique": min_unique,
                        "max_out_of_set": 0,
                    },
                },
                trace_to=None,
                derived_from="evidence",
            )
        )

    return nodes


def _renumber_global_constraint_ids(nodes: list[ConstraintNode]) -> None:
    """Normalize global constraint ids to G1..Gn in current list order."""
    for idx, node in enumerate(nodes, start=1):
        node.cid = f"G{idx}"


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


def _strip_think_prefix(text: str) -> tuple[str, bool]:
    """
    Qwen 等模型可能在正文前输出 <think>...</think>。仅保留 </think> 之后的内容。
    Returns (clean_text, stripped_flag).
    """
    if not text:
        return text, False
    marker = "</think>"
    if marker in text:
        _, tail = text.split(marker, 1)
        return tail.lstrip(), True
    return text, False


def _guard_explicit_references(
    *,
    machine_prompt_raw: str,
    polished_prompt: str,
    explicit_refs: List[Dict[str, Any]],
) -> tuple[str, Dict[str, Any]]:
    """
    Keep explicit reference title/url pairs intact after Step8 polish.
    If any title/url is missing from polished output, fallback to machine_prompt_raw.
    """
    refs = explicit_refs or []
    if not refs:
        return polished_prompt, {"enabled": False, "passed": True, "fallback_used": False, "missing": []}

    missing: List[str] = []
    text = polished_prompt or ""
    for ref in refs:
        rid = str(ref.get("ref_id") or "").strip()
        title = str(ref.get("title") or "").strip()
        url = str(ref.get("url") or "").strip()
        if title and title not in text:
            missing.append(f"{rid}:title")
        if url and url not in text:
            missing.append(f"{rid}:url")

    if missing:
        return machine_prompt_raw, {
            "enabled": True,
            "passed": False,
            "fallback_used": True,
            "missing": missing,
            "explicit_ref_count": len(refs),
        }

    return polished_prompt, {
        "enabled": True,
        "passed": True,
        "fallback_used": False,
        "missing": [],
        "explicit_ref_count": len(refs),
    }


def _strip_explicit_reference_list_for_step8(
    machine_prompt: str,
) -> tuple[str, Dict[str, Any]]:
    """
    Remove the explicit reference list block before sending prompt to Step8 LLM.
    """
    if not machine_prompt:
        return machine_prompt, {"enabled": True, "removed_lines": 0}

    lines = machine_prompt.splitlines()
    out: List[str] = []
    in_list = False
    removed = 0
    replaced_header = False

    for line in lines:
        stripped = line.strip()
        if not in_list and stripped.startswith("- Required evidence references"):
            in_list = True
            removed += 1
            out.append("   - Use the fixed reference list provided at the end of this instruction.")
            replaced_header = True
            continue

        if in_list:
            if stripped.startswith("•"):
                removed += 1
                continue
            if not stripped:
                # skip blank lines immediately after list bullets
                removed += 1
                continue
            in_list = False

        out.append(line)

    result = "\n".join(out).strip() + "\n"
    return result, {
        "enabled": True,
        "removed_lines": removed,
        "replaced_header": replaced_header,
    }


def _append_explicit_reference_list_postfix(
    prompt_text: str,
    explicit_refs: List[Dict[str, Any]],
) -> str:
    """
    Append immutable explicit reference list block after Step8.
    """
    refs = explicit_refs or []
    if not refs:
        return prompt_text

    lines: List[str] = []
    lines.append("Reference List:")
    for ref in refs:
        rid = str(ref.get("ref_id") or "").strip()
        title = str(ref.get("title") or "").strip() or "(untitled)"
        url = str(ref.get("url") or "").strip()
        if rid and url:
            lines.append(f"- [{rid}] {title} | {url}")
        elif rid:
            lines.append(f"- [{rid}] {title}")
        elif url:
            lines.append(f"- {title} | {url}")
        else:
            lines.append(f"- {title}")
    return (prompt_text or "").rstrip() + "\n\n" + "\n".join(lines) + "\n"


def _slugify_style(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_")
    return slug or "variant"



# ------------------------------------------------------------
# Core pipeline for one sample
# ------------------------------------------------------------

def run_pipeline_once(sample_id: str,
                       original_instruction: str,
                       model_answer: str,
                       base_data_dir: str = "data",
                       evidence: Optional[Dict[str, Any]] = None,
                       survey_mode: bool = False,
                       knowledge_style: str = "abstract",
                       generation_granularity: str = "section",
                       survey_block_mode: str = "heading",
                       explicit_reference_strategy: str = "in_prompt_guarded",
                       *,
                       enable_step8_polish: bool = True,
                       enable_step7_5: bool = False,
                       step7_5_templates: Optional[list[str]] = None,
                       step7_5_limit: Optional[int] = None,
                       step7_5_seed: Optional[int] = None,
                       step7_5_heuristic_ratio: float = 0.5,
                       enable_augment: bool = False,
                       enable_augment_diversity: bool = False,
                       augment_seed: int = 0,
                       augment_priority_ratio: float = 0.5,
                       augment_curriculum: bool = True,
                       augment_m1: bool = True,
                       augment_m2: bool = True,
                       enable_step8_5: bool = False,
                       step8_5_variants: int = 3,
                       step8_5_seed: Optional[int] = None,
                       step8_5_styles: Optional[list[str]] = None) -> Dict[str, Any]:
    """
    Run Steps 1 -> 8 on a single (instruction, answer) pair.

    Returns a dict with useful artifacts and file paths.
    Also writes:
        data/graphs/<sample_id>.graph.json        # Step6 graph snapshot (machine readable)
        data/graphs/<sample_id>.graph.mmd         # Step6 Mermaid visualization
        data/instructions/<sample_id>.prompt.txt  # Step8 polished machine_prompt
        data/reports/<sample_id>.eval.json        # Step7 eval_protocol (+ meta)
        data/reports/<sample_id>.bundle.json      # Full bundle (debug/forensics)
        data/reports/<sample_id>.evidence.json    # Evidence copy (optional)
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

    knowledge_style_norm = (knowledge_style or "abstract").strip().lower()
    if knowledge_style_norm not in {"abstract", "explicit"}:
        knowledge_style_norm = "abstract"
    granularity_norm = (generation_granularity or "section").strip().lower()
    if granularity_norm not in {"section", "whole"}:
        granularity_norm = "section"
    survey_block_mode_norm = (survey_block_mode or "heading").strip().lower()
    if survey_block_mode_norm not in {"heading", "paragraph", "llm"}:
        survey_block_mode_norm = "heading"
    explicit_reference_strategy_norm = (explicit_reference_strategy or "in_prompt_guarded").strip().lower()
    if explicit_reference_strategy_norm not in {"in_prompt_guarded", "postfix_after_step8"}:
        explicit_reference_strategy_norm = "in_prompt_guarded"

    # Step 2: segment the answer into ordered blocks
    idx_llm = len(LLM_CALL_EVENTS)
    if granularity_norm == "whole":
        segmentation = {
            "blocks": [
                {
                    "block_id": "B1",
                    "intent": "Introduction",
                    "text_span": model_answer.strip(),
                    "order_index": 0,
                }
            ],
            "order": ["B1"],
        }
    else:
        segmentation = segment_response(
            model_answer,
            survey_mode=survey_mode,
            survey_block_mode=survey_block_mode_norm,
        )
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
    if evidence:
        citation_nodes = _build_evidence_citation_constraints(
            evidence=evidence,
            answer_text=model_answer,
            start_index=len(global_nodes) + 1,
        )
        if citation_nodes:
            global_nodes.extend(citation_nodes)
    _record_llm_status(llm_step_statuses, "Step3 global_constraints", idx_llm)

    # Step 4: local constraints per block (back-translation)
    idx_llm = len(LLM_CALL_EVENTS)
    if granularity_norm == "whole":
        step4_out = {
            "block_constraints": {},
            "block_logic": {},
        }
    else:
        step4_out = extract_block_constraints(
            segmentation=segmentation,
            seed_task=seed_task,
        )
    _record_llm_status(llm_step_statuses, "Step4 block_constraints", idx_llm)

    knowledge_out: Dict[str, Any] = {"global_constraints": [], "block_constraints": {}, "meta": {"enabled": False}}
    if survey_mode and evidence:
        idx_llm = len(LLM_CALL_EVENTS)
        knowledge_out = extract_knowledge_constraints(
            segmentation=segmentation,
            evidence=evidence,
            survey_mode=True,
            knowledge_style=knowledge_style_norm,
            generation_granularity=granularity_norm,
        )
        global_nodes.extend(knowledge_out.get("global_constraints", []))
        if granularity_norm != "whole":
            for bid, nodes in (knowledge_out.get("block_constraints") or {}).items():
                step4_out.setdefault("block_constraints", {}).setdefault(bid, [])
                step4_out["block_constraints"][bid].extend(nodes)
                step4_out.setdefault("block_logic", {}).setdefault(bid, "AND")
        _record_llm_status(llm_step_statuses, "Step3.5 knowledge_constraints", idx_llm)

    _renumber_global_constraint_ids(global_nodes)

    # Step 5: generate conditional branches / selections
    idx_llm = len(LLM_CALL_EVENTS)
    if granularity_norm == "whole" or survey_mode:
        step5_out = {
            "block_constraints": step4_out.get("block_constraints", {}),
            "block_logic": step4_out.get("block_logic", {}),
            "selections": [],
            "extra_blocks": [],
        }
    else:
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
    if evidence:
        survey_meta = evidence.get("survey", {}) if isinstance(evidence, dict) else {}
        graph.meta["evidence"] = {
            "provided": True,
            "reference_count": len(_extract_evidence_ref_map(evidence)),
            "title": str(survey_meta.get("title", "")).strip(),
            "corpusId": str(survey_meta.get("corpusId", "")).strip(),
        }
    if survey_mode:
        graph.meta["survey_mode"] = True
        graph.meta["prompt_profile"] = "survey"
        graph.meta["knowledge_style"] = knowledge_style_norm
        graph.meta["explicit_reference_strategy"] = explicit_reference_strategy_norm
        graph.meta["generation_granularity"] = granularity_norm
        graph.meta["survey_block_mode"] = survey_block_mode_norm
        graph.meta["knowledge_constraints"] = knowledge_out.get("meta", {})

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

    explicit_refs = ((knowledge_out.get("meta") or {}).get("explicit_reference_list") or [])
    step8_input_prompt = machine_prompt_raw
    explicit_postfix_meta: Dict[str, Any] = {"enabled": False}
    if (
        survey_mode
        and knowledge_style_norm == "explicit"
        and explicit_reference_strategy_norm == "postfix_after_step8"
        and explicit_refs
    ):
        step8_input_prompt, strip_meta = _strip_explicit_reference_list_for_step8(machine_prompt_raw)
        explicit_postfix_meta = {
            "enabled": True,
            "strip_meta": strip_meta,
        }

    idx_llm = len(LLM_CALL_EVENTS)
    polish_result = refine_instruction_prompt(
        machine_prompt=step8_input_prompt,
        seed_task=seed_task,
        enable=enable_step8_polish,
    )
    _record_llm_status(llm_step_statuses, "Step8 prompt_refinement", idx_llm)
    machine_prompt = polish_result.get("text", step8_input_prompt)
    machine_prompt, stripped_think = _strip_think_prefix(machine_prompt)
    polish_result["stripped_think"] = stripped_think

    explicit_guard_meta: Dict[str, Any] = {"enabled": False, "passed": True, "fallback_used": False, "missing": []}
    if (
        survey_mode
        and knowledge_style_norm == "explicit"
        and explicit_reference_strategy_norm == "in_prompt_guarded"
    ):
        machine_prompt, explicit_guard_meta = _guard_explicit_references(
            machine_prompt_raw=machine_prompt_raw,
            polished_prompt=machine_prompt,
            explicit_refs=explicit_refs,
        )
        polish_result["explicit_reference_guard"] = explicit_guard_meta
        if explicit_guard_meta.get("fallback_used"):
            polish_result["reason"] = "explicit_reference_guard_fallback"
            polish_result["used_llm"] = False
    elif (
        survey_mode
        and knowledge_style_norm == "explicit"
        and explicit_reference_strategy_norm == "postfix_after_step8"
    ):
        machine_prompt = _append_explicit_reference_list_postfix(machine_prompt, explicit_refs)
        polish_result["explicit_reference_postfix"] = {
            **explicit_postfix_meta,
            "reference_count": len(explicit_refs),
        }

    bundle["machine_prompt_original"] = machine_prompt_raw
    bundle["machine_prompt"] = machine_prompt
    bundle["step8_polish"] = {k: v for k, v in polish_result.items() if k != "text"}
    prompt_length = len(machine_prompt or "")

    step7_5_result = {}
    step7_5_variants_written = []
    if enable_step7_5:
        idx_llm = len(LLM_CALL_EVENTS)
        step7_5_result = render_prompt_variant(
            graph,
            template_pool=step7_5_templates,
            template_seed=step7_5_seed,
            template_limit=step7_5_limit,
            heuristic_ratio=step7_5_heuristic_ratio,
            selection_key=sample_id,
        )
        variant = step7_5_result.get("variant", {})
        machine_variant = (variant.get("machine_prompt") or "").strip()
        if machine_variant:
            template_key = _slugify_style(variant.get("template", "template"))
            machine_path = os.path.join(
                instructions_dir,
                f"{sample_id}.machine.tmpl_{template_key}.txt",
            )
            write_text(machine_path, machine_variant)

            if enable_step8_polish:
                polish_variant = refine_instruction_prompt(
                    machine_prompt=machine_variant,
                    seed_task=seed_task,
                    enable=True,
                )
                prompt_text = polish_variant.get("text", machine_variant)
            else:
                polish_variant = {
                    "used_llm": False,
                    "reason": "disabled",
                    "attempts": 0,
                }
                prompt_text = machine_variant

            prompt_text, stripped_think = _strip_think_prefix(prompt_text)
            polish_variant["stripped_think"] = stripped_think
            prompt_path = os.path.join(
                instructions_dir,
                f"{sample_id}.prompt.tmpl_{template_key}.txt",
            )
            write_text(prompt_path, prompt_text)
            step7_5_variants_written.append({
                "template": variant.get("template"),
                "label": variant.get("label"),
                "description": variant.get("description"),
                "machine_path": machine_path,
                "prompt_path": prompt_path,
                "prompt_length_chars": len(prompt_text),
                "step8_polish": {k: v for k, v in polish_variant.items() if k != "text"},
            })
        _record_llm_status(llm_step_statuses, "Step7.5 prompt_templates", idx_llm)

    if step7_5_result:
        bundle["step7_5_templates"] = {
            "meta": {
                "enabled": enable_step7_5,
                "template_pool": step7_5_result.get("selection", {}).get("template_pool"),
                "unknown_templates": step7_5_result.get("selection", {}).get("unknown_templates"),
                "template_seed": step7_5_seed,
                "template_limit": step7_5_limit,
                "heuristic_ratio": step7_5_heuristic_ratio,
                "selected_template": step7_5_result.get("selection", {}).get("template"),
                "selected_by": step7_5_result.get("selection", {}).get("selected_by"),
                "heuristic_template": step7_5_result.get("selection", {}).get("heuristic_template"),
                "step8_enabled": enable_step8_polish,
            },
            "variants": step7_5_variants_written,
        }

    augment_outputs: list[str] = []
    augment_diversity_outputs: list[Dict[str, Any]] = []
    if enable_augment:
        idx_llm = len(LLM_CALL_EVENTS)
        rng = random.Random(augment_seed)
        run_meta = {
            "seed": augment_seed,
            "priority_ratio": augment_priority_ratio,
            "enable_curriculum": augment_curriculum,
            "enable_m1": augment_m1,
            "enable_m2": augment_m2,
            "enable_step8": enable_step8_polish,
        }
        augment_outputs = augment_graph(
            graph=graph,
            sample_id=sample_id,
            base_dir=base_data_dir,
            rng=rng,
            priority_ratio=augment_priority_ratio,
            enable_curriculum=augment_curriculum,
            enable_m1=augment_m1,
            enable_m2=augment_m2,
            enable_step8=enable_step8_polish,
            run_id=ts_utc,
            run_meta=run_meta,
        )
        _record_llm_status(llm_step_statuses, "Graph augmenter", idx_llm)

    if enable_augment_diversity:
        idx_llm = len(LLM_CALL_EVENTS)
        rng = random.Random(augment_seed)
        augmented_entries = augment_graphs_only(
            graph=graph,
            sample_id=sample_id,
            base_dir=base_data_dir,
            rng=rng,
            priority_ratio=augment_priority_ratio,
            enable_curriculum=augment_curriculum,
            enable_m1=augment_m1,
            enable_m2=augment_m2,
        )
        for entry in augmented_entries:
            graph_path = (entry.get("paths") or {}).get("graph_json")
            if not graph_path:
                continue
            with open(graph_path, "r", encoding="utf-8") as f:
                graph_data = json.load(f)
            aug_graph = _graph_from_serialized(graph_data)
            template_result = render_prompt_variant(
                aug_graph,
                template_pool=step7_5_templates,
                template_seed=step7_5_seed,
                template_limit=step7_5_limit,
                heuristic_ratio=step7_5_heuristic_ratio,
                selection_key=entry.get("sample_id", ""),
            )
            variant = template_result.get("variant", {})
            machine_variant = (variant.get("machine_prompt") or "").strip()
            if not machine_variant:
                continue
            template_key = _slugify_style(variant.get("template", "template"))
            variant_sample_id = entry.get("sample_id") or sample_id
            machine_path = os.path.join(
                instructions_dir,
                f"{variant_sample_id}.machine.tmpl_{template_key}.txt",
            )
            write_text(machine_path, machine_variant)

            if enable_step8_polish:
                polish_variant = refine_instruction_prompt(
                    machine_prompt=machine_variant,
                    seed_task=aug_graph.seed_task,
                    enable=True,
                )
                prompt_text = polish_variant.get("text", machine_variant)
            else:
                polish_variant = {
                    "used_llm": False,
                    "reason": "disabled",
                    "attempts": 0,
                }
                prompt_text = machine_variant

            prompt_text, stripped_think = _strip_think_prefix(prompt_text)
            polish_variant["stripped_think"] = stripped_think
            prompt_path = os.path.join(
                instructions_dir,
                f"{variant_sample_id}.prompt.tmpl_{template_key}.txt",
            )
            write_text(prompt_path, prompt_text)
            augment_diversity_outputs.append({
                "sample_id": variant_sample_id,
                "template": variant.get("template"),
                "selected_by": template_result.get("selection", {}).get("selected_by"),
                "machine_path": machine_path,
                "prompt_path": prompt_path,
                "prompt_length_chars": len(prompt_text),
                "step8_polish": {k: v for k, v in polish_variant.items() if k != "text"},
            })
        _record_llm_status(llm_step_statuses, "Step6.5/7.5 augmented templates", idx_llm)

    step8_5_result = {}
    step8_5_variants_written = []
    if enable_step8_5 and step8_5_variants > 0:
        idx_llm = len(LLM_CALL_EVENTS)
        step8_5_result = diversify_instruction_prompt(
            machine_prompt=machine_prompt_raw,
            seed_task=seed_task,
            enable=enable_step8_5,
            num_variants=step8_5_variants,
            style_seed=step8_5_seed,
            style_pool=step8_5_styles,
        )
        _record_llm_status(llm_step_statuses, "Step8.5 prompt_diversification", idx_llm)

        used_slugs: Dict[str, int] = {}
        for variant in step8_5_result.get("variants", []):
            text = (variant.get("text") or "").strip()
            if not text:
                continue
            text, stripped_think = _strip_think_prefix(text)
            style_slug = variant.get("style_slug") or _slugify_style(variant.get("style", "variant"))
            used_slugs[style_slug] = used_slugs.get(style_slug, 0) + 1
            suffix = f"_{used_slugs[style_slug]}" if used_slugs[style_slug] > 1 else ""
            file_slug = f"{style_slug}{suffix}"
            variant_path = os.path.join(instructions_dir, f"{sample_id}.prompt.{file_slug}.txt")
            write_text(variant_path, text)
            step8_5_variants_written.append({
                "style": variant.get("style"),
                "style_slug": file_slug,
                "path": variant_path,
                "prompt_length_chars": len(text),
                "used_llm": variant.get("used_llm", False),
                "reason": variant.get("reason", ""),
                "attempts": variant.get("attempts", 0),
                "stripped_think": stripped_think,
            })

    if step8_5_result:
        bundle["step8_5_diversify"] = {
            "meta": {
                "enabled": enable_step8_5,
                "requested_variants": step8_5_result.get("requested_variants"),
                "generated_variants": len(step8_5_variants_written),
                "style_seed": step8_5_result.get("style_seed"),
                "style_pool": step8_5_result.get("style_pool"),
                "unknown_styles": step8_5_result.get("unknown_styles", []),
                "result_reason": step8_5_result.get("reason"),
            },
            "variants": step8_5_variants_written,
        }

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

    evidence_path = ""
    if evidence:
        evidence_path = os.path.join(reports_dir, f"{sample_id}.evidence.json")
        write_json(evidence_path, evidence)

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
        "evidence_path": evidence_path,            # reports/<id>.evidence.json
        "prompt_length": prompt_length,            # char length of final prompt
        "constraint_total_count": total_constraints,
        "selection_count": selection_count,
        "bundle": bundle,
        "polish_result": polish_result,
        "step7_5_result": step7_5_result,
        "step7_5_variants": step7_5_variants_written,
        "augment_outputs": augment_outputs,
        "augment_diversity_outputs": augment_diversity_outputs,
        "step8_5_result": step8_5_result,
        "step8_5_variants": step8_5_variants_written,
        "llm_statuses": llm_step_statuses,
        "evidence_loaded": bool(evidence),
        "evidence_reference_count": len(_extract_evidence_ref_map(evidence)) if evidence else 0,
        "survey_mode": survey_mode,
        "knowledge_style": knowledge_style_norm,
        "explicit_reference_strategy": explicit_reference_strategy_norm,
        "generation_granularity": granularity_norm,
        "survey_block_mode": survey_block_mode_norm,
        "knowledge_constraints": knowledge_out.get("meta", {}),
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
        "--evidence-file",
        default="",
        help="Optional evidence.json path. If provided, injects citation-driven global constraints.",
    )
    parser.add_argument(
        "--survey-mode",
        action="store_true",
        help="Enable survey-specific behavior (heading-based segmentation + knowledge constraints + survey prompt profile).",
    )
    parser.add_argument(
        "--knowledge-style",
        type=str,
        choices=["abstract", "explicit"],
        default="abstract",
        help="How to render knowledge constraints in instruction: abstract guidance or explicit source list.",
    )
    parser.add_argument(
        "--generation-granularity",
        type=str,
        choices=["section", "whole"],
        default="section",
        help="Constraint granularity: section uses block constraints; whole uses global-only constraints.",
    )
    parser.add_argument(
        "--survey-block-mode",
        type=str,
        choices=["heading", "paragraph", "llm"],
        default="heading",
        help="Survey segmentation mode when generation-granularity=section.",
    )
    parser.add_argument(
        "--explicit-reference-strategy",
        type=str,
        choices=["in_prompt_guarded", "postfix_after_step8"],
        default="in_prompt_guarded",
        help=(
            "How to handle explicit reference list around Step8. "
            "in_prompt_guarded keeps list in Step8 input and falls back on mismatch; "
            "postfix_after_step8 strips list before Step8 and appends canonical list after polish."
        ),
    )
    parser.add_argument(
        "--skip-step8-polish",
        action="store_true",
        help="Disable the Step 8 LLM-based polish pass (default enabled, can also set PIPELINE_ENABLE_STEP8=0).",
    )
    parser.add_argument(
        "--enable-augment",
        action="store_true",
        help="Enable curriculum/multi-turn graph augmentation (graph_augmenter).",
    )
    parser.add_argument(
        "--enable-augment-diversity",
        action="store_true",
        help="Enable graph-only augmentation + template rendering (Step6.5 -> Step7.5 -> Step8).",
    )
    parser.add_argument(
        "--augment-mode",
        type=str,
        default="auto",
        choices=["auto", "none", "augment", "augment_diversity", "both"],
        help="Override augmentation selection (auto uses --enable-augment/--enable-augment-diversity + env).",
    )
    parser.add_argument(
        "--augment-seed",
        type=int,
        default=0,
        help="Random seed for augmentation steps.",
    )
    parser.add_argument(
        "--augment-priority-ratio",
        type=float,
        default=0.5,
        help="Probability of priority injection during augmentation.",
    )
    parser.add_argument(
        "--augment-disable-curriculum",
        action="store_true",
        help="Disable curriculum generation during augmentation.",
    )
    parser.add_argument(
        "--augment-disable-m1",
        action="store_true",
        help="Disable M1 multi-turn generation during augmentation.",
    )
    parser.add_argument(
        "--augment-disable-m2",
        action="store_true",
        help="Disable M2 multi-turn generation during augmentation.",
    )
    parser.add_argument(
        "--enable-step7-5",
        action="store_true",
        help="Enable Step 7.5 multi-template rendering (default disabled; can also set PIPELINE_ENABLE_STEP7_5=1).",
    )
    parser.add_argument(
        "--step7-5-templates",
        type=str,
        default="",
        help="Comma-separated template names for Step 7.5 (empty means default pool).",
    )
    parser.add_argument(
        "--step7-5-limit",
        type=int,
        default=-1,
        help="Limit the number of Step 7.5 templates to render (negative means no limit).",
    )
    parser.add_argument(
        "--step7-5-seed",
        type=int,
        default=-1,
        help="Random seed for Step 7.5 template selection (negative means default).",
    )
    parser.add_argument(
        "--step7-5-heuristic-ratio",
        type=float,
        default=0.5,
        help="Probability of choosing the heuristic template instead of seeded random in Step 7.5.",
    )
    parser.add_argument(
        "--enable-step8-5",
        action="store_true",
        help="Enable the Step 8.5 diversified prompt pass (default disabled; can also set PIPELINE_ENABLE_STEP8_5=1).",
    )
    parser.add_argument(
        "--step8-5-variants",
        type=int,
        default=3,
        help="Number of diversified prompts to generate in Step 8.5.",
    )
    parser.add_argument(
        "--step8-5-seed",
        type=int,
        default=-1,
        help="Random seed for Step 8.5 style selection (negative means random).",
    )
    parser.add_argument(
        "--step8-5-styles",
        type=str,
        default="",
        help="Comma-separated style names for Step 8.5 (empty means default pool).",
    )

    args = parser.parse_args()

    original_instruction = _read_file(args.instruction_file)
    model_answer = _read_file(args.answer_file)
    evidence = _read_json(args.evidence_file) if args.evidence_file else {}

    if not original_instruction.strip():
        raise ValueError("instruction-file is empty or missing")
    if not model_answer.strip():
        raise ValueError("answer-file is empty or missing")

    env_flag = os.getenv("PIPELINE_ENABLE_STEP8", "1").lower()
    step8_enabled_default = env_flag not in {"0", "false", "no"}
    enable_step8 = step8_enabled_default and not args.skip_step8_polish
    env_flag_augment = os.getenv("PIPELINE_ENABLE_AUGMENT", "0").lower()
    augment_enabled_default = env_flag_augment in {"1", "true", "yes"}
    enable_augment = augment_enabled_default or args.enable_augment
    env_flag_augment_diversity = os.getenv("PIPELINE_ENABLE_AUGMENT_DIVERSITY", "0").lower()
    augment_diversity_default = env_flag_augment_diversity in {"1", "true", "yes"}
    enable_augment_diversity = augment_diversity_default or args.enable_augment_diversity
    if args.augment_mode != "auto":
        if args.augment_mode == "none":
            enable_augment = False
            enable_augment_diversity = False
        elif args.augment_mode == "augment":
            enable_augment = True
            enable_augment_diversity = False
        elif args.augment_mode == "augment_diversity":
            enable_augment = False
            enable_augment_diversity = True
        elif args.augment_mode == "both":
            enable_augment = True
            enable_augment_diversity = True
    env_flag_7_5 = os.getenv("PIPELINE_ENABLE_STEP7_5", "0").lower()
    step7_5_enabled_default = env_flag_7_5 in {"1", "true", "yes"}
    enable_step7_5 = step7_5_enabled_default or args.enable_step7_5
    step7_5_templates = [s.strip() for s in args.step7_5_templates.split(",") if s.strip()]
    step7_5_seed = None if args.step7_5_seed < 0 else args.step7_5_seed
    step7_5_limit = None if args.step7_5_limit < 0 else args.step7_5_limit
    env_flag_8_5 = os.getenv("PIPELINE_ENABLE_STEP8_5", "0").lower()
    step8_5_enabled_default = env_flag_8_5 in {"1", "true", "yes"}
    enable_step8_5 = step8_5_enabled_default or args.enable_step8_5
    step8_5_seed = None if args.step8_5_seed < 0 else args.step8_5_seed
    step8_5_styles = [s.strip() for s in args.step8_5_styles.split(",") if s.strip()]

    result = run_pipeline_once(
        sample_id=args.sample_id,
        original_instruction=original_instruction,
        model_answer=model_answer,
        base_data_dir=args.data_dir,
        evidence=evidence if evidence else None,
        survey_mode=args.survey_mode,
        knowledge_style=args.knowledge_style,
        explicit_reference_strategy=args.explicit_reference_strategy,
        generation_granularity=args.generation_granularity,
        survey_block_mode=args.survey_block_mode,
        enable_step8_polish=enable_step8,
        enable_step7_5=enable_step7_5,
        step7_5_templates=step7_5_templates or None,
        step7_5_limit=step7_5_limit,
        step7_5_seed=step7_5_seed,
        step7_5_heuristic_ratio=args.step7_5_heuristic_ratio,
        enable_augment=enable_augment,
        enable_augment_diversity=enable_augment_diversity,
        augment_seed=args.augment_seed,
        augment_priority_ratio=args.augment_priority_ratio,
        augment_curriculum=not args.augment_disable_curriculum,
        augment_m1=not args.augment_disable_m1,
        augment_m2=not args.augment_disable_m2,
        enable_step8_5=enable_step8_5,
        step8_5_variants=args.step8_5_variants,
        step8_5_seed=step8_5_seed,
        step8_5_styles=step8_5_styles or None,
    )

    # Print a short human summary to stdout
    print("===== PIPELINE DONE =====")
    print(f"sample_id                  : {args.sample_id}")
    print(f"seed_task                  : {result['seed_task']}")
    print(f"blocks                     : {len(result['segmentation'].get('blocks', []))}")
    print(f"global_constraints         : {result['global_constraints_count']}")
    print(f"total_constraints          : {result['constraint_total_count']}")
    print(f"conditional_branches       : {result['selection_count']}")
    print(f"evidence_loaded            : {result['evidence_loaded']}")
    print(f"evidence_reference_count   : {result['evidence_reference_count']}")
    print(f"survey_mode                : {result['survey_mode']}")
    print(f"knowledge_style            : {result['knowledge_style']}")
    print(f"explicit_reference_strategy: {result['explicit_reference_strategy']}")
    print(f"generation_granularity     : {result['generation_granularity']}")
    print(f"survey_block_mode          : {result['survey_block_mode']}")
    print(f"prompt_length_chars        : {result['prompt_length']}")
    print("--- artifacts ---")
    print(f"graph_json_path            : {result['graph_paths']['graph_json']}")
    print(f"graph_mermaid_path         : {result['graph_paths']['mermaid_mmd']}")
    print(f"prompt_path_raw (step7)    : {result['prompt_path_machine']}")
    print(f"prompt_path (to eval LLM)  : {result['prompt_path']}")
    print(f"eval_protocol_path         : {result['eval_path']}")
    print(f"bundle_debug_path          : {result['bundle_path']}")
    if result.get("evidence_path"):
        print(f"evidence_path              : {result['evidence_path']}")
    polish_info = result.get("polish_result") or {}
    print(f"step8_polish_used          : {polish_info.get('used_llm', False)} ({polish_info.get('reason', 'n/a')})")
    print(f"step8_stripped_think       : {polish_info.get('stripped_think', False)}")
    if enable_augment:
        print(f"augment_enabled            : {enable_augment}")
        print(f"augment_outputs            : {len(result.get('augment_outputs', []) or [])}")
    if enable_augment_diversity:
        print(f"augment_diversity_enabled  : {enable_augment_diversity}")
        print(f"augment_diversity_outputs  : {len(result.get('augment_diversity_outputs', []) or [])}")
    if enable_step7_5:
        step7_5_info = result.get("step7_5_result", {}) or {}
        variants_written = len(result.get("step7_5_variants", []) or [])
        print(f"step7_5_enabled            : {enable_step7_5}")
        print(f"step7_5_variants_written   : {variants_written}")
        print(f"step7_5_templates          : {', '.join(step7_5_info.get('template_pool', []) or [])}")
    if enable_step8_5:
        step8_5_info = result.get("step8_5_result", {}) or {}
        variants_written = len(result.get("step8_5_variants", []) or [])
        print(f"step8_5_enabled            : {enable_step8_5}")
        print(f"step8_5_variants_written   : {variants_written}")
        print(f"step8_5_reason             : {step8_5_info.get('reason', 'n/a')}")
    print("--- LLM call status by step ---")
    for entry in result.get("llm_statuses", []):
        print(f"{entry['step']:<32}: {entry['status']} (calls={entry['calls']})")


if __name__ == "__main__":
    main()
