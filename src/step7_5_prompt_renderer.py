"""
Step 7.5 - Multi-template Prompt Rendering

Purpose
- Render multiple machine_prompt variants directly from the Step6 ConstraintGraph.
- Provide diverse but deterministic prompt structures without LLM calls.

Notes
- This step runs in parallel with Step 7.
- Outputs are intended for Step 8 polishing.
"""

from __future__ import annotations

import hashlib
import random
import argparse
import json
import os
from collections import defaultdict
from typing import Dict, Any, List, Optional, Tuple

from .graph_schema import ConstraintGraph, BlockSpec, BlockConstraintSet, ConstraintNode, SelectionNode
from .graph_augmenter import _graph_from_serialized, _default_sample_id_from_path
from .utils.export_utils import write_text


TEMPLATE_PROFILES: Dict[str, Dict[str, Any]] = {
    "stage_blueprint": {
        "label": "Stage-first blueprint",
        "description": "Follow block order, show branch logic in-place.",
    },
    "branch_first": {
        "label": "Branch-first map",
        "description": "Explain IF/ELSE branches up front, then stage flow.",
    },
    "grouped_by_check": {
        "label": "Grouped by verifier type",
        "description": "Group constraints by verifier check per section.",
    },
    "priority_layered": {
        "label": "Priority layered",
        "description": "Separate priority=2 vs priority=1 constraints explicitly.",
    },
}

TEMPLATE_ORDER = list(TEMPLATE_PROFILES.keys())


def _render_block_plan(block_specs: List[BlockSpec],
                       block_sets: List[BlockConstraintSet]) -> List[Dict[str, Any]]:
    set_map = {bcs.block_id: bcs for bcs in block_sets}
    sorted_specs = sorted(block_specs, key=lambda b: (b.order_index, b.block_id))
    rendered: List[Dict[str, Any]] = []
    for spec in sorted_specs:
        if spec.is_alternate:
            continue
        bset = set_map.get(spec.block_id)
        requirements = []
        if bset:
            for cnode in bset.constraints:
                requirements.append({
                    "cid": cnode.cid,
                    "desc": cnode.desc,
                    "verifier": cnode.verifier_spec,
                    "priority_level": cnode.priority_level,
                })
            logic_type = bset.logic_type
        else:
            logic_type = "AND"
        rendered.append({
            "block_id": spec.block_id,
            "role": spec.intent,
            "logic_type": logic_type,
            "requirements": requirements,
        })
    return rendered


def _render_global_constraints(global_nodes: List[ConstraintNode]) -> List[Dict[str, Any]]:
    return [
        {
            "cid": node.cid,
            "desc": node.desc,
            "verifier": node.verifier_spec,
            "priority_level": node.priority_level,
        }
        for node in global_nodes
    ]


def _build_block_label_lookup(block_specs: List[BlockSpec]) -> Dict[str, str]:
    sorted_specs = sorted(
        [spec for spec in block_specs if not spec.is_alternate],
        key=lambda b: (b.order_index, b.block_id),
    )
    lookup: Dict[str, str] = {}
    used_labels: Dict[str, int] = {}

    def _unique_label(label: str) -> str:
        base = label.strip() or "Stage"
        if base not in used_labels:
            used_labels[base] = 1
            return base
        used_labels[base] += 1
        return f"{base} ({used_labels[base]})"

    for idx, spec in enumerate(sorted_specs, start=1):
        role = (spec.intent or "").strip()
        lookup[spec.block_id] = _unique_label(role or f"Stage {idx}")
    for spec in block_specs:
        if not spec.is_alternate:
            continue
        label = (spec.intent or "").strip()
        if not label and spec.origin_block and spec.origin_block in lookup:
            label = f"Alternate: {lookup[spec.origin_block]}"
        lookup[spec.block_id] = _unique_label(label or "Alternate stage")
    return lookup


def _label_for_block(block_id: str,
                     block_label_lookup: Dict[str, str],
                     *,
                     fallback: str) -> str:
    label = (block_label_lookup.get(block_id) or "").strip()
    return label or fallback


def _render_selections(selections: List[SelectionNode],
                       block_specs: List[BlockSpec],
                       block_sets: List[BlockConstraintSet],
                       global_nodes: List[ConstraintNode]) -> List[Dict[str, Any]]:
    cid_to_node: Dict[str, ConstraintNode] = {}
    for node in global_nodes:
        cid_to_node[node.cid] = node
    for bcs in block_sets:
        for node in bcs.constraints:
            cid_to_node[node.cid] = node

    block_label_lookup = _build_block_label_lookup(block_specs)
    rendered = []
    for sel in selections:
        def _resolve(cids: List[str], fallback_block: str) -> List[Dict[str, Any]]:
            items = []
            for cid in cids:
                node = cid_to_node.get(cid)
                items.append({
                    "cid": cid,
                    "desc": node.desc if node else None,
                    "verifier": node.verifier_spec if node else None,
                    "priority_level": node.priority_level if node else 2,
                    "trace_to": node.trace_to if node else fallback_block,
                })
            return items

        real_list = _resolve(sel.branch_real.constraints, sel.branch_real.block_id or sel.trace_to)
        alt_list = _resolve(sel.branch_alt.constraints, sel.branch_alt.block_id or sel.trace_to)

        alt_by_block: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        alt_path_ids = list(sel.alt_path_blocks or [sel.branch_alt.block_id])
        for item in alt_list:
            block_id = item.get("trace_to") or sel.branch_alt.block_id
            if block_id not in alt_path_ids:
                alt_path_ids.append(block_id)
            alt_by_block[block_id].append(item)

        real_by_block: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        real_path_ids = []
        default_real_block = sel.branch_real.block_id or sel.trace_to
        real_path_ids.append(default_real_block)
        for item in real_list:
            block_id = item.get("trace_to") or default_real_block
            if block_id not in real_path_ids:
                real_path_ids.append(block_id)
            real_by_block[block_id].append(item)
        if sel.trace_to and sel.trace_to not in real_path_ids:
            real_path_ids.insert(0, sel.trace_to)

        rendered.append({
            "sid": sel.sid,
            "trace_to": sel.trace_to,
            "where": block_label_lookup.get(sel.trace_to, "Stage"),
            "condition": sel.condition,
            "selection_type": sel.selection_type,
            "merge_point": sel.merge_point,
            "truncated": sel.truncated,
            "branch_real": real_list,
            "branch_alt": alt_list,
            "alt_by_block": alt_by_block,
            "real_by_block": real_by_block,
            "alt_path_blocks": alt_path_ids,
            "real_path_blocks": real_path_ids,
        })
    return rendered


def _condition_statement(cond: str) -> str:
    text = (cond or "").strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered.startswith("if "):
        text = text[3:].strip()
    if not text:
        return ""
    if text[0].islower():
        text = text[0].upper() + text[1:]
    if not text.endswith("."):
        text += "."
    return text


def _negated_condition_statement(cond: str) -> str:
    statement = _condition_statement(cond).rstrip(".")
    if not statement:
        return ""
    lowered = statement.lower()
    if " is " in lowered:
        parts = statement.split(" is ", 1)
        return f"{parts[0]} is not {parts[1]}."
    if " are " in lowered:
        parts = statement.split(" are ", 1)
        return f"{parts[0]} are not {parts[1]}."
    if " should " in lowered:
        parts = statement.split(" should ", 1)
        return f"{parts[0]} should not {parts[1]}."
    if " must " in lowered:
        parts = statement.split(" must ", 1)
        return f"{parts[0]} must not {parts[1]}."
    return f"It is not the case that {statement.lower()}."


def _group_by_check(items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        verifier = item.get("verifier") or {}
        check = (verifier.get("check") or "unspecified").strip() or "unspecified"
        grouped[check].append(item)
    return grouped


def _collect_checks(graph: ConstraintGraph) -> List[str]:
    checks: List[str] = []
    for node in graph.global_constraints:
        verifier = node.verifier_spec or {}
        check = (verifier.get("check") or "").strip()
        if check:
            checks.append(check)
    for bcs in graph.block_constraint_sets:
        for node in bcs.constraints:
            verifier = node.verifier_spec or {}
            check = (verifier.get("check") or "").strip()
            if check:
                checks.append(check)
    return checks


def _has_priority_one(graph: ConstraintGraph) -> bool:
    for node in graph.global_constraints:
        if node.priority_level == 1:
            return True
    for bcs in graph.block_constraint_sets:
        for node in bcs.constraints:
            if node.priority_level == 1:
                return True
    return False


def _heuristic_template(graph: ConstraintGraph) -> str:
    if graph.selections:
        return "branch_first"
    if _has_priority_one(graph):
        return "priority_layered"
    if len(set(_collect_checks(graph))) >= 2:
        return "grouped_by_check"
    return "stage_blueprint"


def _stable_seed(base_seed: Optional[int], selection_key: str) -> int:
    seed = 0 if base_seed is None else int(base_seed)
    payload = f"{seed}:{selection_key}".encode("utf-8")
    digest = hashlib.md5(payload).hexdigest()
    return int(digest, 16) % (2**31)


def _append_constraints(lines: List[str],
                        indent: str,
                        obligations: List[Dict[str, Any]],
                        default_msg: str) -> None:
    if not obligations:
        lines.append(f"{indent}- {default_msg}")
        return
    for item in obligations:
        desc = (item.get("desc") or "").strip()
        if desc:
            lines.append(f"{indent}- {desc}")


def _append_constraints_grouped(lines: List[str],
                                indent: str,
                                obligations: List[Dict[str, Any]],
                                default_msg: str) -> None:
    if not obligations:
        lines.append(f"{indent}- {default_msg}")
        return
    grouped = _group_by_check(obligations)
    for check_name in sorted(grouped.keys()):
        lines.append(f"{indent}- {check_name}:")
        for item in grouped[check_name]:
            desc = (item.get("desc") or "").strip()
            if desc:
                lines.append(f"{indent}  * {desc}")


def _append_constraints_by_priority(lines: List[str],
                                    indent: str,
                                    obligations: List[Dict[str, Any]],
                                    default_msg: str) -> None:
    if not obligations:
        lines.append(f"{indent}- {default_msg}")
        return
    primary = [item for item in obligations if item.get("priority_level") != 1]
    secondary = [item for item in obligations if item.get("priority_level") == 1]
    if primary:
        lines.append(f"{indent}- Primary constraints:")
        for item in primary:
            desc = (item.get("desc") or "").strip()
            if desc:
                lines.append(f"{indent}  * {desc}")
    if secondary:
        lines.append(f"{indent}- Secondary constraints:")
        for item in secondary:
            desc = (item.get("desc") or "").strip()
            if desc:
                lines.append(f"{indent}  * {desc}")
    if not primary and not secondary:
        lines.append(f"{indent}- {default_msg}")


def _render_stage_blueprint(graph: ConstraintGraph,
                            global_rules: List[Dict[str, Any]],
                            block_plan: List[Dict[str, Any]],
                            selections_view: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("SYSTEM INSTRUCTIONS:")
    lines.append("")
    lines.append("1. MISSION BRIEF")
    if graph.seed_task.strip():
        lines.append(f"- Primary directive: {graph.seed_task.strip()}")
    else:
        lines.append("- Follow the staged plan and constraints below to construct the final answer.")
    if graph.meta.get("turn_notice"):
        lines.append("- Only constraints in the current graph are active.")
    lines.append("")

    lines.append("2. NON-NEGOTIABLE GLOBAL RULES")
    if global_rules:
        lines.append("All global rules must be satisfied throughout the response:")
        _append_constraints(lines, indent="  ", obligations=global_rules, default_msg="No global rules provided.")
    else:
        lines.append("- No additional global constraints were supplied.")
    lines.append("")

    selection_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sel in selections_view:
        selection_map[sel["trace_to"]].append(sel)

    block_label_lookup = _build_block_label_lookup(graph.block_specs)

    lines.append("3. STRUCTURED RESPONSE BLUEPRINT")
    lines.append("Follow stages in order and apply branch logic where stated.")
    for idx, blk in enumerate(block_plan, start=1):
        block_id = blk["block_id"]
        role = blk["role"] or f"Stage {idx}"
        logic_type = (blk["logic_type"] or "AND").lower()
        logic_hint = "Cover all listed duties for this stage." \
            if not logic_type.startswith("sub") else \
            "Address listed duties as a short sequence."
        lines.append(f"- Stage {idx}: {role}")
        lines.append(f"  Logic cue: {logic_hint}")
        stage_selections = selection_map.get(block_id, [])
        if not stage_selections:
            _append_constraints(
                lines,
                indent="  ",
                obligations=blk["requirements"],
                default_msg="Follow the intent of this stage even if no explicit duty exists.",
            )
            continue
        for sel in stage_selections:
            condition = _condition_statement(sel.get("condition", "")) or "the condition applies."
            lines.append(f"  IF {condition.rstrip('.')}:")
            alt_blocks = sel.get("alt_path_blocks") or [sel.get("trace_to")]
            alt_by_block = sel.get("alt_by_block") or {}
            for block_id in alt_blocks:
                _append_constraints(
                    lines,
                    indent="    ",
                    obligations=alt_by_block.get(block_id, []),
                    default_msg="Carry out the alternate storyline requirements for this stage.",
                )
            lines.append("  OTHERWISE:")
            real_blocks = sel.get("real_path_blocks") or [sel.get("trace_to")]
            real_by_block = sel.get("real_by_block") or {}
            for block_id in real_blocks:
                _append_constraints(
                    lines,
                    indent="    ",
                    obligations=real_by_block.get(block_id, []),
                    default_msg="Return to the duties defined for this stage.",
                )
    lines.append("")

    if selections_view:
        lines.append("4. CURRENT CONDITION ASSUMPTION")
        lines.append("Unless explicitly told otherwise, assume each trigger below is FALSE:")
        for sel in selections_view:
            cond = (sel.get("condition") or "").strip()
            stage_label = _label_for_block(sel["trace_to"], block_label_lookup, fallback="Stage")
            negated = _negated_condition_statement(cond)
            statement = negated or "The specified condition does not apply."
            lines.append(f"- {stage_label}: {statement}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def _render_branch_first(graph: ConstraintGraph,
                         global_rules: List[Dict[str, Any]],
                         block_plan: List[Dict[str, Any]],
                         selections_view: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("SYSTEM INSTRUCTIONS:")
    lines.append("")
    lines.append("1. MISSION BRIEF")
    if graph.seed_task.strip():
        lines.append(f"- Primary directive: {graph.seed_task.strip()}")
    else:
        lines.append("- Follow the staged plan and constraints below to construct the final answer.")
    if graph.meta.get("turn_notice"):
        lines.append("- Only constraints in the current graph are active.")
    lines.append("")

    lines.append("2. NON-NEGOTIABLE GLOBAL RULES")
    if global_rules:
        _append_constraints(lines, indent="", obligations=global_rules, default_msg="No global rules provided.")
    else:
        lines.append("- No additional global constraints were supplied.")
    lines.append("")

    block_label_lookup = _build_block_label_lookup(graph.block_specs)

    lines.append("3. STRUCTURED RESPONSE BLUEPRINT")
    if selections_view:
        lines.append("Branch map:")
        for sel in selections_view:
            condition = _condition_statement(sel.get("condition", "")) or "the condition applies."
            where = sel.get("where") or _label_for_block(sel.get("trace_to", ""), block_label_lookup, fallback="Stage")
            lines.append(f"- Decision at {where}:")
            lines.append(f"  IF {condition.rstrip('.')}:")
            alt_blocks = sel.get("alt_path_blocks") or [sel.get("trace_to")]
            alt_by_block = sel.get("alt_by_block") or {}
            for block_id in alt_blocks:
                _append_constraints(
                    lines,
                    indent="    ",
                    obligations=alt_by_block.get(block_id, []),
                    default_msg="Carry out the alternate storyline requirements for this stage.",
                )
            lines.append("  OTHERWISE:")
            real_blocks = sel.get("real_path_blocks") or [sel.get("trace_to")]
            real_by_block = sel.get("real_by_block") or {}
            for block_id in real_blocks:
                _append_constraints(
                    lines,
                    indent="    ",
                    obligations=real_by_block.get(block_id, []),
                    default_msg="Return to the duties defined for this stage.",
                )
    else:
        lines.append("No conditional branches were supplied.")

    lines.append("")
    lines.append("Stage flow (follow in order):")
    for idx, blk in enumerate(block_plan, start=1):
        role = blk["role"] or f"Stage {idx}"
        lines.append(f"- Stage {idx}: {role}")
        _append_constraints(
            lines,
            indent="  ",
            obligations=blk["requirements"],
            default_msg="Follow the intent of this stage even if no explicit duty exists.",
        )
    lines.append("")

    if selections_view:
        lines.append("4. CURRENT CONDITION ASSUMPTION")
        lines.append("Unless explicitly told otherwise, assume each trigger below is FALSE:")
        for sel in selections_view:
            cond = (sel.get("condition") or "").strip()
            stage_label = _label_for_block(sel["trace_to"], block_label_lookup, fallback="Stage")
            negated = _negated_condition_statement(cond)
            statement = negated or "The specified condition does not apply."
            lines.append(f"- {stage_label}: {statement}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def _render_grouped_by_check(graph: ConstraintGraph,
                             global_rules: List[Dict[str, Any]],
                             block_plan: List[Dict[str, Any]],
                             selections_view: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("SYSTEM INSTRUCTIONS:")
    lines.append("")
    lines.append("1. MISSION BRIEF")
    if graph.seed_task.strip():
        lines.append(f"- Primary directive: {graph.seed_task.strip()}")
    else:
        lines.append("- Follow the staged plan and constraints below to construct the final answer.")
    if graph.meta.get("turn_notice"):
        lines.append("- Only constraints in the current graph are active.")
    lines.append("")

    block_label_lookup = _build_block_label_lookup(graph.block_specs)
    selection_blocks: set[str] = set()
    for sel in selections_view:
        trace_to = sel.get("trace_to")
        if trace_to:
            selection_blocks.add(trace_to)
        for block_id in sel.get("alt_path_blocks") or []:
            if block_id:
                selection_blocks.add(block_id)
        for block_id in sel.get("real_path_blocks") or []:
            if block_id:
                selection_blocks.add(block_id)

    lines.append("2. CONSTRAINT SUMMARY BY TYPE")
    lines.append("Only constraint types with two or more instances are summarized here.")
    summary_candidates: List[Dict[str, Any]] = []
    for item in global_rules:
        desc = (item.get("desc") or "").strip()
        verifier = item.get("verifier") or {}
        check = (verifier.get("check") or "unspecified").strip() or "unspecified"
        if desc:
            summary_candidates.append({
                "cid": item.get("cid"),
                "desc": desc,
                "check": check,
                "scope": "Global",
            })
    for bcs in graph.block_constraint_sets:
        if bcs.block_id in selection_blocks:
            continue
        scope_label = _label_for_block(bcs.block_id, block_label_lookup, fallback="Stage")
        scope = "Stage" if scope_label.lower() == "stage" else f"Stage: {scope_label}"
        for node in bcs.constraints:
            desc = (node.desc or "").strip()
            verifier = node.verifier_spec or {}
            check = (verifier.get("check") or "unspecified").strip() or "unspecified"
            if desc:
                summary_candidates.append({
                    "cid": node.cid,
                    "desc": desc,
                    "check": check,
                    "scope": scope,
                })
    check_counts: Dict[str, int] = defaultdict(int)
    for item in summary_candidates:
        check_counts[item["check"]] += 1
    summary_checks = {check for check, count in check_counts.items() if count >= 2}
    grouped_all: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    summarized_cids: set = set()
    for item in summary_candidates:
        if item["check"] not in summary_checks:
            continue
        grouped_all[item["check"]][item["scope"]].append(item["desc"])
        if item.get("cid"):
            summarized_cids.add(item["cid"])
    if grouped_all:
        for check_name in sorted(grouped_all.keys()):
            lines.append(f"- {check_name}:")
            scopes = list(grouped_all[check_name].keys())
            scopes.sort(key=lambda s: (s != "Global", s))
            for scope in scopes:
                lines.append(f"  {scope}:")
                for desc in grouped_all[check_name][scope]:
                    lines.append(f"    * {desc}")
    else:
        lines.append("- No constraint types met the summary threshold.")
    lines.append("")

    selection_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sel in selections_view:
        selection_map[sel["trace_to"]].append(sel)

    lines.append("3. RESPONSE BLUEPRINT & REMAINING RULES")
    remaining_globals = [
        item for item in global_rules
        if item.get("cid") not in summarized_cids
    ]
    if remaining_globals:
        lines.append("Remaining global rules (not included in the summary):")
        _append_constraints(
            lines,
            indent="  ",
            obligations=remaining_globals,
            default_msg="No remaining global rules.",
        )
    lines.append("Follow stages in order. Selection stages list their local constraints below.")
    for idx, blk in enumerate(block_plan, start=1):
        block_id = blk["block_id"]
        role = blk["role"] or f"Stage {idx}"
        lines.append(f"- Stage {idx}: {role}")
        stage_selections = selection_map.get(block_id, [])
        if not stage_selections:
            remaining_local = [
                item for item in blk["requirements"]
                if item.get("cid") not in summarized_cids
            ]
            if remaining_local:
                _append_constraints(
                    lines,
                    indent="  ",
                    obligations=remaining_local,
                    default_msg="No additional stage-specific rules.",
                )
            continue
        for sel in stage_selections:
            condition = _condition_statement(sel.get("condition", "")) or "the condition applies."
            lines.append(f"  IF {condition.rstrip('.')}:")
            alt_blocks = sel.get("alt_path_blocks") or [sel.get("trace_to")]
            alt_by_block = sel.get("alt_by_block") or {}
            for block_id in alt_blocks:
                _append_constraints(
                    lines,
                    indent="    ",
                    obligations=alt_by_block.get(block_id, []),
                    default_msg="Carry out the alternate storyline requirements for this stage.",
                )
            lines.append("  OTHERWISE:")
            real_blocks = sel.get("real_path_blocks") or [sel.get("trace_to")]
            real_by_block = sel.get("real_by_block") or {}
            for block_id in real_blocks:
                _append_constraints(
                    lines,
                    indent="    ",
                    obligations=real_by_block.get(block_id, []),
                    default_msg="Return to the duties defined for this stage.",
                )
    lines.append("")

    if selections_view:
        lines.append("4. CURRENT CONDITION ASSUMPTION")
        lines.append("Unless explicitly told otherwise, assume each trigger below is FALSE:")
        for sel in selections_view:
            cond = (sel.get("condition") or "").strip()
            stage_label = _label_for_block(sel["trace_to"], block_label_lookup, fallback="Stage")
            negated = _negated_condition_statement(cond)
            statement = negated or "The specified condition does not apply."
            lines.append(f"- {stage_label}: {statement}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def _render_priority_layered(graph: ConstraintGraph,
                             global_rules: List[Dict[str, Any]],
                             block_plan: List[Dict[str, Any]],
                             selections_view: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("SYSTEM INSTRUCTIONS:")
    lines.append("")
    lines.append("1. MISSION BRIEF")
    if graph.seed_task.strip():
        lines.append(f"- Primary directive: {graph.seed_task.strip()}")
    else:
        lines.append("- Follow the staged plan and constraints below to construct the final answer.")
    if graph.meta.get("turn_notice"):
        lines.append("- Only constraints in the current graph are active.")
    lines.append("")

    block_label_lookup = _build_block_label_lookup(graph.block_specs)
    selection_blocks: set[str] = set()
    for sel in selections_view:
        trace_to = sel.get("trace_to")
        if trace_to:
            selection_blocks.add(trace_to)
        for block_id in sel.get("alt_path_blocks") or []:
            if block_id:
                selection_blocks.add(block_id)
        for block_id in sel.get("real_path_blocks") or []:
            if block_id:
                selection_blocks.add(block_id)

    lines.append("2. CONSTRAINT SUMMARY BY PRIORITY")
    lines.append("Constraints grouped by priority, then by scope.")
    grouped_priority: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for item in global_rules:
        desc = (item.get("desc") or "").strip()
        priority = "Primary" if item.get("priority_level", 2) != 1 else "Secondary"
        if desc:
            grouped_priority[priority]["Global"].append(desc)
    for bcs in graph.block_constraint_sets:
        if bcs.block_id in selection_blocks:
            continue
        scope_label = _label_for_block(bcs.block_id, block_label_lookup, fallback="Stage")
        scope = "Stage" if scope_label.lower() == "stage" else f"Stage: {scope_label}"
        for node in bcs.constraints:
            desc = (node.desc or "").strip()
            priority = "Primary" if node.priority_level != 1 else "Secondary"
            if desc:
                grouped_priority[priority][scope].append(desc)
    if grouped_priority:
        for priority in ["Primary", "Secondary"]:
            if priority not in grouped_priority:
                continue
            lines.append(f"- {priority}:")
            scopes = list(grouped_priority[priority].keys())
            scopes.sort(key=lambda s: (s != "Global", s))
            for scope in scopes:
                lines.append(f"  {scope}:")
                for desc in grouped_priority[priority][scope]:
                    lines.append(f"    * {desc}")
    else:
        lines.append("- No additional global constraints were supplied.")
    lines.append("")

    selection_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sel in selections_view:
        selection_map[sel["trace_to"]].append(sel)

    lines.append("3. RESPONSE BLUEPRINT")
    lines.append("Follow stages in order. Selection stages list their local constraints below.")
    for idx, blk in enumerate(block_plan, start=1):
        block_id = blk["block_id"]
        role = blk["role"] or f"Stage {idx}"
        lines.append(f"- Stage {idx}: {role}")
        stage_selections = selection_map.get(block_id, [])
        if not stage_selections:
            continue
        for sel in stage_selections:
            condition = _condition_statement(sel.get("condition", "")) or "the condition applies."
            lines.append(f"  IF {condition.rstrip('.')}:")
            alt_blocks = sel.get("alt_path_blocks") or [sel.get("trace_to")]
            alt_by_block = sel.get("alt_by_block") or {}
            for block_id in alt_blocks:
                _append_constraints_by_priority(
                    lines,
                    indent="    ",
                    obligations=alt_by_block.get(block_id, []),
                    default_msg="Carry out the alternate storyline requirements for this stage.",
                )
            lines.append("  OTHERWISE:")
            real_blocks = sel.get("real_path_blocks") or [sel.get("trace_to")]
            real_by_block = sel.get("real_by_block") or {}
            for block_id in real_blocks:
                _append_constraints_by_priority(
                    lines,
                    indent="    ",
                    obligations=real_by_block.get(block_id, []),
                    default_msg="Return to the duties defined for this stage.",
                )
    lines.append("")

    if selections_view:
        lines.append("4. CURRENT CONDITION ASSUMPTION")
        lines.append("Unless explicitly told otherwise, assume each trigger below is FALSE:")
        for sel in selections_view:
            cond = (sel.get("condition") or "").strip()
            stage_label = _label_for_block(sel["trace_to"], block_label_lookup, fallback="Stage")
            negated = _negated_condition_statement(cond)
            statement = negated or "The specified condition does not apply."
            lines.append(f"- {stage_label}: {statement}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


TEMPLATE_RENDERERS = {
    "stage_blueprint": _render_stage_blueprint,
    "branch_first": _render_branch_first,
    "grouped_by_check": _render_grouped_by_check,
    "priority_layered": _render_priority_layered,
}


def _normalize_template_name(name: str) -> str:
    return (name or "").strip().lower().replace("-", "_")


def _resolve_template_pool(template_pool: Optional[List[str]]) -> Tuple[List[str], List[str]]:
    if not template_pool:
        return TEMPLATE_ORDER[:], []
    resolved = []
    unknown = []
    for name in template_pool:
        key = _normalize_template_name(name)
        if key in TEMPLATE_RENDERERS and key not in resolved:
            resolved.append(key)
        else:
            unknown.append(name)
    if not resolved:
        resolved = TEMPLATE_ORDER[:]
    return resolved, unknown


def select_template_for_graph(graph: ConstraintGraph,
                              template_pool: Optional[List[str]] = None,
                              *,
                              template_seed: Optional[int] = None,
                              heuristic_ratio: float = 0.5,
                              selection_key: str = "",
                              template_limit: Optional[int] = None) -> Dict[str, Any]:
    resolved_pool, unknown = _resolve_template_pool(template_pool)
    if not resolved_pool:
        return {
            "template": "stage_blueprint",
            "selected_by": "fallback",
            "heuristic_template": "stage_blueprint",
            "template_pool": resolved_pool,
            "unknown_templates": unknown,
        }

    if template_limit is not None:
        limit = max(1, int(template_limit))
        if limit < len(resolved_pool):
            rng_pool = random.Random(_stable_seed(template_seed, f"{selection_key}:pool"))
            rng_pool.shuffle(resolved_pool)
            resolved_pool = resolved_pool[:limit]

    rng = random.Random(_stable_seed(template_seed, selection_key))
    use_heuristic = rng.random() < max(0.0, min(1.0, float(heuristic_ratio)))
    heuristic_choice = _heuristic_template(graph)

    if use_heuristic and heuristic_choice in resolved_pool:
        chosen = heuristic_choice
        selected_by = "heuristic"
    else:
        chosen = rng.choice(resolved_pool)
        selected_by = "seeded_random"

    return {
        "template": chosen,
        "selected_by": selected_by,
        "heuristic_template": heuristic_choice,
        "template_pool": resolved_pool,
        "unknown_templates": unknown,
    }


def render_prompt_variant(graph: ConstraintGraph,
                          template_pool: Optional[List[str]] = None,
                          *,
                          template_seed: Optional[int] = None,
                          heuristic_ratio: float = 0.5,
                          selection_key: str = "",
                          template_limit: Optional[int] = None) -> Dict[str, Any]:
    selection = select_template_for_graph(
        graph,
        template_pool=template_pool,
        template_seed=template_seed,
        heuristic_ratio=heuristic_ratio,
        selection_key=selection_key,
        template_limit=template_limit,
    )
    template_key = selection["template"]
    renderer = TEMPLATE_RENDERERS.get(template_key)
    if not renderer:
        template_key = "stage_blueprint"
        renderer = TEMPLATE_RENDERERS.get(template_key)

    global_rules = _render_global_constraints(graph.global_constraints)
    block_plan = _render_block_plan(graph.block_specs, graph.block_constraint_sets)
    selections_view = _render_selections(
        graph.selections,
        graph.block_specs,
        graph.block_constraint_sets,
        graph.global_constraints,
    )
    machine_prompt = renderer(graph, global_rules, block_plan, selections_view)

    return {
        "variant": {
            "template": template_key,
            "label": TEMPLATE_PROFILES.get(template_key, {}).get("label", template_key),
            "description": TEMPLATE_PROFILES.get(template_key, {}).get("description", ""),
            "machine_prompt": machine_prompt,
        },
        "selection": selection,
    }


def render_prompt_variants(graph: ConstraintGraph,
                           template_pool: Optional[List[str]] = None,
                           *,
                           template_seed: Optional[int] = None,
                           template_limit: Optional[int] = None) -> Dict[str, Any]:
    """
    Render multiple machine_prompt variants from the graph.

    Returns dict:
        {
            "variants": [ { "template": str, "label": str, "machine_prompt": str }, ... ],
            "template_pool": [str],
            "unknown_templates": [str],
            "template_seed": int | None,
        }
    """
    resolved_pool, unknown = _resolve_template_pool(template_pool)
    if template_seed is not None:
        rng = random.Random(template_seed)
        rng.shuffle(resolved_pool)
    if template_limit is not None:
        resolved_pool = resolved_pool[: max(0, int(template_limit))]

    global_rules = _render_global_constraints(graph.global_constraints)
    block_plan = _render_block_plan(graph.block_specs, graph.block_constraint_sets)
    selections_view = _render_selections(
        graph.selections,
        graph.block_specs,
        graph.block_constraint_sets,
        graph.global_constraints,
    )

    variants = []
    for template_key in resolved_pool:
        renderer = TEMPLATE_RENDERERS.get(template_key)
        if not renderer:
            continue
        machine_prompt = renderer(graph, global_rules, block_plan, selections_view)
        variants.append({
            "template": template_key,
            "label": TEMPLATE_PROFILES.get(template_key, {}).get("label", template_key),
            "description": TEMPLATE_PROFILES.get(template_key, {}).get("description", ""),
            "machine_prompt": machine_prompt,
        })

    return {
        "variants": variants,
        "template_pool": resolved_pool,
        "unknown_templates": unknown,
        "template_seed": template_seed,
    }


def _load_graph_from_json(path: str) -> ConstraintGraph:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return _graph_from_serialized(data)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step7.5: render a machine prompt from a .graph.json file.",
    )
    parser.add_argument("--graph-json", type=str, help="Path to a .graph.json file.")
    parser.add_argument("--graphs-dir", type=str, help="Directory containing .graph.json files.")
    parser.add_argument("--sample-id", type=str, default="", help="Override sample_id for output naming.")
    parser.add_argument("--output-dir", type=str, default="", help="Base output dir (expects instructions/).")
    parser.add_argument("--templates", type=str, default="", help="Comma-separated template names (empty means default).")
    parser.add_argument("--template-limit", type=int, default=-1, help="Limit template pool size (negative means no limit).")
    parser.add_argument("--template-seed", type=int, default=-1, help="Seed for deterministic template selection.")
    parser.add_argument("--heuristic-ratio", type=float, default=0.5, help="Probability of heuristic selection.")
    parser.add_argument("--include-augmented", action="store_true", help="Also process graphs with '__' in the sample_id.")
    args = parser.parse_args()

    if not args.graph_json and not args.graphs_dir:
        raise SystemExit("Provide --graph-json or --graphs-dir.")

    template_pool = [s.strip() for s in args.templates.split(",") if s.strip()]
    template_limit = None if args.template_limit < 0 else args.template_limit
    template_seed = None if args.template_seed < 0 else args.template_seed

    def _render_one(path: str) -> Dict[str, Any]:
        sample_id = args.sample_id or _default_sample_id_from_path(path)
        if not args.include_augmented and ("__" in os.path.basename(path) or "__" in sample_id):
            return {}
        base_dir = args.output_dir or os.path.dirname(os.path.dirname(path))
        instructions_dir = os.path.join(base_dir, "instructions")
        graph = _load_graph_from_json(path)
        result = render_prompt_variant(
            graph,
            template_pool=template_pool or None,
            template_seed=template_seed,
            template_limit=template_limit,
            heuristic_ratio=args.heuristic_ratio,
            selection_key=sample_id,
        )
        variant = result.get("variant", {})
        template_key = variant.get("template", "template")
        machine_prompt = (variant.get("machine_prompt") or "").strip()
        if not machine_prompt:
            return {}
        out_path = os.path.join(instructions_dir, f"{sample_id}.machine.tmpl_{template_key}.txt")
        write_text(out_path, machine_prompt)
        return {
            "sample_id": sample_id,
            "template": template_key,
            "output_path": out_path,
            "selected_by": result.get("selection", {}).get("selected_by"),
        }

    outputs: List[Dict[str, Any]] = []
    if args.graph_json:
        out = _render_one(args.graph_json)
        if out:
            outputs.append(out)
    else:
        for name in os.listdir(args.graphs_dir):
            if not name.endswith(".graph.json"):
                continue
            if not args.include_augmented and "__" in name:
                continue
            out = _render_one(os.path.join(args.graphs_dir, name))
            if out:
                outputs.append(out)

    print(f"step7_5_outputs: {len(outputs)}")
    for item in outputs:
        print(f"- {item['sample_id']} => {item['output_path']} ({item['template']}, {item['selected_by']})")


if __name__ == "__main__":
    main()
