import argparse
import copy
import json
import os
import random
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

from .graph_schema import (
    ConstraintGraph,
    ConstraintNode,
    BlockSpec,
    BlockConstraintSet,
    SelectionNode,
    SelectionBranch,
)
from .step7_instruction_synthesis import synthesize_instruction_bundle
from .step8_prompt_refinement import refine_instruction_prompt
from .utils.export_utils import save_graph_outputs, write_json, write_text


_HARD_NUMERIC_CHECKS = {
    "min_word_count",
    "min_paragraphs",
    "min_numbered_items",
    "must_list_n_subpoints",
    "decimal_places",
}


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _strip_think_prefix(text: str) -> tuple[str, bool]:
    if not text:
        return text, False
    marker = "</think>"
    if marker in text:
        _, tail = text.split(marker, 1)
        return tail.lstrip(), True
    return text, False


def _node_from_dict(data: Dict[str, Any]) -> ConstraintNode:
    return ConstraintNode(
        cid=data.get("cid", ""),
        desc=data.get("desc", ""),
        scope=data.get("scope", ""),
        verifier_spec=data.get("verifier_spec") or data.get("verifier") or {},
        priority_level=data.get("priority_level", 2),
        trace_to=data.get("trace_to"),
        derived_from=data.get("derived_from"),
        change_type=data.get("change_type"),
    )


def _graph_from_serialized(data: Dict[str, Any]) -> ConstraintGraph:
    block_specs_raw = data.get("block_specs") or data.get("blocks") or []
    block_specs = [
        BlockSpec(
            block_id=b.get("block_id", ""),
            intent=b.get("intent", ""),
            text_span=b.get("text_span", ""),
            order_index=b.get("order_index", 0),
            is_alternate=b.get("is_alternate", False),
            origin_block=b.get("origin_block"),
        )
        for b in block_specs_raw
    ]

    global_constraints = [_node_from_dict(n) for n in data.get("global_constraints", [])]

    block_constraint_sets = []
    for bcs in data.get("block_constraint_sets", []):
        block_constraint_sets.append(
            BlockConstraintSet(
                block_id=bcs.get("block_id", ""),
                logic_type=bcs.get("logic_type", "AND"),
                constraints=[_node_from_dict(n) for n in bcs.get("constraints", [])],
            )
        )

    selections = []
    for s in data.get("selections", []):
        selections.append(
            SelectionNode(
                sid=s.get("sid", ""),
                condition=s.get("condition", ""),
                trace_to=s.get("trace_to", ""),
                branch_real=SelectionBranch(
                    block_id=(s.get("branch_real") or {}).get("block_id", ""),
                    constraints=(s.get("branch_real") or {}).get("constraints", []) or [],
                ),
                branch_alt=SelectionBranch(
                    block_id=(s.get("branch_alt") or {}).get("block_id", ""),
                    constraints=(s.get("branch_alt") or {}).get("constraints", []) or [],
                ),
                derived_from=s.get("derived_from", "step5"),
                selection_type=s.get("selection_type", "local"),
                merge_point=s.get("merge_point"),
                truncated=s.get("truncated", False),
                alt_path_blocks=s.get("alt_path_blocks", []) or [],
            )
        )

    return ConstraintGraph(
        seed_task=data.get("seed_task", ""),
        global_constraints=global_constraints,
        block_specs=block_specs,
        block_constraint_sets=block_constraint_sets,
        selections=selections,
        meta=data.get("meta", {}) or {},
    )


def _ensure_priority_levels(graph: ConstraintGraph) -> None:
    for node in graph.global_constraints:
        if node.priority_level not in (1, 2):
            node.priority_level = 2
    for bcs in graph.block_constraint_sets:
        for node in bcs.constraints:
            if node.priority_level not in (1, 2):
                node.priority_level = 2


def _iter_constraint_nodes(graph: ConstraintGraph) -> List[ConstraintNode]:
    nodes: List[ConstraintNode] = []
    nodes.extend(graph.global_constraints)
    for bcs in graph.block_constraint_sets:
        nodes.extend(bcs.constraints)
    return nodes


def _constraint_fingerprint(node: ConstraintNode) -> Tuple[str, str, str, int, str]:
    return (
        (node.desc or "").strip(),
        (node.scope or "").strip(),
        json.dumps(node.verifier_spec or {}, sort_keys=True, ensure_ascii=True),
        int(node.priority_level or 2),
        (node.trace_to or "").strip(),
    )


def _constraint_map_by_cid(graph: ConstraintGraph) -> Dict[str, ConstraintNode]:
    mapped: Dict[str, ConstraintNode] = {}
    for node in _iter_constraint_nodes(graph):
        mapped[node.cid] = node
    return mapped


def _annotate_turn_change_types(turns: List[ConstraintGraph],
                                explicit_modify_cids_by_turn: Optional[List[set[str]]] = None) -> None:
    if not turns:
        return
    explicit_modify_cids_by_turn = explicit_modify_cids_by_turn or []
    prev_nodes: Dict[str, ConstraintNode] = {}
    for turn_idx, turn_graph in enumerate(turns):
        curr_nodes = _constraint_map_by_cid(turn_graph)
        explicit_modify = (
            explicit_modify_cids_by_turn[turn_idx]
            if turn_idx < len(explicit_modify_cids_by_turn)
            else set()
        )
        if turn_idx == 0:
            for node in curr_nodes.values():
                node.change_type = "add"
            prev_nodes = curr_nodes
            continue

        for cid, node in curr_nodes.items():
            if cid in explicit_modify:
                node.change_type = "modify"
                continue
            prev_node = prev_nodes.get(cid)
            if prev_node is None:
                node.change_type = "add"
            elif _constraint_fingerprint(prev_node) != _constraint_fingerprint(node):
                node.change_type = "modify"
            else:
                node.change_type = "unchanged"
        prev_nodes = curr_nodes


def _total_constraint_count(graph: ConstraintGraph) -> int:
    return len(graph.global_constraints) + sum(len(bcs.constraints) for bcs in graph.block_constraint_sets)


def _selection_cids(graph: ConstraintGraph) -> set:
    cids = set()
    for sel in graph.selections:
        cids.update(sel.branch_real.constraints)
        cids.update(sel.branch_alt.constraints)
    return cids


def _would_break_selection(graph: ConstraintGraph, cid: str) -> bool:
    for sel in graph.selections:
        if cid in sel.branch_real.constraints and len(sel.branch_real.constraints) <= 1:
            return True
        if cid in sel.branch_alt.constraints and len(sel.branch_alt.constraints) <= 1:
            return True
    return False


def _remove_cids(graph: ConstraintGraph, remove_cids: List[str]) -> None:
    remove_set = set(remove_cids)
    if remove_set:
        graph.global_constraints = [n for n in graph.global_constraints if n.cid not in remove_set]
        for bcs in graph.block_constraint_sets:
            bcs.constraints = [n for n in bcs.constraints if n.cid not in remove_set]

        for sel in graph.selections:
            sel.branch_real.constraints = [cid for cid in sel.branch_real.constraints if cid not in remove_set]
            sel.branch_alt.constraints = [cid for cid in sel.branch_alt.constraints if cid not in remove_set]


def _prune_empty_selections(graph: ConstraintGraph) -> None:
    kept = []
    for sel in graph.selections:
        if sel.branch_real.constraints and sel.branch_alt.constraints:
            kept.append(sel)
    graph.selections = kept


def _pick_removal_candidates(graph: ConstraintGraph) -> List[str]:
    candidates = []
    for node in graph.global_constraints:
        if not _would_break_selection(graph, node.cid):
            candidates.append(node.cid)
    for bcs in graph.block_constraint_sets:
        for node in bcs.constraints:
            if not _would_break_selection(graph, node.cid):
                candidates.append(node.cid)
    return candidates


def _round_to_ten(value: float) -> int:
    return int(((int(value) + 9) // 10) * 10)


def _mutate_numeric_constraint(node: ConstraintNode, rng: random.Random) -> Optional[ConstraintNode]:
    check = (node.verifier_spec or {}).get("check")
    args = dict((node.verifier_spec or {}).get("args", {}))
    if check not in _HARD_NUMERIC_CHECKS:
        return None

    if check == "min_word_count":
        base = args.get("min_words")
        if not isinstance(base, int):
            return None
        if base <= 80:
            low, high = 0.5, 1.8
        elif base <= 200:
            low, high = 0.3, 1.2
        else:
            low, high = 0.2, 0.8
        new_val = _round_to_ten(base * (1 + rng.uniform(low, high)))
        if new_val <= base:
            new_val = base + 10
        args["min_words"] = new_val
        desc = f"Provide a sufficiently detailed explanation with at least {new_val} words."
    elif check == "min_numbered_items":
        key = "n" if "n" in args else "min_items"
        base = args.get(key)
        if not isinstance(base, int):
            return None
        if base <= 3:
            low, high = 0.5, 2.5
        elif base <= 6:
            low, high = 0.4, 1.6
        elif base <= 10:
            low, high = 0.3, 1.0
        else:
            low, high = 0.2, 0.6
        new_val = int(base * (1 + rng.uniform(low, high)))
        if new_val <= base:
            new_val = base + 1
        args[key] = new_val
        desc = f"Provide a numbered list with at least {new_val} items."
    elif check == "must_list_n_subpoints":
        if "min_items" in args:
            key = "min_items"
        elif "n" in args:
            key = "n"
        elif "min_subpoints" in args:
            key = "min_subpoints"
        else:
            return None
        base = args.get(key)
        if not isinstance(base, int):
            return None
        if base <= 3:
            low, high = 0.5, 2.5
        elif base <= 6:
            low, high = 0.4, 1.6
        elif base <= 10:
            low, high = 0.3, 1.0
        else:
            low, high = 0.2, 0.6
        new_val = int(base * (1 + rng.uniform(low, high)))
        if new_val <= base:
            new_val = base + 1
        args[key] = new_val
        desc = f"Provide a bulleted list in this block with at least {new_val} items."
    elif check == "min_paragraphs":
        key = "min_paras" if "min_paras" in args else "min_paragraphs"
        base = args.get(key)
        if not isinstance(base, int):
            return None
        low, high = 0.5, 1.5
        new_val = int(base * (1 + rng.uniform(low, high)))
        if new_val <= base:
            new_val = base + 1
        args[key] = new_val
        desc = f"Organize this block into at least {new_val} logical paragraphs."
    elif check == "decimal_places":
        base = args.get("places")
        if not isinstance(base, int) or base >= 6:
            return None
        low, high = 0.3, 1.0
        delta = max(1, int(base * rng.uniform(low, high) + 0.99))
        new_val = min(6, base + delta)
        if new_val <= base:
            new_val = min(6, base + 1)
        args["places"] = new_val
        desc = f"Keep numerical values in this block to {new_val} decimal places consistently."
    else:
        return None

    return ConstraintNode(
        cid=f"{node.cid}__m2_alt",
        desc=desc,
        scope=node.scope,
        verifier_spec={"check": check, "args": args},
        priority_level=1,
        trace_to=node.trace_to,
        derived_from="augmenter_m2",
    )


def _pick_block_candidates(graph: ConstraintGraph) -> List[str]:
    alt_blocks = {b.block_id for b in graph.block_specs if b.is_alternate}
    return [bcs.block_id for bcs in graph.block_constraint_sets if bcs.block_id not in alt_blocks]


def _apply_priority_injection(graph: ConstraintGraph,
                              rng: random.Random,
                              min_blocks: int = 1,
                              max_blocks: int = 2) -> Dict[str, Any]:
    block_ids = _pick_block_candidates(graph)
    if not block_ids:
        return {"priority_enhanced": False}
    pick_blocks = min(max_blocks, len(block_ids))
    num_blocks = rng.randint(min_blocks, pick_blocks) if pick_blocks >= min_blocks else pick_blocks
    chosen_blocks = rng.sample(block_ids, num_blocks) if num_blocks > 0 else []
    chosen_cids = []

    for bcs in graph.block_constraint_sets:
        if bcs.block_id not in chosen_blocks or not bcs.constraints:
            continue
        node = rng.choice(bcs.constraints)
        node.priority_level = 1
        chosen_cids.append(node.cid)

    return {
        "priority_enhanced": bool(chosen_cids),
        "priority_blocks": chosen_blocks,
        "priority_cids": chosen_cids,
    }


def _filter_selections_by_allowed(graph: ConstraintGraph, allowed_cids: set) -> None:
    kept = []
    for sel in graph.selections:
        real = [cid for cid in sel.branch_real.constraints if cid in allowed_cids]
        alt = [cid for cid in sel.branch_alt.constraints if cid in allowed_cids]
        if real and alt:
            sel.branch_real.constraints = real
            sel.branch_alt.constraints = alt
            kept.append(sel)
    graph.selections = kept


def _split_constraints(rng: random.Random, constraints: List[ConstraintNode], segments: int) -> List[List[ConstraintNode]]:
    if segments <= 1 or len(constraints) <= 1:
        return [constraints]
    segments = min(segments, len(constraints))
    shuffled = constraints[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    sizes: List[int]
    if segments >= 3 and n >= 4:
        size1 = max(2, min(n - 2, int(round(n * rng.uniform(0.3, 0.45)))))
        size2 = max(1, min(n - size1 - 1, int(round(n * rng.uniform(0.2, 0.3)))))
        size3 = n - size1 - size2
        if size3 <= 0:
            size3 = 1
            if size2 > 1:
                size2 -= 1
            elif size1 > 2:
                size1 -= 1
        sizes = [size1, size2, size3]
    else:
        if n <= 3:
            size1 = 1
        else:
            size1 = max(2, min(n - 1, int(round(n * rng.uniform(0.35, 0.5)))))
        sizes = [size1, n - size1]

    chunks: List[List[ConstraintNode]] = []
    cursor = 0
    for size in sizes:
        chunks.append(shuffled[cursor:cursor + size])
        cursor += size
    if cursor < n:
        chunks[-1].extend(shuffled[cursor:])
    return chunks


def _build_curriculum_chain(graph: ConstraintGraph, rng: random.Random) -> Optional[Dict[int, ConstraintGraph]]:
    _ensure_priority_levels(graph)
    total = _total_constraint_count(graph)
    if total < 15:
        return None

    chain: Dict[int, ConstraintGraph] = {5: copy.deepcopy(graph)}
    working = copy.deepcopy(graph)

    for level in [4, 3, 2, 1]:
        remaining_steps = level - 1
        min_remaining = 5 + remaining_steps
        max_removable = _total_constraint_count(working) - min_remaining
        candidates = _pick_removal_candidates(working)
        max_remove = min(3, max_removable, len(candidates))
        if max_remove < 1:
            return None
        remove_count = rng.randint(1, max_remove)
        remove_cids = rng.sample(candidates, remove_count)
        _remove_cids(working, remove_cids)
        _prune_empty_selections(working)
        chain[level] = copy.deepcopy(working)

    return chain


def _build_m1_turns(graph: ConstraintGraph, rng: random.Random, min_blocks: int = 3, max_blocks: int = 6) -> Optional[List[ConstraintGraph]]:
    _ensure_priority_levels(graph)

    if not graph.global_constraints and not any(bcs.constraints for bcs in graph.block_constraint_sets):
        return None

    global_activation: Dict[str, int] = {}
    global_nodes = copy.deepcopy(graph.global_constraints)
    if global_nodes:
        rng.shuffle(global_nodes)
        n = len(global_nodes)
        if n == 1:
            global_activation[global_nodes[0].cid] = 1
        elif n == 2:
            global_activation[global_nodes[0].cid] = 1
            global_activation[global_nodes[1].cid] = 2
        else:
            c1 = max(1, int(round(n * 0.40)))
            c2 = max(1, int(round(n * 0.30)))
            if c1 + c2 >= n:
                c2 = max(1, n - c1 - 1)
            c3 = n - c1 - c2
            if c3 <= 0:
                c3 = 1
                if c2 > 1:
                    c2 -= 1
                elif c1 > 1:
                    c1 -= 1

            for node in global_nodes[:c1]:
                global_activation[node.cid] = 1
            for node in global_nodes[c1:c1 + c2]:
                global_activation[node.cid] = 2
            for node in global_nodes[c1 + c2:]:
                global_activation[node.cid] = 3

    block_activation: Dict[str, Dict[str, int]] = {}
    toggle_two = False
    for bcs in graph.block_constraint_sets:
        nodes = copy.deepcopy(bcs.constraints)
        if not nodes:
            continue
        rng.shuffle(nodes)
        per_block: Dict[str, int] = {}
        if len(nodes) == 1:
            per_block[nodes[0].cid] = 1
        elif len(nodes) == 2:
            per_block[nodes[0].cid] = 1
            per_block[nodes[1].cid] = 2 if not toggle_two else 3
            toggle_two = not toggle_two
        else:
            per_block[nodes[0].cid] = 1
            per_block[nodes[1].cid] = 2
            per_block[nodes[2].cid] = 3
            next_turn = 2
            for node in nodes[3:]:
                per_block[node.cid] = next_turn
                next_turn = 3 if next_turn == 2 else 2
        block_activation[bcs.block_id] = per_block

    turns: List[ConstraintGraph] = []
    for turn_idx in range(1, 4):
        g = copy.deepcopy(graph)
        g.global_constraints = [
            node for node in g.global_constraints
            if global_activation.get(node.cid, 1) <= turn_idx
        ]
        for bcs in g.block_constraint_sets:
            activation = block_activation.get(bcs.block_id, {})
            bcs.constraints = [
                node for node in bcs.constraints
                if activation.get(node.cid, 1) <= turn_idx
            ]
        allowed = {n.cid for n in g.global_constraints}
        for bcs in g.block_constraint_sets:
            allowed.update(n.cid for n in bcs.constraints)
        _filter_selections_by_allowed(g, allowed)
        turns.append(g)

    totals = [_total_constraint_count(t) for t in turns]
    if not (totals[0] < totals[1] < totals[2]):
        return None

    _annotate_turn_change_types(turns)
    return turns


def _replace_selection_cid(graph: ConstraintGraph, old_cid: str, new_cid: str) -> None:
    for sel in graph.selections:
        sel.branch_real.constraints = [new_cid if cid == old_cid else cid for cid in sel.branch_real.constraints]
        sel.branch_alt.constraints = [new_cid if cid == old_cid else cid for cid in sel.branch_alt.constraints]


def _build_m2_turns(graph: ConstraintGraph, rng: random.Random) -> Optional[List[ConstraintGraph]]:
    _ensure_priority_levels(graph)
    alt_blocks = {b.block_id for b in graph.block_specs if b.is_alternate}
    selection_cids = _selection_cids(graph)
    candidates_by_check: Dict[str, List[Tuple[str, ConstraintNode]]] = {}
    selection_candidates_by_check: Dict[str, List[Tuple[str, ConstraintNode]]] = {}
    for bcs in graph.block_constraint_sets:
        if bcs.block_id in alt_blocks:
            continue
        for node in bcs.constraints:
            check = (node.verifier_spec or {}).get("check")
            if check not in _HARD_NUMERIC_CHECKS:
                continue
            if node.cid in selection_cids:
                selection_candidates_by_check.setdefault(check, []).append((bcs.block_id, node))
            else:
                candidates_by_check.setdefault(check, []).append((bcs.block_id, node))

    def _pick_from(check_map: Dict[str, List[Tuple[str, ConstraintNode]]]) -> Optional[Tuple[str, ConstraintNode]]:
        if not check_map:
            return None
        checks = sorted(check_map.keys())
        weights = [2 if check != "min_word_count" else 1 for check in checks]
        chosen_check = rng.choices(checks, weights=weights, k=1)[0]
        return rng.choice(check_map[chosen_check])

    preferred = {k: v for k, v in candidates_by_check.items() if k != "min_word_count"}
    fallback_sel = {k: v for k, v in selection_candidates_by_check.items() if k != "min_word_count"}

    pick = _pick_from(preferred)
    if pick is None:
        pick = _pick_from(fallback_sel)
    if pick is None:
        pick = _pick_from(candidates_by_check)
    if pick is None:
        pick = _pick_from(selection_candidates_by_check)
    if pick is None:
        return None

    block_id, node = pick
    mutated = _mutate_numeric_constraint(node, rng)
    if not mutated:
        return None

    turn1 = copy.deepcopy(graph)
    for bcs in turn1.block_constraint_sets:
        if bcs.block_id != block_id:
            continue
        for idx, existing in enumerate(bcs.constraints):
            if existing.cid == node.cid:
                bcs.constraints[idx] = mutated
                break
    _replace_selection_cid(turn1, node.cid, mutated.cid)

    turn2 = copy.deepcopy(graph)
    _replace_selection_cid(turn2, mutated.cid, node.cid)
    turns = [turn1, turn2]
    # Turn-2 restores the prior constraint text/value and should be treated as a modify event.
    _annotate_turn_change_types(turns, explicit_modify_cids_by_turn=[set(), {node.cid}])
    return turns


def _count_blocks(graph: ConstraintGraph) -> tuple[int, int]:
    primary_blocks = [b for b in graph.block_specs if not b.is_alternate]
    alt_blocks = [b for b in graph.block_specs if b.is_alternate]
    return len(primary_blocks), len(alt_blocks)


def _count_constraints(graph: ConstraintGraph) -> tuple[int, int, int, int]:
    global_count = len(graph.global_constraints)
    block_count = sum(len(bcs.constraints) for bcs in graph.block_constraint_sets)
    priority_1 = 0
    for node in graph.global_constraints:
        if node.priority_level == 1:
            priority_1 += 1
    for bcs in graph.block_constraint_sets:
        for node in bcs.constraints:
            if node.priority_level == 1:
                priority_1 += 1
    total = global_count + block_count
    return global_count, block_count, total, priority_1


def _variant_label(graph: ConstraintGraph, sample_id: str) -> str:
    meta = graph.meta or {}
    if meta.get("curriculum_level"):
        return f"g{meta.get('curriculum_level')}"
    if meta.get("mode") == "m1" and meta.get("turn_index"):
        return f"m1_t{meta.get('turn_index')}"
    if meta.get("mode") == "m2" and meta.get("turn_index"):
        return f"m2_t{meta.get('turn_index')}"
    if "__" in sample_id:
        return sample_id.split("__", 1)[1]
    return "base"


def _save_sample(graph: ConstraintGraph,
                 sample_id: str,
                 base_dir: str,
                 *,
                 enable_step8: bool = True) -> Dict[str, Any]:
    graphs_dir = os.path.join(base_dir, "graphs")
    instructions_dir = os.path.join(base_dir, "instructions")
    reports_dir = os.path.join(base_dir, "reports")
    ts_utc = datetime.now(timezone.utc).isoformat()

    saved_paths = save_graph_outputs(
        graph,
        sample_id=sample_id,
        base_dir=graphs_dir,
    )

    bundle = synthesize_instruction_bundle(graph)
    machine_prompt_raw = bundle.get("machine_prompt", "")

    raw_prompt_path = os.path.join(instructions_dir, f"{sample_id}.machine.txt")
    write_text(raw_prompt_path, machine_prompt_raw)

    polish_result = refine_instruction_prompt(
        machine_prompt=machine_prompt_raw,
        seed_task=graph.seed_task,
        enable=enable_step8,
    )
    machine_prompt = polish_result.get("text", machine_prompt_raw)
    machine_prompt, stripped_think = _strip_think_prefix(machine_prompt)
    polish_result["stripped_think"] = stripped_think
    bundle["machine_prompt_original"] = machine_prompt_raw
    bundle["machine_prompt"] = machine_prompt
    bundle["step8_polish"] = {k: v for k, v in polish_result.items() if k != "text"}

    prompt_path = os.path.join(instructions_dir, f"{sample_id}.prompt.txt")
    write_text(prompt_path, machine_prompt)

    eval_record = {
        "sample_id": sample_id,
        "timestamp_utc": ts_utc,
        "eval_protocol": bundle.get("eval_protocol", {}),
        "meta": bundle.get("meta", {}),
    }
    eval_path = os.path.join(reports_dir, f"{sample_id}.eval.json")
    write_json(eval_path, eval_record)

    bundle_debug = {
        "sample_id": sample_id,
        "timestamp_utc": ts_utc,
        **bundle,
    }
    bundle_path = os.path.join(reports_dir, f"{sample_id}.bundle.json")
    write_json(bundle_path, bundle_debug)

    block_count, alt_block_count = _count_blocks(graph)
    global_count, block_constraint_count, total_constraints, priority_1_count = _count_constraints(graph)
    prompt_text = machine_prompt or ""
    prompt_length_chars = len(prompt_text)
    prompt_length_words = len(prompt_text.split())
    machine_length_chars = len(machine_prompt_raw or "")

    return {
        "graph_json": saved_paths.get("graph_json"),
        "graph_mmd": saved_paths.get("mermaid_mmd"),
        "machine_prompt": raw_prompt_path,
        "prompt": prompt_path,
        "eval": eval_path,
        "bundle": bundle_path,
        "stats": {
            "sample_id": sample_id,
            "variant_type": _variant_label(graph, sample_id),
            "curriculum_level": graph.meta.get("curriculum_level"),
            "mode": graph.meta.get("mode"),
            "turn_index": graph.meta.get("turn_index"),
            "turn_total": graph.meta.get("turn_total"),
            "blocks": block_count,
            "alt_blocks": alt_block_count,
            "global_constraints": global_count,
            "block_constraints": block_constraint_count,
            "total_constraints": total_constraints,
            "priority_level_1": priority_1_count,
            "conditional_branches": len(graph.selections),
            "prompt_length_chars": prompt_length_chars,
            "prompt_length_words": prompt_length_words,
            "machine_prompt_length_chars": machine_length_chars,
            "step8_used_llm": bool(bundle.get("step8_polish", {}).get("used_llm")),
            "step8_reason": bundle.get("step8_polish", {}).get("reason"),
            "paths": {
                "graph_json": saved_paths.get("graph_json"),
                "prompt": prompt_path,
                "eval": eval_path,
                "bundle": bundle_path,
            },
        },
    }


def _write_parent_log(parent_id: str,
                      variants: List[Dict[str, Any]],
                      base_dir: str,
                      run_id: str,
                      run_meta: Dict[str, Any]) -> None:
    logs_dir = os.path.join(base_dir, "logs", "augmenter")
    os.makedirs(logs_dir, exist_ok=True)
    payload = {
        "parent_sample_id": parent_id,
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_meta": run_meta,
        "variants": variants,
    }
    out_path = os.path.join(logs_dir, f"{parent_id}.summary.json")
    write_json(out_path, payload)


def augment_graph(graph: ConstraintGraph,
                  sample_id: str,
                  base_dir: str,
                  rng: random.Random,
                  *,
                  priority_ratio: float = 0.5,
                  enable_curriculum: bool = True,
                  enable_m1: bool = True,
                  enable_m2: bool = True,
                  enable_step8: bool = True,
                  run_id: Optional[str] = None,
                  run_meta: Optional[Dict[str, Any]] = None) -> List[str]:
    outputs: List[str] = []
    log_entries: List[Dict[str, Any]] = []
    run_meta = run_meta or {}

    if enable_curriculum:
        chain = _build_curriculum_chain(graph, rng)
        if chain:
            for level in [1, 2, 3, 4, 5]:
                g = copy.deepcopy(chain[level])
                meta = dict(g.meta or {})
                meta.update({
                    "parent_sample_id": sample_id,
                    "curriculum_level": level,
                })
                g.meta = meta
                if rng.random() < priority_ratio:
                    info = _apply_priority_injection(g, rng)
                    g.meta.update(info)
                _ensure_priority_levels(g)
                saved = _save_sample(g, f"{sample_id}__g{level}", base_dir, enable_step8=enable_step8)
                log_entries.append(saved.get("stats", {}))
                outputs.append(f"{sample_id}__g{level}")

    if enable_m1:
        turns = _build_m1_turns(graph, rng)
        if turns:
            for idx, g in enumerate(turns, start=1):
                meta = dict(g.meta or {})
                meta.update({
                    "parent_sample_id": sample_id,
                    "mode": "m1",
                    "sequence_id": f"{sample_id}__m1",
                    "turn_index": idx,
                    "turn_total": len(turns),
                    "turn_notice": True,
                    "delta_only_instruction": idx > 1,
                    "delta_reference": "previous_turn",
                })
                g.meta = meta
                saved = _save_sample(g, f"{sample_id}__m1_t{idx}", base_dir, enable_step8=enable_step8)
                log_entries.append(saved.get("stats", {}))
                outputs.append(f"{sample_id}__m1_t{idx}")

    if enable_m2:
        turns = _build_m2_turns(graph, rng)
        if turns:
            for idx, g in enumerate(turns, start=1):
                meta = dict(g.meta or {})
                meta.update({
                    "parent_sample_id": sample_id,
                    "mode": "m2",
                    "sequence_id": f"{sample_id}__m2",
                    "turn_index": idx,
                    "turn_total": len(turns),
                    "turn_notice": True,
                    "delta_only_instruction": idx > 1,
                    "delta_reference": "previous_turn",
                })
                g.meta = meta
                saved = _save_sample(g, f"{sample_id}__m2_t{idx}", base_dir, enable_step8=enable_step8)
                log_entries.append(saved.get("stats", {}))
                outputs.append(f"{sample_id}__m2_t{idx}")

    if log_entries:
        _write_parent_log(
            parent_id=sample_id,
            variants=log_entries,
            base_dir=base_dir,
            run_id=run_id or datetime.now(timezone.utc).isoformat(),
            run_meta=run_meta,
        )

    return outputs


def _default_sample_id_from_path(path: str) -> str:
    base = os.path.basename(path)
    if base.endswith(".graph.json"):
        return base[:-len(".graph.json")]
    return os.path.splitext(base)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment ConstraintGraph outputs (curriculum, M1, M2).")
    parser.add_argument("--graph-json", type=str, help="Path to a .graph.json file.")
    parser.add_argument("--graphs-dir", type=str, help="Directory containing .graph.json files.")
    parser.add_argument("--sample-id", type=str, default="", help="Override sample_id for output naming.")
    parser.add_argument("--output-dir", type=str, default="", help="Base output dir (same layout as pipeline).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--priority-ratio", type=float, default=0.5, help="Probability of priority injection per sample.")
    parser.add_argument("--disable-curriculum", action="store_true", help="Disable curriculum generation.")
    parser.add_argument("--disable-m1", action="store_true", help="Disable M1 multi-turn generation.")
    parser.add_argument("--disable-m2", action="store_true", help="Disable M2 multi-turn generation.")
    parser.add_argument("--disable-step8", action="store_true", help="Disable Step8 prompt polish.")
    parser.add_argument("--include-augmented", action="store_true", help="Also process graphs with '__' in the sample_id.")
    args = parser.parse_args()

    if not args.graph_json and not args.graphs_dir:
        raise SystemExit("Provide --graph-json or --graphs-dir.")

    rng = random.Random(args.seed)
    output_dir = args.output_dir
    run_id = datetime.now(timezone.utc).isoformat()
    run_meta = {
        "seed": args.seed,
        "priority_ratio": args.priority_ratio,
        "enable_curriculum": not args.disable_curriculum,
        "enable_m1": not args.disable_m1,
        "enable_m2": not args.disable_m2,
        "enable_step8": not args.disable_step8,
    }

    def _run_one(path: str) -> None:
        sample_id = args.sample_id or _default_sample_id_from_path(path)
        if not args.include_augmented and ("__" in os.path.basename(path) or "__" in sample_id):
            return
        base_dir = output_dir or os.path.dirname(os.path.dirname(path))
        graph_data = _load_json(path)
        graph = _graph_from_serialized(graph_data)
        augment_graph(
            graph=graph,
            sample_id=sample_id,
            base_dir=base_dir,
            rng=rng,
            priority_ratio=args.priority_ratio,
            enable_curriculum=not args.disable_curriculum,
            enable_m1=not args.disable_m1,
            enable_m2=not args.disable_m2,
            enable_step8=not args.disable_step8,
            run_id=run_id,
            run_meta=run_meta,
        )

    if args.graph_json:
        _run_one(args.graph_json)
    else:
        for name in os.listdir(args.graphs_dir):
            if not name.endswith(".graph.json"):
                continue
            if not args.include_augmented and "__" in name:
                continue
            _run_one(os.path.join(args.graphs_dir, name))


if __name__ == "__main__":
    main()
