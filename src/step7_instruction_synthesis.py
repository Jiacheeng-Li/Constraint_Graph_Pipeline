

"""
step7_instruction_synthesis.py

第七步：基于约束关系图生成高复杂度指令规范

输入项：
    - 约束关系图对象（来自step6_graph_assembly.assemble_constraint_graph）
      或由serialize_graph(graph)生成的JSON安全快照

输出项：
    1. machine_prompt：
        直接提供给待评估模型的自然语言指令提示。包含：
        - 整体任务说明
        - 全局约束条件（风格、安全性、长度、语言等）
        - 分阶段结构要求（逐模块职责说明）
        - 条件分支逻辑（if/else执行要求）

    2. eval_protocol：
        面向评估器/评判器的结构化评分协议。
        将每个约束与其验证器规格检查对齐，并说明分支评分规则。

    3. meta：
        从graph.meta复制的溯源信息。

注意事项：
    - 第七步不调用任何大语言模型，仅执行确定性模板生成
    - 不重写/美化用户原始任务，以最可检验、可执行的形式呈现任务与约束
    - 机器提示语将直接输入被评估模型，非JSON格式，需符合规范式、命令式、显式化要求
    - 评估协议适用于自动化评估代码、质量审查或裁定流程

机器提示语的风格要求：
    - 整体框架和标题必须使用英语
    - 单个约束描述可保留原始语言（通常为英语祈使句）
    - 分支逻辑必须使用明确的"IF...THEN...ELSE..."句式表达
    - 每个阶段按功能职责而非具体措辞进行描述
"""

from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional
from .graph_schema import ConstraintGraph, BlockSpec, BlockConstraintSet, ConstraintNode, SelectionNode
from .step6_graph_assembly import serialize_graph


# ------------------------------------------------------------
# Helper 1. 生成分阶段（逐段落）规划视图
# （把每个 block 的顺序、功能角色、约束义务整理出来）
# ------------------------------------------------------------

def _render_block_plan(block_specs: List[BlockSpec],
                        block_sets: List[BlockConstraintSet]) -> List[Dict[str, Any]]:
    """
    Merge block_specs (ordering + intent) and block_constraint_sets (constraints + logic_type)
    into a linear stage plan.

    Returns a list of dicts, one per stage/block:
        {
            "block_id": "B2",
            "role": "Main Analysis / Examples",
            "logic_type": "AND" | "sub-chain",
            "requirements": [
                {"cid": "B2_C1", "desc": "Give at least two concrete real-world examples.", "verifier": {...}},
                ...
            ]
        }
    If a block has no explicit constraints, we still include it with an empty requirements list.
    """
    # 建立 block_id -> BlockConstraintSet 的映射，方便后续按块查约束
    set_map: Dict[str, BlockConstraintSet] = {
        bcs.block_id: bcs for bcs in block_sets
    }

    # 按照 order_index 排序，保持原回答的真实推进顺序
    sorted_specs = sorted(block_specs, key=lambda b: (b.order_index, b.block_id))

    rendered: List[Dict[str, Any]] = []
    for spec in sorted_specs:
        if spec.is_alternate:
            continue
        bset = set_map.get(spec.block_id)
        if not bset:
            # 这个 block 没有显式的本地约束（极少数情况也可能出现）
            rendered.append({
                "block_id": spec.block_id,
                "role": spec.intent,
                "logic_type": "AND",
                "requirements": [],
            })
            continue

        reqs = []
        for cnode in bset.constraints:
            reqs.append({
                "cid": cnode.cid,
                "desc": cnode.desc,
                "verifier": cnode.verifier_spec,
            })
        rendered.append({
            "block_id": spec.block_id,
            "role": spec.intent,
            "logic_type": bset.logic_type,
            "requirements": reqs,
        })

    return rendered


# ------------------------------------------------------------
# Helper 2. 汇总全局约束
# （这些是在整篇回答中无条件必须满足的规则）
# ------------------------------------------------------------

def _render_global_constraints(global_nodes: List[ConstraintNode]) -> List[Dict[str, Any]]:
    out = []
    for node in global_nodes:
        out.append({
            "cid": node.cid,
            "desc": node.desc,
            "verifier": node.verifier_spec,
        })
    return out


# ------------------------------------------------------------
# Helper 3. 处理条件化分支（SelectionNode）
# （把 if/then/else 的分支结构展开成可读+可打分的形式）
# ------------------------------------------------------------

def _render_selections(selections: List[SelectionNode],
                        block_specs: List[BlockSpec],
                        block_sets: List[BlockConstraintSet]) -> List[Dict[str, Any]]:
    """
    Convert SelectionNode objects into an explicit IF / THEN / ELSE view for scoring and prompting.

    Returns: list of dicts like
        {
            "sid": "SEL_B3",
            "where": "B3 Conclusion / Outlook / Recommendation",
            "condition": "If the stance is critical/negative",
            "branch_real": [ {"cid": ..., "desc": ..., "verifier": ...}, ... ],
            "branch_alt":  [ {"cid": ..., "desc": ..., "verifier": ...}, ... ],
        }
    """
    # 建立 cid -> ConstraintNode 的索引，后面用它把 cid 还原为具体约束描述
    cid_to_node: Dict[str, ConstraintNode] = {}
    for bcs in block_sets:
        for cnode in bcs.constraints:
            cid_to_node[cnode.cid] = cnode

    # block_id -> intent，用来生成可读的 "where" 字段（告诉分支发生在哪个阶段）
    block_intent_map: Dict[str, str] = {bs.block_id: bs.intent for bs in block_specs}

    rendered: List[Dict[str, Any]] = []
    for sel in selections:
        real_list = []
        for cid in sel.branch_real.constraints:
            node = cid_to_node.get(cid)
            if node:
                real_list.append({
                    "cid": cid,
                    "desc": node.desc,
                    "verifier": node.verifier_spec,
                })
            else:
                real_list.append({"cid": cid, "desc": None, "verifier": None})

        alt_list = []
        for cid in sel.branch_alt.constraints:
            node = cid_to_node.get(cid)
            if node:
                alt_list.append({
                    "cid": cid,
                    "desc": node.desc,
                    "verifier": node.verifier_spec,
                })
            else:
                alt_list.append({"cid": cid, "desc": None, "verifier": None})

        # Group alternate-branch constraints by their true owning block via trace_to
        alt_by_block: Dict[str, List[Dict[str, Any]]] = {}
        alt_path_ids = list(sel.alt_path_blocks or [sel.branch_alt.block_id])
        for block_id in alt_path_ids:
            grouped: List[Dict[str, Any]] = []
            for cid in sel.branch_alt.constraints:
                node = cid_to_node.get(cid)
                node_block = node.trace_to if (node and node.trace_to) else sel.branch_alt.block_id
                if node and node_block == block_id:
                    grouped.append({
                        "cid": cid,
                        "desc": node.desc,
                        "verifier": node.verifier_spec,
                    })
            if not grouped:
                fallback_items = [
                    item for item in alt_list
                    if item.get("cid", "").startswith(f"{block_id}_")
                ]
                grouped.extend(fallback_items)
            alt_by_block[block_id] = grouped

        # Group real-path constraints for later hierarchical rendering
        real_by_block: Dict[str, List[Dict[str, Any]]] = {}
        real_path_blocks: List[str] = []
        default_real_block = sel.branch_real.block_id or sel.trace_to
        real_path_blocks.append(default_real_block)
        for cid in sel.branch_real.constraints:
            node = cid_to_node.get(cid)
            if not node:
                continue
            block_id = node.trace_to or default_real_block
            real_by_block.setdefault(block_id, []).append({
                "cid": cid,
                "desc": node.desc,
                "verifier": node.verifier_spec,
            })
            if block_id not in real_path_blocks:
                real_path_blocks.append(block_id)
        if sel.trace_to and sel.trace_to not in real_path_blocks:
            real_path_blocks.insert(0, sel.trace_to)

        rendered.append({
            "sid": sel.sid,
            "trace_to": sel.trace_to,
            "where": f"{sel.trace_to} {block_intent_map.get(sel.trace_to, '')}",
            "condition": sel.condition,
            "selection_type": sel.selection_type,
            "merge_point": sel.merge_point,
            "truncated": sel.truncated,
            "branch_real_block": sel.branch_real.block_id,
            "branch_alt_block": sel.branch_alt.block_id,
            "alt_path_blocks": list(sel.alt_path_blocks or [sel.branch_alt.block_id]),
            "branch_real": real_list,
            "branch_alt": alt_list,
            "alt_by_block": alt_by_block,
            "real_by_block": real_by_block,
            "real_path_blocks": real_path_blocks,
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


# ------------------------------------------------------------
# Helper 4. 组装 machine_prompt
# （给待评测模型看的最终高复杂度指令）
# ------------------------------------------------------------

def _mk_machine_prompt(seed_task: str,
                       global_rules: List[Dict[str, Any]],
                       block_plan: List[Dict[str, Any]],
                       selections_view: List[Dict[str, Any]],
                       graph: ConstraintGraph) -> str:
    """
    Build the final instruction prompt to feed directly to the evaluated model.

    Goals:
    - English framework / headings.
    - The model must follow global constraints, the staged structure, and any applicable branch.
    - We explicitly communicate IF/THEN/ELSE behavior for conditional branches.
    - We do NOT expose internal IDs (cid) as something the model must output.
    """

    lines: List[str] = []

    lines.append("SYSTEM INSTRUCTIONS:")
    lines.append("")

    mission = seed_task.strip()
    lines.append("1. MISSION BRIEF & DELIVERABLE")
    if mission:
        lines.append(f"   - Primary directive: {mission}")
    else:
        lines.append("   - Follow the staged plan and constraints below to construct the final answer.")
    lines.append("")

    lines.append("2. NON-NEGOTIABLE GLOBAL RULES")
    if global_rules:
        lines.append("   Respect every rule simultaneously across the full response:")
        for idx, rule in enumerate(global_rules, start=1):
            desc = (rule.get("desc") or "").strip()
            if desc:
                lines.append(f"   {idx}. {desc}")
    else:
        lines.append("   - No additional global constraints were supplied.")
    lines.append("")

    selection_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sel in selections_view:
        selection_map[sel["trace_to"]].append(sel)

    block_label_lookup = {spec.block_id: spec.intent for spec in graph.block_specs}

    lines.append("3. STRUCTURED RESPONSE BLUEPRINT")
    lines.append("   Work chronologically through each stage. Tasks must already be described with the condition that governs when they apply.")

    def _append_path(block_ids: List[str],
                     grouped: Dict[str, List[Dict[str, Any]]],
                     default_msg: str,
                     indent: str,
                     show_steps: bool = True,
                     target: Optional[List[str]] = None) -> None:
        target_list = target if target is not None else lines
        for order, block_id in enumerate(block_ids, start=1):
            label = block_label_lookup.get(block_id, block_id)
            obligations = grouped.get(block_id, [])
            if show_steps:
                target_list.append(f"{indent}- Step {order}: {label}")
                if obligations:
                    for item in obligations:
                        desc = (item.get("desc") or "").strip()
                        if desc:
                            target_list.append(f"{indent}    • {desc}")
                else:
                    target_list.append(f"{indent}    • {default_msg.format(stage=label)}")
            else:
                if obligations:
                    for item in obligations:
                        desc = (item.get("desc") or "").strip()
                        if desc:
                            target_list.append(f"{indent}• {desc}")
                else:
                    target_list.append(f"{indent}• {default_msg.format(stage=label)}")

    def _inline_stage_block(stage_idx: int,
                            indent: str,
                            step_no: int) -> Tuple[List[str], str]:
        blk = block_plan[stage_idx]
        block_id = blk["block_id"]
        role = blk["role"] or block_id
        logic_type = (blk["logic_type"] or "AND").lower()
        logic_hint = "Parallel coverage: hit every bullet for this stage." \
            if not logic_type.startswith("sub") else \
            "Sequential micro-steps: address bullets in a flowing order."
        reqs = blk["requirements"]
        stage_selections = selection_map.get(block_id, [])

        snippet: List[str] = []
        snippet.append(f"{indent}- Step {step_no}: {role}")
        snippet.append(f"{indent}    Logic cue: {logic_hint}")
        body_indent = indent + "    "

        if not stage_selections:
            if reqs:
                snippet.append(f"{body_indent}Duties:")
                for r in reqs:
                    desc = (r.get("desc") or "").strip()
                    if desc:
                        snippet.append(f"{body_indent}  • {desc}")
            else:
                snippet.append(f"{body_indent}Duties: Follow the intent of this stage even if no explicit bullet exists.")
            return snippet, block_id

        for sel in stage_selections:
            sel_type = (sel.get("selection_type") or "local").upper()
            condition_text = _condition_statement(sel.get("condition", ""))
            condition_clause = condition_text or "the described condition applies"

            snippet.append(f"{body_indent}{sel_type} branch – when {condition_clause.rstrip('.')}:")
            alt_blocks = sel.get("alt_path_blocks") or [sel.get("branch_alt_block")]
            alt_by_block = sel.get("alt_by_block") or {}
            _append_path(
                alt_blocks,
                alt_by_block,
                "Carry out the alternate storyline requirements for {stage}.",
                indent=body_indent + "  ",
                show_steps=(sel_type == "GLOBAL"),
                target=snippet,
            )
            snippet.append(f"{body_indent}Otherwise at this same decision point:")
            real_blocks = sel.get("real_path_blocks") or [sel.get("branch_real_block") or sel["trace_to"]]
            real_by_block = sel.get("real_by_block") or {}
            _append_path(
                real_blocks,
                real_by_block,
                "Return to the duties defined for {stage}.",
                indent=body_indent + "  ",
                show_steps=(sel_type == "GLOBAL"),
                target=snippet,
            )
            if (sel.get("selection_type") or "").lower() == "global":
                snippet.append(f"{body_indent}Global path note: When this condition reshapes the storyline, continue following the nested steps immediately below.")
        return snippet, block_id

    def _inline_followups_from(start_idx: int,
                               indent: str,
                               start_step: int,
                               consumed: set) -> List[str]:
        """Inline downstream stages when a global branch stays on the default path."""
        timeline: List[str] = []
        next_step = start_step
        for future_idx in range(start_idx + 1, len(block_plan)):
            future_block = block_plan[future_idx]["block_id"]
            if future_block in consumed:
                continue
            lines_chunk, consumed_id = _inline_stage_block(future_idx, indent, next_step)
            if lines_chunk:
                timeline.extend(lines_chunk)
                consumed.add(consumed_id)
                next_step += 1
        return timeline

    consumed_blocks: set = set()

    for idx, blk in enumerate(block_plan, start=1):
        stage_id = blk["block_id"]
        if stage_id in consumed_blocks:
            continue
        role = blk["role"] or f"Stage {idx}"
        logic_type = (blk["logic_type"] or "AND").lower()
        logic_hint = "Parallel coverage: hit every bullet for this stage." \
            if not logic_type.startswith("sub") else \
            "Sequential micro-steps: address bullets in a flowing order."
        reqs = blk["requirements"]

        lines.append("")
        lines.append(f"   Stage {idx} – {role}")
        lines.append(f"      Logic cue: {logic_hint}")
        stage_selections = selection_map.get(stage_id, [])
        if not stage_selections:
            if reqs:
                lines.append("      Timeline duties:")
                for r in reqs:
                    desc = (r.get("desc") or "").strip()
                    if desc:
                        lines.append(f"        • {desc}")
            else:
                lines.append("      Timeline duties: Follow the intent of this stage even if no explicit bullet exists.")
            consumed_blocks.add(stage_id)
            continue

        for sel in stage_selections:
            sel_type = (sel.get("selection_type") or "local").upper()
            condition_text = _condition_statement(sel.get("condition", ""))
            condition_clause = condition_text or "the described condition applies"

            lines.append(f"      {sel_type} branch – when {condition_clause.rstrip('.')}:")
            alt_blocks = sel.get("alt_path_blocks") or [sel.get("branch_alt_block")]
            alt_by_block = sel.get("alt_by_block") or {}
            _append_path(
                alt_blocks,
                alt_by_block,
                "Carry out the alternate storyline requirements for {stage}.",
                indent="        ",
                show_steps=(sel_type == "GLOBAL"),
            )
            lines.append("      Otherwise at this same decision point:")
            real_blocks = sel.get("real_path_blocks") or [sel.get("branch_real_block") or sel["trace_to"]]
            real_by_block = sel.get("real_by_block") or {}
            _append_path(
                real_blocks,
                real_by_block,
                "Return to the duties defined for {stage}.",
                indent="        ",
                show_steps=(sel_type == "GLOBAL"),
            )
            if (sel.get("selection_type") or "").lower() == "global":
                followup = _inline_followups_from(
                    idx - 1,
                    indent="        ",
                    start_step=len(real_blocks) + 1,
                    consumed=consumed_blocks,
                )
                if followup:
                    lines.append("        Continue executing the timeline under this branch:")
                    lines.extend(followup)
                lines.append("      Global path note: Once you enter this alternate storyline, remain on it until it naturally merges back into the main sequence.")
        consumed_blocks.add(stage_id)
    lines.append("")

    if selections_view:
        lines.append("4. CURRENT CONDITION ASSUMPTION")
        lines.append("   Unless the evaluator states otherwise, assume each trigger below is currently FALSE.")
        for sel in selections_view:
            cond = (sel.get("condition") or "").strip()
            stage_label = block_label_lookup.get(sel["trace_to"], sel["trace_to"])
            negated = _negated_condition_statement(cond)
            statement = negated or "The specified condition does not apply."
            lines.append(f"   - {stage_label}: {statement}")
        lines.append("")

    # Evaluation reminder
    lines.append("5. EVALUATION NOTICE")
    lines.append("- Automated checks will verify every global rule, each stage’s duties, and the branch-specific obligations you triggered.")
    lines.append("- Once you pick a branch, treat all bullets under that branch as mandatory, and do not mix incompatible paths unless explicitly told to merge.")
    lines.append("- Language, tone, structural cues, and conditional behaviors are all inspected.")
    lines.append("")

    return "\n".join(lines).strip() + "\n"


# ------------------------------------------------------------
# Helper 5. 组装 eval_protocol
# （给自动评分/裁决系统看的打分协议）
# ------------------------------------------------------------

def _mk_eval_protocol(seed_task: str,
                      global_rules: List[Dict[str, Any]],
                      block_plan: List[Dict[str, Any]],
                      selections_view: List[Dict[str, Any]],
                      graph_meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the evaluation protocol consumed by automated scoring / adjudication.

    Semantics:
    - Every constraint in `global_rules` is considered mandatory for the entire answer
      (e.g. required language, tone, min length, safety style).
    - For each stage in `block_plan`:
        * If logic_type == "AND": ALL listed requirements must be satisfied.
        * If logic_type == "sub-chain": The model's answer is expected to cover the sub-steps
          in a coherent forward progression.
    - For each conditional branch in `selections_view`:
        * The model MUST implicitly or explicitly pick ONE branch.
        * The chosen branch's requirements are then all mandatory (logical AND).
    """

    # 构建按阶段(block)打分的规范：每个阶段有哪些必须满足的要求
    block_specs_eval = []
    for blk in block_plan:
        if blk.get("is_alternate"):
            continue
        block_entry = {
            "block_id": blk["block_id"],
            "role": blk["role"],
            "logic_type": blk["logic_type"],
            "is_alternate": blk.get("is_alternate", False),
            "origin_block": blk.get("origin_block"),
            "requirements": [],
        }
        for r in blk["requirements"]:
            block_entry["requirements"].append({
                "cid": r["cid"],
                "desc": r["desc"],
                "verifier": r["verifier"],
            })
        block_specs_eval.append(block_entry)

    # 构建分支选择阶段的打分规范：模型必须选择其中一条分支，然后满足该分支的所有要求
    sel_specs_eval = []
    for sel in selections_view:
        sel_specs_eval.append({
            "sid": sel["sid"],
            "where": sel["where"],
            "condition": sel["condition"],
            "selection_type": sel.get("selection_type"),
            "merge_point": sel.get("merge_point"),
            "truncated": sel.get("truncated"),
            "branch_real_block": sel.get("branch_real_block"),
            "branch_alt_block": sel.get("branch_alt_block"),
            "alt_path_blocks": sel.get("alt_path_blocks"),
            "branch_real": [
                {
                    "cid": it["cid"],
                    "desc": it["desc"],
                    "verifier": it["verifier"],
                }
                for it in sel["branch_real"]
            ],
            "branch_alt": [
                {
                    "cid": it["cid"],
                    "desc": it["desc"],
                    "verifier": it["verifier"],
                }
                for it in sel["branch_alt"]
            ],
            "scoring_rule": {
                "must_choose_one_branch": True,
                "branch_real_logic": "AND",
                "branch_alt_logic": "AND",
            },
        })

    protocol = {
        "seed_task": seed_task,
        "global_scoring": [
            {
                "cid": g["cid"],
                "desc": g["desc"],
                "verifier": g["verifier"],
                "logic": "MANDATORY_GLOBAL",
            }
            for g in global_rules
        ],
        "block_scoring": block_specs_eval,
        "conditional_scoring": sel_specs_eval,
        "meta": graph_meta,
    }
    return protocol


# ------------------------------------------------------------
# 主入口：从 ConstraintGraph 产出第7步需要的所有结果
# ------------------------------------------------------------

def synthesize_instruction_bundle(graph: ConstraintGraph) -> Dict[str, Any]:
    """
    Given a ConstraintGraph (assembled in Step 6), produce the Step 7 artifacts:

        - machine_prompt (str):
            Final instruction prompt to give to the evaluated model.
            English framing + explicit duties + branch logic.

        - eval_protocol (dict):
            Scoring protocol for evaluators / automatic verifiers. This maps each
            constraint to its verifier and explains how branches should be graded.

        - graph_serialized (dict):
            A JSON-safe snapshot of the graph (serialize_graph(graph)).

        - meta (dict):
            Provenance metadata from the graph.

    This function is deterministic. It does not call any LLM.
    """
    # 拆解 ConstraintGraph：把图结构转成我们需要的几个视图（全局约束/分阶段约束/分支约束）
    graph_snap = serialize_graph(graph)
    seed_task = graph.seed_task
    global_rules = _render_global_constraints(graph.global_constraints)
    block_plan = _render_block_plan(graph.block_specs, graph.block_constraint_sets)
    selections_view = _render_selections(graph.selections, graph.block_specs, graph.block_constraint_sets)

    machine_prompt = _mk_machine_prompt(
        seed_task=seed_task,
        global_rules=global_rules,
        block_plan=block_plan,
        selections_view=selections_view,
        graph=graph,
    )

    eval_protocol = _mk_eval_protocol(
        seed_task=seed_task,
        global_rules=global_rules,
        block_plan=block_plan,
        selections_view=selections_view,
        graph_meta=graph.meta,
    )

    return {
        "machine_prompt": machine_prompt,
        "eval_protocol": eval_protocol,
        "graph_serialized": graph_snap,
        "meta": graph.meta,
    }


# ------------------------------------------------------------
# 自检示例（本文件直接运行时）
# ------------------------------------------------------------
if __name__ == "__main__":
    from .graph_schema import ConstraintNode, SelectionBranch
    from .step6_graph_assembly import assemble_constraint_graph

    # 构造一个和第6步示例类似的演示用图
    seed_task_demo = (
        "Analyze the geopolitical implications of the modern space race in a neutral analytical tone, "
        "providing real-world examples and forward-looking assessment."
    )

    segmentation_demo = {
        "blocks": [
            {
                "block_id": "B1",
                "intent": "Opening / Context setup",
                "text_span": "The modern space race is both technological and geopolitical...",
                "order_index": 0,
            },
            {
                "block_id": "B2",
                "intent": "Main Analysis / Examples",
                "text_span": "For instance, recent launch programs by ...",
                "order_index": 1,
            },
            {
                "block_id": "B3",
                "intent": "Conclusion / Outlook / Recommendation",
                "text_span": "In conclusion, these trends indicate long-term strategic risk...",
                "order_index": 2,
            },
        ],
        "order": ["B1", "B2", "B3"],
    }

    global_constraints_demo = [
        ConstraintNode(
            cid="G1",
            desc="The answer must be written primarily in English.",
            scope="global",
            verifier_spec={"check": "require_language", "args": {"lang": "en"}},
            trace_to=None,
            derived_from="step3",
        ),
        ConstraintNode(
            cid="G2",
            desc="The answer must maintain a neutral, analytical tone without inflammatory language.",
            scope="global",
            verifier_spec={"check": "tone_neutral_llm_judge", "args": {}},
            trace_to=None,
            derived_from="step3",
        ),
    ]

    block_constraints_demo = {
        "B1": [
            ConstraintNode(
                cid="B1_C1",
                desc="Explain why the modern space race matters geopolitically.",
                scope="local",
                verifier_spec={"check": "tone_neutral_llm_judge", "args": {}},
                trace_to="B1",
                derived_from="step4",
            ),
            ConstraintNode(
                cid="B1_C2",
                desc="Provide neutral historical background context.",
                scope="local",
                verifier_spec={"check": "forbid_first_person", "args": {}},
                trace_to="B1",
                derived_from="step4",
            ),
        ],
        "B2": [
            ConstraintNode(
                cid="B2_C1",
                desc="Give at least two concrete real-world examples.",
                scope="local",
                verifier_spec={"check": "must_list_n_subpoints", "args": {"n": 2}},
                trace_to="B2",
                derived_from="step4",
            ),
            ConstraintNode(
                cid="B2_C2",
                desc="Contrast national vs private/commercial actors.",
                scope="local",
                verifier_spec={"check": "must_cover_topics", "args": {"topics": ["state actors", "private actors"]}},
                trace_to="B2",
                derived_from="step4",
            ),
        ],
        "B3": [
            ConstraintNode(
                cid="B3_C1",
                desc="State long-term geopolitical risks and strategic implications.",
                scope="local",
                verifier_spec={"check": "tone_neutral_llm_judge", "args": {}},
                trace_to="B3",
                derived_from="step4",
            ),
            ConstraintNode(
                cid="B3_C2",
                desc="Offer at least one forward-looking recommendation.",
                scope="local",
                verifier_spec={"check": "actionability_judge", "args": {}},
                trace_to="B3",
                derived_from="step4",
            ),
            ConstraintNode(
                cid="B3_C3",
                desc="Adopt an explicitly critical tone highlighting concrete failures.",
                scope="local",
                verifier_spec={"check": "tone_negative_llm_judge", "args": {}},
                trace_to="B3",
                derived_from="step5",
            ),
            ConstraintNode(
                cid="B3_C4",
                desc="Propose at least one concrete next-step action to address the identified problems.",
                scope="local",
                verifier_spec={"check": "actionability_judge", "args": {}},
                trace_to="B3",
                derived_from="step5",
            ),
        ],
    }

    # 示范各 block 的逻辑类型：B2 是 sub-chain（顺序子步骤），其他是 AND（并列都要满足）
    block_logic_demo = {
        "B1": "AND",
        "B2": "sub-chain",
        "B3": "AND",
    }

    # SelectionNode 示例：branch_real 代表默认路径，branch_alt 代表在条件成立时的替代路径
    alt_block_spec_demo = BlockSpec(
        block_id="B3_ALT",
        intent="Conclusion / Outlook / Recommendation",
        text_span="Alternate conclusion branch text...",
        order_index=2,
        is_alternate=True,
        origin_block="B3",
    )
    block_constraints_demo["B3_ALT"] = [
        ConstraintNode(
            cid="B3_ALT_C1",
            desc="Adopt an explicitly critical tone highlighting concrete failures.",
            scope="local",
            verifier_spec={"check": "tone_negative_llm_judge", "args": {}},
            trace_to="B3_ALT",
            derived_from="step5",
        ),
        ConstraintNode(
            cid="B3_ALT_C2",
            desc="Propose at least one concrete next-step action to address the identified problems.",
            scope="local",
            verifier_spec={"check": "actionability_judge", "args": {}},
            trace_to="B3_ALT",
            derived_from="step5",
        ),
    ]
    block_logic_demo["B3_ALT"] = "AND"

    selections_demo = [
        SelectionNode(
            sid="SEL_B3",
            condition="If the stance is critical/negative",
            trace_to="B3",
            branch_real=SelectionBranch(block_id="B3", constraints=["B3_C1", "B3_C2"]),
            branch_alt=SelectionBranch(block_id="B3_ALT", constraints=["B3_ALT_C1", "B3_ALT_C2"]),
            derived_from="step5",
            selection_type="local",
            merge_point="B4",
            alt_path_blocks=["B3_ALT"],
        )
    ]

    from .graph_schema import BlockSpec, BlockConstraintSet
    from .graph_schema import SelectionNode as _SN  # clarity only

    step5_output_demo = {
        "block_constraints": block_constraints_demo,
        "block_logic": block_logic_demo,
        "selections": selections_demo,
        "extra_blocks": [alt_block_spec_demo],
    }

    graph_demo = assemble_constraint_graph(
        seed_task=seed_task_demo,
        segmentation=segmentation_demo,
        global_constraints=global_constraints_demo,
        step5_output=step5_output_demo,
    )

    bundle = synthesize_instruction_bundle(graph_demo)

    import pprint
    pp = pprint.PrettyPrinter(indent=2, width=100)
    print("\n===== MACHINE PROMPT =====\n")
    print(bundle["machine_prompt"])  # 给待评测模型的最终复杂指令

    print("\n===== EVAL PROTOCOL =====\n")
    pp.pprint(bundle["eval_protocol"])  # 打分/裁决协议（自动评测参考）

    print("\n===== GRAPH SNAPSHOT =====\n")
    pp.pprint(bundle["graph_serialized"])  # 第6步拼装出的约束图快照

    print("\n===== META =====\n")
    pp.pprint(bundle["meta"])  # 溯源元数据（哪些步骤贡献了哪些信息）
