

"""
step7_instruction_synthesis.py

第七步：基于约束关系图生成高复杂度指令规范

输入项：
    - 约束关系图对象（来自step6_graph_assembly.assemble_constraint_graph）
      或由serialize_graph(graph)生成的JSON安全快照

输出项：
    1. 机器提示语：
        直接提供给待评估模型的自然语言指令提示。包含：
        - 整体任务说明
        - 全局约束条件（风格、安全性、长度、语言等）
        - 分阶段结构要求（逐模块职责说明）
        - 条件分支逻辑（if/else执行要求）

    2. 评估协议：
        面向评估器/评判器的结构化评分协议。
        将每个约束与其验证器规格检查对齐，并说明分支评分规则。

    3. 元数据：
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

from typing import Dict, Any, List, Tuple
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
    sorted_specs = sorted(block_specs, key=lambda b: b.order_index)

    rendered: List[Dict[str, Any]] = []
    for spec in sorted_specs:
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

        rendered.append({
            "sid": sel.sid,
            "where": f"{sel.trace_to} {block_intent_map.get(sel.trace_to, '')}",
            "condition": sel.condition,
            "branch_real": real_list,
            "branch_alt": alt_list,
        })

    return rendered


# ------------------------------------------------------------
# Helper 4. 组装 machine_prompt
# （给待评测模型看的最终高复杂度指令）
# ------------------------------------------------------------

def _mk_machine_prompt(seed_task: str,
                       global_rules: List[Dict[str, Any]],
                       block_plan: List[Dict[str, Any]],
                       selections_view: List[Dict[str, Any]]) -> str:
    """
    Build the final instruction prompt to feed directly to the evaluated model.

    Goals:
    - English framework / headings.
    - The model must follow global constraints, the staged structure, and any applicable branch.
    - We explicitly communicate IF/THEN/ELSE behavior for conditional branches.
    - We do NOT expose internal IDs (cid) as something the model must output.
    """

    lines: List[str] = []

    # 0. 总任务（seed_task）：告诉模型它要完成的核心目标
    lines.append("OVERALL TASK:")
    lines.append(f"- {seed_task.strip()}")
    lines.append("")

    # 1. 全局硬性/软性约束：整篇回答必须同时满足的条件
    lines.append("GLOBAL MANDATORY CONSTRAINTS:")
    lines.append("You MUST satisfy ALL of the following properties throughout the entire answer:")
    for rule in global_rules:
        desc = rule.get("desc", "").strip()
        if desc:
            lines.append(f"- {desc}")
    lines.append("")

    # 2. 分阶段的输出结构约束：逐段落告诉模型在每个阶段必须完成什么职责
    lines.append("REQUIRED OUTPUT STRUCTURE (STAGED PLAN):")
    lines.append("Your answer MUST be organized into the following stages / sections.\n" \
                 "You do not need to literally copy these headings, but you MUST perform the listed duties.")

    for blk in block_plan:
        role = blk["role"] or blk["block_id"]
        logic_type = blk["logic_type"]
        reqs = blk["requirements"]

        if logic_type.lower().startswith("sub"):
            # sub-chain：这一段内部是按顺序推进的小步骤
            logic_phrase = "Follow these sub-steps IN ORDER."
        else:
            # AND：这一段内部是并列要点，全部都要覆盖
            logic_phrase = "Cover ALL of the following points (logical AND)."

        lines.append("")
        lines.append(f"Stage {blk['block_id']} — {role}:")
        lines.append(f"- This stage is '{role}'. You MUST: {logic_phrase}")
        for r in reqs:
            rdesc = r["desc"] or "(unspecified requirement)"
            lines.append(f"  • {rdesc}")
    lines.append("")

    # 3. 条件化分支（IF / THEN / ELSE）：模型需要根据条件只选用其中一条路径
    if selections_view:
        lines.append("CONDITIONAL BRANCH REQUIREMENTS:")
        lines.append("Some stages include branching logic. You MUST clearly choose ONE branch that applies, "
                     "and then satisfy ONLY that branch's obligations.")
        for sel in selections_view:
            cond = sel["condition"].strip() if sel["condition"] else "(no condition text)"
            where = sel["where"].strip()

            lines.append("")
            lines.append(f"In stage {where}:")
            lines.append(f"IF {cond} THEN you MUST satisfy ALL of these requirements:")
            for item in sel["branch_alt"]:
                lines.append(f"    • {item.get('desc') or '(unspecified requirement)'}")
            lines.append("ELSE (if the condition does not apply), you MUST satisfy ALL of these requirements:")
            for item in sel["branch_real"]:
                lines.append(f"    • {item.get('desc') or '(unspecified requirement)'}")
        lines.append("")

    # 4. 评测提醒：明确告诉模型它会被自动检查，并且不能混合分支
    lines.append("EVALUATION NOTICE:")
    lines.append("- Your answer will be automatically checked against the global constraints, the stage duties, "
                 "and (if applicable) the branch you chose.")
    lines.append("- You MUST clearly follow EXACTLY ONE branch for each conditional stage. Do NOT merge branches.")
    lines.append("- Automated verifiers will check things like required language, tone, length, inclusion of "
                 "specific elements, actionable recommendations, numbered items, etc.")
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
        block_entry = {
            "block_id": blk["block_id"],
            "role": blk["role"],
            "logic_type": blk["logic_type"],
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
    selections_demo = [
        SelectionNode(
            sid="SEL_B3",
            condition="If the stance is critical/negative",
            trace_to="B3",
            branch_real=SelectionBranch(constraints=["B3_C1", "B3_C2"]),
            branch_alt=SelectionBranch(constraints=["B3_C3", "B3_C4"]),
            derived_from="step5",
        )
    ]

    from .graph_schema import BlockSpec, BlockConstraintSet
    from .graph_schema import SelectionNode as _SN  # clarity only

    step5_output_demo = {
        "block_constraints": block_constraints_demo,
        "block_logic": block_logic_demo,
        "selections": selections_demo,
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