

"""
Step 6 - Constraint Graph Assembly

Purpose / 目标
- Normalize artifacts from Steps 1-5 into a single ConstraintGraph dataclass so later stages can serialize, prompt, or visualize the structure consistently.
- Preserve ordering, scope, and provenance metadata without inventing new constraints.

Inputs
- seed_task from Step 1.
- segmentation from Step 2 (including intents/order).
- global_constraints from Step 3.
- step5_output bundle containing augmented block constraints, logic types, selections, and any synthetic alt blocks.

Outputs
- ConstraintGraph: canonical data object consumed by Step 7+ and exported via utils.export_utils.
- serialize_graph helper: JSON-safe dict used for persistence/debugging.

Why this matters
- Centralizes the pipeline contract: every downstream consumer reads the same structure rather than juggling step-specific formats.
- Ensures selection nodes only reference constraint IDs that truly exist and that block order stays deterministic.
"""

import string
from typing import Dict, Any, List
from .graph_schema import (
    ConstraintGraph,
    BlockSpec,
    ConstraintNode,
    BlockConstraintSet,
    SelectionNode,
)


def _build_block_specs(segmentation: Dict[str, Any],
                      extra_blocks: List[BlockSpec]) -> List[BlockSpec]:
    """
    根据 step2_segmentation 的输出，构造 BlockSpec 列表。

    segmentation["blocks"] 的元素示例：
    {
        "block_id": "B1",
        "intent": "Opening / Context setup",
        "text_span": "...",
        "order_index": 0
    }

    如果 step2 没提供 order_index，我们用遍历顺序赋值。
    extra_blocks 用于挂载 Step5 生成的替代块 (B3_ALT 等)。
    """
    block_specs: List[BlockSpec] = []
    for idx, b in enumerate(segmentation.get("blocks", [])):
        block_specs.append(
            BlockSpec(
                block_id=b.get("block_id", f"B{idx+1}"),
                intent=b.get("intent", ""),
                text_span=b.get("text_span", ""),
                order_index=b.get("order_index", idx),
                is_alternate=False,
                origin_block=None,
            )
        )
    block_specs.extend(extra_blocks)
    block_specs.sort(key=lambda bs: (bs.order_index, bs.block_id))
    return block_specs


def _build_block_constraint_sets(
    block_constraints: Dict[str, List[ConstraintNode]],
    block_logic: Dict[str, str],
) -> List[BlockConstraintSet]:
    """
    把 step4/step5 合并后的 block_constraints + block_logic 转成 BlockConstraintSet 列表。

    block_constraints: {
        "B1": [ConstraintNode(...), ConstraintNode(...)],
        "B2": [...]
    }

    block_logic: {
        "B1": "AND",
        "B2": "sub-chain"
    }

    输出：List[BlockConstraintSet]
    每个元素形如：
        BlockConstraintSet(
            block_id="B1",
            logic_type="AND",
            constraints=[ConstraintNode(...), ...]
        )
    """
    sets: List[BlockConstraintSet] = []

    for bid, node_list in block_constraints.items():
        logic = block_logic.get(bid, "AND")
        sets.append(
            BlockConstraintSet(
                block_id=bid,
                logic_type=("sub-chain" if str(logic).lower().startswith("sub") else "AND"),
                constraints=node_list,
            )
        )

    # 按 block_id 的自然顺序进行稳定排序（B1, B2, ...），尽量保持 determinism。
    sets.sort(key=lambda s: s.block_id)
    return sets


def assemble_constraint_graph(
    seed_task: str,
    segmentation: Dict[str, Any],
    global_constraints: List[ConstraintNode],
    step5_output: Dict[str, Any],
) -> ConstraintGraph:
    """
    Step 6 主入口：

    输入：
    - seed_task:           Step1 输出（单句核心任务，英文祈使式）
    - segmentation:        Step2 输出（block 列表 + 顺序）
    - global_constraints:  Step3 输出（List[ConstraintNode]，scope="global"）
    - step5_output:        Step5 输出：{
           "block_constraints": {block_id: [ConstraintNode,...]},
           "block_logic": {block_id: "AND"|"sub-chain"},
           "selections": [SelectionNode,...]
      }

    返回：ConstraintGraph 实例

    具体步骤：
    1. block_specs <- segmentation
    2. block_constraint_sets <- step5_output.block_constraints + block_logic
    3. selections <- step5_output.selections

    备注：
    - global_constraints 应该在 Step3 中已经包含了硬性/软性要求并绑定了 verifier_spec。
    - block_constraints 来自 Step4 + Step5（Step5 可能追加伪分支约束）；这些 nodes 的 scope="local"。
    - selections 已经引用了这些 block 的约束 cid，我们此处只是挂上去。
    """

    extra_blocks: List[BlockSpec] = step5_output.get("extra_blocks", [])
    block_specs = _build_block_specs(segmentation, extra_blocks)

    block_constraint_sets = _build_block_constraint_sets(
        block_constraints=step5_output.get("block_constraints", {}),
        block_logic=step5_output.get("block_logic", {}),
    )

    selections: List[SelectionNode] = step5_output.get("selections", [])

    graph = ConstraintGraph(
        seed_task=seed_task,
        global_constraints=global_constraints,
        block_specs=block_specs,
        block_constraint_sets=block_constraint_sets,
        selections=selections,
        meta={
            "assembled_from": {
                "step1": "seed_task",
                "step2": "segmentation",
                "step3": "global_constraints",
                "step4": "block_constraints + block_logic",
                "step5": "selections + augmented block_constraints",
            }
        },
    )

    return graph


# ---------------------------------------------------------
# 序列化方便下游 Step7 / 调试 / 存盘
# ---------------------------------------------------------

def serialize_graph(graph: ConstraintGraph) -> Dict[str, Any]:
    """
    把 ConstraintGraph dataclass 转成一个纯 JSON-safe dict，
    用于：
      - 人类可读调试
      - 直接保存为中间产物
      - 喂给 Step7 指令生成器

    注意：为了可解释性，我们在序列化时会把 SelectionNode 也展开成 dict，
    其中 branch_real / branch_alt 会显示引用到的约束 cid 列表。
    """

    def _ser_constraint_node(n: ConstraintNode) -> Dict[str, Any]:
        return {
            "cid": n.cid,
            "desc": n.desc,
            "scope": n.scope,
            "verifier_spec": n.verifier_spec,
            "priority_level": n.priority_level,
            "trace_to": n.trace_to,
            "derived_from": n.derived_from,
            "change_type": n.change_type,
        }

    def _ser_block_spec(b: BlockSpec) -> Dict[str, Any]:
        return {
            "block_id": b.block_id,
            "intent": b.intent,
            "text_span": b.text_span,
            "order_index": b.order_index,
            "is_alternate": b.is_alternate,
            "origin_block": b.origin_block,
        }

    def _ser_block_constraint_set(bcs: BlockConstraintSet) -> Dict[str, Any]:
        return {
            "block_id": bcs.block_id,
            "logic_type": bcs.logic_type,
            "constraints": [_ser_constraint_node(n) for n in bcs.constraints],
        }

    def _ser_selection(sel: SelectionNode) -> Dict[str, Any]:
        return {
            "sid": sel.sid,
            "condition": sel.condition,
            "trace_to": sel.trace_to,
            "branch_real": {
                "block_id": sel.branch_real.block_id,
                "constraints": list(sel.branch_real.constraints),
            },
            "branch_alt": {
                "block_id": sel.branch_alt.block_id,
                "constraints": list(sel.branch_alt.constraints),
            },
            "derived_from": sel.derived_from,
            "selection_type": sel.selection_type,
            "merge_point": sel.merge_point,
            "truncated": sel.truncated,
            "alt_path_blocks": list(sel.alt_path_blocks),
        }

    out = {
        "seed_task": graph.seed_task,
        "global_constraints": [
            _ser_constraint_node(n) for n in graph.global_constraints
        ],
        "block_specs": [_ser_block_spec(b) for b in graph.block_specs],
        "block_constraint_sets": [
            _ser_block_constraint_set(bcs) for bcs in graph.block_constraint_sets
        ],
        "selections": [_ser_selection(s) for s in graph.selections],
        "meta": graph.meta,
    }
    return out


def make_mermaid(graph: ConstraintGraph, max_desc_len: int = 60) -> str:
    """
    生成 Mermaid 流程图（flowchart LR）。
    约定：
    - 起点 seed_task 是一个 double-circle 节点 (用三层括号强调)。
    - seed_task 有向边指向全局约束 subgraph。
    - Global Constraints 画成一个 subgraph，里面每条全局约束是圆形节点 ((...))，
      并用无向边 --- 串接表示 AND。
    - 然后按 block 顺序画每个 block 的 subgraph：
        * label = "B1 Opening / Context setup"
        * 每条局部约束是方框节点 ["..."]，根据 block_logic:
            - "AND"      -> 用 --- 连接
            - "sub-chain"-> 用 --> 连接
      并把 subgraph 的首个局部约束节点当成锚点，连接主路径。
    - 如果某个 block 有 selection(s):
        * 在该 block 的锚点后画一个菱形 DEC_xxx{"condition"}。
        * 对每个 selection:
            - 生成两个 subgraphs: SEL_sid_REAL[...] / SEL_sid_ALT[...]
            - 里面放 branch_real / branch_alt 的约束节点，节点之间用 --- (AND)
            - 画 DEC_xxx --> subgraphName
    """

    lines = []
    lines.append("flowchart LR")

    # helper: shorten long text for node labels
    punctuation_table = str.maketrans("", "", string.punctuation + "，。！？、“”‘’：；（）【】《》—…·|")

    def _clean_text(txt: str) -> str:
        cleaned = (txt or "").translate(punctuation_table)
        return " ".join(cleaned.split())

    def _short(txt: str) -> str:
        t = _clean_text(txt.strip().replace("\n", " "))
        if len(t) > max_desc_len:
            t = t[:max_desc_len - 3].rstrip() + "..."
        return t

    # ----- Seed task node -----
    seed_label = _short(graph.seed_task or "Seed Task")
    # triple-paren for visual emphasis as 'double circle'
    lines.append(f'    SEED((({seed_label})))')

    # ----- Global constraints subgraph -----
    # We'll create all global constraint nodes as circles ((...))
    # and chain them with --- to show AND.
    if graph.global_constraints:
        lines.append(f'    subgraph GLOBAL[Global Constraints]')
        glob_node_ids = []
        for gc in graph.global_constraints:
            nid = gc.cid if gc.cid else f"G_{len(glob_node_ids)+1}"
            label_desc = gc.desc or gc.cid or "global rule"
            label = _short(f"{nid}: {label_desc}")
            # circle node
            lines.append(f'        {nid}(({label}))')
            glob_node_ids.append(nid)

        # AND-chain using ---
        if len(glob_node_ids) >= 2:
            chain = " --- ".join(glob_node_ids)
            lines.append(f'        {chain}')
        # if only one node, it stands alone
        lines.append("    end")

        # Connect SEED --> first global node
        first_global = glob_node_ids[0]
        lines.append(f'    SEED --> {first_global}')
        global_end = glob_node_ids[-1]
    else:
        # no global constraints: just connect SEED forward later
        global_end = "SEED"

    # ----- Block subgraphs -----
    # We need to:
    # 1. Sort blocks by order_index so the chain is B1 -> B2 -> ...
    # 2. For each block, render its constraints in a subgraph.
    #
    # Also: Keep a mapping from block_id -> first constraint node id,
    # which we'll use to connect chain edges.
    block_anchor_map = {}
    block_end_map: Dict[str, str] = {}
    main_blocks: List[str] = []

    # Sort block_specs in natural order (primary order_index, secondary block_id)
    sorted_blocks = sorted(graph.block_specs, key=lambda b: (b.order_index, b.block_id))
    ordering: Dict[str, int] = {b.block_id: idx for idx, b in enumerate(sorted_blocks)}

    # Collect all alternate block IDs for edge-origin logic
    alt_block_ids = {b.block_id for b in sorted_blocks if getattr(b, "is_alternate", False)}

    # Build helper maps for quick lookup
    # - constraints per block
    # - logic type per block
    logic_map = {bcs.block_id: bcs.logic_type for bcs in graph.block_constraint_sets}
    constraints_map = {
        bcs.block_id: bcs.constraints for bcs in graph.block_constraint_sets
    }

    for b in sorted_blocks:
        bid = b.block_id
        intent_label = _clean_text(b.intent or bid)
        subgraph_name = bid  # Mermaid subgraph label uses this as header ID
        lines.append(f'    subgraph {subgraph_name}[{bid} {intent_label}]')

        block_nodes = constraints_map.get(bid, [])
        anchor_id = None

        # Render each local constraint in this block as a rectangular node ["..."]
        rendered_ids = []
        for cnode in block_nodes:
            cid = cnode.cid or f"{bid}_C{len(rendered_ids)+1}"
            label_desc = cnode.desc or ""
            label = _short(f"{cid}: {label_desc}")
            lines.append(f'        {cid}["{label}"]')
            rendered_ids.append(cid)
            if anchor_id is None:
                anchor_id = cid

        # Connect them with --- (AND) or --> (sub-chain)
        if len(rendered_ids) >= 2:
            connector = "---"
            if str(logic_map.get(bid, "AND")).lower().startswith("sub"):
                connector = "-->"
            chain_expr = f" {connector} ".join(rendered_ids)
            lines.append(f'        {chain_expr}')

        lines.append("    end")

        if anchor_id:
            block_anchor_map[bid] = anchor_id
            block_end_map[bid] = rendered_ids[-1]
        else:
            synthetic_anchor = f"{bid}_EMPTY"
            lines.append(f'    {synthetic_anchor}["{bid} (no explicit constraints)"]')
            block_anchor_map[bid] = synthetic_anchor
            block_end_map[bid] = synthetic_anchor

        if not b.is_alternate:
            main_blocks.append(bid)

    # Connect main chain blocks (using last -> first constraint)
    selection_map = {sel.trace_to: sel for sel in graph.selections}

    if main_blocks:
        first_main = main_blocks[0]
        if first_main not in selection_map:
            lines.append(f'    {global_end} --> {block_anchor_map[first_main]}')

        for idx in range(len(main_blocks) - 1):
            curr_bid = main_blocks[idx]
            next_bid = main_blocks[idx + 1]
            if next_bid in selection_map:
                continue
            curr_end = block_end_map.get(curr_bid)
            next_start = block_anchor_map.get(next_bid)
            if curr_end and next_start:
                lines.append(f'    {curr_end} --> {next_start}')

    # ----- Selection branches -----
    # 对于每个 SelectionNode：
    #   1. 创建菱形决策节点 DEC_<sid>{"condition"}
    #   2. 从对应 block 的锚点连到该菱形
    #   3. 为 branch_real / branch_alt 各建一个子图子分支
    #   4. 决策菱形分别指向两个分支的第一个节点

    for sel in graph.selections:
        sid = sel.sid
        cond_label = _short(sel.condition or sid)

        # 决策菱形
        dec_id = f"DEC_{sid}"
        lines.append(f'    {dec_id}{{"{cond_label}"}}')

        origin_bid = sel.trace_to
        prev_end_node = None
        if origin_bid in main_blocks:
            idx = main_blocks.index(origin_bid)
            if idx == 0:
                prev_end_node = global_end
            else:
                prev_block = main_blocks[idx - 1]
                prev_end_node = block_end_map.get(prev_block)
        else:
            prev_end_node = global_end
        if prev_end_node:
            lines.append(f'    {prev_end_node} --> {dec_id}')

        # 真实/默认分支（branch_real）指向原块锚点
        real_anchor = block_anchor_map.get(sel.branch_real.block_id)
        if real_anchor:
            lines.append(f'    {dec_id} --> {real_anchor}')

        # 条件分支指向替代块锚点（可能是多段链）
        alt_path_blocks = sel.alt_path_blocks or [sel.branch_alt.block_id]
        first_alt_block = alt_path_blocks[0]
        first_alt_anchor = block_anchor_map.get(first_alt_block)
        if first_alt_anchor:
            lines.append(f'    {dec_id} --> {first_alt_anchor}')

        # Chain alternate blocks.
        # For ALT blocks, outgoing edges originate from the block's last internal constraint node (end)
        # rather than the first constraint (anchor).
        for idx_chain in range(len(alt_path_blocks) - 1):
            current_block = alt_path_blocks[idx_chain]
            next_block = alt_path_blocks[idx_chain + 1]
            # For ALT blocks, use the last internal constraint as the origin; otherwise use anchor
            origin_node = (
                block_end_map.get(current_block)
                if current_block in alt_block_ids else
                block_anchor_map.get(current_block)
            )
            next_anchor = block_anchor_map.get(next_block)
            if origin_node and next_anchor:
                lines.append(f'    {origin_node} --> {next_anchor}')

        # For ALT blocks, the outgoing edge should originate from the last internal constraint
        final_alt_block_id = alt_path_blocks[-1]
        final_origin_node = (
            block_end_map.get(final_alt_block_id)
            if final_alt_block_id in alt_block_ids else
            block_anchor_map.get(final_alt_block_id)
        )
        if final_origin_node and sel.merge_point:
            merge_sel = selection_map.get(sel.merge_point)
            if merge_sel:
                lines.append(f'    {final_origin_node} --> DEC_{merge_sel.sid}')
            else:
                merge_anchor = block_anchor_map.get(sel.merge_point)
                if merge_anchor:
                    lines.append(f'    {final_origin_node} --> {merge_anchor}')
        elif final_origin_node and not sel.merge_point and sel.selection_type.lower() == "local":
            order_idx = ordering.get(alt_path_blocks[-1], 0) + 1
            next_main = next((bid for bid in main_blocks if ordering.get(bid, 0) >= order_idx), None)
            next_sel = next((s for s in graph.selections if ordering.get(s.trace_to, 0) >= order_idx), None)
            if next_sel:
                lines.append(f'    {final_origin_node} --> DEC_{next_sel.sid}')
            elif next_main:
                next_anchor = block_anchor_map.get(next_main)
                if next_anchor:
                    lines.append(f'    {final_origin_node} --> {next_anchor}')

    # 最终把所有 Mermaid 行合并
    return "\n".join(lines)



if __name__ == "__main__":
    # 烟雾测试：拼一张假的图
    # 1. seed_task
    seed_task_demo = "Analyze the geopolitical implications of the modern space race in a neutral analytical tone, providing real-world examples and forward-looking assessment."

    # 2. segmentation (step2)
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

    # 3. global constraints (step3)
    from .graph_schema import ConstraintNode
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

    # 4/5. block constraints + selections
    from .graph_schema import SelectionNode, SelectionBranch
    block_constraints_demo = {
        "B3": [
            ConstraintNode(
                cid="B3_C1",
                desc="Summarize long-term geopolitical implications in a neutral analytical tone.",
                scope="local",
                verifier_spec={"check": "tone_neutral_llm_judge", "args": {}},
                trace_to="B3",
                derived_from="step4",
            ),
            ConstraintNode(
                cid="B3_C2",
                desc="Offer at least two concrete future risks or scenarios.",
                scope="local",
                verifier_spec={"check": "must_list_n_subpoints", "args": {"n": 2}},
                trace_to="B3",
                derived_from="step4",
            ),
        ],
        "B3_ALT": [
            ConstraintNode(
                cid="B3_ALT_C1",
                desc="Adopt an explicitly critical tone highlighting concrete problems or failures.",
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
        ],
    }

    block_logic_demo = {"B3": "AND", "B3_ALT": "AND"}

    selections_demo = [
        SelectionNode(
            sid="SEL_B3",
            condition="If the stance is critical/negative",
            trace_to="B3",
            branch_real=SelectionBranch(block_id="B3", constraints=["B3_C1", "B3_C2"]),
            branch_alt=SelectionBranch(block_id="B3_ALT", constraints=["B3_ALT_C1", "B3_ALT_C2"]),
            derived_from="step5",
            selection_type="local",
            merge_point=None,
            alt_path_blocks=["B3_ALT"],
        )
    ]

    extra_blocks_demo = [
        BlockSpec(
            block_id="B3_ALT",
            intent="Conclusion / Outlook / Recommendation",
            text_span="Alternate conclusion branch text...",
            order_index=2,
            is_alternate=True,
            origin_block="B3",
        )
    ]

    step5_output_demo = {
        "block_constraints": block_constraints_demo,
        "block_logic": block_logic_demo,
        "selections": selections_demo,
        "extra_blocks": extra_blocks_demo,
    }

    graph_demo = assemble_constraint_graph(
        seed_task=seed_task_demo,
        segmentation=segmentation_demo,
        global_constraints=global_constraints_demo,
        step5_output=step5_output_demo,
    )

    import pprint
    pp = pprint.PrettyPrinter(indent=2, width=100)
    pp.pprint(serialize_graph(graph_demo))
    print("\n===== Mermaid Diagram (preview only) =====\n")
    print(make_mermaid(graph_demo))
    print("\n(Preview only; official saving is handled by pipeline_runner.)\n")
