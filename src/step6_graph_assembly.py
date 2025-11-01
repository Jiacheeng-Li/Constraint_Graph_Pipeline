

"""
step6_graph_assembly.py

Step 6: 组装完整的 ConstraintGraph

到目前为止我们已经有：
- step1_seed_task.extract_seed_task()          -> seed_task: str
- step2_segmentation.segment_response()        -> segmentation: {"blocks": [...], "order": [...]}
- step3_global_constraints.extract_global_constraints() -> global_nodes: List[ConstraintNode]
- step4_back_traslation.extract_block_constraints()     -> {
      "block_constraints": {block_id: [ConstraintNode,...]},
      "block_logic": {block_id: "AND"|"sub-chain"}
  }
- step5_selection_augment.generate_selection_branches() -> {
      "block_constraints": {block_id: [ConstraintNode,...(augmented)]},
      "block_logic": {block_id: logic},
      "selections": [SelectionNode,...]
  }

本步骤（Step 6）做三件事：
1. 把所有这些拼成一个 ConstraintGraph 数据结构（见 graph_schema.py）。
2. 生成 block-level 的约束组（BlockConstraintSet），并标记逻辑关系（AND / sub-chain）。
3. 封装 selections (条件分支)，并确保其中引用的 cid 都存在于 block_constraints 里。

产物：
- ConstraintGraph 实例，可被下游 Step 7 用于生成复杂指令文本。
- 同时我们会给一个 `serialize_graph(graph)` 用于导出 JSON 结构，方便调试 / 存盘。

注意：
- 我们不会在 Step6 中做评测；只是结构装配。
- 我们不会在这里修改/扩展约束本身，避免引入和 Step3/4/5 不一致的内容。
"""

from typing import Dict, Any, List
from .graph_schema import (
    ConstraintGraph,
    BlockSpec,
    ConstraintNode,
    BlockConstraintSet,
    SelectionNode,
)


def _build_block_specs(segmentation: Dict[str, Any]) -> List[BlockSpec]:
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
    """
    block_specs: List[BlockSpec] = []
    for idx, b in enumerate(segmentation.get("blocks", [])):
        block_specs.append(
            BlockSpec(
                block_id=b.get("block_id", f"B{idx+1}"),
                intent=b.get("intent", ""),
                text_span=b.get("text_span", ""),
                order_index=b.get("order_index", idx),
            )
        )
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

    block_specs = _build_block_specs(segmentation)

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
            "trace_to": n.trace_to,
            "derived_from": n.derived_from,
        }

    def _ser_block_spec(b: BlockSpec) -> Dict[str, Any]:
        return {
            "block_id": b.block_id,
            "intent": b.intent,
            "text_span": b.text_span,
            "order_index": b.order_index,
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
                "constraints": list(sel.branch_real.constraints),
            },
            "branch_alt": {
                "constraints": list(sel.branch_alt.constraints),
            },
            "derived_from": sel.derived_from,
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
    def _short(txt: str) -> str:
        t = txt.strip().replace("\n", " ")
        if len(t) > max_desc_len:
            t = t[:max_desc_len - 3].rstrip() + "..."
        # Balance parentheses for readability (avoid dangling "(" after truncation)
        open_count = t.count("(")
        close_count = t.count(")")
        if open_count > close_count:
            t = t + (")" * (open_count - close_count))
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
            label = _short(gc.desc or gc.cid or "global rule")
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
    else:
        # no global constraints: just connect SEED forward later
        first_global = "SEED"

    # ----- Block subgraphs -----
    # We need to:
    # 1. Sort blocks by order_index so the chain is B1 -> B2 -> ...
    # 2. For each block, render its constraints in a subgraph.
    #
    # Also: Keep a mapping from block_id -> first constraint node id,
    # which we'll use to connect chain edges.
    block_anchor_map = {}  # block_id -> first constraint node id

    # Build helper maps for quick lookup
    # - constraints per block
    # - logic type per block
    logic_map = {bcs.block_id: bcs.logic_type for bcs in graph.block_constraint_sets}
    constraints_map = {
        bcs.block_id: bcs.constraints for bcs in graph.block_constraint_sets
    }

    # Sort block_specs in natural order of appearance
    sorted_blocks = sorted(graph.block_specs, key=lambda b: b.order_index)

    prev_anchor = first_global  # we chain from global to first block

    for b in sorted_blocks:
        bid = b.block_id
        intent_label = b.intent or bid
        subgraph_name = bid  # Mermaid subgraph label uses this as header ID
        lines.append(f'    subgraph {subgraph_name}[{bid} {intent_label}]')

        block_nodes = constraints_map.get(bid, [])
        anchor_id = None

        # Render each local constraint in this block as a rectangular node ["..."]
        rendered_ids = []
        for cnode in block_nodes:
            cid = cnode.cid or f"{bid}_C{len(rendered_ids)+1}"
            label = _short(cnode.desc or cid)
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
            # connect prev anchor to this block's anchor
            lines.append(f'    {prev_anchor} --> {anchor_id}')
            prev_anchor = anchor_id
        else:
            # 块内没有显式局部约束，我们仍然给它一个锚点，方便主链连接
            synthetic_anchor = f"{bid}_EMPTY"
            lines.append(f'    {synthetic_anchor}["{bid} (no explicit constraints)"]')
            block_anchor_map[bid] = synthetic_anchor
            lines.append(f'    {prev_anchor} --> {synthetic_anchor}')
            prev_anchor = synthetic_anchor

    # ----- Selection branches -----
    # 对于每个 SelectionNode：
    #   1. 创建菱形决策节点 DEC_<sid>{"condition"}
    #   2. 从对应 block 的锚点连到该菱形
    #   3. 为 branch_real / branch_alt 各建一个子图子分支
    #   4. 决策菱形分别指向两个分支的第一个节点

    # 为了能把 cid 解析成 desc，需要一份全局索引
    all_constraints_index = {}
    for bcs in graph.block_constraint_sets:
        for cn in bcs.constraints:
            all_constraints_index[cn.cid] = cn

    for sel in graph.selections:
        sid = sel.sid
        cond_label = _short(sel.condition or sid)

        # 决策菱形
        dec_id = f"DEC_{sid}"
        lines.append(f'    {dec_id}{{"{cond_label}"}}')

        origin_bid = sel.trace_to
        origin_anchor = block_anchor_map.get(origin_bid, prev_anchor)
        lines.append(f'    {origin_anchor} --> {dec_id}')

        def _render_branch(branch_name: str,
                           header_label: str,
                           constraint_ids: List[str]):
            """
            渲染一个分支子图：
            subgraph <branch_name>[<header_label>]
                <alias>["desc"]
                ...
                c1 --- c2 --- c3   # 用 '---' 表示 AND 关系
            end
            返回该分支第一个节点的 cid（用于连线）
            """
            lines.append(f'    subgraph {branch_name}[{header_label}]')
            rendered = []
            for cid in constraint_ids:
                cn = all_constraints_index.get(cid)
                label_desc = cn.desc if cn and cn.desc else cid
                label = _short(f"{cid}: {label_desc}")
                alias = f"{branch_name}_{cid}"
                lines.append(f'        {alias}["{label}"]')
                rendered.append(alias)

            if len(rendered) >= 2:
                chain_expr = " --- ".join(rendered)
                lines.append(f'        {chain_expr}')

            lines.append("    end")
            return rendered[0] if rendered else None

        # 真实/默认分支（branch_real）
        real_ids = list(sel.branch_real.constraints)
        real_header = "Branch: default path"
        real_anchor = _render_branch(f"{sid}_REAL", real_header, real_ids)

        # 条件分支（branch_alt）
        alt_ids = list(sel.branch_alt.constraints)
        alt_header = "Branch: conditional path"
        alt_anchor = _render_branch(f"{sid}_ALT", alt_header, alt_ids)

        # 从菱形连到两个分支的锚点
        if real_anchor:
            lines.append(f'    {dec_id} --> {real_anchor}')
        if alt_anchor:
            lines.append(f'    {dec_id} --> {alt_anchor}')

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
            ConstraintNode(
                cid="B3_C3",
                desc="Adopt an explicitly critical tone highlighting concrete problems or failures.",
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
        ]
    }

    block_logic_demo = {"B3": "AND"}

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

    import pprint
    pp = pprint.PrettyPrinter(indent=2, width=100)
    pp.pprint(serialize_graph(graph_demo))
    print("\n===== Mermaid Diagram (preview only) =====\n")
    print(make_mermaid(graph_demo))
    print("\n(Preview only; official saving is handled by pipeline_runner.)\n")
