"""
Graph schema definitions for the Constraint Graph pipeline.

Each dataclass below captures one layer of the graph:
- ConstraintNode: atomic rule with verifier metadata.
- BlockSpec: ordered segment descriptors from Step 2.
- BlockConstraintSet: grouping of local constraints + logic per block.
- SelectionBranch / SelectionNode: conditional IF/ELSE structures referencing constraint IDs.
- ConstraintGraph: top-level bundle used by Step 7+, including serialization helpers.

Having a single module own these contracts keeps steps 3-8 aligned and avoids ad-hoc dicts.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ConstraintNode:
    """
    单条约束节点（约束是图里的基本原子单元）。

    字段含义：
    - cid: 约束ID，全局唯一。
    - desc: 该约束的自然语言描述（可读性高，后续会被写回复杂指令）。
    - scope: 约束适用范围，例如:
        "local"         仅在当前块内起作用
        "transitional"  约束这个块到下一个块的过渡
        "global"        全文级别
    - verifier_spec: 一个dict，描述如何自动验证这条约束。
        例如 {"check": "min_word_count", "args": {"min_words":150}}
        这些check会在verifier_registry中注册。
    - priority_level: 约束优先级（2=必须满足；1=尽可能满足，且不得违反2）。
    - trace_to: 该约束来源于哪个block（例如"B2"），用于追踪来源。
    - derived_from: 该约束是由pipeline的哪个步骤生成的（"step3", "step4", "step5"等）。
    """
    cid: str
    desc: str
    scope: str
    verifier_spec: Dict[str, Any]
    priority_level: int = 2
    trace_to: Optional[str] = None
    derived_from: Optional[str] = None


@dataclass
class BlockSpec:
    """
    一个逻辑块（回答被切分后的片段信息）。

    我们不会直接把 BlockSpec 当成约束节点，但会把它保留进图，
    方便：
    - 构建 CHAIN 边（块的顺序）
    - 合成最终指令时描述每个阶段的功能(intent)

    字段含义：
    - block_id: 例如"B1"、"B2"...
    - intent: 这个块在语义/功能上的角色（背景介绍、主体分析、结论…）
    - text_span: 该块对应的原始回答文本片段
    - order_index: 在回答整体顺序中的位置，用于重建chain
    - is_alternate: 是否为选择分支中新生成的替代块
    - origin_block: 如果是替代块，对应的原始块ID
    """
    block_id: str
    intent: str
    text_span: str
    order_index: int
    is_alternate: bool = False
    origin_block: Optional[str] = None


@dataclass
class BlockConstraintSet:
    """
    某个 block 对应的一组局部约束以及逻辑关系。

    字段含义：
    - block_id: 对应的 BlockSpec.block_id
    - logic_type: "AND" 或 "sub-chain"（其它值会被上游规范化）
    - constraints: 该 block 下的 ConstraintNode 列表
    """
    block_id: str
    logic_type: str
    constraints: List[ConstraintNode] = field(default_factory=list)


@dataclass
class SelectionBranch:
    """
    Selection 分支中的一条路径。

    字段含义：
    - block_id: 该分支对应的 block（原块或替代块）
    - constraints: 该分支下需要满足的约束ID列表（cid列表）。
      这些cid必须能在 ConstraintGraph.block_constraints 或 global_constraints 里找到。
    """
    block_id: str
    constraints: List[str]


@dataclass
class SelectionNode:
    """
    Selection 节点表示“条件化分支”（if/else）。

    字段含义：
    - sid: selection节点ID。
    - condition: 触发该分支的可观测条件，例如"If the stance is negative"。
      这必须是后续可由评测逻辑判断/分类的信号。
    - trace_to: 该分支是从哪一个 block 演化/派生出来的（溯源）。
    - branch_real: 真正回答走过的那条路径（真实链路）对应的约束ID集。
    - branch_alt: 我们合成的另一条“伪分支”路径的约束ID集。
    - derived_from: 我们是在pipeline哪一步造出的这个Selection（通常是 step5）。
    - selection_type: "local" | "global"
    - merge_point: 若为 local，分支合流到的块ID；global 默认为 None
    - truncated: 若因为上限被截断，则为 True
    """
    sid: str
    condition: str
    trace_to: str
    branch_real: SelectionBranch
    branch_alt: SelectionBranch
    derived_from: str = "step5"
    selection_type: str = "local"
    merge_point: Optional[str] = None
    truncated: bool = False
    alt_path_blocks: List[str] = field(default_factory=list)


@dataclass
class ConstraintGraph:
    """
    整体约束图的数据结构。

    字段含义：
    - seed_task: 种子任务（Step1提炼的原子任务描述）。
    - global_constraints: 全局约束节点列表（ConstraintNode）。
    - block_specs: BlockSpec 列表，记录回答里每个逻辑块的信息及顺序。
    - block_constraint_sets: BlockConstraintSet 列表，记录每个 block 的约束集合及逻辑。
    - selections: SelectionNode 列表，表示条件化分支。
    - meta: 额外的溯源元数据（由 Step6 组装时写入）。
    """
    seed_task: str
    global_constraints: List[ConstraintNode]
    block_specs: List[BlockSpec]
    block_constraint_sets: List[BlockConstraintSet]
    selections: List[SelectionNode]
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        """
        将整张图序列化为纯Python字典，方便：
        - 保存到graph.json
        - 传给评测模块进行自动打分
        - 用于instruction合成
        """
        return {
            "seed_task": self.seed_task,
            "blocks": [
                {
                    "block_id": b.block_id,
                    "intent": b.intent,
                    "text_span": b.text_span,
                    "order_index": b.order_index,
                    "is_alternate": b.is_alternate,
                    "origin_block": b.origin_block,
                } for b in self.block_specs
            ],
            "global_constraints": [
                {
                    "cid": c.cid,
                    "desc": c.desc,
                    "scope": c.scope,
                    "verifier_spec": c.verifier_spec,
                    "priority_level": c.priority_level,
                    "trace_to": c.trace_to,
                    "derived_from": c.derived_from,
                } for c in self.global_constraints
            ],
            "block_constraint_sets": [
                {
                    "block_id": bcs.block_id,
                    "logic_type": bcs.logic_type,
                    "constraints": [
                        {
                            "cid": c.cid,
                            "desc": c.desc,
                            "scope": c.scope,
                            "verifier_spec": c.verifier_spec,
                            "priority_level": c.priority_level,
                            "trace_to": c.trace_to,
                            "derived_from": c.derived_from,
                        } for c in bcs.constraints
                    ],
                }
                for bcs in self.block_constraint_sets
            ],
            "selections": [
                {
                    "sid": s.sid,
                    "condition": s.condition,
                    "trace_to": s.trace_to,
                    "branch_real": {
                        "block_id": s.branch_real.block_id,
                        "constraints": s.branch_real.constraints
                    },
                    "branch_alt": {
                        "block_id": s.branch_alt.block_id,
                        "constraints": s.branch_alt.constraints
                    },
                    "derived_from": s.derived_from,
                    "selection_type": s.selection_type,
                    "merge_point": s.merge_point,
                    "truncated": s.truncated,
                    "alt_path_blocks": s.alt_path_blocks,
                } for s in self.selections
            ],
            "meta": self.meta,
        }
