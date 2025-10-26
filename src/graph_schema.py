

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import uuid


def _gen_id(prefix: str) -> str:
    """
    Helper to generate short stable-looking IDs for nodes/edges/etc.
    We keep this here in case we later want deterministic hashing.
    """
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


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
    - trace_to: 该约束来源于哪个block（例如"B2"），用于追踪来源。
    - derived_from: 该约束是由pipeline的哪个步骤生成的（"step3", "step4", "step5"等）。
    """
    cid: str
    desc: str
    scope: str
    verifier_spec: Dict[str, Any]
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
    """
    block_id: str
    intent: str
    text_span: str
    order_index: int


@dataclass
class SelectionBranch:
    """
    Selection 分支中的一条路径。

    字段含义：
    - constraints: 该分支下需要满足的约束ID列表（cid列表）。
      这些cid必须能在 ConstraintGraph.block_constraints 或 global_constraints 里找到。
    """
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
    """
    sid: str
    condition: str
    trace_to: str
    branch_real: SelectionBranch
    branch_alt: SelectionBranch
    derived_from: str = "step5"


@dataclass
class GraphEdge:
    """
    图中的一条有向边。

    字段含义：
    - source: 源节点ID（可以是block_id / cid / selection.sid）
    - target: 目标节点ID
    - edge_type: 边类型：
        "CHAIN"     表示顺序依赖（Block A → Block B）
        "AND"       表示并列必需（Block → 该Block下所有必须满足的约束）
        "SELECTION" 表示这里存在条件分支（Block → Selection 节点）
    - derived_from: 该边是在哪个步骤推出来的（step2/step4/step5等）
    - trace_to: 可选，指向来源块，帮助调试/追踪。
    """
    source: str
    target: str
    edge_type: str
    derived_from: str
    trace_to: Optional[str] = None


@dataclass
class ConstraintGraph:
    """
    整体约束图的数据结构。

    字段含义：
    - seed_task: 种子任务（Step1提炼的原子任务描述）。
    - global_constraints: 全局约束节点列表（ConstraintNode）。
    - block_constraints: dict: block_id -> List[ConstraintNode]
      每个block对应若干条约束，这些约束节点就是图中的可验证原子要求。
    - selections: SelectionNode列表，表示条件化分支。
    - blocks: BlockSpec列表，记录回答里每个逻辑块的信息及顺序。
    - edges: GraphEdge列表，包含CHAIN / AND / SELECTION依赖关系。
    - cnode: List[str]，一组关键约束ID，满足这些可视为“主路径达标”。
    """
    seed_task: str
    global_constraints: List[ConstraintNode]
    block_constraints: Dict[str, List[ConstraintNode]]
    selections: List[SelectionNode]
    blocks: List[BlockSpec]
    edges: List[GraphEdge]
    cnode: List[str] = field(default_factory=list)

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
                } for b in self.blocks
            ],
            "global_constraints": [
                {
                    "cid": c.cid,
                    "desc": c.desc,
                    "scope": c.scope,
                    "verifier_spec": c.verifier_spec,
                    "trace_to": c.trace_to,
                    "derived_from": c.derived_from,
                } for c in self.global_constraints
            ],
            "block_constraints": {
                bid: [
                    {
                        "cid": c.cid,
                        "desc": c.desc,
                        "scope": c.scope,
                        "verifier_spec": c.verifier_spec,
                        "trace_to": c.trace_to,
                        "derived_from": c.derived_from,
                    } for c in clist
                ]
                for bid, clist in self.block_constraints.items()
            },
            "selections": [
                {
                    "sid": s.sid,
                    "condition": s.condition,
                    "trace_to": s.trace_to,
                    "branch_real": {
                        "constraints": s.branch_real.constraints
                    },
                    "branch_alt": {
                        "constraints": s.branch_alt.constraints
                    },
                    "derived_from": s.derived_from,
                } for s in self.selections
            ],
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "type": e.edge_type,
                    "derived_from": e.derived_from,
                    "trace_to": e.trace_to,
                } for e in self.edges
            ],
            "cnode": self.cnode,
        }