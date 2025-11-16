"""
step5_selection_augment.py

Step 5: 基于局部约束构造条件化分支 (Selection Augmentation)

目标：
我们目前有：
- 每个 block 的局部约束 (来自 Step4)，例如：
    B2_C1: "Provide at least two concrete real-world examples." -> must_list_n_subpoints(n=2)
    B2_C2: "Maintain a neutral, analytical tone." -> tone_neutral_llm_judge

问题：采样得到的真实回答只走了一条路径。我们想要引入“备选路径 / 条件分支”，
让最终的指令出现 if/else 风格：
    - 如果场景是 A，则必须满足约束组 A1/A2/A3；
    - 否则（或如果是 B），必须满足约束组 B1/B2/B3。

为什么这么做：
- 复杂指令往往是条件化的要求（"如果用户立场是负面，则…，否则…"、"如果发现严重风险，则…"）。
- SelectionNode 会变成图里的一类节点，后续在 step6_graph_assembly 中并入 ConstraintGraph。

产物（返回给后续步骤）：
{
  "block_constraints": {block_id: [ConstraintNode,...], ...},
  "block_logic": {block_id: "AND"|"sub-chain"},
  "selections": [SelectionNode,...]
}

SelectionNode 结构 (from graph_schema.py):
@dataclass
class SelectionNode:
    sid: str
    condition: str
    trace_to: str      # 来自哪个 block
    branch_real: SelectionBranch  # 真实路径要求（cid 列表）
    branch_alt:  SelectionBranch  # 我们合成的伪路径要求（cid 列表）
    derived_from: str = "step5"

SelectionBranch:
    constraints: List[str]  # cid 列表

在 Step5 中要做两件事：
1. 先基于 block_constraints 生成一条“真实分支” (branch_real) = 该 block 的现有局部约束。
2. 调用 LLM，请它基于同一 block 生成一套“对立 / 替代 / 条件化”的分支要求 (branch_alt)，
   这些要求必须：
   - 合法可检（使用我们已有的 verifier 函数名）
   - 在语义上与真实分支形成条件化区别，比如：
        * branch_real: neutral_tone   vs   branch_alt: negative_tone + actionability
        * branch_real: third-person   vs   branch_alt: first-person eyewitness style
        * branch_real: 2 examples    vs   branch_alt: numbered action steps
   - 不能无中生有完全无关的主题；必须仍然与同一 block intent 对齐。

我们会把 SelectionNode 附加到输出中。

策略：
- 我们不会为所有 block 都生成 selection，避免爆炸。我们可在这里挑一部分 block：
 
- LLM 返回的 branch_alt 是一组约束 (desc + {check,args}...)，我们会把它们注入为全新的 ConstraintNode，
  并把它们的 cid 记录在 selection.branch_alt.constraints。

- 注意：这些新合成的约束节点也要落进 block_constraints[...]，否则后续组图时 graph 没法引用到它们。

Fallback：
- 如果 LLM 失败，我们也能人造一个简单的条件分支：
    condition = "If the stance is critical/negative"
    branch_real = 原约束cid们
    branch_alt  = 新增两条约束：
        - 语气必须体现批评/不满 (tone_negative_llm_judge)
        - 文末必须提出具体改进步骤 (actionability_judge)

    这样保证我们永远至少输出一个 selection。
"""

import json
import random
import hashlib
import copy
from typing import Dict, Any, List, Optional, Tuple
from .graph_schema import (
    ConstraintNode,
    SelectionNode,
    SelectionBranch,
    BlockSpec,
)
from .utils.text_clean import make_snippet, clip

from .utils.deepseek_client import call_chat_completions, DeepSeekError
_DEEPSEEK_API_KEY_DEFAULT = ""
_DEEPSEEK_ENDPOINT = ""
_DEEPSEEK_MODEL = ""

# Selection configuration (can be tuned for experiments)
SELECTION_CONFIG = {
    "local_min": 1,
    "local_max": 3,
    "global_min": 0,
    "global_max": 1,
}

# Library of fallback constraint templates used to keep alternate
# global branches distinct when the LLM does not return an explicit chain.
_GLOBAL_ALT_FOLLOWUP_LIBRARY = [
    [
        {
            "desc": "Diagnose at least two critical failures driving the alternate scenario.",
            "verifier": {"check": "must_list_n_subpoints", "args": {"n": 2}},
        },
        {
            "desc": "Maintain an openly critical tone that assigns accountability for those failures.",
            "verifier": {"check": "tone_negative_llm_judge", "args": {}},
        },
    ],
    [
        {
            "desc": "Lay out an urgent recovery plan using numbered steps.",
            "verifier": {"check": "min_numbered_items", "args": {"n": 2}},
        },
        {
            "desc": "Highlight the need for rapid collaborative action to stabilize the situation.",
            "verifier": {"check": "must_include_keywords", "args": {"keywords": ["urgent", "collaboration"]}},
        },
    ],
    [
        {
            "desc": "Close with a cautious outlook that stresses long-term vigilance.",
            "verifier": {"check": "must_include_keywords", "args": {"keywords": ["vigilance", "long-term"]}},
        },
        {
            "desc": "Warn about the risks if corrective measures are not executed swiftly.",
            "verifier": {"check": "must_include_keywords", "args": {"keywords": ["risk", "swiftly"]}},
        },
    ],
]



def _plan_selections(block_constraints: Dict[str, List[ConstraintNode]],
                     segmentation: Dict[str, Any],
                     rng: random.Random) -> List[Tuple[str, str]]:
    """
    结合配置生成 selection 任务列表，返回 [(block_id, selection_type), ...]
    """
    ordered_blocks = [
        b.get("block_id")
        for b in segmentation.get("blocks", [])
        if b.get("block_id") in block_constraints and block_constraints[b.get("block_id")]
    ]

    if not ordered_blocks:
        return []

    total_blocks = len(ordered_blocks)
    available = list(ordered_blocks)

    # Ensure we can satisfy minimum locals
    local_min_cfg = SELECTION_CONFIG["local_min"]
    local_min_cap = min(local_min_cfg, total_blocks)

    max_global_allowed = max(0, total_blocks - local_min_cap)
    global_max_possible = min(SELECTION_CONFIG["global_max"], max_global_allowed)
    global_min = min(SELECTION_CONFIG["global_min"], global_max_possible)
    global_target = 0
    if global_max_possible >= global_min:
        global_target = rng.randint(global_min, global_max_possible)

    global_blocks = []
    if global_target > 0:
        global_blocks = rng.sample(available, global_target)
        available = [b for b in available if b not in global_blocks]

    # Decide local selections
    local_max_possible = min(SELECTION_CONFIG["local_max"], len(available))
    local_blocks = []
    if local_max_possible > 0:
        local_min = min(SELECTION_CONFIG["local_min"], local_max_possible)
        local_target = rng.randint(local_min, local_max_possible) if local_max_possible >= local_min else local_max_possible
        if local_target > 0:
            local_blocks = rng.sample(available, local_target)
            available = [b for b in available if b not in local_blocks]

    plan = [(bid, "global") for bid in global_blocks] + [(bid, "local") for bid in local_blocks]
    order_index = {bid: idx for idx, bid in enumerate(ordered_blocks)}
    plan.sort(key=lambda item: order_index.get(item[0], 0))
    return plan


def _format_block_constraints_for_llm(constraints: List[ConstraintNode]) -> str:
    """
    把真实分支（当前block的约束们）转成可喂给LLM的文字，便于它生成一个互补/对立分支。
    我们向 LLM 展示：
    - 这些约束的自然语言描述 desc
    - 这些约束对应的 verifier (check + args)
    """
    lines = []
    for c in constraints:
        lines.append(
            json.dumps(
                {
                    "cid": c.cid,
                    "desc": c.desc,
                    "verifier": c.verifier_spec,
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(lines)


def _call_deepseek_selection_aug(block_id: str,
                                 block_intent: str,
                                 block_text: str,
                                 seed_task: str,
                                 real_constraints_for_llm: str,
                                 selection_type_hint: str) -> Dict[str, Any]:
    """
    让 LLM 生成一个条件化分支：
    - 给一个清晰 condition（如 "If the stance is critical/negative"）
    - branch_alt: 一组新的约束( desc + verifier.check + verifier.args )

    期望输出 JSON：
    {
      "condition": "If the stance is critical/negative",
      "alt_constraints": [
        {
          "desc": "Use an explicitly critical tone to highlight problems.",
          "verifier": {"check": "tone_negative_llm_judge", "args": {}}
        },
        {
          "desc": "Propose at least one concrete next-step action.",
          "verifier": {"check": "actionability_judge", "args": {}}
        }
      ]
    }
    另外：
    - 如果当前 block 位于回答的较早部分（例如引言/背景段），
      允许生成的替代分支描述一种“走向完全不同后续路线并最终得出不同结尾/建议”的条件，
      而不需要与当前 REAL BRANCH 在结构上严格对齐。
    """

    # 规范 block_text，保持语义原样：仅清理空白，不做摘要
    block_text_clean = make_snippet(block_text)
    # 极端长文本才会硬截断，避免 prompt 过大；这是唯一可能丢信息的点
    if len(block_text_clean) > 12000:
        block_text_clean = clip(block_text_clean, 12000)

    selection_hint_text = selection_type_hint.lower()

    system_prompt = (
        "You are generating a conditional alternative branch for evaluation.\n"
        "We have a specific block from an answer. That block ALREADY satisfies some constraints (REAL BRANCH).\n"
        "Your job is to propose ONE ALTERNATIVE BRANCH for that SAME block.\n\n"
        "CRITICAL RULES:\n"
        "1. The alternative branch must stay on the SAME topic / role / rhetorical function as this block.\n"
        "   - It must be realistic for this block's purpose (e.g., summary, recommendation, warning, critique).\n"
        "   - You are NOT allowed to introduce obligations that would be off-topic or irrelevant.\n"
        "   - You are NOT allowed to hallucinate a totally different subject matter (e.g. budget, legal risk) if it's not implied by this block.\n"
        "2. The alternative branch MUST impose MEANINGFULLY DIFFERENT obligations from the REAL BRANCH.\n"
        "   - Not just minor paraphrase.\n"
        "   - Examples of meaningful difference: neutral analysis vs urgent critical escalation;\n"
        "     descriptive overview vs step-by-step action plan;\n"
        "     third-person analyst voice vs first-person eyewitness tone;\n"
        "     calm assessment vs aggressive call-to-action.\n"
        "3. The alternative branch must be VERIFIABLE.\n"
        "   - Each requirement must map to a verifier {check,args}.\n"
        "   - The condition should describe WHEN to use this alternative branch (e.g. 'If the situation is high-risk and urgent',\n"
        "     'If the stance is openly critical/negative'), and it must be plausible for this same block.\n"
        "4. Do NOT introduce unrelated tasks (no random new sections, no unrelated domains).\n"
        "   The alternative branch should feel like a different mode/style of THIS SAME block, under a specific scenario.\n\n"
        "OUTPUT FORMAT (MUST BE VALID JSON):\n"
        "{\n"
        "  \"condition\": \"If ... (a realistic trigger condition for this same block)\",\n"
        "  \"alt_constraints\": [\n"
        "    {\n"
        "      \"desc\": \"<new requirement that applies under that condition>\",\n"
        "      \"verifier\": {\n"
        "         \"check\": \"<verifier function name>\",\n"
        "         \"args\": { }\n"
        "      }\n"
        "    }, ...\n"
        "  ]\n"
        "}\n\n"
        "You MUST also include a field \"selection_type\" with value \"local\" or \"global\" matching the user's requested branch type.\n"
        "If selection_type == \"local\": the alternate branch must merge back into the main timeline at the provided merge_point.\n"
        "If selection_type == \"global\": the alternate path continues as an independent storyline (merge_point may be null).\n"
        "You may optionally include \"merge_point\" and \"alt_chain\" for additional alternate blocks.\n"
        "ABOUT verifier.check:\n"
        "- If one of these fits, you MAY use it:\n"
        "    tone_neutral_llm_judge\n"
        "    tone_negative_llm_judge\n"
        "    non_extremeness_judge\n"
        "    role_consistency_judge\n"
        "    actionability_judge\n"
        "    forbid_first_person\n"
        "    min_word_count\n"
        "    must_list_n_subpoints\n"
        "    min_numbered_items\n"
        "    must_include_keywords\n"
        "    keyword_min_frequency\n"
        "    must_cover_topics\n"
        "    min_char_count\n"
        "    require_language\n"
        "    has_sections\n"
        "    must_end_with_template\n"
        "- OTHERWISE, you MUST create a NEW descriptive snake_case name that reflects the obligation, e.g.\n"
        "    must_escalate_urgency\n"
        "    first_person_witness_tone\n"
        "    provide_step_by_step_plan\n"
        "  This is allowed.\n"
        "  Any new verifier.check MUST still describe a requirement that is plausible for THIS SAME BLOCK.\n"
        "  You are NOT allowed to invent a requirement that would make no sense for this block's role.\n\n"
        "RULES FOR NEW verifier.check NAMES:\n"
        "- Use snake_case only [a-z0-9_].\n"
        "- The name must clearly reflect the obligation in 'desc'.\n"
        "- 'args' must be a JSON object (possibly empty), e.g. {\"min_items\": 3}.\n\n"
        "RULES FOR 'desc':\n"
        "- 'desc' must be English, imperative, concrete, and verifiable.\n"
        "- 'desc' MUST describe what the alternative branch NOW REQUIRES under that condition.\n"
        "- Do NOT mention block ids.\n"
        "- Do NOT just restate the REAL BRANCH wording. It must impose a meaningfully different obligation.\n\n"
        "FINAL RULE:\n"
        "Return ONLY the JSON. NO explanations outside JSON.\n"
    )

    user_prompt = (
        "SEED TASK (overall assignment/user ask):\n" + seed_task.strip() + "\n\n"
        f"CURRENT BLOCK ID: {block_id}\n"
        f"BLOCK INTENT / ROLE IN THE ANSWER: {block_intent}\n\n"
        "BLOCK TEXT (actual content from the answer; keep topic consistent with this):\n"
        f"{block_text_clean}\n\n"
        "REAL BRANCH CONSTRAINTS (what this block currently enforces):\n"
        f"{real_constraints_for_llm}\n\n"
        "Now create ONE realistic ALTERNATIVE BRANCH for this SAME block:\n"
        "- The alternative branch must apply under a clear conditional trigger ('If ...').\n"
        "- It must impose meaningfully DIFFERENT obligations from the REAL BRANCH (not trivial paraphrase).\n"
        "- It must stay on-topic and plausible for this block's role.\n"
        "- You MAY create new verifier.check names in snake_case if needed, with JSON args.\n"
        f"- The branch TYPE must be '{selection_hint_text}'.\n"
        "- If the branch type is 'local', specify merge_point where this alternate branch rejoins the main timeline.\n"
        "- If the branch type is 'global', you may optionally provide alt_chain to continue the alternate storyline.\n"
        "Return ONLY the JSON spec described above.\n"
    )

    try:
        content = call_chat_completions(
            messages=[{"role": "user", "content": user_prompt}],
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=900,
            timeout=20,
        ).strip()
        start = content.find("{")
        end = content.rfind("}") + 1
        json_str = content[start:end]
        parsed = json.loads(json_str)
        return parsed
    except (DeepSeekError, Exception):
        return {}


def _fallback_selection(block_id: str,
                        block_constraints: List[ConstraintNode],
                        selection_type: str) -> Dict[str, Any]:
    """
    LLM 失败时构造一个兜底的 selection 说明，支持 local / global。
    """
    if selection_type == "global":
        return {
            "condition": "If a radically different storyline is required",
            "selection_type": "global",
            "merge_point": None,
            "alt_constraints": [
                {
                    "desc": "Reimagine this stage with a fundamentally different narrative focus.",
                    "verifier": {"check": "must_include_keywords", "args": {"keywords": ["new path"]}},
                },
                {
                    "desc": "Introduce a new conflict that redirects the subsequent storyline.",
                    "verifier": {"check": "must_include_keywords", "args": {"keywords": ["conflict"]}},
                },
            ],
        }
    return {
        "condition": "If the stance is critical/negative",
        "selection_type": "local",
        "alt_constraints": [
            {
                "desc": "Adopt an explicitly critical tone highlighting concrete problems or failures.",
                "verifier": {"check": "tone_negative_llm_judge", "args": {}},
            },
            {
                "desc": "Propose at least one concrete next-step action to address the identified problems.",
                "verifier": {"check": "actionability_judge", "args": {}},
            },
        ],
    }


def _build_global_followup_fallback(base_alt_block_id: str,
                                    tail_block_ids: List[str],
                                    block_info_map: Dict[str, Dict[str, Any]],
                                    start_order_index: int) -> List[Tuple[BlockSpec, List[ConstraintNode]]]:
    """
    Construct deterministic alternate follow-up blocks for a global selection when
    the LLM does not return an alt_chain. These blocks intentionally diverge from
    the main storyline by enforcing different constraints (keywords, tone, numbered items).
    """
    results: List[Tuple[BlockSpec, List[ConstraintNode]]] = []
    current_order = start_order_index

    for idx, original_bid in enumerate(tail_block_ids, start=1):
        template_idx = min(idx - 1, len(_GLOBAL_ALT_FOLLOWUP_LIBRARY) - 1)
        templates = _GLOBAL_ALT_FOLLOWUP_LIBRARY[template_idx]

        new_block_id = f"{base_alt_block_id}_NEXT{idx}"
        original_info = block_info_map.get(original_bid, {})
        base_intent = original_info.get("intent") or original_bid
        new_intent = base_intent

        current_order += 1
        new_spec = BlockSpec(
            block_id=new_block_id,
            intent=new_intent,
            text_span=f"Alternate storyline continuation derived from {original_bid}.",
            order_index=current_order,
            is_alternate=True,
            origin_block=original_bid,
        )

        new_nodes: List[ConstraintNode] = []
        for cons_idx, tmpl in enumerate(templates, start=1):
            desc = tmpl["desc"]
            verifier_spec = copy.deepcopy(tmpl["verifier"])
            new_nodes.append(
                ConstraintNode(
                    cid=f"{new_block_id}_C{cons_idx}",
                    desc=desc,
                    scope="local",
                    verifier_spec=verifier_spec,
                    trace_to=new_block_id,
                    derived_from="step5",
                )
            )

        results.append((new_spec, new_nodes))

    return results


def generate_selection_branches(segmentation: Dict[str, Any],
                                 seed_task: str,
                                 step4_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step5 主入口：

    输入：
    - segmentation: Step2 输出
    - seed_task: Step1 输出
    - step4_output: Step4 输出 = {
        "block_constraints": {block_id: [ConstraintNode,...]},
        "block_logic": {block_id: "AND" | "sub-chain"}
      }

    输出：
    - 扩展后的结构：{
        "block_constraints": {... 包含原本约束 + 新增伪分支中产生的新约束 ...},
        "block_logic": {... 原样透传 ...},
        "selections": [SelectionNode, ...],
        "extra_blocks": [BlockSpec,...]   # 新增的替代块定义（用于局部/全局分支）
      }
    """

    block_constraints: Dict[str, List[ConstraintNode]] = {
        bid: list(nodes) for bid, nodes in step4_output["block_constraints"].items()
    }
    block_logic = dict(step4_output["block_logic"])  # shallow copy
    extra_blocks: List[BlockSpec] = []

    order_list = segmentation.get("order") or [b.get("block_id") for b in segmentation.get("blocks", [])]
    order_index = {bid: idx for idx, bid in enumerate(order_list)}

    ordered_block_ids = [
        b.get("block_id")
        for b in segmentation.get("blocks", [])
        if b.get("block_id") in block_constraints and block_constraints[b.get("block_id")]
    ]
    seed_material = "|".join(ordered_block_ids).encode("utf-8")
    seed_int = int.from_bytes(hashlib.sha256(seed_material).digest(), "big")
    rng = random.Random(seed_int)

    selection_plan = _plan_selections(block_constraints, segmentation, rng)
    selections: List[SelectionNode] = []

    if not selection_plan:
        return {
            "block_constraints": block_constraints,
            "block_logic": block_logic,
            "selections": selections,
            "extra_blocks": extra_blocks,
        }

    main_blocks_in_order = list(ordered_block_ids)

    for chosen_block_id, selection_type_target in selection_plan:
        real_nodes = block_constraints.get(chosen_block_id, [])
        real_cids = [n.cid for n in real_nodes]

        block_info = next(
            (b for b in segmentation.get("blocks", []) if b.get("block_id") == chosen_block_id),
            None,
        ) or {}
        block_intent = block_info.get("intent", "")
        block_text = block_info.get("text_span", "")
        real_constraints_for_llm = _format_block_constraints_for_llm(real_nodes)

        alt_spec = _call_deepseek_selection_aug(
            block_id=chosen_block_id,
            block_intent=block_intent,
            block_text=block_text,
            seed_task=seed_task,
            real_constraints_for_llm=real_constraints_for_llm,
            selection_type_hint=selection_type_target,
        )
        if not alt_spec or "alt_constraints" not in alt_spec:
            alt_spec = _fallback_selection(chosen_block_id, real_nodes, selection_type_target)

        condition_text = alt_spec.get("condition", "If an alternative stance applies")
        selection_type = selection_type_target.lower()
        if selection_type not in {"local", "global"}:
            selection_type = "local"

        alt_block_id = alt_spec.get("alt_block_id") or f"{chosen_block_id}_ALT"
        alt_block_intent = alt_spec.get("alt_block_intent") or block_intent
        alt_block_intent = alt_block_intent.replace("(alternate)", "").strip()
        alt_block_text = alt_spec.get("alt_block_text", block_text)

        # Determine merge point for local branches
        merge_point: Optional[str] = None
        if selection_type == "local":
            merge_point = alt_spec.get("merge_point")
            if not merge_point:
                # default merge point is the next block in segmentation order
                order_list = segmentation.get("order") or [b.get("block_id") for b in segmentation.get("blocks", [])]
                try:
                    idx = order_list.index(chosen_block_id)
                    merge_point = order_list[idx + 1] if idx + 1 < len(order_list) else None
                except ValueError:
                    merge_point = None
        else:
            merge_point = alt_spec.get("merge_point")

        truncated_flag = bool(alt_spec.get("truncated", False))

        # Prepare alternate block constraint nodes
        alt_constraints_raw = alt_spec.get("alt_constraints", [])
        new_alt_nodes: List[ConstraintNode] = []
        next_idx = 1
        for item in alt_constraints_raw:
            desc = item.get("desc", "").strip()
            verif = item.get("verifier", {})
            check_name = verif.get("check")
            args_obj = verif.get("args", {}) or {}
            if not desc or not check_name:
                continue
            new_node = ConstraintNode(
                cid=f"{alt_block_id}_C{next_idx}",
                desc=desc,
                scope="local",
                verifier_spec={"check": check_name, "args": args_obj},
                trace_to=alt_block_id,
                derived_from="step5",
            )
            new_alt_nodes.append(new_node)
            next_idx += 1

        # Register alternate block constraints and logic
        block_constraints.setdefault(alt_block_id, [])
        block_constraints[alt_block_id].extend(new_alt_nodes)
        block_logic[alt_block_id] = "AND"

        alt_block_spec = BlockSpec(
            block_id=alt_block_id,
            intent=alt_block_intent,
            text_span=alt_block_text,
            order_index=block_info.get("order_index", 0),
            is_alternate=True,
            origin_block=chosen_block_id,
        )
        extra_blocks.append(alt_block_spec)

        alt_cids = [n.cid for n in new_alt_nodes]
        alt_path_blocks: List[str] = [alt_block_id]

        # Handle extended alternate chains for global selections
        alt_chain = alt_spec.get("alt_chain", [])
        previous_order_index = alt_block_spec.order_index
        chain_created = False
        for idx_chain, block_def in enumerate(alt_chain, start=1):
            chain_block_id = f"{alt_block_id}_N{idx_chain}"
            chain_intent = (block_def.get("intent") or block_intent).replace("(alternate)", "").strip()
            chain_text = block_def.get("text_span", alt_block_text)
            chain_constraints_raw = block_def.get("constraints", [])
            chain_logic = block_def.get("logic", "AND")

            chain_nodes: List[ConstraintNode] = []
            for item_idx, item in enumerate(chain_constraints_raw, start=1):
                desc = item.get("desc", "").strip()
                verif = item.get("verifier", {})
                check_name = verif.get("check")
                args_obj = verif.get("args", {}) or {}
                if not desc or not check_name:
                    continue
                chain_nodes.append(
                    ConstraintNode(
                        cid=f"{chain_block_id}_C{item_idx}",
                        desc=desc,
                        scope="local",
                        verifier_spec={"check": check_name, "args": args_obj},
                        trace_to=chain_block_id,
                        derived_from="step5",
                    )
                )
            if not chain_nodes:
                continue

            block_constraints.setdefault(chain_block_id, [])
            block_constraints[chain_block_id].extend(chain_nodes)
            block_logic[chain_block_id] = chain_logic

            chain_spec = BlockSpec(
                block_id=chain_block_id,
                intent=chain_intent,
                text_span=chain_text,
                order_index=previous_order_index + idx_chain + 1,
                is_alternate=True,
                origin_block=chosen_block_id,
            )
            extra_blocks.append(chain_spec)
            alt_path_blocks.append(chain_block_id)
            alt_cids.extend([node.cid for node in chain_nodes])
            chain_created = True

        if selection_type == "global":
            last_reference = alt_path_blocks[-1] if alt_path_blocks else chosen_block_id
            base_order = order_index.get(last_reference, order_index.get(chosen_block_id, 0))
            remaining_main = [
                bid for bid in main_blocks_in_order
                if order_index.get(bid, 10**6) > order_index.get(chosen_block_id, 0)
            ]
            block_info_map = {
                blk.get("block_id"): blk for blk in segmentation.get("blocks", [])
            }

            fallback_followups = _build_global_followup_fallback(
                base_alt_block_id=alt_block_id,
                tail_block_ids=remaining_main,
                block_info_map=block_info_map,
                start_order_index=previous_order_index,
            )

            for follow_spec, follow_nodes in fallback_followups:
                block_constraints.setdefault(follow_spec.block_id, [])
                block_constraints[follow_spec.block_id].extend(follow_nodes)
                block_logic[follow_spec.block_id] = "AND"
                extra_blocks.append(follow_spec)
                alt_path_blocks.append(follow_spec.block_id)
                alt_cids.extend([node.cid for node in follow_nodes])
                previous_order_index = follow_spec.order_index

        selection_node = SelectionNode(
            sid=f"SEL_{chosen_block_id}",
            condition=condition_text,
            trace_to=chosen_block_id,
            branch_real=SelectionBranch(block_id=chosen_block_id, constraints=real_cids),
            branch_alt=SelectionBranch(block_id=alt_block_id, constraints=alt_cids),
            derived_from="step5",
            selection_type=selection_type,
            merge_point=merge_point,
            truncated=truncated_flag,
            alt_path_blocks=alt_path_blocks,
        )
        selections.append(selection_node)

    # 返回增广后的结果
    return {
        "block_constraints": block_constraints,
        "block_logic": block_logic,
        "selections": selections,
        "extra_blocks": extra_blocks,
    }


# main 测试流程保持不变，无需更改
