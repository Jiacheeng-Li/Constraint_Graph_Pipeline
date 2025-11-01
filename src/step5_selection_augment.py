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
import requests
import random
import hashlib
from typing import Dict, Any, List
from .graph_schema import ConstraintNode, SelectionNode, SelectionBranch
from .utils.text_clean import make_snippet, clip

_DEEPSEEK_API_KEY_DEFAULT = "sk-4bb3e24d26674a30b2cc7e2ff1bfc763"
_DEEPSEEK_ENDPOINT = "https://api.deepseek.com/v1/chat/completions"
_DEEPSEEK_MODEL = "deepseek-chat"



def _choose_blocks_for_selection(block_constraints: Dict[str, List[ConstraintNode]],
                                 segmentation: Dict[str, Any]) -> List[str]:
    """
    选择若干个 block 来生成 selection 分支，而不是只选一个。

    设计目标：
    - 我们希望在不同层级（开头、中段、结尾）都可能出现条件分支，而不是永远只在结尾出现。
    - 这些分支可以在下游指令里变成 if/else 风格的条件化路径。

    策略 (v2):
    1. 我们先收集所有“有至少一条局部约束”的 block_id，保持它们在原回答中的顺序。
    2. 我们计算一个候选集：
       - 优先包含 intent 类似 Conclusion / Recommendation / Outlook / Summary 的块（总结类）；
       - 同时也包含前半部分和中间的块，以便产生早期分支（可以走到完全不同的结尾）。
    3. 在候选集中随机抽取 1~3 个 block_id（如果候选很少，就全用）。

    注意：返回顺序按照原文出现顺序，不打乱。后续我们会对这些 block_id 逐个生成 selection。
    """
    # block_id -> intent
    intent_map = {
        b.get("block_id"): b.get("intent", "")
        for b in segmentation.get("blocks", [])
    }

    # 1. 有约束的 block_ids，保持原顺序
    ordered_block_ids = [
        b.get("block_id")
        for b in segmentation.get("blocks", [])
        if b.get("block_id") in block_constraints and block_constraints[b.get("block_id")]
    ]

    if not ordered_block_ids:
        return []

    # 2. 识别“总结/展望/建议”类 intent
    preferred_keywords = [
        "conclusion", "outlook", "recommendation", "summary", "next step", "future"
    ]
    summary_like = []
    other_like = []
    for bid in ordered_block_ids:
        intent = intent_map.get(bid, "").lower()
        if any(k in intent for k in preferred_keywords):
            summary_like.append(bid)
        else:
            other_like.append(bid)

    # 3. 组候选集：优先总结类，其次其余
    candidates = summary_like + other_like
    # 去重保持顺序
    seen = set()
    deduped = []
    for bid in candidates:
        if bid not in seen:
            deduped.append(bid)
            seen.add(bid)

    # 4. 基于候选集生成稳定随机子集。
    seed_material = "|".join(deduped).encode("utf-8")
    seed_int = int.from_bytes(hashlib.sha256(seed_material).digest(), "big")
    rng = random.Random(seed_int)

    max_pick = min(len(deduped), 3)
    pick_n = rng.randint(1, max_pick)

    picked_set = set(rng.sample(deduped, pick_n))
    final_list = [bid for bid in ordered_block_ids if bid in picked_set]

    return final_list


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
                                 real_constraints_for_llm: str) -> Dict[str, Any]:
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
        "Return ONLY the JSON spec described above.\n"
    )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {_DEEPSEEK_API_KEY_DEFAULT}",
    }

    payload = {
        "model": _DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 900,
    }

    try:
        resp = requests.post(
            _DEEPSEEK_ENDPOINT, headers=headers, data=json.dumps(payload), timeout=20
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()

        start = content.find("{")
        end = content.rfind("}") + 1
        json_str = content[start:end]
        parsed = json.loads(json_str)
        return parsed
    except Exception:
        return {}


def _fallback_selection(block_id: str,
                         block_constraints: List[ConstraintNode]) -> Dict[str, Any]:
    """
    如果 LLM 生成分支失败，我们合成人工分支：
    - condition: "If the stance is critical/negative"
    - alt_constraints: 两条：
        1. 语气必须体现批评/不满 (tone_negative_llm_judge)
        2. 必须提出下一步改进措施/行动 (actionability_judge)
    这些 verifier 都存在于 soft_checks，并已在 registry 注册。
    """
    return {
        "condition": "If the stance is critical/negative",
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
        "selections": [SelectionNode, ...]
      }
    """

    block_constraints: Dict[str, List[ConstraintNode]] = {
        bid: list(nodes) for bid, nodes in step4_output["block_constraints"].items()
    }
    block_logic = dict(step4_output["block_logic"])  # shallow copy

    # 1. 决定在哪些 block 上造分支
    chosen_block_ids = _choose_blocks_for_selection(block_constraints, segmentation)
    selections: List[SelectionNode] = []

    if not chosen_block_ids:
        # 如果完全挑不到合适的块，就返回无selection的结果
        return {
            "block_constraints": block_constraints,
            "block_logic": block_logic,
            "selections": selections,
        }

    for chosen_block_id in chosen_block_ids:
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
        )
        if not alt_spec or "alt_constraints" not in alt_spec:
            alt_spec = _fallback_selection(chosen_block_id, real_nodes)

        condition_text = alt_spec.get("condition", "If an alternative stance applies")
        alt_constraints_raw = alt_spec.get("alt_constraints", [])

        new_alt_nodes: List[ConstraintNode] = []
        next_idx = len(real_nodes) + 1
        for item in alt_constraints_raw:
            desc = item.get("desc", "").strip()
            verif = item.get("verifier", {})
            check_name = verif.get("check")
            args_obj = verif.get("args", {}) or {}
            if not desc or not check_name:
                continue
            new_node = ConstraintNode(
                cid=f"{chosen_block_id}_C{next_idx}",
                desc=desc,
                scope="local",
                verifier_spec={"check": check_name, "args": args_obj},
                trace_to=chosen_block_id,
                derived_from="step5",
            )
            new_alt_nodes.append(new_node)
            block_constraints[chosen_block_id].append(new_node)
            next_idx += 1

        alt_cids = [n.cid for n in new_alt_nodes]

        selection_node = SelectionNode(
            sid=f"SEL_{chosen_block_id}",
            condition=condition_text,
            trace_to=chosen_block_id,
            branch_real=SelectionBranch(constraints=real_cids),
            branch_alt=SelectionBranch(constraints=alt_cids),
            derived_from="step5",
        )
        selections.append(selection_node)

    # 返回增广后的结果
    return {
        "block_constraints": block_constraints,
        "block_logic": block_logic,
        "selections": selections,
    }


# main 测试流程保持不变，无需更改
