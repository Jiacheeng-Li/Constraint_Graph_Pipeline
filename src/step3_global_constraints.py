"""
step3_global_constraints.py

Step 3: 全局约束抽取 (Global Constraint Extraction)

- 我们把全局约束分成两类：
  A. 硬性可程序校验的全局约束（hard global constraints）
     例如：最少词数、必须包含结构段落、禁止第一人称、必须为英文。
     这些可以直接由我们本地代码给出并附上 verifier_spec，
     不依赖 LLM，因此总是可用，保证下游评测有稳定基线。

  B. 软性 / 语气 / 安全 / 质量类全局约束（soft / semantic global constraints）
     例如：中立分析语气、不得煽动性攻击、输出必须保持专业而非情绪化。
     这些需要语用判断，继续调用 deepseek 生成或确认，
     并为它们附上 LLM-based 的 verifier（如 tone_neutral_llm_judge, non_extremeness_judge）。


输出：List[ConstraintNode]
- 每个 ConstraintNode:
    cid: 全局唯一ID（G1, G2, ...）
    desc: 人类可读描述
    scope: "global"
    verifier_spec: {"check": <fn-name>, "args": {...}}
    derived_from: "step3"

依赖：
- deepseek-chat (用于软性约束)
- ConstraintNode schema
- 硬性规则来自我们自己的启发式：
  - 字数下限 (min_word_count)
  - 语言判断 (require_language)
  - 结构段落 (has_sections) [仅当回答明显分块时]
  - 禁止第一人称 (forbid_first_person) [可选]
"""

import json
import requests
from typing import List, Dict, Any

from .graph_schema import ConstraintNode
from .utils.parsing import extract_constraints
from .utils.text_clean import make_snippet, summarize_blocks_outline, clip

_DEEPSEEK_API_KEY_DEFAULT = "sk-4bb3e24d26674a30b2cc7e2ff1bfc763"
_DEEPSEEK_ENDPOINT = "https://api.deepseek.com/v1/chat/completions"
_DEEPSEEK_MODEL = "deepseek-chat"


# -------------------------------------------------
# 工具：从当前回答中推测硬性全局约束基线
# -------------------------------------------------

def _estimate_word_count(text: str) -> int:
    import re
    tokens = re.findall(r"\w+", text)
    return len(tokens)


def _guess_language(text: str) -> str:
    """
    粗暴判断文本主要语言：
    - 如果包含较多中文汉字 => 'zh'
    - 否则默认 'en'
    我们不做复杂检测，这只是为了构造 require_language。
    """
    import re
    zh_chars = re.findall(r"[\u4e00-\u9fff]", text)
    if len(zh_chars) >= 10:
        return "zh"
    return "en"


def _has_intro_body_conclusion(segmentation: Dict[str, Any]) -> bool:
    """
    根据 Step2 的 segmentation 结果，看看是否能观察到典型结构：
    - 存在开篇类块 (Opening / Intro / Background / Context)
    - 存在主体分析类块 (Main Analysis / Discussion / Evaluation / Argument)
    - 存在总结/展望类块 (Conclusion / Summary / Outlook / Recommendation)

    如果这些intent基本存在，就可以生成一个 has_sections 约束。
    否则别强行要求。
    """
    intents = [blk.get("intent", "").lower() for blk in segmentation.get("blocks", [])]

    def any_contains(keys):
        return any(any(k in intent for k in keys) for intent in intents)

    has_opening = any_contains(["opening", "intro", "context", "background"])
    has_body = any_contains(["analysis", "discussion", "main", "argument", "evaluation"])
    has_conclusion = any_contains(["conclusion", "summary", "outlook", "recommendation"])

    return has_opening and has_body and has_conclusion


def _build_hard_global_constraints(response_text: str,
                                   segmentation: Dict[str, Any]) -> List[ConstraintNode]:
    """
    基于可观测信号，构造稳定的硬性全局约束节点。
    我们不会幻想不存在的要求，只根据文本本身的客观属性：
    - 字数下限：设为 floor(word_count * 0.8) 向下取整，但至少 100 词。
      （思路：我们希望后续回答别比示例短太多，否则不合格）
    - 语言：根据文本主语言生成 require_language(lang=...)
    - 结构段落：如果 segmentation 看起来有开头/主体/结论，就要求 has_sections
    这些都会被标记为 scope="global"。
    """
    nodes: List[ConstraintNode] = []
    cid_counter = 1

    # 1. 字数下限约束
    wc = _estimate_word_count(response_text)
    if wc > 0:
        target_min = max(100, int(wc * 0.8))
        nodes.append(
            ConstraintNode(
                cid=f"G{cid_counter}",
                desc=f"The answer must be at least {target_min} words long (approximately comparable length to the provided reference).",
                scope="global",
                verifier_spec={
                    "check": "min_word_count",
                    "args": {"min_words": target_min},
                },
                trace_to=None,
                derived_from="step3",
            )
        )
        cid_counter += 1

    # 2. 主语言约束
    lang = _guess_language(response_text)
    nodes.append(
        ConstraintNode(
            cid=f"G{cid_counter}",
            desc=("The answer must be written primarily in Chinese." if lang == "zh"
                  else "The answer must be written primarily in English."),
            scope="global",
            verifier_spec={
                "check": "require_language",
                "args": {"lang": lang},
            },
            trace_to=None,
            derived_from="step3",
        )
    )
    cid_counter += 1

    # 3. 结构性约束（仅当回答真的有明显结构）
    if _has_intro_body_conclusion(segmentation):
        nodes.append(
            ConstraintNode(
                cid=f"G{cid_counter}",
                desc="The answer must include an Opening/Intro section, a Main Analysis/Body section, and a Conclusion/Outlook section in logical progression.",
                scope="global",
                verifier_spec={
                    "check": "has_sections",
                    "args": {"sections": ["Opening", "Body", "Conclusion"]},
                },
                trace_to=None,
                derived_from="step3",
            )
        )
        cid_counter += 1

    # 4. 禁用第一人称（只在回答主要是第三人称分析风格时添加）
    # 启发式：如果文本里几乎没有 "I " / "we ", 我们假定它是客观第三人称分析，
    # 那么我们就可以把 forbid_first_person 设为一个约束。
    lower_txt = response_text.lower()
    first_person_hits = any(token in lower_txt for token in [" i ", " we ", " my ", " our "])  # 粗暴启发式
    if not first_person_hits:
        nodes.append(
            ConstraintNode(
                cid=f"G{cid_counter}",
                desc="The answer should maintain an objective, third-person analytic voice without using first-person pronouns.",
                scope="global",
                verifier_spec={
                    "check": "forbid_first_person",
                    "args": {},
                },
                trace_to=None,
                derived_from="step3",
            )
        )
        cid_counter += 1

    return nodes


# -------------------------------------------------
# LLM: 生成软性/语气/安全类全局约束
# -------------------------------------------------


def _call_deepseek_soft_constraints(response_text: str,
                                    segmentation: Dict[str, Any]) -> str:
    """
    调用 deepseek 让它给出“语气 / 安全 / 风格”类全局约束。

    非常重要：
    - 我们现在要求它只能根据回答本身已经呈现出来的风格/语气/姿态来提炼约束，
      不能脑补“理想上应该是什么样”。
    - 我们提供给它的 TEXT SNIPPET 是原文本身（仅做空白规整），
      不摘要、不改写、不自动截断；只有在极端过长时才 clip() 以防 token 爆炸。
    - outline 只是结构位置参考，不能用来发明没出现的要求。

    期望 deepseek 输出：JSON list，每项类似：
        {
          "desc": "The answer must maintain a neutral, analytical tone.",
          "verifier": {"check": "tone_neutral_llm_judge", "args": {}}
        }
    """

    # 处理原文：保持语义，去除多余空白，不默认截断
    answer_clean = make_snippet(response_text)
    if len(answer_clean) > 20000:
        # 极端长文本才触发截断；这是显式的、可审计的内容丢失点
        answer_clean = clip(answer_clean, 20000)

    outline_str = summarize_blocks_outline(segmentation)
    
    system_prompt = """You are an instruction analyst.
Your job is to infer ONLY global style/tone/safety requirements that the FULL ANSWER is ALREADY FOLLOWING.
You MUST base every requirement on observable evidence in the provided TEXT SNIPPET.
Do NOT invent idealized rules that are not clearly demonstrated in that text.
The OUTLINE is just structural context (which block does what), NOT evidence.
If you cannot justify a requirement from the snippet, you must NOT output it.

Soft global constraints are about tone, safety, stance, professional voice, neutrality, actionability, or analyst persona consistency across the entire answer.
Do NOT restate local factual obligations (e.g. "must list three risks") that only apply to one block; those belong to local block constraints, not global style.  🔁

Every constraint must be grounded in observable evidence in the TEXT SNIPPET.
Do NOT invent requirements that do not clearly appear in the text.

You must return ONLY valid JSON: a list of objects.
Each object MUST have: {desc, verifier:{check,args}}.

About verifier.check:
- If one of these fits, use it:
  tone_neutral_llm_judge
  tone_negative_llm_judge
  non_extremeness_judge
  role_consistency_judge
  actionability_judge
- Otherwise, you MUST create a new descriptive snake_case name
  that reflects the requirement, e.g. "must_include_case_studies", "balanced_argumentation", "risk_mitigation_guidance".
  This is allowed.
Any new verifier.check you create MUST still describe a requirement that is clearly exhibited by the TEXT SNIPPET. 🔁
You are NOT allowed to invent a requirement that the snippet does not follow, just to create a new check name. 🔁

Rules for new verifier names:
- snake_case only [a-z0-9_]
- It must reflect the obligation in desc.
- args must be a JSON object (possibly empty) describing any parameters needed to check this rule, e.g. {"min_items": 3}.

If nothing applies, return an empty JSON list [].

Rules:
- desc must be English, imperative, concrete, verifiable.
- desc should describe the style/voice/safety stance the answer actually exhibits.
- Do NOT include word count, paragraph structure, language choice, or first-person bans here.
  Those are handled elsewhere.
- Do NOT output explanations outside JSON."""


    user_prompt = (
        "GLOBAL OUTLINE (structure only; DO NOT invent rules from this):\n"
        f"{outline_str}\n\n"
        "TEXT SNIPPET (this is the FULL ANSWER content as given to the user;\n"
        "ALL requirements MUST be grounded in this text, do NOT hallucinate):\n"
        f"{answer_clean}\n\n"
        "Extract the global style/tone/safety constraints that the answer is ALREADY following.\n"
        "Return ONLY the JSON list.\n"
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
        "temperature": 0.0,
        "max_tokens": 800,
    }

    try:
        resp = requests.post(
            _DEEPSEEK_ENDPOINT, headers=headers, data=json.dumps(payload), timeout=20
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
        return content
    except Exception:
        # 兜底：返回一个空 JSON list 字符串，让上层解析时得到 []
        return "[]"


# -------------------------------------------------
# 主入口：结合硬约束 + 软约束
# -------------------------------------------------

def extract_global_constraints(response_text: str,
                               segmentation: Dict[str, Any]) -> List[ConstraintNode]:
    """
    Step3 主入口：

    1. 基于回答文本 + segmentation，构造硬性全局约束（本地可验证）。
       这些约束永远会存在，确保我们至少能做程序化打分。

    2. 调用 deepseek 提取软性/语气/安全类的全局约束；可能返回0条，也可能多条。

    3. 把二者合并，按顺序编号 G1, G2, ... ，得到最终的全局约束列表。

    注意：
    - 不再强行生成通用fallback约束。
    - 软性约束缺席时，我们仍有硬性约束可用。
    """

    hard_nodes = _build_hard_global_constraints(response_text, segmentation)
    soft_raw_str = _call_deepseek_soft_constraints(response_text, segmentation)
    soft_items = extract_constraints(soft_raw_str)  # list[dict]

    soft_nodes: List[ConstraintNode] = []
    for item in soft_items:
        # extract_constraints() 已经尽量标准化字段名：cid/desc/scope/verifier_spec
        desc = item.get("desc", "").strip()
        verifier_spec = item.get("verifier_spec", {}) or item.get("verifier", {}) or {}
        check_name = verifier_spec.get("check")
        args_obj = verifier_spec.get("args", {}) or {}

        if not desc or not check_name:
            continue

        soft_nodes.append(
            ConstraintNode(
                cid="TEMP",  # 后续统一重排ID
                desc=desc,
                scope="global",
                verifier_spec={"check": check_name, "args": args_obj},
                trace_to=None,
                derived_from="step3",
            )
        )

    # 合并并重新编号 cid
    all_nodes: List[ConstraintNode] = []
    for node in hard_nodes + soft_nodes:
        all_nodes.append(node)
    for idx, node in enumerate(all_nodes, start=1):
        node.cid = f"G{idx}"

    return all_nodes


if __name__ == "__main__":
    demo_resp = (
        "The modern space race is not only a technical contest but a geopolitical instrument. "
        "In this analysis, we outline historical context, assess key actors, and discuss future risks.\n\n"
        "First, we review how national prestige and commercial incentives shaped recent launches.\n\n"
        "Finally, we conclude with implications for global stability and practical next-step recommendations."
    )
    demo_seg = {
        "blocks": [
            {"block_id": "B1", "intent": "Opening / Context setup", "text_span": "..."},
            {"block_id": "B2", "intent": "Main Analysis", "text_span": "..."},
            {"block_id": "B3", "intent": "Conclusion / Outlook / Recommendation", "text_span": "..."},
        ],
        "order": ["B1", "B2", "B3"],
    }

    out_nodes = extract_global_constraints(demo_resp, demo_seg)
    print(json.dumps([
        {
            "cid": n.cid,
            "desc": n.desc,
            "scope": n.scope,
            "verifier_spec": n.verifier_spec,
            "derived_from": n.derived_from,
        } for n in out_nodes
    ], indent=2, ensure_ascii=False))