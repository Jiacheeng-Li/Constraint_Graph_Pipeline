
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

def _summarize_blocks_for_prompt(segmentation: Dict[str, Any]) -> str:
    lines = []
    for blk in segmentation.get("blocks", []):
        bid = blk.get("block_id", "?")
        intent = blk.get("intent", "?")
        lines.append(f"{bid}: {intent}")
    return "\n".join(lines)


def _call_deepseek_soft_constraints(response_text: str,
                                    segmentation: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    让 deepseek 只负责推断“语气 / 安全 / 立场 / 风格”类的全局约束，
    并映射到我们已有的 soft verifiers，例如：
      - tone_neutral_llm_judge
      - non_extremeness_judge
      - actionability_judge (如果全篇明显要给建议)
      - role_consistency_judge (如果明显要求特定叙述身份)

    它输出一个 JSON list，每项形如：
    {
      "desc": "The answer must maintain a neutral, analytical tone without emotional or inflammatory language.",
      "verifier": {
          "check": "tone_neutral_llm_judge",
          "args": {}
      }
    }
    """

    block_outline = _summarize_blocks_for_prompt(segmentation)

    system_prompt = (
        "You are an instruction analyst.\n"
        "Infer ONLY global style/tone/safety requirements for the answer.\n"
        "These are high-level constraints on tone, stance, safety, or narrative voice that apply to the ENTIRE answer.\n"
        "Examples:\n"
        "- The answer must maintain a neutral, analytical tone. -> tone_neutral_llm_judge\n"
        "- The answer must avoid inflammatory or insulting language. -> non_extremeness_judge\n"
        "- The answer should present concrete recommendations or next steps. -> actionability_judge\n"
        "- The answer must keep a consistent third-person analyst voice. -> role_consistency_judge\n"
        "Do NOT include word count, language, paragraph structure, or first-person bans here; those are handled elsewhere.\n"
        "Return ONLY valid JSON: a list of objects with fields {desc, verifier:{check,args}}.\n"
        "Use ONLY these verifier names: tone_neutral_llm_judge, non_extremeness_judge, actionability_judge, role_consistency_judge\n"
        "If nothing applies, return an empty JSON list []."
    )

    user_prompt = (
        "ANSWER TEXT (full):\n" + response_text.strip() + "\n\n"
        "BLOCK OUTLINE (order and intent):\n" + block_outline + "\n\n"
        "Infer the global style/tone/safety constraints. Output ONLY the JSON list."
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

        start = content.find("[")
        end = content.rfind("]") + 1
        json_str = content[start:end]
        parsed = json.loads(json_str)
        return parsed
    except Exception:
        return []


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
    soft_raw = _call_deepseek_soft_constraints(response_text, segmentation)

    soft_nodes: List[ConstraintNode] = []
    for item in soft_raw:
        desc = item.get("desc", "").strip()
        verif = item.get("verifier", {})
        check_name = verif.get("check")
        args_obj = verif.get("args", {}) or {}
        if not desc or not check_name:
            continue
        soft_nodes.append(
            ConstraintNode(
                cid="TEMP",  # 先占位，后面统一重排ID
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