"""
step4_back_traslation.py

Step 4: 反向约束抽取 (Back-translation of each block)

目标：
对 step2 分块得到的每个 block，让 LLM 解释：
  - 这个 block 在“完成 seed_task 的过程中”到底在做什么？
  - 这个 block 在内容上、语气上、结构上，满足了哪些具体的可验证要求？
  - 这些要求哪些是硬性 AND（必须全部做），哪些是内部顺序式 sub-chain（先做A再做B）？

输出：
我们要返回两部分：
1. block_constraints: { block_id: [ConstraintNode, ...], ... }
   - 对于每个 block，抽出它满足的约束，每条约束都带 verifier_spec
   - scope="local"，trace_to=该 block_id，derived_from="step4"

2. block_logic: { block_id: "AND" | "sub-chain" }
   - 如果该块基本是在并列罗列要点/事实/解释 -> AND
   - 如果该块是明显的多步推进（例如：先定义，再对比，再给结论）-> sub-chain
   （我们先让模型判断；失败时 fallback=AND）

说明：
- 这些约束节点后面会进入 Step5 作为候选，生成分支 (Selection)。
- 我们在这里不会强行 eval 文本，只是生成约束定义 + verifier_spec。

策略：
A. 遍历 segmentation["blocks"]；对每个block：
   1. 调用 deepseek，请它输出该 block 的局部约束列表，JSON格式。
   2. 每条约束包含：desc / verifier.check / verifier.args。
   3. 还要它告诉我们这个block内部更像 AND 还是 sub-chain。

B. fallback：
   - 如果 deepseek 挂了，该 block 给两条最宽泛约束：
       * 中立分析口吻 (tone_neutral_llm_judge)
       * 至少50词 (min_word_count)
     block_logic = "AND"

边界：
- 直接使用 deepseek-chat
- 注意：本步引用 ConstraintNode 定义
"""

import json
import requests
from typing import Dict, Any, List

from .graph_schema import ConstraintNode, BlockSpec
from .utils.parsing import extract_constraints, safe_json_load
from .utils.text_clean import make_snippet, summarize_blocks_outline, clip

_DEEPSEEK_API_KEY_DEFAULT = "sk-4bb3e24d26674a30b2cc7e2ff1bfc763"
_DEEPSEEK_ENDPOINT = "https://api.deepseek.com/v1/chat/completions"
_DEEPSEEK_MODEL = "deepseek-chat"


def _call_deepseek_block_constraints(block: BlockSpec,
                                     seed_task: str,
                                     segmentation: Dict[str, Any]) -> str:
    """
    让 deepseek 针对单个 block 生成：
    - 该 block 的逻辑类型 (logic: "AND" | "sub-chain")
    - 该 block 实际在做/必须做的可验证要求列表 (constraints: [...])

    重要：
    - 该函数现在只做 LLM 调用并返回原始字符串，不做解析。
    - 解析逻辑统一放到 utils/parsing (extract_constraints / safe_json_load)。

    我们要求 deepseek 输出严格 JSON：
    {
      "logic": "AND" | "sub-chain",
      "constraints": [
        {
          "desc": "Explain historical background in neutral tone.",
          "verifier": {
             "check": "tone_neutral_llm_judge",
             "args": {}
          }
        },
        ...
      ]
    }
    """

    # 使用 text_clean 保留原文语义，仅做空白规整；不默认截断
    block_text_clean = make_snippet(block.text_span)

    # 作为安全阀：如果块内容极端长，显式硬截断（唯一会丢信息的地方）
    if len(block_text_clean) > 12000:
        block_text_clean = clip(block_text_clean, 12000)

    # 给模型一点上下文：整个回答的结构是什么（只是结构提示，不是证据）
    outline_str = summarize_blocks_outline(segmentation)

    system_prompt = (
        "You are an instruction reverse-engineer.\n"
        "Goal: For ONE block of an assistant's answer, infer what concrete obligations that block actually satisfies.\n"
        "You must ONLY claim obligations that are directly evidenced in the provided TEXT SNIPPET.\n"
        "Do NOT invent requirements that are not literally supported by that snippet.\n"
        "The OUTLINE is only high-level context of where this block sits in the full answer.\n"
        "The SEED TASK is the global assignment.\n"
        "But your constraints MUST be grounded strictly in the snippet's actual wording/content.\n\n"
        "Return ONLY valid JSON with this structure:\n"
        "{\n"
        "  \"logic\": \"AND\" or \"sub-chain\",\n"
        "  \"constraints\": [\n"
        "    {\n"
        "      \"desc\": \"<what this block MUST accomplish (as evidenced in the text)>\",\n"
        "      \"verifier\": {\n"
        "         \"check\": \"<verifier function name>\",\n"
        "         \"args\": { }\n"
        "      }\n"
        "    }, ...\n"
        "  ]\n"
        "}\n\n"
        "Rules for 'logic':\n"
        "- 'logic' = 'AND' if the block lists parallel obligations/facts.\n"
        "- 'logic' = 'sub-chain' if the block clearly performs a multi-step progression (define -> compare -> conclude).\n\n"
        "Grounding rules:\n"
        "- Every constraint MUST be directly supported by evidence in the TEXT SNIPPET.\n"
        "- You MUST NOT add ideal/aspirational requirements that are not clearly present in the snippet.\n"
        "- The OUTLINE and the SEED TASK are context only; you CANNOT invent constraints from them.\n\n"
        "About verifier.check:\n"
        "- If one of these fits, use it:\n"
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
        "- Otherwise, you MUST create a new descriptive snake_case name that reflects the requirement actually shown in the snippet, e.g.\n"
        "    must_compare_policy_options\n"
        "    must_include_two_examples\n"
        "    must_reference_timeframe\n"
        "  This is allowed.\n\n"
        "Rules for new verifier names:\n"
        "- snake_case only [a-z0-9_]\n"
        "- The name must reflect the obligation in 'desc'.\n"
        "- 'args' must be a JSON object (possibly empty) describing any parameters needed to check, e.g. {\"min_items\": 3}.\n\n"
        "Rules for 'desc':\n"
        "- 'desc' must be English, imperative, concrete, and verifiable.\n"
        "- 'desc' MUST NOT mention block ids.\n"
        "- 'desc' MUST describe what the block IS doing / MUST accomplish per the snippet, not what it SHOULD do ideally.\n\n"
        "Output rule:\n"
        "- Do NOT output any explanation outside JSON.\n"
    )

    user_prompt = (
        "GLOBAL OUTLINE (high-level structure only; DO NOT invent obligations from this):\n"
        f"{outline_str}\n\n"
        "SEED TASK (the global assignment):\n"
        f"{seed_task.strip()}\n\n"
        "CURRENT BLOCK YOU ARE ANALYZING:\n"
        f"BLOCK ID: {block.block_id}\n"
        f"BLOCK INTENT: {block.intent}\n\n"
        "TEXT SNIPPET (this is the only allowed evidence; base ALL constraints ONLY on this text):\n"
        f"{block_text_clean}\n\n"
        "Now produce the JSON.\n"
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
        "max_tokens": 900,
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
        # 兜底：返回一个最小可解析 JSON，供上游 safe_json_load 使用
        return "{\n  \"logic\": \"AND\",\n  \"constraints\": []\n}"


def _parse_block_llm_result(raw_str: str) -> Dict[str, Any]:
    """
    将 deepseek 返回的原始字符串解析为结构化 dict：
    {
      "logic": "AND" | "sub-chain",
      "constraints": [ {"desc":..., "verifier":{check,args}}, ... ]
    }

    解析策略：
    1. safe_json_load(raw_str) —— 这会尝试提取最像 JSON 的片段并解析。
    2. 从解析结果里拿到 logic 和 constraints 字段；无则兜底。
    3. constraints 字段再用 extract_constraints() 走一遍，以统一字段名 (cid/desc/verifier_spec)。
       注意：extract_constraints 期望的是一个 JSON list 或者 {constraints:[...]}
       所以我们把 {"constraints": [...]} 丢给它即可。
    """
    try:
        parsed_any = safe_json_load(raw_str)
    except ValueError:
        return {}

    if not isinstance(parsed_any, dict):
        return {}

    logic_val = parsed_any.get("logic", "AND")
    raw_constraints_field = {
        "constraints": parsed_any.get("constraints", [])
    }

    # 使用 extract_constraints() 做统一清洗，得到 list[dict]
    cons_items = extract_constraints(json.dumps(raw_constraints_field, ensure_ascii=False))

    return {
        "logic": logic_val,
        "constraints": cons_items,
    }


def _fallback_block_constraints(block: BlockSpec) -> Dict[str, Any]:
    """
    如果 LLM 失败：
    - 给出两个非常通用但几乎总是合理的局部约束。
    - logic = "AND"。
    返回格式与 _parse_block_llm_result() 对齐：
      {
        "logic": "AND",
        "constraints": [
           {"desc": ..., "verifier_spec": {"check":..., "args":...}}, ...
        ]
      }
    """
    return {
        "logic": "AND",
        "constraints": [
            {
                "desc": "Maintain a neutral, analytical tone without emotional or inflammatory language.",
                "verifier_spec": {
                    "check": "tone_neutral_llm_judge",
                    "args": {},
                },
            },
            {
                "desc": "Provide a sufficiently detailed explanation of the topic with at least 50 words.",
                "verifier_spec": {
                    "check": "min_word_count",
                    "args": {"min_words": 50},
                },
            },
        ],
    }


def extract_block_constraints(segmentation: Dict[str, Any],
                              seed_task: str) -> Dict[str, Any]:
    """
    Step4 主入口：
    输入：
      - segmentation: Step2 的输出 {"blocks": [...], "order": [...]}
      - seed_task: Step1 的输出

    输出：
      {
        "block_constraints": {
            "B1": [ConstraintNode, ...],
            "B2": [...],
            ...
        },
        "block_logic": {
            "B1": "AND" | "sub-chain",
            ...
        }
      }
    """

    block_constraints: Dict[str, List[ConstraintNode]] = {}
    block_logic: Dict[str, str] = {}

    for block_dict in segmentation.get("blocks", []):
        block = BlockSpec(
            block_id=block_dict["block_id"],
            intent=block_dict.get("intent", "TBD"),
            text_span=block_dict.get("text_span", ""),
            order_index=block_dict.get("order_index", 0),
        )

        raw_str = _call_deepseek_block_constraints(block, seed_task, segmentation)
        parsed = _parse_block_llm_result(raw_str)
        if not parsed or "constraints" not in parsed:
            parsed = _fallback_block_constraints(block)

        logic_tag = parsed.get("logic", "AND")
        block_logic[block.block_id] = "sub-chain" if str(logic_tag).lower().startswith("sub") else "AND"

        local_nodes: List[ConstraintNode] = []
        cid_idx = 1
        for item in parsed.get("constraints", []):
            # item 可能是 extract_constraints() 的输出风格：
            # {"cid":..., "desc":..., "verifier_spec":{check,args}, ...}
            desc = item.get("desc", "").strip()
            verifier_spec = item.get("verifier_spec", {}) or item.get("verifier", {}) or {}
            check_name = verifier_spec.get("check")
            args_obj = verifier_spec.get("args", {}) or {}
            if not desc or not check_name:
                continue

            node = ConstraintNode(
                cid=f"{block.block_id}_C{cid_idx}",
                desc=desc,
                scope="local",
                verifier_spec={"check": check_name, "args": args_obj},
                trace_to=block.block_id,
                derived_from="step4",
            )
            local_nodes.append(node)
            cid_idx += 1

        block_constraints[block.block_id] = local_nodes

    return {
        "block_constraints": block_constraints,
        "block_logic": block_logic,
    }


if __name__ == "__main__":
    # quick smoke test
    demo_seg = {
        "blocks": [
            {
                "block_id": "B1",
                "intent": "Opening / Context setup",
                "text_span": "The modern space race signals not only technological ambition but geopolitical signaling...",
                "order_index": 0,
            },
            {
                "block_id": "B2",
                "intent": "Main Analysis / Drivers / Examples",
                "text_span": "For example, recent launch programs by nation X and private company Y illustrate...",
                "order_index": 1,
            },
        ],
        "order": ["B1", "B2"],
    }

    demo_seed = (
        "Analyze the geopolitical implications of the modern space race in a neutral analytical tone, "
        "providing real-world examples and forward-looking assessment."
    )

    out = extract_block_constraints(demo_seg, demo_seed)
    print(json.dumps({
        "block_logic": out["block_logic"],
        "block_constraints": {
            bid: [
                {
                    "cid": n.cid,
                    "desc": n.desc,
                    "scope": n.scope,
                    "verifier_spec": n.verifier_spec,
                    "trace_to": n.trace_to,
                    "derived_from": n.derived_from,
                }
                for n in nlist
            ]
            for bid, nlist in out["block_constraints"].items()
        }
    }, indent=2, ensure_ascii=False))