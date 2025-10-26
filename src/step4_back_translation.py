

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
from typing import Dict, Any, List, Tuple
from .graph_schema import ConstraintNode, BlockSpec

_DEEPSEEK_API_KEY_DEFAULT = "sk-4bb3e24d26674a30b2cc7e2ff1bfc763"
_DEEPSEEK_ENDPOINT = "https://api.deepseek.com/v1/chat/completions"
_DEEPSEEK_MODEL = "deepseek-chat"


def _call_deepseek_block_constraints(block: BlockSpec,
                                     seed_task: str) -> Dict[str, Any]:
    """
    让 deepseek 针对单个 block 生成：
    - block_constraints: 该 block 实际在做/必须做的要求列表
    - block_logic: "AND" or "sub-chain"

    期望 deepseek 输出 JSON：
    {
      "logic": "AND" | "sub-chain",
      "constraints": [
        {
          "desc": "Explain historical background of the modern space race in neutral tone.",
          "verifier": {
            "check": "tone_neutral_llm_judge",
            "args": {}
          }
        },
        {
          "desc": "Provide at least two concrete real-world examples.",
          "verifier": {
            "check": "must_list_n_subpoints",
            "args": {"n": 2}
          }
        }
      ]
    }
    """

    system_prompt = (
        "You are an instruction reverse-engineer.\n"
        "You will receive: (1) the overall task the assistant is solving, and (2) ONE block of the assistant's answer.\n"
        "Your job is to infer what local requirements this block is satisfying.\n\n"
        "For that single block, output ONLY valid JSON with this structure:\n"
        "{\n"
        "  \"logic\": \"AND\" or \"sub-chain\",\n"
        "  \"constraints\": [\n"
        "    {\n"
        "      \"desc\": \"<what this block MUST accomplish>\",\n"
        "      \"verifier\": {\n"
        "         \"check\": \"<verifier function name>\",\n"
        "         \"args\": { <key-value args, can be empty> }\n"
        "      }\n"
        "    }, ...\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- 'logic' should be 'AND' if the block is basically a list of parallel obligations or facts.\n"
        "- 'logic' should be 'sub-chain' if the block clearly performs a multi-step progression (e.g. define -> compare -> conclude).\n"
        "- Each constraint MUST be checkable by one verifier. Use ONLY these verifier names when possible:\n"
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
        "- Write desc in English, imperative style, concrete and verifiable.\n"
        "- Do NOT mention block ids in desc.\n"
        "- Do NOT output any explanation outside JSON.\n"
    )

    user_prompt = (
        "OVERALL TASK (seed_task):\n" + seed_task.strip() + "\n\n"
        "CURRENT BLOCK (the assistant's answer chunk):\n"
        f"INTENT: {block.intent}\n"
        f"TEXT:\n{block.text_span}\n\n"
        "Infer what this block is REQUIRED to do. Output ONLY the JSON."
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

        # 取 JSON 体
        start = content.find("{")
        end = content.rfind("}") + 1
        json_str = content[start:end]
        parsed = json.loads(json_str)
        return parsed
    except Exception:
        return {}


def _fallback_block_constraints(block: BlockSpec) -> Dict[str, Any]:
    """
    如果 LLM 失败：
    - 给出两个非常通用但几乎总是合理的局部约束。
    - logic 视为 AND。
    这些是低风险的，因为：
    - 大部分 informative 段落都应该是中立分析或信息表达，而不是煽动。
    - 大部分段落都有一定长度（>50词）作为最低要求，可以衡量稀薄回答。
    """
    return {
        "logic": "AND",
        "constraints": [
            {
                "desc": "Maintain a neutral, analytical tone without emotional or inflammatory language.",
                "verifier": {"check": "tone_neutral_llm_judge", "args": {}},
            },
            {
                "desc": "Provide a sufficiently detailed explanation of the topic with at least 50 words.",
                "verifier": {"check": "min_word_count", "args": {"min_words": 50}},
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

        result = _call_deepseek_block_constraints(block, seed_task)
        if not result or "constraints" not in result:
            result = _fallback_block_constraints(block)

        logic_tag = result.get("logic", "AND")
        block_logic[block.block_id] = "sub-chain" if logic_tag.lower().startswith("sub") else "AND"

        # 把每条约束变成 ConstraintNode
        local_nodes: List[ConstraintNode] = []
        cid_idx = 1
        for item in result.get("constraints", []):
            desc = item.get("desc", "").strip()
            verif = item.get("verifier", {})
            check_name = verif.get("check")
            args_obj = verif.get("args", {}) or {}
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