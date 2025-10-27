"""
utils/parsing.py

统一的解析/健壮化工具：
- safe_json_load(text)
- extract_blocks(llm_raw)
- extract_constraints(llm_raw)

目标：
1. LLM 很多时候不会老老实实返回干净 JSON，而是包了一层解释性自然语言。
2. 我们想在 step2 / step3 / step4 里做统一的“宽松解析”逻辑，而不是每个 step 都手写正则+try/except。
3. 这些函数都只在内存中操作字符串，不做IO。

约定：
- "blocks" 用于回答分块（step2_segmentation 的输出）
    期望结构类似：
    {
      "blocks":[
        {"block_id":"B1","intent":"...","text_span":"...","order_index":0},
        {"block_id":"B2","intent":"...","text_span":"...","order_index":1}
      ],
      "order":["B1","B2",...]
    }

- "constraints" 用于约束抽取（step3_global_constraints / step4_back_translation）
    期望结构类似：
    [
      {"cid":"C1","desc":"...","scope":"global","verifier_spec":{...}},
      {"cid":"C2","desc":"...","scope":"local","verifier_spec":{...}}
    ]

IMPORTANT:
- 这些工具函数并不会“理解语义”或“合成新的内容”，它们只是把 LLM 的原始回答里
  可能出现的 JSON 片段挖出来、解析成 Python 结构，并做一些字段兜底。
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union


# ------------------------------------------------------------
# 基础工具：从一段文本里粗暴提取第一个 JSON 片段并解析
# ------------------------------------------------------------

def _find_json_candidate(text: str) -> Optional[str]:
    """
    从 text 中尽量找到第一个看起来像 JSON 对象/数组的片段，返回该片段字符串。
    策略：
      1. 优先找以 { 开头、匹配到平衡的 } 的片段
      2. 不行再找以 [ 开头、匹配到平衡的 ] 的片段
    这是一个近似方法，不保证在严重畸形输出下成功。
    """
    if not text:
        return None

    # 尝试找对象 {...}
    obj_match = _extract_balanced_braces(text, opener="{", closer="}")
    if obj_match is not None:
        return obj_match.strip()

    # 尝试找数组 [...]
    arr_match = _extract_balanced_braces(text, opener="[", closer="]")
    if arr_match is not None:
        return arr_match.strip()

    return None


def _extract_balanced_braces(text: str, opener: str, closer: str) -> Optional[str]:
    """
    给定 opener='{', closer='}', 或 opener='[', closer=']',
    扫描 text，找到第一个 opener 并尝试向后走到平衡的 closer。
    如果成功，返回包含整个平衡块的子串。
    如果失败，返回 None。
    """
    start_idx = text.find(opener)
    if start_idx == -1:
        return None

    depth = 0
    for i in range(start_idx, len(text)):
        ch = text[i]
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                # i 是匹配到的最后一个 closer
                return text[start_idx:i+1]
    return None


def safe_json_load(text: str) -> Union[Dict[str, Any], List[Any]]:
    """
    最关键的基础函数：
    - 给一段 LLM 的原始输出（可能含有自然语言+解释+JSON示例）
    - 返回解析后的 Python 对象（dict 或 list）
    - 如果彻底无法解析，就抛 ValueError

    逻辑：
      1. 先尝试直接 json.loads
      2. 如果失败，尝试 _find_json_candidate 抽子串再 json.loads
      3. 如果还失败，尝试做一些常见修补（尾随逗号、单引号 -> 双引号 等）
         * 注意：我们做的是尽量安全的最小修补，不去“脑补”缺字段
    """
    text = text.strip()
    if not text:
        raise ValueError("safe_json_load: empty input")

    # 直接尝试
    try:
        return json.loads(text)
    except Exception:
        pass

    # 抽取 JSON candidate
    candidate = _find_json_candidate(text)
    if candidate:
        try:
            return json.loads(candidate)
        except Exception:
            # 尝试轻度修补
            fixed = _light_json_fix(candidate)
            try:
                return json.loads(fixed)
            except Exception:
                raise ValueError(
                    "safe_json_load: cannot parse after candidate extraction+light fix"
                )
    else:
        # 没找到任何疑似JSON
        raise ValueError("safe_json_load: no JSON-looking segment found")


def _light_json_fix(s: str) -> str:
    """
    对 candidate 做一些保守修补，常见在大模型输出里出现的错误：
      - 把单引号包裹的 dict/list 尝试替换成双引号（非常粗糙）
      - 去掉末尾多余的逗号
    这不是完美JSON修复器，只是提高鲁棒性，让解析率更高。
    """
    # 粗暴方案1：如果外层是用单引号的 dict/list，替换成双引号
    # 例如: {'a': 'b'} -> {"a": "b"}
    # 用正则把 `'key'` -> "key"
    # 以及把 `': '` 之类保持间距
    fixed = s

    # 把 Python风格的单引号key/value 换成双引号（非常粗略，但能救不少LLM输出）
    fixed = re.sub(r"\'([^']*)\'", r'"\1"', fixed)

    # 去掉可能的末尾逗号，比如 {"a":1,} -> {"a":1}
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)

    return fixed


# ------------------------------------------------------------
# block 解析
# ------------------------------------------------------------

def extract_blocks(llm_raw: str) -> Dict[str, Any]:
    """
    用于 step2_segmentation 的 post-processing。

    输入：
        llm_raw: LLM的原始输出，期望其包含 JSON，形如：
        {
          "blocks":[
            {"block_id":"B1","intent":"Opening","text_span":"...","order_index":0},
            {"block_id":"B2","intent":"Analysis",...}
          ],
          "order":["B1","B2"]
        }

    返回：
        {
          "blocks":[{block_id,...},...],
          "order":[...]
        }
    如果字段缺失，会补上空列表/空对象结构，以尽量不中断后续流程。
    """
    try:
        parsed = safe_json_load(llm_raw)
    except ValueError:
        # 如果实在解析不了，就给一个兜底空结构
        return {"blocks": [], "order": []}

    if not isinstance(parsed, dict):
        # 有些LLM可能直接给 list blocks -> 我们包一层
        if isinstance(parsed, list):
            # 猜测是 blocks 列表
            return {"blocks": parsed, "order": [b.get("block_id") for b in parsed if isinstance(b, dict)]}
        return {"blocks": [], "order": []}

    blocks = parsed.get("blocks", [])
    order = parsed.get("order", [])

    # 兜底类型修正
    if not isinstance(blocks, list):
        blocks = []
    if not isinstance(order, list):
        # 自动从 blocks 抽一次
        order = [b.get("block_id") for b in blocks if isinstance(b, dict)]

    # 补 order_index，如果缺了
    for idx, b in enumerate(blocks):
        if isinstance(b, dict):
            if "order_index" not in b:
                b["order_index"] = idx

    return {
        "blocks": blocks,
        "order": order,
    }


# ------------------------------------------------------------
# constraint 解析
# ------------------------------------------------------------

def extract_constraints(llm_raw: str) -> List[Dict[str, Any]]:
    """
    用于 step3_global_constraints / step4_back_translation 的 post-processing。

    输入：
        llm_raw: LLM的原始输出，期望其包含数组或对象中含有约束项。例如：
        [
          {
            "cid":"G1",
            "desc":"Must be written in English.",
            "scope":"global",
            "verifier_spec":{"check":"require_language","args":{"lang":"en"}}
          },
          {
            "cid":"G2",
            "desc":"Include background, analysis, and conclusion.",
            "scope":"global.structure",
            "verifier_spec":{"check":"require_sections","args":{"sections":["Opening","Body","Conclusion"]}}
          }
        ]

    返回：
        一个 list[dict]，每个元素至少包含:
        {
          "cid": <str or auto-generated>,
          "desc": <str>,
          "scope": <str or "local">,
          "verifier_spec": <dict or {}>,
        }

    如果解析失败，返回空列表。
    """
    try:
        parsed = safe_json_load(llm_raw)
    except ValueError:
        return []

    # 允许 parsed 是 {"constraints":[...]} 或 直接 [...]
    if isinstance(parsed, dict):
        if "constraints" in parsed and isinstance(parsed["constraints"], list):
            items = parsed["constraints"]
        # 有些LLM可能命成 "global_constraints" / "block_constraints"
        elif "global_constraints" in parsed and isinstance(parsed["global_constraints"], list):
            items = parsed["global_constraints"]
        elif "block_constraints" in parsed and isinstance(parsed["block_constraints"], list):
            items = parsed["block_constraints"]
        else:
            # 如果是 dict 但是不是我们认识的键，那就尝试把它当成单条约束包起来
            items = [parsed]
    elif isinstance(parsed, list):
        items = parsed
    else:
        return []

    cleaned: List[Dict[str, Any]] = []
    auto_idx = 0

    for it in items:
        if not isinstance(it, dict):
            continue

        cid = it.get("cid")
        if not cid:
            cid = f"AUTO_C{auto_idx}"
            auto_idx += 1

        desc = it.get("desc") or it.get("description") or ""
        scope = it.get("scope", "local")
        verifier_spec = it.get("verifier_spec", {}) or it.get("verifier", {}) or {}

        cleaned.append({
            "cid": cid,
            "desc": desc,
            "scope": scope,
            "verifier_spec": verifier_spec,
            # 如果后续需要，还可以把原始字段也原样带上，例如：
            # "raw": it
        })

    return cleaned