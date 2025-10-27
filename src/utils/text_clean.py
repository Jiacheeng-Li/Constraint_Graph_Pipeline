"""
utils/text_clean.py

该模块提供文本清理、截断与结构性摘要（outline）工具，供 step2/3/4 的 prompt 构造阶段复用。

本版设计目标（新版非常重要的约束）：
1. 不擅自删改语义：LLM 在做 back-translation / 约束抽取时，必须看到真实文本证据，
   否则它会“脑补应该做的要求”，而不是“当前文本真实做了什么”。
2. 默认情况下，我们只做空白规范（whitespace normalization），不做内容删减。
3. 只有当调用方 *主动* 调用 clip(...) 时，才会截断文本长度。这是一个显式的“风险自担”操作，
   用于极端长文本避免 token 爆炸；不是默认行为。

函数一览：
- normalize_whitespace(text):
    轻量规范空白：统一换行、压缩过度空格/空行。不会删句子、不会摘要。

- clip(text, max_chars):
    硬截断到指定字符上限。⚠ 这是唯一会丢信息的函数。
    只有上游显式、主动调用时才会触发。正常路径不要自动用它。

- make_snippet(text):
    生成可以直接塞进 prompt 里的正文片段，但不截断、不总结。
    它只是调用 normalize_whitespace，坚持“保留原句原措辞”。

- summarize_blocks_outline(segmentation, ...):
    只给出 block 的顺序和 intent，帮助 LLM理解整体结构位置。
    这个 outline 是结构性辅助，不是证据文本。
"""

from typing import Dict, Any, List
import re


def normalize_whitespace(text: str) -> str:
    """
    轻量规范空白：
    - 统一换行符为 \n
    - 把制表符 \t 换成单个空格
    - 把单行内的长空格串(>=3个空格)压成单个空格
    - 把连续 >=3 行空行压成 2 行，避免 prompt 中出现巨大片空白

    非常重要：
    - 我们不会删除整句内容
    - 我们不会重写措辞
    - 我们不会进行语义总结

    也就是说，返回值依然是“原文”，只是去除了视觉噪音式的空白。
    """
    if not text:
        return ""

    # 统一换行符
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    # 制表符 -> 空格
    t = t.replace("\t", " ")

    # 压缩一行内过长的多空格
    t = re.sub(r"[ ]{3,}", " ", t)

    # 压缩超长空行（3+ 连续换行 -> 2个换行）
    t = re.sub(r"\n{3,}", "\n\n", t)

    # 去掉首尾的纯空白
    return t.strip()


def clip(text: str, max_chars: int) -> str:
    """
    显式截断：将文本裁剪到不超过 max_chars 个字符。
    ⚠ 这是唯一会主动丢信息的函数。

    使用场景：
    - 只有当文本极端长（比如几十KB）并且会直接导致 prompt 超 Token，我们才会调用它。
    - 正常 step2 / step3 / step4 的流程中，不应该默认 clip。

    截断策略：
    - 硬截断（text[:max_chars]）
    - 不做“智能句边界截断”，不做意译，不做补写 "...[TRUNCATED]"。
      原因：我们不想引入任何我们自己的总结性语言。
    """
    if not text:
        return ""
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def make_snippet(text: str) -> str:
    """
    返回可以安全塞进 LLM prompt 的正文片段，但不做删减。

    行为：
    - 调用 normalize_whitespace() 清理空白和极端不可读的缩进/多余空行
    - 不截断！不截断！不截断！
    - 不添加 "...[TRUNCATED]" 之类的人造标记
    - 不总结、不改写、不重排句子

    换句话说：
    这个函数的输出仍然是原文语义，只是更干净，方便放进 prompt。
    """
    return normalize_whitespace(text)


def summarize_blocks_outline(
    segmentation: Dict[str, Any],
    max_blocks: int = 12,
    max_intent_chars: int = 200,
) -> str:
    """
    根据 step2 的 segmentation 结果，生成一段轻量 block outline。

    segmentation 的典型结构：
    {
      "blocks": [
        {"block_id": "B1", "intent": "Opening / geopolitical framing ...", "text_span": "...", "order_index": 0},
        {"block_id": "B2", "intent": "Historical background ...", ...},
        ...
      ],
      "order": ["B1","B2",...]
    }

    我们输出一段结构性说明，用来告诉 LLM：
        B1 (Opening / geopolitical framing ...)
        B2 (Historical background ...)
        B3 (Risk assessment / recommendations ...)
        ...

    注意：
    - 这个 outline 是“结构性提示”，不是证据文本本身。
    - 我们仍然会对 intent 做 very light 清洗（normalize_whitespace）
    - 如果 intent 太长，只做硬字符截断（[:max_intent_chars] + " ...")，
      这是提示性摘要，但它不是强约束来源。
    - 后续在 prompt 里我们要明确告诉模型：
        你只能根据后面给的 TEXT SNIPPET 本身来推断约束；
        不要凭 outline 去脑补不存在的要求。
    """
    blocks = segmentation.get("blocks", []) or []

    # 确保是按回答原顺序
    blocks_sorted = sorted(
        [b for b in blocks if isinstance(b, dict)],
        key=lambda x: x.get("order_index", 0),
    )[:max_blocks]

    lines: List[str] = []
    for b in blocks_sorted:
        bid = b.get("block_id", "?")
        intent = b.get("intent", "?") or "?"

        # intent 清洗空白
        cleaned_intent = normalize_whitespace(intent)

        # intent 若非常长，只做硬截断（避免 prompt 撑爆）
        if len(cleaned_intent) > max_intent_chars:
            cleaned_intent = cleaned_intent[:max_intent_chars].rstrip() + " ..."

        lines.append(f"{bid} ({cleaned_intent})")

    return "\n".join(lines)