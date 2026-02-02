"""
Step 2 - Response Segmentation

Purpose / 目标
- Break a high-quality reference answer into ordered blocks so later stages can reason about local duties.
- Capture each block's semantic intent (Opening, Background, Analysis, Recommendation, etc.) to drive structure constraints and traceability.

Workflow
1. Feed the full answer into the LLM with a strict JSON schema request.
2. Ask for sequential blocks labelled with block_id, intent, text_span, and order_index.
3. Parse the response with utils.parsing.extract_blocks so we survive minor JSON mistakes.
4. If the LLM output is empty or invalid after several attempts, fall back to deterministic paragraph splitting that preserves bullet runs.

Inputs / 输入
- response_text: the exemplar answer we are reverse engineering.

Outputs / 输出
- dict with `blocks` (list of block specs) and `order` (array of block_ids).

Why this matters
- Each block feeds Step 4 back-translation, Step 5 selection planning, and Step 6 chain ordering.
- Intent labels allow Step 3 to infer structural global constraints such as Intro/Body/Conclusion requirements.

Dependencies
- utils.deepseek_client.call_chat_completions for the primary segmentation.
- utils.parsing.extract_blocks for resilient parsing and `_split_paragraphs_with_bullet_groups` for fallback.
"""

import json
import re
import time
from typing import Dict, Any

from .utils.deepseek_client import call_chat_completions, DeepSeekError
from .utils.parsing import extract_blocks

# Deprecated per-unified client configuration (kept for minimal diff)
_DEEPSEEK_API_KEY_DEFAULT = ""
_DEEPSEEK_ENDPOINT = ""
_DEEPSEEK_MODEL = ""


def _call_deepseek_segmentation(response_text: str) -> str:
    """
    调用 LLM，请它把回答分块并标注 intent。

    返回：LLM 原始输出字符串（可能是 JSON，也可能是包含解释+JSON）。
    解析工作由上层 extract_blocks() 统一处理。
    """

    system_prompt = (
        "You are a response segmenter. "
        "Your job is to split the given answer text into logical blocks that each serve a specific purpose. "
        "Each block should represent a coherent paragraph or reasoning unit. "
        "For each block, identify its semantic intent "
        "(e.g., Opening, Context setup, Background, Main Analysis, Counterargument, Recommendation, Conclusion, Outlook). "
        "Output JSON ONLY in the format below:\n\n"
        "{\n"
        "  \"blocks\": [\n"
        "    {\"block_id\": \"B1\", \"intent\": \"...\", \"text_span\": \"...\"},\n"
        "    {\"block_id\": \"B2\", \"intent\": \"...\", \"text_span\": \"...\"}\n"
        "  ],\n"
        "  \"order\": [\"B1\", \"B2\", ...]\n"
        "}\n\n"
        "Rules:\n"
        "- Preserve the original text order.\n"
        "- Keep each block text_span concise (50-200 words typically).\n"
        "- Ensure each block intent label is unique; do not reuse the same intent string.\n"
        "- Do not include explanations outside JSON.\n"
    )

    user_prompt = (
        "Split the following answer into logical blocks with intent labels. "
        "Return ONLY valid JSON, no explanations.\n\n"
        f"ANSWER TEXT:\n{response_text.strip()}"
    )

    try:
        content = call_chat_completions(
            messages=[
                {"role": "user", "content": user_prompt},
            ],
            system_prompt=system_prompt,
            temperature=0.2,
            max_tokens=4096,
            timeout=180,
            retries=4,
            retry_backoff_sec=1.2,
        ).strip()
        return content
    except DeepSeekError:
        # 上层会根据 "{}" 得知本次调用失败
        return "{}"


_BULLET_LINE_PATTERN = re.compile(r"^\s*(?:[-*•]+|\d+[\.\)]|[A-Za-z][\.\)])\s+")


def _split_paragraphs_with_bullet_groups(response_text: str) -> list[str]:
    """
    Split text into paragraphs while merging consecutive bullet lines (and the blank
    lines between them) into a single paragraph so that list continuity is preserved.
    """
    paragraphs: list[str] = []
    current_lines: list[str] = []
    bullet_mode = False

    def flush() -> None:
        nonlocal current_lines, bullet_mode
        chunk = "\n".join(current_lines).strip()
        if chunk:
            paragraphs.append(chunk)
        current_lines = []
        bullet_mode = False

    for raw_line in response_text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            # 空行逻辑：列表模式下保留空行，否则结束当前段
            if bullet_mode:
                current_lines.append("")
            else:
                flush()
            continue

        is_bullet_line = bool(_BULLET_LINE_PATTERN.match(line))
        if is_bullet_line:
            if not bullet_mode and current_lines:
                flush()
            bullet_mode = True
            current_lines.append(line)
            continue

        if bullet_mode:
            flush()
        current_lines.append(line)

    flush()
    return paragraphs


def segment_response(response_text: str) -> Dict[str, Any]:
    """
    Step2 主函数：
    输入：完整回答文本
    输出：包含 blocks 列表和顺序的字典

    逻辑：
    1. 多次调用 LLM，尝试得到合法的分块 JSON。
    2. 若多次尝试后仍然没有 blocks，则使用本地段落切分兜底。
    3. 最后补全缺失的 block_id / order_index / order。
    """

    # 1) 先尝试通过 LLM + extract_blocks 获得分块
    max_attempts = 3
    parsed: Dict[str, Any] = {}
    last_error: str | None = None

    for attempt in range(1, max_attempts + 1):
        raw_llm = _call_deepseek_segmentation(response_text)
        parsed = extract_blocks(raw_llm)
        if parsed.get("blocks"):
            break

        last_error = f"[Step2] attempt {attempt}/{max_attempts} produced no blocks."
        print(last_error)

        if attempt < max_attempts:
            # 简单线性退避，避免打爆后端
            time.sleep(1.0 * attempt)
    else:
        # 2) LLM 多次失败：使用本地段落切分兜底
        paragraphs = _split_paragraphs_with_bullet_groups(response_text)
        if paragraphs:
            blocks = []
            order = []
            for i, para in enumerate(paragraphs, start=1):
                block_id = f"B{i}"
                blocks.append(
                    {
                        "block_id": block_id,
                        "intent": f"Paragraph {i}",
                        "text_span": para,
                        "order_index": i - 1,
                    }
                )
                order.append(block_id)
            parsed = {"blocks": blocks, "order": order}
        else:
            # 输入本身为空或只有空白，确实没法分块
            msg = last_error or "[Step2] segmentation produced no blocks; raising to mark failure."
            raise DeepSeekError(msg)

    # 3) 补全 block_id / order_index / order
    blocks = parsed.get("blocks", [])
    for idx, blk in enumerate(blocks):
        if "block_id" not in blk or not blk["block_id"]:
            blk["block_id"] = f"B{idx+1}"
        if "order_index" not in blk:
            blk["order_index"] = idx

    if "order" not in parsed or not parsed["order"]:
        parsed["order"] = [b["block_id"] for b in blocks]

    # 4) Ensure intent labels are unique to avoid merging stages downstream.
    used_intents: Dict[str, int] = {}
    for idx, blk in enumerate(blocks, start=1):
        intent = (blk.get("intent") or "").strip() or f"Stage {idx}"
        if intent not in used_intents:
            used_intents[intent] = 1
            blk["intent"] = intent
            continue
        used_intents[intent] += 1
        blk["intent"] = f"{intent} ({used_intents[intent]})"

    return parsed


if __name__ == "__main__":
    demo_text = (
        "The modern space race represents not just a technological competition but a geopolitical one. "
        "In the opening decades of the 21st century, new players such as China and private companies entered the field. "
        "\n\n"
        "This essay first reviews the historical roots of space exploration, then evaluates how current missions "
        "affect global cooperation and rivalry. "
        "\n\n"
        "In conclusion, understanding the space race helps us grasp how nations project power and inspire innovation."
    )

    result = segment_response(demo_text)
    print(json.dumps(result, indent=2, ensure_ascii=False))
