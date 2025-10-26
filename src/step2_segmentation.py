

"""
step2_segmentation.py

Step 2: 回答分块 (Response Segmentation)

目标：
将模型的完整回答文本切分为若干语义逻辑块（Block），
并识别每块的语义功能(intent)，如：
  - Opening / Introduction
  - Background / Context
  - Main Analysis / Discussion
  - Conclusion / Summary / Outlook
  - Recommendation / Suggestion

为什么要分块：
- 每个块会在 Step 4 中单独做 back-translation，抽出局部约束；
- 块的 intent 能帮助后续自动生成结构约束（如 “必须包含开头、主体、结论”）。

本版本 (vLLM):
使用 DeepSeek LLM 读取整段回答文本，
要求它：
  1. 按语义自然边界切块；
  2. 给出每块的 intent 标签；
  3. 输出 JSON 格式。

输出格式：
{
  "blocks": [
    {"block_id": "B1", "intent": "Opening / Context setup", "text_span": "..."},
    {"block_id": "B2", "intent": "Main Analysis", "text_span": "..."},
    {"block_id": "B3", "intent": "Conclusion / Outlook", "text_span": "..."}
  ],
  "order": ["B1", "B2", "B3"]
}

依赖：
DeepSeek API（deepseek-chat）
"""

import json
import requests
from typing import Dict, Any

_DEEPSEEK_API_KEY_DEFAULT = "sk-4bb3e24d26674a30b2cc7e2ff1bfc763"
_DEEPSEEK_ENDPOINT = "https://api.deepseek.com/v1/chat/completions"
_DEEPSEEK_MODEL = "deepseek-chat"


def _call_deepseek_segmentation(response_text: str) -> Dict[str, Any]:
    """
    调用 DeepSeek 模型，请它把回答分块并标注 intent。
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
        "- Keep each block text_span concise (50–200 words typically).\n"
        "- Do not include explanations outside JSON.\n"
    )

    user_prompt = (
        "Split the following answer into logical blocks with intent labels. "
        "Return ONLY valid JSON, no explanations.\n\n"
        f"ANSWER TEXT:\n{response_text.strip()}"
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

        # 提取 JSON
        start = content.find("{")
        end = content.rfind("}") + 1
        json_str = content[start:end]
        parsed = json.loads(json_str)

        # 补充编号
        for i, blk in enumerate(parsed.get("blocks", []), start=1):
            blk["block_id"] = blk.get("block_id", f"B{i}")

        parsed["order"] = [b["block_id"] for b in parsed.get("blocks", [])]
        return parsed

    except Exception:
        # fallback: 简单按段落切
        paras = [p.strip() for p in response_text.split("\n\n") if p.strip()]
        blocks = []
        for i, p in enumerate(paras, start=1):
            blocks.append({"block_id": f"B{i}", "intent": "TBD", "text_span": p})
        return {"blocks": blocks, "order": [b["block_id"] for b in blocks]}


def segment_response(response_text: str) -> Dict[str, Any]:
    """
    Step2 主函数：
    输入：完整回答文本
    输出：包含 blocks 列表和顺序的字典
    """
    result = _call_deepseek_segmentation(response_text)
    return result


if __name__ == "__main__":
    demo_text = (
        "The modern space race represents not just a technological competition but a geopolitical one. "
        "In the opening decades of the 21st century, new players such as China and private companies entered the field. "
        "\\n\\n"
        "This essay first reviews the historical roots of space exploration, then evaluates how current missions "
        "affect global cooperation and rivalry. "
        "\\n\\n"
        "In conclusion, understanding the space race helps us grasp how nations project power and inspire innovation."
    )

    print(json.dumps(segment_response(demo_text), indent=2, ensure_ascii=False))