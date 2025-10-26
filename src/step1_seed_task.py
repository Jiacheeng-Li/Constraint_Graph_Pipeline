"""
step1_seed_task.py

Step 1: 种子任务抽取 (Seed Task Extraction)

当前版本（vLLM-full-context）：使用 deepseek 直接读取整份指令文本，
识别“用户最终想要的主要交付物”，并将其压缩成单句英文祈使式任务描述。

为什么要这么做：
- 真实场景里，用户经常先给背景、限制、口径，然后在后面才说真正的任务；
  如果我们只截第一句，很容易拿到的是背景而不是任务。
- seed_task 作为整张 ConstraintGraph 的根节点，必须对应“我要你最终产出什么”。

策略：
1. 把完整的 instruction_text（未经截断）喂给 LLM。
2. 让 LLM 输出 exactly ONE imperative English sentence，描述最终交付义务。
   - 必须以动词开头 (Analyze / Summarize / Write / Draft / Explain / etc.)
   - 只能一句话，不要客套，不要多余上下文。
   - 如果用户要求了特定语气/风格/视角（"neutral tone", "formal", "first-person"），要把这些也放进句子里。
3. 如果 deepseek 挂了或输出不合法，则 fallback：
   - 用启发式清洗+前置 "Analyze ..." 生成一个兜底 seed_task。

返回：string seed_task
"""

from typing import Optional
import re
import requests
import json

_DEEPSEEK_API_KEY_DEFAULT = "sk-4bb3e24d26674a30b2cc7e2ff1bfc763"
_DEEPSEEK_ENDPOINT = "https://api.deepseek.com/v1/chat/completions"
_DEEPSEEK_MODEL = "deepseek-chat"


# ---------- 基础清洗（仅用于 fallback） ----------

def _clean_polite_prefix(text: str) -> str:
    """
    去掉"请你…/我想让你…/能否…"等礼貌性开头；英文里去掉"please can you"等。
    仅用于 fallback 情况。
    """
    t = text.strip()

    polite_zh = [
        r"^请你(帮我)?", r"^请帮我", r"^请为我", r"^麻烦你", r"^我想让你", r"^能否帮我", r"^可以帮我",
    ]
    for pat in polite_zh:
        t = re.sub(pat, "", t)

    polite_en = [
        r"^please\s+", r"^could\s+you\s+", r"^can\s+you\s+", r"^would\s+you\s+",
        r"^i\s+would\s+like\s+you\s+to\s+", r"^i\s+want\s+you\s+to\s+",
        r"^help\s+me\s+",
    ]
    for pat in polite_en:
        t = re.sub(pat, "", t, flags=re.IGNORECASE)

    return t.strip()


def _first_sentence(text: str) -> str:
    """
    fallback时使用：尝试抓instruction_text的第一句/第一段，作为兜底素材。
    我们仍然保留这个步骤，但它不再是主路径（主路径是LLM看全文）。
    """
    if not text:
        return ""
    para = text.strip().split("\n\n")[0].strip()
    m = re.split(r"(?<=[。！？!?\.])\s+", para)
    if m:
        return m[0].strip()
    return para


def _fallback_imperative_form(text: str) -> str:
    """
    把兜底文本粗暴转成祈使式指令；如果不是英文，就简单地前置"Analyze"。
    """
    t = text.strip()
    if not t:
        return "Analyze the given topic."

    # 尝试去掉中文/英文里“我想让你… / I want you to …”模式
    t = re.sub(r"^(我|我想|我需要|我希望|我希望你|我想让你|我想要你|我想请你)[^,，。:.：]*", "", t)
    t = re.sub(r"^(i\s+need\s+you\s+to|i\s+want\s+you\s+to|i\s+would\s+like\s+you\s+to)\s+",
                "",
                t,
                flags=re.IGNORECASE)

    # 如果是英文动词开头（Explain/Analyze/...），直接返回
    words = t.split()
    if words:
        first_word = words[0]
        if re.match(r"^[A-Za-z]+$", first_word):
            return t

    # 否则简单地加 Analyze 前缀
    return f"Analyze {t}" if t else "Analyze the given topic."


# ---------- deepseek 调用：主路径 ----------

def _call_deepseek_seed_task_full_context(instruction_text: str,
                                          api_key: Optional[str] = None,
                                          endpoint: Optional[str] = None,
                                          model: Optional[str] = None) -> str:
    """
    主路径：把整份 instruction_text 给 deepseek，请它抽取唯一主交付物。

    我们强制它输出：
    - 仅一条句子
    - 英文
    - 祈使式（动词开头）

    同时告诉它：如果有多个子步骤，只保留最终产出任务，不要罗列流程。
    """
    system_prompt = (
        "You are a task distiller. "
        "Your job is to read the entire user request and identify the SINGLE primary deliverable "
        "the assistant is being asked to produce. "
        "Ignore background context, motivation, disclaimers, policy notes, or tooling setup. "
        "Focus ONLY on the final product the assistant should create.\n\n"
        "Your output requirements:\n"
        "1. Output EXACTLY ONE sentence in English.\n"
        "2. The sentence MUST be imperative and MUST start with a strong verb like Analyze / Summarize / Write / Draft / Explain / Propose / Generate.\n"
        "3. Include key style constraints if they are explicitly required (e.g. 'neutral tone', 'formal style', 'first-person').\n"
        "4. DO NOT include politeness or meta-instructions like 'please'.\n"
        "5. DO NOT include multiple steps; choose the final deliverable, not setup steps.\n"
        "6. Return ONLY that one sentence, with no extra words."
    )

    user_prompt = (
        "USER REQUEST (full context):\n" + instruction_text.strip() + "\n\n"
        "Now output the single distilled task sentence."
    )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key or _DEEPSEEK_API_KEY_DEFAULT}",
    }

    payload = {
        "model": model or _DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 64,
    }

    try:
        resp = requests.post(endpoint or _DEEPSEEK_ENDPOINT,
                             headers=headers,
                             data=json.dumps(payload),
                             timeout=10)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
        # 取第一行，确保只保留一行
        first_line = content.splitlines()[0].strip()
        return first_line
    except Exception:
        return ""


# ---------- 对外主入口 ----------

def extract_seed_task(instruction_text: str) -> str:
    """
    Step1 (LLM full context version):

    1. 直接把完整 instruction_text 丢给 deepseek，要求它总结“最终产出任务”成单句英文祈使式。\n

    2. 如果 deepseek 正常返回：用它的输出作为 seed_task。\n

    3. 如果 deepseek 返回空 / 调用失败：
       - 我们 fallback 到启发式（抓第一句，清礼貌前缀，加 'Analyze ...'）。

    返回值：seed_task (str)
    """
    # 主路径：LLM基于全context总结
    distilled = _call_deepseek_seed_task_full_context(instruction_text)
    if distilled:
        return distilled.strip()

    # 兜底策略：启发式处理首句
    first_line = _first_sentence(instruction_text)
    cleaned_core = _clean_polite_prefix(first_line)
    fallback_task = _fallback_imperative_form(cleaned_core)
    return fallback_task.strip()


if __name__ == "__main__":
    demo = (
        "我们现在要准备一份给外部合作伙伴的说明文档，背景涉及到我们最近的卫星监管策略变化。"
        "下面我会给你一些内部观点和潜在冲突点。"
        "最终我想让你做的是：请用中立、正式、无攻击性的口吻，起草一份对外声明，解释这些变动的原因和影响。"
    )
    print(extract_seed_task(demo))