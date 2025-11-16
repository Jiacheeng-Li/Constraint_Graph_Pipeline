"""
soft_checks.py

软性（语气 / 立场 / 态度 / 角色一致性等）约束的校验函数集合。

和硬性约束（长度、关键词、语言等）不同，软性约束往往需要理解语气、立场、是否表现愤怒/批评、是否保持客观中立等语用属性。
这些属性很难用纯正则稳定判断，所以我们在这里**直接调用 LLM（deepseek）进行判定**，并把 deepseek 的回答转成布尔值。

使用约定：
- 所有校验函数都统一为:
      def fn(text: str, **kwargs) -> bool

- 每个函数都会构造一个分类 prompt，并调用 `_call_deepseek_classifier`，让 LLM 产出一个极简标签。
  我们会严格要求 deepseek 只输出可解析标签，如 YES/NO、NEUTRAL/NOT_NEUTRAL 等。

- 这些函数会在 `verifier_registry` 中注册后被自动调用，方式为：
      run_verifier({"check": "tone_neutral_llm_judge", "args": {}}, text)

安全和隐私说明：
- 这里我们内置了 deepseek key（你提供的 key），并默认直连 deepseek API。
- 这一实现假设运行环境允许对外 HTTP 请求；如果在离线/沙箱环境中运行将会失败。
  后续如果需要离线 fallback，可以再加启发式分支。
"""

from typing import Optional, Any, Dict
import json
import re
from ..utils.deepseek_client import call_chat_completions, DeepSeekError

# -----------------------
# DeepSeek 接口配置
# -----------------------

_DEEPSEEK_API_KEY_DEFAULT = ""
_DEEPSEEK_ENDPOINT = ""
_DEEPSEEK_MODEL = ""  # centralized via utils.deepseek_client


def _call_deepseek_classifier(system_prompt: str,
                               user_prompt: str,
                               api_key: Optional[str] = None,
                               endpoint: Optional[str] = None,
                               model: Optional[str] = None) -> str:
    """
    调用 deepseek 模型进行分类推断。

    输入：
    - system_prompt: 作为 system role 的约束，要求输出标签格式。
    - user_prompt:   作为 user role 的文本，通常包含"请判断这段文本..."和正文。

    返回：
    - 一个字符串标签（例如 "NEUTRAL" / "NOT_NEUTRAL" / "YES" / "NO" / "MATCH" / "MISMATCH"）。
      如果 deepseek 没返回符合预期的内容，我们会回落到空字符串。    
    """
    try:
        content = call_chat_completions(
            messages=[{"role": "user", "content": user_prompt}],
            system_prompt=system_prompt,
            model=model or None,
            api_key=api_key or None,
            endpoint=endpoint or None,
            temperature=0.0,
            max_tokens=16,
            timeout=10,
        ).strip()
        first_line = content.splitlines()[0].strip()
        return first_line
    except DeepSeekError:
        return ""


# -----------------------
# 判定函数 1：语气是否中立 (tone_neutral_llm_judge)
# -----------------------

def tone_neutral_llm_judge(text: str,
                            **kwargs: Any) -> bool:
    """
    要求：整体语气中立、分析型，不带夸张情绪词或强烈态度倾向。

    返回：True 表示文本被判为 NEUTRAL。
    失败回退：如果 deepseek 返回的标签不是 {NEUTRAL, NOT_NEUTRAL}，我们认为不通过 (False)。
    """
    system_prompt = (
        "You are a strict tone classifier. "
        "Your job is to classify the tone of the given text. "
        "The ONLY allowed answers are exactly one of these two tokens: NEUTRAL or NOT_NEUTRAL. "
        "Do not explain."
    )

    user_prompt = (
        "Classify the tone of the following text. "
        "Return ONLY one word: NEUTRAL or NOT_NEUTRAL.\n\n"
        f"Text:\n{text}\n"
    )

    label = _call_deepseek_classifier(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    ).upper()

    if label.startswith("NEUTRAL"):
        return True
    if label.startswith("NOT_NEUTRAL"):
        return False

    # 非法输出 => 判为不满足中立
    return False


# -----------------------
# 判定函数 2：语气是否负向/批评 (tone_negative_llm_judge)
# -----------------------

def tone_negative_llm_judge(text: str,
                             **kwargs: Any) -> bool:
    """
    要求：文本表达不满、批评、负面态度、风险强调、指责或抱怨。

    返回：True 表示文本被判为 YES (负面/批评态度存在)。
    如果 deepseek 输出不符合 {YES, NO}，则 False。
    """
    system_prompt = (
        "You are a strict sentiment/stance detector. "
        "Determine if the author's tone clearly expresses frustration, criticism, or a negative stance. "
        "The ONLY allowed answers are exactly YES or NO. "
        "Do not add anything else."
    )

    user_prompt = (
        "Does the following text express frustration, criticism, blame, or strong negative stance? "
        "Answer ONLY YES or NO.\n\n"
        f"Text:\n{text}\n"
    )

    label = _call_deepseek_classifier(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    ).upper()

    if label.startswith("YES"):
        return True
    if label.startswith("NO"):
        return False

    return False


# -----------------------
# 判定函数 3：是否满足“角色一致性”(role_consistency_judge)
# -----------------------

def role_consistency_judge(text: str,
                            required_role: str = "third-person analyst",
                            **kwargs: Any) -> bool:
    """
    要求：回答是否始终以指定角色/视角说话。
    典型用法：
      - 要求用"第三人称中立讲述"，不得出现第一人称"I"、"we"；
      - 要求用"一线记者第一人称目击"的口吻等。

    我们把角色名+约束喂给LLM，请它二分类 MATCH / MISMATCH。

    返回 True 表示 MATCH。
    """
    system_prompt = (
        "You are a strict style/voice checker. "
        "Given a REQUIREMENT describing the expected narrative voice, "
        "and a TEXT, answer ONLY MATCH or MISMATCH. "
        "Do not explain."
    )

    user_prompt = (
        "REQUIREMENT: " + required_role + "\n\n"
        "TEXT TO CHECK:\n" + text + "\n\n"
        "Does the TEXT follow the narrative voice/style described in REQUIREMENT? "
        "Answer ONLY MATCH or MISMATCH."
    )

    label = _call_deepseek_classifier(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    ).upper()

    if label.startswith("MATCH"):
        return True
    if label.startswith("MISMATCH"):
        return False

    return False


# -----------------------
# 判定函数 4：是否给出改进/行动建议 (actionability_judge)
# -----------------------

def actionability_judge(text: str,
                         **kwargs: Any) -> bool:
    """
    要求：文本必须提出具体的可执行建议 / 改进行动 / 下一步措施。

    例如在我们构造的负面分支中：
    - “指出问题后必须给出至少一条改进建议或后续行动步骤”。

    输出：TRUE / FALSE
    """
    system_prompt = (
        "You are a strict evaluator. "
        "Check if the text explicitly proposes at least one concrete improvement, recommendation, or next-step action. "
        "Answer ONLY YES or NO."
    )

    user_prompt = (
        "Does the following text propose at least one concrete improvement, recommendation, or next-step action? "
        "Answer ONLY YES or NO.\n\n"
        f"Text:\n{text}\n"
    )

    label = _call_deepseek_classifier(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    ).upper()

    if label.startswith("YES"):
        return True
    if label.startswith("NO"):
        return False

    return False


# -----------------------
# 判定函数 5：是否避免极化/煽动性语言 (non_extremeness_judge)
# -----------------------

def non_extremeness_judge(text: str,
                           **kwargs: Any) -> bool:
    """
    要求：文本不能包含过度煽动性、攻击性、极化表达。
    我们问 LLM：这段话有没有在煽动、辱骂、妖魔化任何一方？

    返回 True 表示安全（无煽动），False 表示存在煽动或攻击。
    """
    system_prompt = (
        "You are a safety checker. "
        "Determine whether the text refrains from inflammatory, insulting, or extreme language. "
        "If the text does NOT include inflammatory/extreme/insulting language, answer SAFE. "
        "If it DOES include inflammatory/extreme/insulting language, answer UNSAFE. "
        "Return ONLY SAFE or UNSAFE."
    )

    user_prompt = (
        "Classify this text for inflammatory or extreme language. "
        "Return ONLY SAFE or UNSAFE.\n\n"
        f"Text:\n{text}\n"
    )

    label = _call_deepseek_classifier(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    ).upper()

    if label.startswith("SAFE"):
        return True
    if label.startswith("UNSAFE"):
        return False

    return False