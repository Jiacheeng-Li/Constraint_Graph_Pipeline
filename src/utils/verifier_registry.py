
# 此文件已更新以支持更多 LLM-based 软性判定。


"""
verifier_registry.py

统一管理所有可调用的约束校验函数（hard / soft）。

使用方式：
- 每个约束节点都有一个 verifier_spec:
    {
        "check": "min_word_count",
        "args": {"min_words": 150}
    }

- 评测时我们会调用:
    run_verifier(verifier_spec, text)
  这会自动在注册表里找到对应函数，并把 args 作为关键字参数传入。

注意：
- 如果某个 check 在注册表中不存在，我们当前策略是返回 True（即跳过罚分），
  这样不会因为未实现的软性规则导致整条样本被判死。
  后续可以改为返回 None 并单独统计。
"""

from typing import Dict, Any, Callable
from ..verifier import hard_checks, soft_checks

# 全局注册表: check_name -> callable
_REGISTRY: Dict[str, Callable[..., bool]] = {}


def _register_builtin() -> None:
    """把我们目前支持的硬性/软性校验函数全部挂到注册表里。"""

    # 硬性规则（hard_checks）
    _REGISTRY["is_english"] = hard_checks.is_english
    _REGISTRY["forbid_first_person"] = hard_checks.forbid_first_person
    _REGISTRY["forbid_words"] = hard_checks.forbid_words

    _REGISTRY["min_word_count"] = hard_checks.min_word_count
    _REGISTRY["max_word_count"] = hard_checks.max_word_count
    _REGISTRY["min_char_count"] = hard_checks.min_char_count

    _REGISTRY["must_list_n_subpoints"] = hard_checks.must_list_n_subpoints
    _REGISTRY["has_sections"] = hard_checks.has_sections
    _REGISTRY["min_numbered_items"] = hard_checks.min_numbered_items

    _REGISTRY["must_include_keywords"] = hard_checks.must_include_keywords
    _REGISTRY["keyword_min_frequency"] = hard_checks.keyword_min_frequency
    _REGISTRY["must_cover_topics"] = hard_checks.must_cover_topics

    _REGISTRY["require_language"] = hard_checks.require_language
    _REGISTRY["must_end_with_template"] = hard_checks.must_end_with_template

    # 软性规则（soft_checks）——语气、态度、立场等需要语义判断的部分。
    _REGISTRY["tone_neutral_llm_judge"] = soft_checks.tone_neutral_llm_judge
    _REGISTRY["tone_negative_llm_judge"] = soft_checks.tone_negative_llm_judge
    _REGISTRY["role_consistency_judge"] = soft_checks.role_consistency_judge
    _REGISTRY["actionability_judge"] = soft_checks.actionability_judge
    _REGISTRY["non_extremeness_judge"] = soft_checks.non_extremeness_judge


# 初始化注册表
_register_builtin()


def run_verifier(verifier_spec: Dict[str, Any], text: str) -> bool:
    """
    根据 verifier_spec 调用对应的校验器并返回 True/False。

    verifier_spec 形如：
        {
            "check": "min_word_count",
            "args": {"min_words": 150}
        }

    运行逻辑：
    1. 找到对应的函数
    2. 把 text 作为 text=... 传入
    3. 把 args 其余字段作为关键字参数传入

    异常/未知检查项策略：返回 True（宽容通过）。
    """
    if verifier_spec is None:
        return True

    check_name = verifier_spec.get("check")
    args = verifier_spec.get("args", {}) or {}

    fn = _REGISTRY.get(check_name)
    if fn is None:
        # 未实现的/未知的校验，当前策略：视为通过
        return True

    try:
        return fn(text=text, **args)
    except Exception:
        # 安全兜底：任何异常都不要让整个pipeline崩掉
        return False