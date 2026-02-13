
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

- 评测时我们主要调用:
    run_verifier(check_name="min_word_count",
                 check_args={"min_words": 150},
                 answer_text=resp)
  这会自动在注册表里找到对应函数，并把 args 作为关键字参数传入。

注意：
- 如果某个 check 在注册表中不存在，我们当前策略是返回 True（即跳过罚分），
  这样不会因为未实现的软性规则导致整条样本被判死。
  后续可以改为返回 None 并单独统计。
"""

from typing import Dict, Any, Callable, Optional, Tuple
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
    _REGISTRY["require_sections"] = hard_checks.require_sections
    _REGISTRY["min_numbered_items"] = hard_checks.min_numbered_items

    _REGISTRY["must_include_keywords"] = hard_checks.must_include_keywords
    _REGISTRY["keyword_min_frequency"] = hard_checks.keyword_min_frequency
    _REGISTRY["must_cover_topics"] = hard_checks.must_cover_topics

    _REGISTRY["require_language"] = hard_checks.require_language
    _REGISTRY["must_end_with_template"] = hard_checks.must_end_with_template

    # === Added to align with Step3/Step4 hard verifiers ===
    # Length variants
    _REGISTRY["word_count_between"] = hard_checks.word_count_between
    _REGISTRY["word_count_around"] = hard_checks.word_count_around

    # Structure / paragraphs
    _REGISTRY["min_paragraphs"] = hard_checks.min_paragraphs

    # Format consistency
    _REGISTRY["heading_levels_only"] = hard_checks.heading_levels_only
    _REGISTRY["bullet_style_consistent"] = hard_checks.bullet_style_consistent
    _REGISTRY["decimal_places"] = hard_checks.decimal_places
    _REGISTRY["date_format"] = hard_checks.date_format

    # Style / safety
    _REGISTRY["forbid_emojis"] = hard_checks.forbid_emojis
    _REGISTRY["avoid_contractions"] = hard_checks.avoid_contractions
    _REGISTRY["forbid_symbol"] = hard_checks.forbid_symbol

    # Citations
    _REGISTRY["citation_style"] = hard_checks.citation_style
    _REGISTRY["min_citation_markers"] = hard_checks.min_citation_markers
    _REGISTRY["min_distinct_citations"] = hard_checks.min_distinct_citations
    _REGISTRY["citation_refs_from_allowed_set"] = hard_checks.citation_refs_from_allowed_set

    # 软性规则（soft_checks）——语气、态度、立场等需要语义判断的部分。
    _REGISTRY["tone_neutral_llm_judge"] = soft_checks.tone_neutral_llm_judge
    _REGISTRY["tone_negative_llm_judge"] = soft_checks.tone_negative_llm_judge
    _REGISTRY["role_consistency_judge"] = soft_checks.role_consistency_judge
    _REGISTRY["actionability_judge"] = soft_checks.actionability_judge
    _REGISTRY["non_extremeness_judge"] = soft_checks.non_extremeness_judge


# 初始化注册表
_register_builtin()


def run_verifier(
    *,
    check_name: Optional[str] = None,
    check_args: Optional[Dict[str, Any]] = None,
    answer_text: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    verifier_spec: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[bool], Dict[str, Any]]:
    """
    调用指定的校验器并返回 (passed, details)。

    兼容两种调用方式：
      A. 关键字参数方式（推荐）：
            run_verifier(
                check_name="min_word_count",
                check_args={"min_words": 150},
                answer_text=resp,
                context={"cid": "..."}
            )

      B. 旧版接口（verifier_spec + text）：
            run_verifier(verifier_spec={"check": "...", "args": {...}}, answer_text=resp)

    返回：
        passed: bool | None  # None 表示未执行（未知校验或缺少文本）
        details: Dict[str, Any]  # 包含 check / args / 错误信息 等
    """
    # Back-compat: 如果仅提供 verifier_spec，则从中提取配置
    if verifier_spec:
        check_name = check_name or verifier_spec.get("check")
        check_args = check_args or (verifier_spec.get("args") or {})

    details: Dict[str, Any] = {
        "check": check_name,
        "args": check_args or {},
        "context": context or {},
    }

    if not check_name:
        details["note"] = "missing check name"
        return None, details

    fn = _REGISTRY.get(check_name)
    if fn is None:
        details["note"] = "unregistered verifier (treated as skipped)"
        return None, details

    if answer_text is None:
        details["note"] = "no answer_text provided"
        return None, details

    try:
        result = fn(text=answer_text, **(check_args or {}))
        passed = bool(result)
        details["result_raw"] = result
        return passed, details
    except Exception as exc:
        details["error"] = str(exc)
        return False, details
