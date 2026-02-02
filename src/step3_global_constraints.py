"""
Step 3 - Global Constraint Extraction

Purpose / 目标
- Capture document-wide rules (language, structure, tone, safety) that every future answer must honor.
- Provide verifier specs for each rule so the scoring runner can automatically check compliance.

Constraint sources
1. Deterministic heuristics ("hard" constraints): infer length bands, language, sectioning, formatting habits, and safety toggles by inspecting the exemplar answer + segmentation metadata.
2. LLM-backed "soft" constraints: ask DeepSeek to articulate abstract stylistic or quality expectations grounded in the provided text.

Outputs / 输出
- List[ConstraintNode] with scope="global", traceability metadata, and verifier_spec.

Why this matters
- Establishes non-negotiable contract terms before we dive into block-level reasoning.
- Guarantees at least a baseline of machine-checkable rules even if LLM calls fail (heuristics always emit something).

Dependencies
- graph_schema.ConstraintNode
- utils.deepseek_client.call_chat_completions for soft constraint generation.
- utils.parsing.extract_constraints for resilient parsing of LLM JSON.
- utils.text_clean helpers for outlines/snippets referenced inside prompts.
"""

import json
import math
import requests
from typing import List, Dict, Any

from .graph_schema import ConstraintNode
from .utils.parsing import extract_constraints
from .utils.text_clean import make_snippet, summarize_blocks_outline, clip

# Optional description templates (user-provided). We gracefully fall back if unavailable.
try:
    from .utils.templates import DESCS as _DESC_TEMPLATES  # type: ignore
except Exception:
    _DESC_TEMPLATES = {}

# Sampling config: pick a random count within these ranges (clamped by availability).
_STEP3_HARD_SAMPLE_RANGE = (3, 5)
_STEP3_SOFT_SAMPLE_RANGE = (3, 5)

def _desc_from_tpl(key: str, default: str, **kwargs) -> str:
    """
    Pull a description template by key from user-provided templates. If the template value
    is a list or tuple, randomly select one candidate. Fall back to `default` if key not found.
    """
    tpl = _DESC_TEMPLATES.get(key, default)
    import random
    if isinstance(tpl, (list, tuple)) and tpl:
        tpl = random.choice(list(tpl))
    try:
        return str(tpl).format(**kwargs)
    except Exception:
        # final fallback: use default
        return default.format(**kwargs)

# Regex utilities for new hard constraints
import re
_PAR_SPLIT = re.compile(r"(?:\r?\n){2,}")
_MD_HEADING = re.compile(r"^(#{1,6})\s+\S", re.M)
_BULLET_MARK = re.compile(r"^(?:[-*]\s+|\d+\.\s+)", re.M)
_EMOJI = re.compile(r"[\U00010000-\U0010ffff]", re.UNICODE)

from .utils.deepseek_client import call_chat_completions, DeepSeekError
_DEEPSEEK_API_KEY_DEFAULT = ""
_DEEPSEEK_ENDPOINT = ""
_DEEPSEEK_MODEL = ""


# -------------------------------------------------
# 工具：从当前回答中推测硬性全局约束基线
# -------------------------------------------------

def _estimate_word_count(text: str) -> int:
    """
    Estimate word count robustly for mixed Latin/CJK:
    - Latin tokens: \\w+ matches words/numbers/underscore
    - CJK: count individual Han characters (rough proxy)
    """
    tokens = re.findall(r"\w+", text)
    zh_chars = re.findall(r"[\u4e00-\u9fff]", text)
    return len(tokens) + len(zh_chars)


def _round_up_to_ten(value: int) -> int:

    """Round positive integers up to the nearest multiple of 10."""

    value = max(1, value)

    return int(math.ceil(value / 10.0) * 10)



def _round_nearest_ten(value: int) -> int:

    """Round positive integers to the nearest multiple of 10."""

    value = max(1, value)

    return int(math.floor((value + 5) / 10.0) * 10)



def _guess_language(text: str) -> str:
    """
    Heuristic primary language guess:
    - If Han characters dominate (≥ 25 and ≥ 40% of visible letters), return 'zh'
    - Otherwise 'en'
    """
    zh_chars = re.findall(r"[\u4e00-\u9fff]", text)
    letters = re.findall(r"[A-Za-z\u4e00-\u9fff]", text)
    if len(zh_chars) >= 25 and len(zh_chars) >= 0.4 * max(1, len(letters)):
        return "zh"
    return "en"


def _has_intro_body_conclusion(segmentation: Dict[str, Any]) -> bool:
    """
    根据 Step2 的 segmentation 结果，看看是否能观察到典型结构：
    - 存在开篇类块 (Opening / Intro / Background / Context)
    - 存在主体分析类块 (Main Analysis / Discussion / Evaluation / Argument)
    - 存在总结/展望类块 (Conclusion / Summary / Outlook / Recommendation)

    如果这些intent基本存在，就可以生成一个 has_sections 约束。
    否则别强行要求。
    """
    intents = [blk.get("intent", "").lower() for blk in segmentation.get("blocks", [])]

    def any_contains(keys):
        return any(any(k in intent for k in keys) for intent in intents)

    has_opening = any_contains(["opening", "intro", "context", "background"])
    has_body = any_contains(["analysis", "discussion", "main", "argument", "evaluation"])
    has_conclusion = any_contains(["conclusion", "summary", "outlook", "recommendation"])

    return has_opening and has_body and has_conclusion

# ---- New hard constraint detectors ----
def _count_paragraphs(text: str) -> int:
    paras = [p for p in _PAR_SPLIT.split(text.strip()) if p.strip()]
    return len(paras)

def _detect_heading_levels(text: str):
    levels = set()
    for m in _MD_HEADING.finditer(text):
        levels.add(len(m.group(1)))
    return levels

def _detect_bullet_marker(text: str):
    marks = re.findall(r"^([-*]|\d+\.)\s+", text, re.M)
    if not marks:
        return None, False
    first = marks[0]
    mixed = any(m != first for m in marks[1:])
    return first, mixed

def _has_emojis(text: str) -> bool:
    return bool(_EMOJI.search(text))

def _detect_citation_style(text: str):
    # numeric [1], [12] vs author-year (Smith, 2021)
    has_numeric = bool(re.search(r"\[\d{1,3}\]", text))
    has_author_year = bool(re.search(r"\([A-Z][A-Za-z]+,?\s+\d{4}\)", text))
    if has_numeric and not has_author_year:
        return "numeric"
    if has_author_year and not has_numeric:
        return "author_year"
    return None

def _detect_decimal_places(text: str):
    nums = re.findall(r"\b\d+\.(\d+)\b", text)
    if len(nums) < 3:
        return None
    from collections import Counter
    cnt = Counter(len(s) for s in nums)
    most, freq = cnt.most_common(1)[0]
    if freq >= max(3, int(0.6 * len(nums))):
        return most
    return None

def _detect_date_format(text: str):
    if re.search(r"\b\d{4}-\d{2}-\d{2}\b", text):
        return "yyyy-mm-dd"
    if re.search(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}\b", text):
        return "mon dd, yyyy"
    return None

def _detect_contractions_en(text: str) -> int:
    # count common English contractions as a proxy
    return len(re.findall(r"\b(?:don't|doesn't|can't|won't|I'm|it's|that's|we're|they're|I've|you've|isn't|aren't|weren't|hasn't|haven't|shouldn't|couldn't|wouldn't)\b", text, re.I))

# --- Keyword and symbol format helpers ---
_EN_STOP = set("""
a an the and or but if then else when while for to of in on at by from as is are was were be been being this that these those with without into within across over under between among can could should would may might must do does did done doing have has had having not no nor so such very more most other same own just also than too rather quite
""".split())

def _extract_keywords_simple(text: str, lang: str) -> List[str]:
    """
    Lightweight keyword extractor to avoid heavy deps:
    - EN: count word frequencies, remove short tokens (<=3) & stopwords, pick top 1-3 distinct
    - ZH: return [] (we avoid low-quality heuristics for now)
    """
    if lang == "zh":
        return []
    words = re.findall(r"[A-Za-z][A-Za-z\-']+", text.lower())
    # strip apostrophes at the ends
    words = [w.strip("'").strip("-") for w in words]
    words = [w for w in words if len(w) > 3 and w not in _EN_STOP]
    if not words:
        return []
    from collections import Counter
    top = [w for (w, c) in Counter(words).most_common(8)]
    # keep order, take up to 3 unique
    out: List[str] = []
    for w in top:
        if w not in out:
            out.append(w)
        if len(out) == 3:
            break
    return out

_SYMBOL_CANDIDATES = [",", ":", "?", "!"]

def _choose_forbid_symbol(text: str) -> str | None:
    for s in _SYMBOL_CANDIDATES:
        if s not in text:
            return s
    return None


def _build_hard_global_constraints(response_text: str,
                                   segmentation: Dict[str, Any]) -> List[ConstraintNode]:
    """
    基于可观测信号，构造稳定的硬性全局约束节点。
    我们不会幻想不存在的要求，只根据文本本身的客观属性：
    - 字数下限：设为 floor(word_count * 0.8) 向下取整，但至少 100 词。
      （思路：我们希望后续回答别比示例短太多，否则不合格）
    - 语言：根据文本主语言生成 require_language(lang=...)
    - 结构段落：如果 segmentation 看起来有开头/主体/结论，就要求 has_sections
    这些都会被标记为 scope="global"。
    """
    nodes: List[ConstraintNode] = []
    cid_counter = 1
    import random

    # -----------------------------
    # Phase A: collect candidates
    # -----------------------------

    length_candidates: List[ConstraintNode] = []
    language_candidates: List[ConstraintNode] = []
    structure_candidates: List[ConstraintNode] = []
    format_candidates: List[ConstraintNode] = []
    style_safety_candidates: List[ConstraintNode] = []

    # A1) Length candidates
    wc = _estimate_word_count(response_text)
    if wc > 0:
        raw_min = max(100, int(wc * 0.85))
        raw_max = max(raw_min + 10, int(wc * 1.20))
        min_words = _round_up_to_ten(raw_min)
        max_words = _round_up_to_ten(raw_max)
        if max_words <= min_words:
            max_words = min_words + 10
        has_minmax = "digit_format_min_max" in _DESC_TEMPLATES
        has_around = "digit_format_around" in _DESC_TEMPLATES
        has_min = "digit_format_min" in _DESC_TEMPLATES
        has_max = "digit_format_max" in _DESC_TEMPLATES

        if has_minmax:
            length_candidates.append(ConstraintNode(
                cid=f"G{cid_counter}",
                desc=_desc_from_tpl(
                    "digit_format_min_max",
                    "Keep the answer length between {min_words} and {max_words} words.",
                    min_words=min_words, max_words=max_words,
                ),
                scope="global",
                verifier_spec={"check": "word_count_between", "args": {"min_words": min_words, "max_words": max_words}},
                trace_to=None, derived_from="step3",
            ))
        if has_around:
            center = _round_nearest_ten(int(round(wc))); tol = 0.15
            length_candidates.append(ConstraintNode(
                cid=f"G{cid_counter}",
                desc=_desc_from_tpl(
                    "digit_format_around",
                    "Keep the answer length around {center} words (±{tol_pct}%).",
                    center=center, tol_pct=int(tol * 100),
                ),
                scope="global",
                verifier_spec={"check": "word_count_around", "args": {"center": center, "tolerance_pct": tol}},
                trace_to=None, derived_from="step3",
            ))
        if has_min:
            length_candidates.append(ConstraintNode(
                cid=f"G{cid_counter}",
                desc=_desc_from_tpl(
                    "digit_format_min",
                    "The answer must be at least {min_words} words long.",
                    min_words=min_words,
                ),
                scope="global",
                verifier_spec={"check": "min_word_count", "args": {"min_words": min_words}},
                trace_to=None, derived_from="step3",
            ))
        if has_max:
            length_candidates.append(ConstraintNode(
                cid=f"G{cid_counter}",
                desc=_desc_from_tpl(
                    "digit_format_max",
                    "Keep the answer under {max_words} words.",
                    max_words=max_words,
                ),
                scope="global",
                verifier_spec={"check": "max_word_count", "args": {"max_words": max_words}},
                trace_to=None, derived_from="step3",
            ))
        if not (has_minmax or has_around or has_min or has_max):
            length_candidates.append(ConstraintNode(
                cid=f"G{cid_counter}",
                desc=_desc_from_tpl(
                    "min_word_count",
                    "The answer must be at least {min_words} words long.",
                    min_words=min_words,
                ),
                scope="global",
                verifier_spec={"check": "min_word_count", "args": {"min_words": min_words}},
                trace_to=None, derived_from="step3",
            ))

    # A2) Language candidates
    lang = _guess_language(response_text)
    language_candidates.append(
        ConstraintNode(
            cid=f"G{cid_counter}",
            desc=_desc_from_tpl(
                "require_language_zh" if lang == "zh" else "require_language_en",
                ("The answer must be written primarily in Chinese." if lang == "zh"
                 else "The answer must be written primarily in English."),
            ),
            scope="global",
            verifier_spec={"check": "require_language", "args": {"lang": lang}},
            trace_to=None, derived_from="step3",
        )
    )
    if lang == "en":
        contractions = _detect_contractions_en(response_text)
        if contractions == 0:
            language_candidates.append(
                ConstraintNode(
                    cid=f"G{cid_counter}",
                    desc=_desc_from_tpl(
                        "avoid_contractions",
                        "Avoid contractions (use 'do not' instead of 'don't').",
                    ),
                    scope="global",
                    verifier_spec={"check": "avoid_contractions", "args": {}},
                    trace_to=None, derived_from="step3",
                )
            )

    # A3) Structure candidates
    if _has_intro_body_conclusion(segmentation):
        structure_candidates.append(
            ConstraintNode(
                cid=f"G{cid_counter}",
                desc=_desc_from_tpl(
                    "has_sections_intro_body_conclusion",
                    "The answer must include an Opening/Intro section, a Body/Main Analysis section, and a Conclusion/Outlook section in logical progression.",
                ),
                scope="global",
                verifier_spec={"check": "has_sections", "args": {"sections": ["Opening", "Body", "Conclusion"]}},
                trace_to=None, derived_from="step3",
            )
        )
    para_cnt = _count_paragraphs(response_text)
    if para_cnt >= 3:
        structure_candidates.append(
            ConstraintNode(
                cid=f"G{cid_counter}",
                desc=_desc_from_tpl(
                    "min_paragraphs",
                    "Organize the answer into at least {min_paras} paragraphs.",
                    min_paras=para_cnt,
                ),
                scope="global",
                verifier_spec={"check": "min_paragraphs", "args": {"min_paras": para_cnt}},
                trace_to=None, derived_from="step3",
            )
        )

    # A4) Format consistency candidates
    heading_levels = _detect_heading_levels(response_text)
    if heading_levels:
        levels_sorted = sorted(heading_levels)
        format_candidates.append(
            ConstraintNode(
                cid=f"G{cid_counter}",
                desc=_desc_from_tpl(
                    "heading_levels_only",
                    "Use consistent Markdown heading levels: only {levels}.",
                    levels=levels_sorted,
                ),
                scope="global",
                verifier_spec={"check": "heading_levels_only", "args": {"levels": levels_sorted}},
                trace_to=None, derived_from="step3",
            )
        )
    bullet, mixed = _detect_bullet_marker(response_text)
    if bullet:
        format_candidates.append(
            ConstraintNode(
                cid=f"G{cid_counter}",
                desc=_desc_from_tpl(
                    "bullet_style_consistent",
                    "Use a consistent list marker style ('{marker}'); do not mix list markers.",
                    marker=bullet,
                ),
                scope="global",
                verifier_spec={"check": "bullet_style_consistent", "args": {"marker": bullet}},
                trace_to=None, derived_from="step3",
            )
        )
    dec = _detect_decimal_places(response_text)
    if dec is not None:
        format_candidates.append(
            ConstraintNode(
                cid=f"G{cid_counter}",
                desc=_desc_from_tpl(
                    "decimal_places",
                    "Keep numeric values to {places} decimal places consistently.",
                    places=dec,
                ),
                scope="global",
                verifier_spec={"check": "decimal_places", "args": {"places": dec}},
                trace_to=None, derived_from="step3",
            )
        )
    dfmt = _detect_date_format(response_text)
    if dfmt:
        format_candidates.append(
            ConstraintNode(
                cid=f"G{cid_counter}",
                desc=_desc_from_tpl(
                    "date_format_iso" if dfmt=="yyyy-mm-dd" else "date_format_long",
                    ("Use the date format YYYY-MM-DD." if dfmt=="yyyy-mm-dd"
                     else "Use the date format 'Month DD, YYYY'."),
                ),
                scope="global",
                verifier_spec={"check": "date_format", "args": {"style": dfmt}},
                trace_to=None, derived_from="step3",
            )
        )

    # A5) Style & safety candidates
    lower_txt = response_text.lower()
    first_person_hits = any(token in lower_txt for token in [" i ", " we ", " my ", " our "])
    if not first_person_hits:
        style_safety_candidates.append(
            ConstraintNode(
                cid=f"G{cid_counter}",
                desc=_desc_from_tpl(
                    "forbid_first_person",
                    "Maintain an objective, third-person analytic voice; do not use first-person pronouns.",
                ),
                scope="global",
                verifier_spec={"check": "forbid_first_person", "args": {}},
                trace_to=None, derived_from="step3",
            )
        )
    if not _has_emojis(response_text):
        style_safety_candidates.append(
            ConstraintNode(
                cid=f"G{cid_counter}",
                desc=_desc_from_tpl(
                    "forbid_emojis",
                    "Do not use emojis or decorative unicode symbols.",
                ),
                scope="global",
                verifier_spec={"check": "forbid_emojis", "args": {}},
                trace_to=None, derived_from="step3",
            )
        )
    sym = _choose_forbid_symbol(response_text)
    if sym:
        style_safety_candidates.append(
            ConstraintNode(
                cid=f"G{cid_counter}",
                desc=_desc_from_tpl(
                    "symbol_format",
                    "Do not use the symbol '{symbol}'.",
                    symbol=sym,
                ),
                scope="global",
                verifier_spec={"check": "forbid_symbol", "args": {"symbol": sym}},
                trace_to=None, derived_from="step3",
            )
        )
    kws = _extract_keywords_simple(response_text, lang)
    if kws:
        style_safety_candidates.append(
            ConstraintNode(
                cid=f"G{cid_counter}",
                desc=_desc_from_tpl(
                    "keyword_format",
                    "Include the following keywords: {keywords}.",
                    keywords=", ".join(f"\"{k}\"" for k in kws),
                ),
                scope="global",
                verifier_spec={"check": "must_include_keywords", "args": {"keywords": kws}},
                trace_to=None, derived_from="step3",
            )
        )
    cite_style = _detect_citation_style(response_text)
    if cite_style:
        style_safety_candidates.append(
            ConstraintNode(
                cid=f"G{cid_counter}",
                desc=_desc_from_tpl(
                    "citation_style_numeric" if cite_style=="numeric" else "citation_style_author_year",
                    ("Use numeric bracket citations like [1], [2]." if cite_style=="numeric"
                     else "Use author-year citations like (Smith, 2021)."),
                ),
                scope="global",
                verifier_spec={"check": "citation_style", "args": {"style": cite_style}},
                trace_to=None, derived_from="step3",
            )
        )

    # -----------------------------
    # Phase B: randomly select 3-5 hard constraints total (if available)
    # -----------------------------
    all_candidates: List[ConstraintNode] = (
        length_candidates
        + language_candidates
        + structure_candidates
        + format_candidates
        + style_safety_candidates
    )
    if not all_candidates:
        return nodes

    min_hard, max_hard = _STEP3_HARD_SAMPLE_RANGE
    target = random.randint(min_hard, max_hard)
    pick_count = min(len(all_candidates), target)
    nodes.extend(random.sample(all_candidates, pick_count))

    return nodes


# -------------------------------------------------
# LLM: 生成软性/语气/安全类全局约束
# -------------------------------------------------


def _call_deepseek_soft_constraints(response_text: str,
                                    segmentation: Dict[str, Any]) -> str:
    """
    调用 deepseek 反推出“抽象软性全局偏好”：

    目标：从完整回答 (#Response#) 中，推测真实用户在提问时可能隐含的“软性偏好”，例如：
    - 希望论证结构清晰、有逻辑性
    - 希望多解释原因或背景
    - 希望语气更正式 / 更友好
    - 希望例子更丰富或更贴近日常生活
    - 希望避免含糊表达，提高可理解性

    要求：
    - 偏好必须是抽象的（abstract）、人类自然会说的偏好（human-like preference）。
    - 不包含任何具体结构要求：段落数、列表长度、字数、字符数、符号/格式（粗体、斜体、代码块、顺序号等）。
    - 不包含 Response 的具体内容、结论或细节；不能让模型看到“答案骨架”然后照着仿写。
    - 只能从整体风格/写法中抽象出可能的偏好，而不是复述答案本身。

    输出：JSON list，每一项对应一个“抽象软性约束/偏好”，建议条目数在 3-6 条：
        {
          "desc": "<抽象软性偏好，英文>",
          "verifier": {"check": "<snake_case>", "args": {}}
        }

    desc：
    - 必须是英文、命令式或规范式语句，但语义上是“偏好/风格”，不是硬性结构规则。
    - 不得提及段落个数、列表个数、字数、字符数、具体符号、具体结论或样例细节。

    verifier.check：
    - 若适用，优先使用已有的软性校验器：
      tone_neutral_llm_judge
      tone_negative_llm_judge
      non_extremeness_judge
      role_consistency_judge
      actionability_judge
    - 如需表达新的偏好，可以创建新的 snake_case 名称，如
      "preference_clear_reasoning", "preference_rich_examples", "preference_formal_tone" 等。
    - 所有新校验名必须与 desc 语义对应，并且 args 为 JSON object（可为空）。

    若无法抽取合理偏好，可返回空列表 []。
    """

    # 处理原文：保持语义，去除多余空白，不默认截断
    answer_clean = make_snippet(response_text)
    if len(answer_clean) > 20000:
        # 极端长文本才触发截断；这是显式的、可审计的内容丢失点
        answer_clean = clip(answer_clean, 20000)

    outline_str = summarize_blocks_outline(segmentation)
    
    system_prompt = """You are an instruction analyst.
Your job is to infer ONLY abstract, human-like soft preferences a user might have had when asking for this answer.

CRITICAL SCOPE:
- You MUST base your inferences on the overall style and behavior of the FULL ANSWER text.
- You MUST NOT restate concrete content, conclusions, or detailed structure from the answer.
- You MUST keep the preferences abstract and user-like (how a human would describe their preferences), not as a restatement of the answer.

What is an acceptable soft preference?
- High-level expectations about style, clarity, and helpfulness, for example:
  - "Prefer clear and logical reasoning."
  - "Prefer more explanation of reasons and background."
  - "Prefer a formal and professional tone." / "Prefer a friendly and accessible tone."
  - "Prefer using concrete examples drawn from everyday life."
  - "Prefer avoiding vague, ambiguous phrasing to improve comprehensibility."

You MUST AVOID:
- Mentioning paragraph count.
- Mentioning list length or number of bullet points.
- Mentioning character counts or word counts.
- Mentioning explicit formatting (bold, italics, headings, code blocks, numbering).
- Describing the specific content, conclusions, or structure of the given answer.
- Any wording that would let a model mimic the exact outline or skeleton of the original answer.

Your task is to infer only generalized, reasonable user soft preferences which could explain why the answer looks the way it does, but without leaking any concrete answer details.

OUTPUT FORMAT (STRICT JSON):
- Return a JSON list (array) of 3 to 6 items.
- Each item describes ONE abstract soft preference:
  {
    "desc": "<imperative, abstract soft preference in English>",
    "verifier": {"check": "<snake_case>", "args": {}}
  }

Rules for desc:
- English only.
- Imperative or normative form ("Keep the tone ...", "Provide ...", etc.), but semantically a soft preference.
- MUST NOT mention counts, explicit formatting, or concrete answer details.

Rules for verifier.check:
- If suitable, choose from:
  - tone_neutral_llm_judge
  - tone_negative_llm_judge
  - non_extremeness_judge
  - role_consistency_judge
  - actionability_judge
- Otherwise, you MAY create new descriptive snake_case names that reflect the preference,
  e.g., "preference_clear_reasoning", "preference_rich_examples", "preference_formal_tone".
- New names must:
  - be snake_case [a-z0-9_],
  - clearly reflect the obligation in desc,
  - include an args JSON object (possibly empty) describing parameters if any.

If nothing reasonable applies, return an empty JSON list [] and no additional text.

You must return ONLY the JSON list, no explanations, no markdown, no code fences."""


    user_prompt = (
        "GLOBAL OUTLINE (structure only; DO NOT invent rules from this):\n"
        f"{outline_str}\n\n"
        "#Response# (this is the FULL ANSWER content as given to the user):\n"
        "You must infer only abstract, human-like soft preferences from this text, without leaking concrete details.\n\n"
        f"{answer_clean}\n\n"
        "Extract 3 to 6 abstract soft preferences the user might have, following the specification above,\n"
        "and return ONLY the JSON list.\n"
    )

    try:
        content = call_chat_completions(
            messages=[
                {"role": "user", "content": user_prompt},
            ],
            system_prompt=system_prompt,
            temperature=0.0,
            max_tokens=1024,
            timeout=180,
            retries=2,
        ).strip()
        return content
    except DeepSeekError:
        return "[]"


# -------------------------------------------------
# 主入口：结合硬约束 + 软约束
# -------------------------------------------------

def extract_global_constraints(response_text: str,
                               segmentation: Dict[str, Any]) -> List[ConstraintNode]:
    """
    Step3 主入口：

    1. 基于回答文本 + segmentation，构造硬性全局约束（本地可验证）。
       这些约束永远会存在，确保我们至少能做程序化打分。

    2. 调用 deepseek 提取软性/语气/安全类的全局约束；可能返回0条，也可能多条。

    3. 把二者合并，按顺序编号 G1, G2, ... ，得到最终的全局约束列表。

    注意：
    - 不再强行生成通用fallback约束。
    - 软性约束缺席时，我们仍有硬性约束可用。
    """

    hard_nodes = _build_hard_global_constraints(response_text, segmentation)
    soft_raw_str = _call_deepseek_soft_constraints(response_text, segmentation)
    soft_items = extract_constraints(soft_raw_str)  # list[dict]

    soft_nodes: List[ConstraintNode] = []
    for item in soft_items:
        # extract_constraints() 已经尽量标准化字段名：cid/desc/scope/verifier_spec
        desc = item.get("desc", "").strip()
        verifier_spec = item.get("verifier_spec", {}) or item.get("verifier", {}) or {}
        check_name = verifier_spec.get("check")
        args_obj = verifier_spec.get("args", {}) or {}

        if not desc or not check_name:
            continue

        soft_nodes.append(
            ConstraintNode(
                cid="TEMP",  # 后续统一重排ID
                desc=desc,
                scope="global",
                verifier_spec={"check": check_name, "args": args_obj},
                trace_to=None,
                derived_from="step3",
            )
        )

    # Soft sampling: randomly pick 3-5 soft constraints if available
    import random
    if soft_nodes:
        min_soft, max_soft = _STEP3_SOFT_SAMPLE_RANGE
        target_soft = random.randint(min_soft, max_soft)
        soft_nodes = random.sample(soft_nodes, min(len(soft_nodes), target_soft))

    # 合并并重新编号 cid
    all_nodes: List[ConstraintNode] = []
    for node in hard_nodes + soft_nodes:
        all_nodes.append(node)
    for idx, node in enumerate(all_nodes, start=1):
        node.cid = f"G{idx}"

    return all_nodes


if __name__ == "__main__":
    demo_resp = (
        "The modern space race is not only a technical contest but a geopolitical instrument. "
        "In this analysis, we outline historical context, assess key actors, and discuss future risks.\n\n"
        "First, we review how national prestige and commercial incentives shaped recent launches.\n\n"
        "Finally, we conclude with implications for global stability and practical next-step recommendations."
    )
    demo_seg = {
        "blocks": [
            {"block_id": "B1", "intent": "Opening / Context setup", "text_span": "..."},
            {"block_id": "B2", "intent": "Main Analysis", "text_span": "..."},
            {"block_id": "B3", "intent": "Conclusion / Outlook / Recommendation", "text_span": "..."},
        ],
        "order": ["B1", "B2", "B3"],
    }

    out_nodes = extract_global_constraints(demo_resp, demo_seg)
    print(json.dumps([
        {
            "cid": n.cid,
            "desc": n.desc,
            "scope": n.scope,
            "verifier_spec": n.verifier_spec,
            "derived_from": n.derived_from,
        } for n in out_nodes
    ], indent=2, ensure_ascii=False))
