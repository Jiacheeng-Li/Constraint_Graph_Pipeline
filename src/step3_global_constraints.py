"""
step3_global_constraints.py

Step 3: 全局约束抽取 (Global Constraint Extraction)

- 我们把全局约束分成两类：
  A. 硬性可程序校验的全局约束（hard global constraints）
     例如：最少词数、必须包含结构段落、禁止第一人称、必须为英文。
     这些可以直接由我们本地代码给出并附上 verifier_spec，
     不依赖 LLM，因此总是可用，保证下游评测有稳定基线。

  B. 软性 / 语气 / 安全 / 质量类全局约束（soft / semantic global constraints）
     例如：中立分析语气、不得煽动性攻击、输出必须保持专业而非情绪化。
     这些需要语用判断，继续调用 deepseek 生成或确认，
     并为它们附上 LLM-based 的 verifier（如 tone_neutral_llm_judge, non_extremeness_judge）。


输出：List[ConstraintNode]
- 每个 ConstraintNode:
    cid: 全局唯一ID（G1, G2, ...）
    desc: 人类可读描述
    scope: "global"
    verifier_spec: {"check": <fn-name>, "args": {...}}
    derived_from: "step3"

依赖：
- deepseek-chat (用于软性约束)
- ConstraintNode schema
- 硬性规则来自我们自己的启发式：
  - 字数下限 (min_word_count)
  - 语言判断 (require_language)
  - 结构段落 (has_sections) [仅当回答明显分块时]
  - 禁止第一人称 (forbid_first_person) [可选]
"""

import json
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
    added_categories: set[str] = set()
    def _add_node(category: str, node: ConstraintNode) -> bool:
        """
        Add a node only if this category hasn't been used yet.
        Returns True if added, False if skipped due to category cap.
        """
        if category in added_categories:
            return False
        nodes.append(node)
        added_categories.add(category)
        return True

    import random
    def _add_random_from(category: str, candidates: List[ConstraintNode]) -> int:
        """
        From a non-empty list of candidate nodes belonging to the same super-category,
        randomly select ONE and add it via _add_node. Returns 1 if added, else 0.
        """
        if not candidates:
            return 0
        choice = random.choice(candidates)
        if _add_node(category, choice):
            return 1
        return 0

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
        min_words = max(100, int(wc * 0.85))
        max_words = int(wc * 1.20)
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
            center = int(round(wc)); tol = 0.15
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
                     else "Use author–year citations like (Smith, 2021)."),
                ),
                scope="global",
                verifier_spec={"check": "citation_style", "args": {"style": cite_style}},
                trace_to=None, derived_from="step3",
            )
        )

    # -----------------------------
    # Phase B: randomly select one per super-category
    # -----------------------------
    cid_counter += _add_random_from("length", length_candidates)
    cid_counter += _add_random_from("language", language_candidates)
    cid_counter += _add_random_from("structure", structure_candidates)
    cid_counter += _add_random_from("format_consistency", format_candidates)
    cid_counter += _add_random_from("style_safety", style_safety_candidates)

    return nodes


# -------------------------------------------------
# LLM: 生成软性/语气/安全类全局约束
# -------------------------------------------------


def _call_deepseek_soft_constraints(response_text: str,
                                    segmentation: Dict[str, Any]) -> str:
    """
    调用 deepseek 让它给出“语气 / 安全 / 风格”类全局约束。

    非常重要：
    - 我们现在要求它只能根据回答本身已经呈现出来的风格/语气/姿态来提炼约束，
      不能脑补“理想上应该是什么样”。
    - 我们提供给它的 TEXT SNIPPET 是原文本身（仅做空白规整），
      不摘要、不改写、不自动截断；只有在极端过长时才 clip() 以防 token 爆炸。
    - outline 只是结构位置参考，不能用来发明没出现的要求。

    期望 deepseek 输出：JSON list，每项类似：
        {
          "desc": "The answer must maintain a neutral, analytical tone.",
          "verifier": {"check": "tone_neutral_llm_judge", "args": {}}
        }
    """

    # 处理原文：保持语义，去除多余空白，不默认截断
    answer_clean = make_snippet(response_text)
    if len(answer_clean) > 20000:
        # 极端长文本才触发截断；这是显式的、可审计的内容丢失点
        answer_clean = clip(answer_clean, 20000)

    outline_str = summarize_blocks_outline(segmentation)
    
    system_prompt = """You are an instruction analyst.
Your job is to infer ONLY global style/tone/safety requirements that the FULL ANSWER is ALREADY FOLLOWING.
You MUST base every requirement on observable evidence in the provided TEXT SNIPPET.
Do NOT invent idealized rules that are not clearly demonstrated in that text.
The OUTLINE is just structural context (which block does what), NOT evidence.
If you cannot justify a requirement from the snippet, you must NOT output it.

Soft global constraints are about tone, safety, stance, professional voice, neutrality, actionability, or analyst persona consistency across the entire answer.
Do NOT restate local factual obligations (e.g. "must list three risks") that only apply to one block; those belong to local block constraints, not global style.  🔁

Every constraint must be grounded in observable evidence in the TEXT SNIPPET.
Do NOT invent requirements that do not clearly appear in the text.

You must return ONLY valid JSON: a list of objects.
Each object MUST have: {desc, verifier:{check,args}}.

About verifier.check:
- If one of these fits, use it:
  tone_neutral_llm_judge
  tone_negative_llm_judge
  non_extremeness_judge
  role_consistency_judge
  actionability_judge
- Otherwise, you MUST create a new descriptive snake_case name
  that reflects the requirement, e.g. "must_include_case_studies", "balanced_argumentation", "risk_mitigation_guidance".
  This is allowed.
Any new verifier.check you create MUST still describe a requirement that is clearly exhibited by the TEXT SNIPPET. 🔁
You are NOT allowed to invent a requirement that the snippet does not follow, just to create a new check name. 🔁

Rules for new verifier names:
- snake_case only [a-z0-9_]
- It must reflect the obligation in desc.
- args must be a JSON object (possibly empty) describing any parameters needed to check this rule, e.g. {"min_items": 3}.

If nothing applies, return an empty JSON list [].

Rules:
- desc must be English, imperative, concrete, verifiable.
- desc should describe the style/voice/safety stance the answer actually exhibits.
- Do NOT include word count, paragraph structure, language choice, or first-person bans here.
  Those are handled elsewhere.
- Do NOT output explanations outside JSON."""


    user_prompt = (
        "GLOBAL OUTLINE (structure only; DO NOT invent rules from this):\n"
        f"{outline_str}\n\n"
        "TEXT SNIPPET (this is the FULL ANSWER content as given to the user;\n"
        "ALL requirements MUST be grounded in this text, do NOT hallucinate):\n"
        f"{answer_clean}\n\n"
        "Extract the global style/tone/safety constraints that the answer is ALREADY following.\n"
        "Return ONLY the JSON list.\n"
    )

    try:
        content = call_chat_completions(
            messages=[
                {"role": "user", "content": user_prompt},
            ],
            system_prompt=system_prompt,
            temperature=0.0,
            max_tokens=800,
            timeout=20,
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