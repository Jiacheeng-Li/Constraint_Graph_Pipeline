

"""
硬性（可程序化）约束校验函数集合。

这些函数会被 `verifier_registry` 注册，通过 `verifier_spec` 调用。

所有函数都应该保持一个统一的签名风格：
    def fn(text: str, **kwargs) -> bool
其中 text 是待检文本，kwargs 来自 verifier_spec["args"].

示例 verifier_spec:
{
    "check": "min_word_count",
    "args": {"min_words": 150}
}

然后 runtime 会调用:
    min_word_count(text=resp, min_words=150)

注意：
- 这里的实现是启发式/启用简单规则，后续可以替换为更严格的实现。
- 这些检查必须是“可重复、客观、可自动评分”的。

本模块扩展了硬性约束的类型，参考 Crab 的 unified constraints，包括但不限于：
格式要求、长度要求、关键词出现频次、角色人称限制、输出语言限制、禁止词、列表/小节数等。
"""

import re
from typing import List, Optional, Dict



# ====== 基础统计辅助函数 ======

def _count_words(text: str) -> int:
    tokens = re.findall(r"\w+", text)
    return len(tokens)

def _count_chars(text: str, strip_ws: bool = False) -> int:
    if strip_ws:
        # 仅统计非空白字符
        return len(re.findall(r"\S", text))
    return len(text)

def _count_keyword_occurrences(text: str, keyword: str) -> int:
    # 非重叠出现次数，大小写忽略由调用方提前lower
    return len(re.findall(re.escape(keyword), text))

def _contains_first_person(text: str) -> bool:
    lowered = text.lower()
    first_person_patterns = [r"\bI\b", r"\bme\b", r"\bmy\b", r"\bwe\b", r"\bour\b", r"\bus\b"]
    for pat in first_person_patterns:
        if re.search(pat.lower(), lowered):
            return True
    return False



# ====== 1. 语言 / 视角 / 语气类（可硬规则化的部分）======
def is_english(text: str) -> bool:
    """
    判断文本是否主要为英文。
    启发式：
    - 统计 ASCII 字母字符数量 vs 非ASCII字符数量。
    - 如果非ASCII字符很少（或没有），则视为英文。
    - 如果有非ASCII字符，但ASCII字母数量至少是非ASCII的5倍，也视为英文。
    """
    ascii_letters = re.findall(r"[A-Za-z]", text)
    non_ascii = re.findall(r"[^\x00-\x7F]", text)
    if not non_ascii:
        return True
    return len(ascii_letters) >= 5 * len(non_ascii)

def forbid_first_person(text: str) -> bool:
    """
    要求避免第一人称主语（"I", "me", "my", "we", "our"）。
    用于类似“保持第三人称客观叙述”的硬性限制。
    返回 True 表示通过（即没有第一人称）。
    """
    first_person_patterns = [r"\bI\b", r"\bme\b", r"\bmy\b", r"\bwe\b", r"\bour\b", r"\bus\b"]
    lowered = text.lower()
    for pat in first_person_patterns:
        if re.search(pat.lower(), lowered):
            return False
    return True

def forbid_words(text: str, banned: List[str]) -> bool:
    """
    检查文本中是否出现禁用词/禁用表达（如攻击性词汇、侮辱性词汇、粗俗词汇等）。
    返回 True 表示文本不包含这些词。
    """
    lowered = text.lower()
    return all(bw.lower() not in lowered for bw in banned)


# ====== 2. 长度 / 结构 / 列表格式 ======

def min_word_count(text: str, min_words: int) -> bool:
    return _count_words(text) >= int(min_words)

def max_word_count(text: str, max_words: int) -> bool:
    return _count_words(text) <= int(max_words)

def min_char_count(text: str, min_chars: int, ignore_whitespace: bool = False) -> bool:
    if ignore_whitespace:
        return _count_chars(text, strip_ws=True) >= int(min_chars)
    return _count_chars(text, strip_ws=False) >= int(min_chars)

def must_list_n_subpoints(text: str, n: int) -> bool:
    """
    至少出现 n 个要点/条目式列举（-, *, 1., 2) ...）。
    适用于“列出不少于N条建议 / 风险 / 改进点”。
    """
    bullet_pattern = re.compile(r"(^\s*(?:[-*]|\d+[\.)])\s+.+)", re.MULTILINE)
    bullets = bullet_pattern.findall(text)
    return len(bullets) >= int(n)

def has_sections(text: str, sections: List[str]) -> bool:
    """
    要求输出显式包含特定小节/标题提示（Opening / Body / Conclusion 等）。
    我们用子串匹配作为启发式。
    """
    lowered = text.lower()
    return all(sec.lower() in lowered for sec in sections)

def min_numbered_items(text: str, n: int) -> bool:
    """
    要求出现至少 n 个编号点（1.,2.,3. 或 1) 2) 3)）。
    用法类似“给我三个行动建议并编号”。
    """
    pattern = re.compile(r"(^\s*\d+[\.)]\s+.+)", re.MULTILINE)
    items = pattern.findall(text)
    return len(items) >= int(n)

def min_paragraphs(text: str, min_paras: int) -> bool:
    """
    要求文本至少包含 min_paras 个自然段（空行分隔）。
    """
    paragraphs = [p for p in re.split(r"\n\s*\n+", text.strip()) if p.strip()]
    return len(paragraphs) >= int(min_paras)

def bullet_style_consistent(text: str, marker: str) -> bool:
    """
    要求所有项目符号行使用相同的 marker（'-' 或 '*'）。
    """
    bullets = re.findall(r"^\s*([-*])\s+.+", text, flags=re.MULTILINE)
    if not bullets:
        return False
    return all(b == marker for b in bullets)

def decimal_places(text: str, places: int) -> bool:
    """
    要求出现的所有小数位数一致且等于 places。
    """
    matches = re.findall(r"\b\d+\.(\d+)\b", text)
    if not matches:
        return False
    return all(len(m) == int(places) for m in matches)


# ====== 3. 关键词出现频率 / 主题覆盖 ======

def must_include_keywords(text: str, keywords: List[str]) -> bool:
    lowered = text.lower()
    return all(kw.lower() in lowered for kw in keywords)

def keyword_min_frequency(text: str, keyword: str, min_count: int) -> bool:
    """
    检查某个关键术语是否至少出现 min_count 次。
    用于类似 Crab 里那种“必须多次强调某核心术语（如 space race / space exploration）”的硬性约束。
    """
    return _count_keyword_occurrences(text.lower(), keyword.lower()) >= int(min_count)

def must_cover_topics(text: str, topics: List[str]) -> bool:
    """
    要求文本中必须分别提及 topics 列表里的所有主题。
    例如 ["geopolitical impact", "economic competition", "technological leadership"].
    这类似 Crab 里“覆盖关键维度/角度”的显式检查。
    """
    lowered = text.lower()
    return all(topic.lower() in lowered for topic in topics)


# ====== 4. 输出语言 / 输出形式 ======

def require_language(text: str, lang: str) -> bool:
    """
    粗暴语言检测：
    - lang == "en" 时，复用 is_english
    - lang == "zh" 时，要求至少存在一定数量的中文字符
    - 其他语言后续可以扩展
    """
    if lang == "en":
        return is_english(text)
    if lang == "zh":
        # 启发式：是否至少包含10个汉字（[一-鿿]）
        zh_chars = re.findall(r"[\u4e00-\u9fff]", text)
        return len(zh_chars) >= 10
    # 默认：先视为通过
    return True

def must_end_with_template(text: str, template_suffix: str) -> bool:
    """
    检查文本是否以特定模板式结尾。
    用于诸如“最后一句必须是某种总结性声明/安全免责声明/建议性落句”。
    返回 True 表示结尾满足要求。
    """
    return text.strip().lower().endswith(template_suffix.strip().lower())
