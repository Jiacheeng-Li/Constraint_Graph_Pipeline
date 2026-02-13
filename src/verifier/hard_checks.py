"""
Hard verifier checks (deterministic / scriptable).
"""

import re
from typing import List


_CONTRACTION_RE = re.compile(
    r"\b(?:"
    r"don't|doesn't|didn't|can't|cannot|won't|wouldn't|shouldn't|couldn't|mustn't|"
    r"i'm|i've|i'll|i'd|it's|it'd|it'll|that's|there's|we're|we've|we'll|we'd|"
    r"they're|they've|they'll|they'd|you're|you've|you'll|you'd|isn't|aren't|wasn't|weren't|"
    r"hasn't|haven't|hadn't"
    r")\b",
    re.I,
)
_EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF]")
_NUMERIC_CITE_RE = re.compile(r"\[\s*\d{1,4}(?:\s*[-,;]\s*\d{1,4})*\s*\]")
_AUTHOR_YEAR_CITE_RE = re.compile(
    r"\([A-Z][A-Za-z.\-]+(?:\s+et\s+al\.)?,\s*\d{4}[a-z]?\)"
)
_REF_ID_CITE_RE = re.compile(r"\[([A-Za-z][A-Za-z0-9_-]*\d+)\]")


def _count_words(text: str) -> int:
    return len(re.findall(r"\w+", text))


def _count_chars(text: str, strip_ws: bool = False) -> int:
    if strip_ws:
        return len(re.findall(r"\S", text))
    return len(text)


def _count_keyword_occurrences(text: str, keyword: str) -> int:
    return len(re.findall(re.escape(keyword), text))


def _all_citation_markers(text: str) -> List[str]:
    markers: List[str] = []
    markers.extend(m.group(0) for m in _NUMERIC_CITE_RE.finditer(text))
    markers.extend(m.group(0) for m in _AUTHOR_YEAR_CITE_RE.finditer(text))
    markers.extend(m.group(0) for m in _REF_ID_CITE_RE.finditer(text))
    return markers


def _extract_ref_id_citations(text: str) -> List[str]:
    return [m.group(1).strip().lower() for m in _REF_ID_CITE_RE.finditer(text)]


def is_english(text: str) -> bool:
    ascii_letters = re.findall(r"[A-Za-z]", text)
    non_ascii = re.findall(r"[^\x00-\x7F]", text)
    if not non_ascii:
        return True
    return len(ascii_letters) >= 5 * len(non_ascii)


def forbid_first_person(text: str) -> bool:
    lowered = text.lower()
    first_person_patterns = [r"\bi\b", r"\bme\b", r"\bmy\b", r"\bwe\b", r"\bour\b", r"\bus\b"]
    for pat in first_person_patterns:
        if re.search(pat, lowered):
            return False
    return True


def forbid_words(text: str, banned: List[str]) -> bool:
    lowered = text.lower()
    return all(bw.lower() not in lowered for bw in banned)


def min_word_count(text: str, min_words: int) -> bool:
    return _count_words(text) >= int(min_words)


def max_word_count(text: str, max_words: int) -> bool:
    return _count_words(text) <= int(max_words)


def word_count_between(text: str, min_words: int, max_words: int) -> bool:
    wc = _count_words(text)
    return int(min_words) <= wc <= int(max_words)


def word_count_around(text: str, center: int, tolerance_pct: float = 0.15) -> bool:
    wc = _count_words(text)
    center_i = int(center)
    tol = max(0.0, float(tolerance_pct))
    lo = int(center_i * (1.0 - tol))
    hi = int(center_i * (1.0 + tol))
    return lo <= wc <= hi


def min_char_count(text: str, min_chars: int, ignore_whitespace: bool = False) -> bool:
    if ignore_whitespace:
        return _count_chars(text, strip_ws=True) >= int(min_chars)
    return _count_chars(text, strip_ws=False) >= int(min_chars)


def must_list_n_subpoints(text: str, n: int) -> bool:
    bullet_pattern = re.compile(r"(^\s*(?:[-*]|\d+[\.)])\s+.+)", re.MULTILINE)
    bullets = bullet_pattern.findall(text)
    return len(bullets) >= int(n)


def has_sections(text: str, sections: List[str]) -> bool:
    lowered = text.lower()
    return all(sec.lower() in lowered for sec in sections)


def require_sections(text: str, sections: List[str]) -> bool:
    return has_sections(text=text, sections=sections)


def min_numbered_items(text: str, n: int) -> bool:
    pattern = re.compile(r"(^\s*\d+[\.)]\s+.+)", re.MULTILINE)
    items = pattern.findall(text)
    return len(items) >= int(n)


def min_paragraphs(text: str, min_paras: int) -> bool:
    paragraphs = [p for p in re.split(r"\n\s*\n+", text.strip()) if p.strip()]
    return len(paragraphs) >= int(min_paras)


def heading_levels_only(text: str, levels: List[int]) -> bool:
    allowed = {int(x) for x in levels}
    found = [
        len(m.group(1))
        for m in re.finditer(r"^(#{1,6})\s+\S", text, flags=re.MULTILINE)
    ]
    if not found:
        return False
    return all(level in allowed for level in found)


def bullet_style_consistent(text: str, marker: str) -> bool:
    bullets = re.findall(r"^\s*([-*])\s+.+", text, flags=re.MULTILINE)
    if not bullets:
        return False
    return all(b == marker for b in bullets)


def decimal_places(text: str, places: int) -> bool:
    matches = re.findall(r"\b\d+\.(\d+)\b", text)
    if not matches:
        return False
    return all(len(m) == int(places) for m in matches)


def date_format(text: str, style: str) -> bool:
    iso_matches = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text)
    long_matches = re.findall(
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}\b",
        text,
    )
    if style == "yyyy-mm-dd":
        return bool(iso_matches) and not bool(long_matches)
    if style == "mon dd, yyyy":
        return bool(long_matches) and not bool(iso_matches)
    return False


def must_include_keywords(text: str, keywords: List[str]) -> bool:
    lowered = text.lower()
    return all(kw.lower() in lowered for kw in keywords)


def keyword_min_frequency(text: str, keyword: str, min_count: int) -> bool:
    return _count_keyword_occurrences(text.lower(), keyword.lower()) >= int(min_count)


def must_cover_topics(text: str, topics: List[str]) -> bool:
    lowered = text.lower()
    return all(topic.lower() in lowered for topic in topics)


def require_language(text: str, lang: str) -> bool:
    if lang == "en":
        return is_english(text)
    if lang == "zh":
        zh_chars = re.findall(r"[\u4e00-\u9fff]", text)
        return len(zh_chars) >= 10
    return True


def must_end_with_template(text: str, template_suffix: str) -> bool:
    return text.strip().lower().endswith(template_suffix.strip().lower())


def forbid_emojis(text: str) -> bool:
    return not bool(_EMOJI_RE.search(text))


def avoid_contractions(text: str) -> bool:
    return not bool(_CONTRACTION_RE.search(text))


def forbid_symbol(text: str, symbol: str) -> bool:
    return symbol not in text


def citation_style(text: str, style: str) -> bool:
    numeric = bool(_NUMERIC_CITE_RE.search(text))
    author_year = bool(_AUTHOR_YEAR_CITE_RE.search(text))
    ref_id = bool(_REF_ID_CITE_RE.search(text))

    if style == "numeric":
        return numeric and not author_year and not ref_id
    if style == "author_year":
        return author_year and not numeric and not ref_id
    if style == "ref_id":
        return ref_id
    return False


def min_citation_markers(text: str, min_count: int) -> bool:
    return len(_all_citation_markers(text)) >= int(min_count)


def min_distinct_citations(text: str, min_unique: int) -> bool:
    distinct = {m.lower() for m in _all_citation_markers(text)}
    return len(distinct) >= int(min_unique)


def citation_refs_from_allowed_set(
    text: str,
    allowed_ref_ids: List[str],
    min_unique: int = 1,
    max_out_of_set: int = 0,
) -> bool:
    allowed = {x.strip().lower() for x in allowed_ref_ids if x and str(x).strip()}
    if not allowed:
        return False

    found = _extract_ref_id_citations(text)
    if not found:
        return False

    found_set = set(found)
    in_set = {x for x in found_set if x in allowed}
    out_of_set = {x for x in found_set if x not in allowed}
    return len(in_set) >= int(min_unique) and len(out_of_set) <= int(max_out_of_set)
