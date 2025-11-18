"""
Step 4 - Block Back-translation

Purpose / 目标
- For each segmented block, infer the abstract obligations it satisfies so we can reuse them as reusable constraints.
- Distinguish between parallel requirements ("AND") and sequential reasoning ("sub-chain") to preserve local logic.

Workflow
1. Iterate over segmentation["blocks"] from Step 2.
2. Provide DeepSeek with the seed task, a structural outline, and the exact block text (cleaned but not summarized).
3. Ask for JSON describing:
   - logic: "AND" or "sub-chain"
   - constraints: desc + verifier spec + short evidence snippet
4. Normalize/whitelist verifier specs, limit per-block counts, and deduplicate overlapping obligations.
5. If an LLM call fails, fall back to deterministic neutral-tone + length/numbered-list constraints and mark the logic as AND.

Outputs / 输出
- block_constraints: mapping block_id -> List[ConstraintNode] (scope="local", trace_to=block_id, derived_from="step4").
- block_logic: mapping block_id -> "AND" or "sub-chain".

Why this matters
- These nodes become the "real-path" requirements for Step 5 selections and feed directly into the graph.
- Expressing obligations at pattern-level prevents leaking exemplar-specific facts into downstream prompts.

Dependencies
- graph_schema.ConstraintNode / BlockSpec dataclasses.
- utils.deepseek_client.call_chat_completions and parsing helpers.
- utils.text_clean.make_snippet + summarize_blocks_outline for safe prompt context.
"""

import json
import os
import time
import random
import re
from .utils.deepseek_client import call_chat_completions, DeepSeekError
from typing import Dict, Any, List, Tuple

from .graph_schema import ConstraintNode, BlockSpec
from .utils.parsing import extract_constraints, safe_json_load
from .utils.text_clean import make_snippet, summarize_blocks_outline, clip
_STEP4_RAND_SEED = os.getenv("STEP4_RAND_SEED")
if _STEP4_RAND_SEED is not None:
    try:
        random.seed(int(_STEP4_RAND_SEED))
    except Exception:
        random.seed(_STEP4_RAND_SEED)

# --- Block-level constraint config ---
# Hard caps per requirements: at most 4 constraints per block with no more than 1 hard rule.
_STEP4_BLOCK_MAX_CONSTRAINTS = 4
_STEP4_BLOCK_MAX_HARD = 1

_NUM_LIST = re.compile(r"^\s*(?:\d+[\.\)]|[a-z][\.\)])\s+", re.I | re.M)
_BULLET_LIST = re.compile(r"^\s*[-*]\s+", re.M)
_SEQUENCE_CUES = re.compile(r"\b(?:first|second|third|then|next|afterward|finally|step\s*\d+)\b", re.I)
_PAR_SPLIT = re.compile(r"(?:\r?\n){2,}")

# --- Additional regex helpers and block-suitable hard-constraint helpers ---
_BULLET_HEAD = re.compile(r"^\s*([-*])\s+\S", re.M)
_DECIMAL_NUM = re.compile(r"\b\d+\.(\d+)\b")

def _detect_bullet_list(snippet: str) -> Tuple[int, str]:
    """Return (count, dominant_marker) for bullet-style list lines in the snippet."""
    markers = _BULLET_HEAD.findall(snippet)
    if not markers:
        return 0, ""
    counts: Dict[str, int] = {}
    for m in markers:
        counts[m] = counts.get(m, 0) + 1
    # dominant marker by frequency
    dominant = max(counts.items(), key=lambda x: x[1])[0]
    return len(markers), dominant

def _detect_decimal_places(snippet: str) -> int:
    """Detect a dominant decimal place count if numbers like 1.23 appear; return 0 if none."""
    matches = _DECIMAL_NUM.findall(snippet)
    if len(matches) < 2:
        return 0
    freq: Dict[int, int] = {}
    for m in matches:
        l = len(m)
        freq[l] = freq.get(l, 0) + 1
    places, count = max(freq.items(), key=lambda x: x[1])
    # require at least two occurrences with the same decimal length
    return places if count >= 2 else 0

def _count_paragraphs(snippet: str) -> int:
    """Approximate paragraph count by splitting on blank lines."""
    parts = [p for p in _PAR_SPLIT.split(snippet) if p.strip()]
    return len(parts)

def _estimate_word_count(text: str) -> int:
    tokens = re.findall(r"\w+", text)
    zh = re.findall(r"[\u4e00-\u9fff]", text)
    return len(tokens) + len(zh)

def _extract_max_json(s: str) -> str:
    """
    Extract the largest plausible JSON object substring from s (from first '{' to last '}').
    Returns the substring or original s if not found.
    """
    try:
        start = s.index("{")
        end = s.rindex("}")
        if start < end:
            return s[start:end+1]
    except ValueError:
        pass
    return s

def _heuristic_logic(snippet: str, llm_logic: str) -> str:
    """
    Heuristic audit for logic type: returns 'sub-chain' or 'AND'.
    Uses numbered/bulleted lists and sequence cue words.
    """
    score_chain = 0
    score_and = 0
    if _SEQUENCE_CUES.search(snippet):
        score_chain += 2
    if _NUM_LIST.search(snippet):
        score_chain += 2
    if _BULLET_LIST.search(snippet):
        score_and += 1
    # dense parallel clauses: semicolons or many commas in a single paragraph
    if snippet.count(";") >= 2:
        score_and += 1
    # decide
    if score_chain - score_and >= 2:
        return "sub-chain"
    if score_and - score_chain >= 2:
        return "AND"
    # fallback to LLM
    return "sub-chain" if str(llm_logic).lower().startswith("sub") else "AND"

_VERIFIER_WHITELIST = {
    "tone_neutral_llm_judge",
    "tone_negative_llm_judge",
    "non_extremeness_judge",
    "role_consistency_judge",
    "actionability_judge",
    "forbid_first_person",
    "min_word_count",
    "must_list_n_subpoints",
    "min_numbered_items",
    "must_include_keywords",
    "keyword_min_frequency",
    "must_cover_topics",
    "min_char_count",
    "require_language",
    "has_sections",
    "must_end_with_template",
}

_VERIFIER_SYNONYM_MAP = {
    # LLM free-form names -> (canonical_name, args_patch)
    "must_present_two_examples": ("must_list_n_subpoints", {"min_items": 2}),
    "must_present_three_examples": ("must_list_n_subpoints", {"min_items": 3}),
    "must_list_two_items": ("must_list_n_subpoints", {"min_items": 2}),
    "must_use_numbered_list": ("min_numbered_items", {"min_items": 2}),
}

_HARD_VERIFIERS = {
    # Only keep verifiers that Step4's deterministic logic actually emits.
    "min_word_count",
    "min_paragraphs",
    "bullet_style_consistent",
    "decimal_places",
    "min_numbered_items",
    "must_list_n_subpoints",
}

def _is_hard_verifier(check: str) -> bool:
    """Heuristic: decide whether a verifier is hard (structure/content) or soft (tone/preferences)."""
    if not check:
        return False
    # custom:* 默认视为软性
    if check.startswith("custom:"):
        return False
    if check in _HARD_VERIFIERS:
        return True
    # tone_* / preference_* 一律当软性
    if check.startswith("tone_") or check.startswith("preference_"):
        return False
    # 一些特定 judge 也视作软性
    if check in {"non_extremeness_judge", "role_consistency_judge", "actionability_judge"}:
        return False
    # 其余默认软性（更保守）
    return False

def _map_verifier(check: str, args: Dict[str, Any]) -> Tuple[str, Dict[str, Any], bool]:
    """
    Map a possibly custom verifier name to whitelist, else prefix 'custom:'.
    Returns (mapped_check, mapped_args, is_custom)
    """
    if check in _VERIFIER_WHITELIST:
        return check, args or {}, False
    if check in _VERIFIER_SYNONYM_MAP:
        base, patch = _VERIFIER_SYNONYM_MAP[check]
        merged = dict(args or {})
        merged.update(patch)
        return base, merged, False
    return f"custom:{check}", args or {}, True

def _dedup_and_cap(items: List[Dict[str, Any]], cap: int = 5) -> List[Dict[str, Any]]:
    """
    Deduplicate by normalized desc and cap to 'cap' items.
    Priority:
      1) programmatic (verifier not custom)
      2) with evidence_str present
      3) longer desc
    """
    def norm(s: str) -> str:
        s = s.lower()
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    seen = set()
    scored = []
    for it in items:
        desc = it.get("desc", "")
        key = norm(desc)
        if not desc or key in seen:
            continue
        seen.add(key)
        is_custom = it.get("_is_custom", False)
        ev = 1 if it.get("evidence_str") else 0
        scored.append(( (0 if not is_custom else 1, -ev, -len(desc)), it ))
    scored.sort(key=lambda x: x[0])
    out = [it for _, it in scored[:cap]]
    return out

def _find_numbered_items(snippet: str) -> int:
    # approximate count of numbered items
    return len(re.findall(r"^\s*(?:\d+[\.\)]|[a-z][\.\)])\s+", snippet, re.I | re.M))

def _guess_block_min_words(snippet: str) -> int:
    wc = _estimate_word_count(snippet)
    return max(30, int(wc * 0.6))

def _call_deepseek_block_constraints(block: BlockSpec,
                                     seed_task: str,
                                     segmentation: Dict[str, Any]) -> str:
    """
    让 deepseek 针对单个 block 生成：
    - 该 block 的逻辑类型 (logic: "AND" | "sub-chain")
    - 该 block 实际在做/必须做的可验证要求列表 (constraints: [...])

    注意：
    - 这些约束是针对该 block 的“局部硬性/软性要求”，但必须是**抽象的、泛化的**，而不是把原文内容逐句复述。
    - desc 不能泄露具体答案细节（人名、年份、专有名词、具体结论等），而应概括为通用写作义务，如：
      * "Provide at least one real-world example to illustrate the main point."
      * "Compare two contrasting viewpoints before giving a judgment."
      * "Summarize the key causes and their effects in a neutral tone."
    - 内容、语气、结构上的要求都可以出现，但要写成**模式级别（pattern-level）**，而不是复制当前 block 的具体事实。

    每条约束还需提供一个 evidence 字段（极短原文短语）以便后续定位证据，但 evidence 只用于内部溯源，不应在 desc 中泄露具体细节。
    """
    # 使用 text_clean 保留原文语义，仅做空白规整；不默认截断
    block_text_clean = make_snippet(block.text_span)

    # 作为安全阀：如果块内容极端长，显式硬截断（唯一会丢信息的地方）
    if len(block_text_clean) > 12000:
        block_text_clean = clip(block_text_clean, 12000)

    # 给模型一点上下文：整个回答的结构是什么（只是结构提示，不是证据）
    outline_str = summarize_blocks_outline(segmentation)

    system_prompt = (
        "You are an instruction reverse-engineer for LOCAL blocks of an answer.\n"
        "Goal: For ONE block of an assistant's answer, infer what concrete, verifiable OBLIGATIONS that block satisfies,\n"
        "but write them in an ABSTRACT, PATTERN-LEVEL way, not as a paraphrase of the specific content.\n\n"
        "CRITICAL RULES:\n"
        "- You MUST base all obligations on evidence that appears inside the TEXT SNIPPET.\n"
        "- You MUST NOT simply summarize or restate the snippet.\n"
        "- You MUST NOT encode specific facts, names, dates, numbers, or conclusions from the snippet into the description.\n"
        "- Write obligations as generic patterns that could apply to many similar blocks.\n"
        "- It is OK to mention structure or tone (e.g., \"present at least one real-world example\", \"keep a neutral analytical tone\"),\n"
        "  but do NOT leak the actual examples or detailed conclusions.\n\n"
        "Examples of acceptable obligation styles (not tied to specific content):\n"
        "- \"Provide at least one real-world example to illustrate the main claim.\"\n"
        "- \"Compare two contrasting cases before giving a judgment.\"\n"
        "- \"Summarize the key causes and effects in a neutral, analytical tone.\"\n"
        "- \"Highlight practical implications or next steps for the user.\"\n\n"
        "You must return ONLY valid JSON. No commentary. No markdown. No code fences.\n\n"
        "JSON schema:\n"
        "{\n"
        "  \"logic\": \"AND\" | \"sub-chain\",\n"
        "  \"constraints\": [\n"
        "    {\n"
        "      \"desc\": \"<imperative, abstract, verifiable obligation; no specific names/dates>\",\n"
        "      \"verifier\": {\"check\": \"<snake_case>\", \"args\": { }},\n"
        "      \"evidence\": \"<very short phrase copied from the text>\"\n"
        "    }\n"
        "  ]\n"
        "}\n"
    )

    user_prompt = (
        "GLOBAL OUTLINE (only high-level; DO NOT invent from this):\n"
        f"{outline_str}\n\n"
        "SEED TASK (the global assignment):\n"
        f"{seed_task.strip()}\n\n"
        "CURRENT BLOCK YOU ARE ANALYZING:\n"
        f"BLOCK ID: {block.block_id}\n"
        f"BLOCK INTENT: {block.intent}\n\n"
        "TEXT SNIPPET (the ONLY allowed evidence; base ALL constraints ONLY on this text):\n"
        f"{block_text_clean}\n\n"
        "Now return ONLY the JSON.\n"
    )

    try:
        content = call_chat_completions(
            messages=[
                {"role": "user", "content": user_prompt},
            ],
            system_prompt=system_prompt,
            temperature=0.0,
            max_tokens=900,
            timeout=20,
            retries=2,
        ).strip()
        return content
    except DeepSeekError:
        return "{\n  \"logic\": \"AND\",\n  \"constraints\": []\n}"

def _parse_block_llm_result(raw_str: str) -> Dict[str, Any]:
    """
    将 deepseek 返回的原始字符串解析为结构化 dict：
    {
      "logic": "AND" | "sub-chain",
      "constraints": [ {"desc":..., "verifier":{check,args}}, ... ]
    }

    解析策略：
    1. safe_json_load(raw_str) —— 这会尝试提取最像 JSON 的片段并解析。
    2. 从解析结果里拿到 logic 和 constraints 字段；无则兜底。
    3. constraints 字段再用 extract_constraints() 走一遍，以统一字段名 (cid/desc/verifier_spec)。
       注意：extract_constraints 期望的是一个 JSON list 或者 {constraints:[...]}
       所以我们把 {"constraints": [...]} 丢给它即可。
    """
    try:
        parsed_any = safe_json_load(_extract_max_json(raw_str))
    except ValueError:
        return {}

    if not isinstance(parsed_any, dict):
        return {}

    logic_val = parsed_any.get("logic", "AND")
    raw_constraints_field = {
        "constraints": parsed_any.get("constraints", [])
    }

    # 使用 extract_constraints() 做统一清洗，得到 list[dict]
    cons_items = extract_constraints(json.dumps(raw_constraints_field, ensure_ascii=False))

    return {
        "logic": logic_val,
        "constraints": cons_items,
    }

def _fallback_block_constraints(block: BlockSpec) -> Dict[str, Any]:
    """
    如果 LLM 失败：
    - 动态兜底：若文本疑似编号列表，则给出 'min_numbered_items'；否则用 'min_word_count'（基于片段长度）。
    - 始终包含中立语气。
    - logic = "AND"。
    """
    snippet = block.text_span or ""
    min_items = _find_numbered_items(snippet)
    min_words = _guess_block_min_words(snippet)
    constraints = [
        {
            "desc": "Maintain a neutral, analytical tone without emotional or inflammatory language.",
            "verifier_spec": {"check": "tone_neutral_llm_judge", "args": {}},
        }
    ]
    if min_items >= 2:
        constraints.append({
            "desc": f"Provide a numbered list with at least {min_items} items.",
            "verifier_spec": {"check": "min_numbered_items", "args": {"min_items": min_items}},
        })
    else:
        constraints.append({
            "desc": f"Provide a sufficiently detailed explanation with at least {min_words} words.",
            "verifier_spec": {"check": "min_word_count", "args": {"min_words": min_words}},
        })
    return {"logic": "AND", "constraints": constraints}


# --- Deterministic rule-based local hard constraint extractor ---
def _extract_local_hard_constraints(block: BlockSpec) -> List[Dict[str, Any]]:
    """Rule-based extractor for block-level hard constraints.

    These constraints are derived purely from the block text via deterministic functions,
    not from the LLM. They should capture simple, verifiable obligations such as
    minimum word count or minimum number of numbered items, as well as paragraph, bullet, and decimal formatting.
    """
    snippet = block.text_span or ""
    snippet_stripped = snippet.strip()
    if not snippet_stripped:
        return []

    items: List[Dict[str, Any]] = []

    # Length-based local constraint
    min_words = _guess_block_min_words(snippet)
    items.append({
        "desc": f"Provide a sufficiently detailed explanation in this block with at least {min_words} words.",
        "verifier_spec": {"check": "min_word_count", "args": {"min_words": min_words}},
        "evidence_str": "",
        "_is_custom": False,
        "origin": "local_rule",
    })

    # Numbered list constraint if the block appears to use numbered items
    min_items = _find_numbered_items(snippet)
    if min_items >= 2:
        items.append({
            "desc": f"Provide a numbered list in this block with at least {min_items} items.",
            "verifier_spec": {"check": "min_numbered_items", "args": {"min_items": min_items}},
            "evidence_str": "",
            "_is_custom": False,
            "origin": "local_rule",
        })

    # Paragraph-level structure constraint if there are multiple logical paragraphs
    para_count = _count_paragraphs(snippet)
    if para_count >= 2:
        items.append({
            "desc": f"Organize this block into at least {para_count} logical paragraphs.",
            "verifier_spec": {"check": "min_paragraphs", "args": {"min_paras": para_count}},
            "evidence_str": "",
            "_is_custom": False,
            "origin": "local_rule",
        })

    # Bullet-style list constraint if there are multiple bullet lines
    bullet_count, bullet_marker = _detect_bullet_list(snippet)
    if bullet_count >= 2 and bullet_marker:
        items.append({
            "desc": f"Use a consistent bullet style '{bullet_marker}' for list items in this block.",
            "verifier_spec": {"check": "bullet_style_consistent", "args": {"marker": bullet_marker}},
            "evidence_str": "",
            "_is_custom": False,
            "origin": "local_rule",
        })
        items.append({
            "desc": f"Provide a bulleted list in this block with at least {bullet_count} items.",
            "verifier_spec": {"check": "must_list_n_subpoints", "args": {"min_items": bullet_count}},
            "evidence_str": "",
            "_is_custom": False,
            "origin": "local_rule",
        })

    # Decimal formatting constraint if there is a dominant decimal place count
    places = _detect_decimal_places(snippet)
    if places > 0:
        items.append({
            "desc": f"Keep numerical values in this block to {places} decimal places consistently.",
            "verifier_spec": {"check": "decimal_places", "args": {"places": places}},
            "evidence_str": "",
            "_is_custom": False,
            "origin": "local_rule",
        })

    return items

def extract_block_constraints(segmentation: Dict[str, Any],
                              seed_task: str) -> Dict[str, Any]:
    """
    Step4 主入口：
    输入：
      - segmentation: Step2 的输出 {"blocks": [...], "order": [...]}
      - seed_task: Step1 的输出

    输出：
      {
        "block_constraints": {
            "B1": [ConstraintNode, ...],
            "B2": [...],
            ...
        },
        "block_logic": {
            "B1": "AND" | "sub-chain",
            ...
        }
      }
    """

    block_constraints: Dict[str, List[ConstraintNode]] = {}
    block_logic: Dict[str, str] = {}

    for block_dict in segmentation.get("blocks", []):
        block = BlockSpec(
            block_id=block_dict["block_id"],
            intent=block_dict.get("intent", "TBD"),
            text_span=block_dict.get("text_span", ""),
            order_index=block_dict.get("order_index", 0),
        )

        from_fallback = False
        raw_str = _call_deepseek_block_constraints(block, seed_task, segmentation)
        parsed = _parse_block_llm_result(raw_str)
        constraints_list = parsed.get("constraints") if isinstance(parsed, dict) else None
        if not parsed or not constraints_list:
            parsed = _fallback_block_constraints(block)
            constraints_list = parsed.get("constraints", [])
            from_fallback = True

        # Heuristic logic audit (block-level)
        logic_tag = parsed.get("logic", "AND")
        final_logic = _heuristic_logic(block.text_span or "", logic_tag)
        block_logic[block.block_id] = final_logic

        # Normalize, map verifiers, attach evidence text if present, keep local scope
        normalized_items: List[Dict[str, Any]] = []
        for item in constraints_list:
            desc = (item.get("desc") or "").strip()
            if not desc:
                continue
            v = item.get("verifier_spec") or item.get("verifier") or {}
            check_name = (v.get("check") or "").strip()
            args_obj = v.get("args") or {}
            if not check_name:
                continue
            mapped_check, mapped_args, is_custom = _map_verifier(check_name, args_obj)
            evidence_str = (item.get("evidence") or "").strip()
            origin = "local_rule" if from_fallback else "llm"
            normalized_items.append({
                "desc": desc,
                "verifier_spec": {"check": mapped_check, "args": mapped_args},
                "evidence_str": evidence_str,
                "_is_custom": is_custom,
                "origin": origin,
            })

        # Add deterministic local hard constraints derived from the block text itself
        normalized_items.extend(_extract_local_hard_constraints(block))

        # Separate into hard vs soft items based on verifier type (origin-agnostic so LLM + rule-based can mix)
        hard_items: List[Dict[str, Any]] = []
        soft_items: List[Dict[str, Any]] = []
        for it in normalized_items:
            check = it.get("verifier_spec", {}).get("check", "")
            if _is_hard_verifier(check):
                hard_items.append(it)
            else:
                soft_items.append(it)

        # Deduplicate within each group and cap
        hard_items = _dedup_and_cap(hard_items, cap=_STEP4_BLOCK_MAX_CONSTRAINTS)
        soft_items = _dedup_and_cap(soft_items, cap=_STEP4_BLOCK_MAX_CONSTRAINTS)

        max_total = _STEP4_BLOCK_MAX_CONSTRAINTS
        hard_cap = min(_STEP4_BLOCK_MAX_HARD, max_total)

        # Randomly select up to the hard cap first
        if len(hard_items) > hard_cap:
            hard_selected = random.sample(hard_items, hard_cap)
        else:
            hard_selected = list(hard_items)

        soft_slots = max_total - len(hard_selected)
        if soft_slots < 0:
            soft_slots = 0

        if len(soft_items) > soft_slots:
            soft_selected = random.sample(soft_items, soft_slots)
        else:
            soft_selected = list(soft_items)

        combined_selected = hard_selected + soft_selected

        # Fill leftover capacity (if any) with additional soft constraints only
        remaining_slots = max_total - len(combined_selected)
        if remaining_slots > 0:
            soft_leftover = [it for it in soft_items if it not in soft_selected]
            if soft_leftover:
                if len(soft_leftover) > remaining_slots:
                    soft_leftover = random.sample(soft_leftover, remaining_slots)
                combined_selected.extend(soft_leftover)

        # Final filtered items for this block
        filtered_items = combined_selected

        # Materialize ConstraintNode list with local scope and correct trace_to
        local_nodes: List[ConstraintNode] = []
        cid_idx = 1
        for it in filtered_items:
            node = ConstraintNode(
                cid=f"{block.block_id}_C{cid_idx}",
                desc=it["desc"],
                scope="local",  # ensure block-local
                verifier_spec=it["verifier_spec"],
                trace_to=block.block_id,  # ensure trace to this block
                derived_from="step4",
            )
            local_nodes.append(node)
            cid_idx += 1

        block_constraints[block.block_id] = local_nodes

    return {
        "block_constraints": block_constraints,
        "block_logic": block_logic,
    }


if __name__ == "__main__":
    # quick smoke test
    demo_seg = {
        "blocks": [
            {
                "block_id": "B1",
                "intent": "Opening / Context setup",
                "text_span": "The modern space race signals not only technological ambition but geopolitical signaling...",
                "order_index": 0,
            },
            {
                "block_id": "B2",
                "intent": "Main Analysis / Drivers / Examples",
                "text_span": "For example, recent launch programs by nation X and private company Y illustrate...",
                "order_index": 1,
            },
        ],
        "order": ["B1", "B2"],
    }

    demo_seed = (
        "Analyze the geopolitical implications of the modern space race in a neutral analytical tone, "
        "providing real-world examples and forward-looking assessment."
    )

    out = extract_block_constraints(demo_seg, demo_seed)
    print(json.dumps({
        "block_logic": out["block_logic"],
        "block_constraints": {
            bid: [
                {
                    "cid": n.cid,
                    "desc": n.desc,
                    "scope": n.scope,
                    "verifier_spec": n.verifier_spec,
                    "trace_to": n.trace_to,
                    "derived_from": n.derived_from,
                }
                for n in nlist
            ]
            for bid, nlist in out["block_constraints"].items()
        }
    }, indent=2, ensure_ascii=False))
