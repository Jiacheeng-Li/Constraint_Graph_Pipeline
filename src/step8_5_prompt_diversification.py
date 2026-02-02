"""
Step 8.5 - Prompt Diversification

Purpose
- Generate multiple natural-language prompt variants from the Step 7 machine_prompt.
- Keep every constraint, numeric threshold, and branch logic intact while varying style and form.

Notes
- This step is optional and runs in parallel with Step 8.
- It is designed to improve prompt realism and diversity without changing semantics.
"""

from __future__ import annotations

import random
import re
from typing import Dict, Any, Optional, List, Tuple

from .utils.deepseek_client import call_chat_completions, DeepSeekError


SECTION_HEADINGS = [
    ("mission", "1. MISSION BRIEF"),
    ("global", "2. NON-NEGOTIABLE GLOBAL RULES"),
    ("blueprint", "3. STRUCTURED RESPONSE BLUEPRINT"),
    ("conditions", "4. CURRENT CONDITION ASSUMPTION"),
]

SECTION_COVERAGE_THRESHOLDS = {
    "mission": {"min_ratio": 0.18, "min_hits": 4},
    "global": {"min_ratio": 0.14, "min_hits": 5},
    "blueprint": {"min_ratio": 0.05, "min_hits": 8},
    "conditions": {"min_ratio": 0.10, "min_hits": 3},
}


STYLE_PROFILES: Dict[str, Dict[str, Any]] = {
    "briefing": {
        "label": "Mission briefing",
        "voice": "authoritative, concise, directive",
        "paragraphs": "3-5 short paragraphs",
        "structure_templates": [
            "Open with the mission and deliverable, then state global rules, then walk the stages in order, and end with the default assumptions.",
            "Start with the desired outcome, then the non-negotiable rules, then the stage duties, and conclude with default condition assumptions.",
        ],
    },
    "collaborative": {
        "label": "Collaborative request",
        "voice": "cooperative, clear, pragmatic",
        "paragraphs": "4-6 paragraphs",
        "structure_templates": [
            "Open with a brief request and mission, then explain the global rules, then describe each stage duty in sequence, and finish with the default assumptions.",
            "Start with the mission and context, then summarize constraints, then walk through the staged obligations, and end with condition assumptions.",
        ],
    },
    "question_led": {
        "label": "Question-led request",
        "voice": "inquisitive but still directive",
        "paragraphs": "3-5 paragraphs",
        "structure_templates": [
            "Lead with a question about the mission, then answer it by stating constraints, then cover the stage duties in order, and conclude with default assumptions.",
            "Open with a short question or request, then lay out global rules, then the staged duties, then the default assumptions.",
        ],
    },
    "constraints_first": {
        "label": "Constraints first",
        "voice": "precise, compliance-oriented",
        "paragraphs": "3-6 paragraphs",
        "structure_templates": [
            "Begin with non-negotiable global rules, then restate the mission, then walk the stages in order, then close with default assumptions.",
            "Open with the constraints, follow with the mission, then the stage duties, and end with condition assumptions.",
        ],
    },
    "outcome_first": {
        "label": "Outcome first",
        "voice": "results-driven, practical",
        "paragraphs": "3-6 paragraphs",
        "structure_templates": [
            "Start with the desired output and acceptance expectations, then the mission, then the global rules, then the staged duties, and end with default assumptions.",
            "Lead with the target deliverable, then state constraints, then walk through stages in order, and finish with default assumptions.",
        ],
    },
    "narrative": {
        "label": "Narrative walkthrough",
        "voice": "story-like but still precise",
        "paragraphs": "4-7 paragraphs",
        "structure_templates": [
            "Introduce the mission as a scenario, weave in global rules, then narrate the stages in order, and conclude with default assumptions.",
            "Open with a brief scenario for the mission, then present the constraints, then the stages, and end with default assumptions.",
        ],
    },
}

STYLE_ORDER = list(STYLE_PROFILES.keys())


BASE_SYSTEM_PROMPT = """
You are a senior technical editor. Transform the rigid specification into a realistic user request.
Preserve every rule, numeric threshold, keyword requirement, and IF/THEN/ELSE branch logic.
Do not add, remove, weaken, or strengthen any condition. Do not invent new context.
Keep stage duties in the original order when you describe them.
Never collapse conditional branches into a single path; always state both outcomes explicitly.
Avoid headings, bullet lists, numbering, or Markdown formatting unless explicitly allowed.
"""


def _tokenize_lower(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9']+", text.lower()))


def _section_overlap(source: str, target: str) -> Tuple[float, int]:
    src_tokens = _tokenize_lower(source)
    tgt_tokens = _tokenize_lower(target)
    if not src_tokens:
        return 1.0, 0
    hits = len(src_tokens & tgt_tokens)
    ratio = hits / len(src_tokens)
    return ratio, hits


def _extract_sections(machine_prompt: str) -> Dict[str, str]:
    positions = []
    for name, heading in SECTION_HEADINGS:
        idx = machine_prompt.find(heading)
        if idx != -1:
            positions.append((idx, name, heading))
    positions.sort(key=lambda x: x[0])

    sections: Dict[str, str] = {}
    for i, (start_idx, name, _) in enumerate(positions):
        end_idx = positions[i + 1][0] if i + 1 < len(positions) else len(machine_prompt)
        sections[name] = machine_prompt[start_idx:end_idx].strip()
    return sections


def _has_disallowed_markers(text: str) -> bool:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^#{1,6}\s+", stripped):
            return True
        if re.match(r"^[-*]\s+", stripped):
            return True
        if re.match(r"^\d+[.)]\s+", stripped):
            return True
    return False


def _validate_variant(original: str,
                      candidate: str,
                      *,
                      section_texts: Optional[Dict[str, str]] = None,
                      allow_lists: bool = False) -> Tuple[bool, str]:
    candidate_stripped = (candidate or "").strip()
    if not candidate_stripped:
        return False, "empty_output"

    if not allow_lists and _has_disallowed_markers(candidate_stripped):
        return False, "disallowed_formatting"

    orig_len = len(original.split())
    new_len = len(candidate_stripped.split())
    if new_len == 0:
        return False, "empty_output"

    if orig_len:
        ratio = new_len / max(1, orig_len)

        def _min_ratio(word_count: int) -> float:
            if word_count >= 800:
                return 0.18
            if word_count >= 400:
                return 0.22
            if word_count >= 200:
                return 0.26
            if word_count >= 120:
                return 0.32
            return 0.38

        min_ratio = _min_ratio(orig_len)
        if ratio < min_ratio:
            return False, f"length_ratio({ratio:.2f}<{min_ratio:.2f})"

    missing_sections = []
    if section_texts:
        for section_name, section_text in section_texts.items():
            rule = SECTION_COVERAGE_THRESHOLDS.get(
                section_name,
                {"min_ratio": 0.08, "min_hits": 3},
            )
            ratio, hits = _section_overlap(section_text, candidate_stripped)
            if hits < rule["min_hits"] or ratio < rule["min_ratio"]:
                missing_sections.append(f"{section_name}:{ratio:.2f}|{hits}")

    if missing_sections:
        return False, f"coverage_missing({'/'.join(missing_sections)})"

    return True, ""


def _normalize_style_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (name or "").strip().lower()).strip("_")


def _resolve_style_pool(style_pool: Optional[List[str]]) -> Tuple[List[str], List[str]]:
    if not style_pool:
        return STYLE_ORDER[:], []

    resolved = []
    unknown = []
    for name in style_pool:
        key = _normalize_style_name(name)
        if key in STYLE_PROFILES:
            if key not in resolved:
                resolved.append(key)
        else:
            unknown.append(name)
    if not resolved:
        resolved = STYLE_ORDER[:]
    return resolved, unknown


def _choose_styles(style_pool: List[str], num_variants: int, rng: random.Random) -> List[str]:
    if num_variants <= 0:
        return []
    if num_variants <= len(style_pool):
        return rng.sample(style_pool, num_variants)
    chosen: List[str] = []
    while len(chosen) < num_variants:
        batch = style_pool[:]
        rng.shuffle(batch)
        chosen.extend(batch)
    return chosen[:num_variants]


def _build_user_prompt(style_key: str,
                       structure_hint: str,
                       seed_task: str,
                       machine_prompt: str,
                       *,
                       allow_lists: bool) -> str:
    spec = STYLE_PROFILES[style_key]
    formatting = (
        "Use paragraphs only. Do not use headings, bullet lists, numbering, or Markdown formatting."
        if not allow_lists
        else "Use paragraphs; short lists are allowed only when unavoidable."
    )
    return (
        "Rewrite the document into a natural, realistic user request.\n\n"
        "STYLE PROFILE:\n"
        f"- Name: {spec['label']}\n"
        f"- Voice: {spec['voice']}\n"
        f"- Structure: {structure_hint}\n"
        f"- Length: {spec['paragraphs']}\n\n"
        "Non-negotiable content rules:\n"
        "- Preserve every numeric threshold and explicit keyword requirement verbatim.\n"
        "- Keep every global rule, every stage duty, and every IF/THEN/ELSE branch.\n"
        "- Make both outcomes explicit when a condition appears, using an explicit pattern like:\n"
        "  \"If <condition>, then <branch A requirements>; otherwise, <branch B requirements>.\"\n"
        "- After each dual-branch sentence, add one short sentence stating which branch applies\n"
        "  under the default assumptions when the evaluator is silent.\n"
        "- State the overall default assumption about which conditions are active when unspecified.\n"
        "- Do not invent new context or examples not present in the source.\n\n"
        f"Formatting:\n- {formatting}\n\n"
        "SEED TASK CONTEXT:\n"
        f"{seed_task or 'N/A'}\n\n"
        "ORIGINAL DOCUMENT:\n<<<\n"
        f"{machine_prompt.strip()}\n"
        ">>>\n\n"
        "Return ONLY the rewritten prompt."
    )


def _generate_variant(style_key: str,
                      seed_task: str,
                      machine_prompt: str,
                      *,
                      allow_lists: bool,
                      rng: random.Random,
                      max_attempts: int,
                      section_texts: Dict[str, str],
                      api_key: Optional[str],
                      endpoint: Optional[str],
                      model: Optional[str]) -> Dict[str, Any]:
    spec = STYLE_PROFILES[style_key]
    structure_hint = rng.choice(spec["structure_templates"])
    system_prompt = BASE_SYSTEM_PROMPT
    user_prompt = _build_user_prompt(
        style_key,
        structure_hint,
        seed_task,
        machine_prompt,
        allow_lists=allow_lists,
    )

    attempts = max(1, int(max_attempts))
    last_failure = ""

    for attempt_idx in range(attempts):
        prompt_body = user_prompt
        if attempt_idx > 0:
            prompt_body = (
                user_prompt
                + "\n\nREVISION NOTE: The prior rewrite failed validation because it "
                f"missed required content ({last_failure or 'unspecified'}). "
                "Ensure the mission, all global rules, every stage/branch duty, and the default "
                "condition assumptions are explicitly present in prose."
            )
        try:
            rewritten = call_chat_completions(
                messages=[{"role": "user", "content": prompt_body}],
                system_prompt=system_prompt,
                api_key=api_key,
                endpoint=endpoint,
                model=model,
                temperature=0.7,
                max_tokens=4096,
                timeout=180,
                retries=3,
                retry_backoff_sec=1.5,
            ).strip()
        except DeepSeekError as err:
            return {
                "style": style_key,
                "style_slug": style_key,
                "text": "",
                "used_llm": False,
                "reason": f"llm_error: {err}",
                "attempts": attempt_idx + 1,
            }

        is_valid, failure_code = _validate_variant(
            machine_prompt,
            rewritten,
            section_texts=section_texts,
            allow_lists=allow_lists,
        )
        if is_valid:
            return {
                "style": style_key,
                "style_slug": style_key,
                "text": rewritten,
                "used_llm": True,
                "reason": "success",
                "attempts": attempt_idx + 1,
            }

        last_failure = failure_code

    return {
        "style": style_key,
        "style_slug": style_key,
        "text": "",
        "used_llm": True,
        "reason": f"validation_failed:{last_failure or 'unknown'}",
        "attempts": attempts,
    }


def diversify_instruction_prompt(machine_prompt: str,
                                 seed_task: str = "",
                                 *,
                                 enable: bool = True,
                                 api_key: Optional[str] = None,
                                 endpoint: Optional[str] = None,
                                 model: Optional[str] = None,
                                 num_variants: int = 3,
                                 style_seed: Optional[int] = None,
                                 style_pool: Optional[List[str]] = None,
                                 max_attempts: int = 2,
                                 allow_lists: bool = False) -> Dict[str, Any]:
    """
    Generate multiple natural-language prompt variants.

    Returns dict:
        {
            "variants": [ { "style": ..., "text": ..., "used_llm": bool, "reason": str, "attempts": int }, ... ],
            "used_llm": bool,
            "reason": str,
            "style_seed": int,
            "style_pool": [str],
            "unknown_styles": [str],
            "requested_variants": int,
        }
    """
    cleaned = (machine_prompt or "").strip()
    result: Dict[str, Any] = {
        "variants": [],
        "used_llm": False,
        "reason": "disabled" if not enable else "not_run",
        "style_seed": style_seed,
        "style_pool": [],
        "unknown_styles": [],
        "requested_variants": num_variants,
    }

    if not cleaned:
        result["reason"] = "empty_prompt"
        return result

    if not enable or num_variants <= 0:
        result["reason"] = "disabled"
        return result

    if style_seed is None:
        style_seed = random.randint(0, 2**31 - 1)
    rng = random.Random(style_seed)

    resolved_pool, unknown = _resolve_style_pool(style_pool)
    result["style_seed"] = style_seed
    result["style_pool"] = resolved_pool
    result["unknown_styles"] = unknown

    selected_styles = _choose_styles(resolved_pool, num_variants, rng)
    section_texts = _extract_sections(machine_prompt)

    for style_key in selected_styles:
        variant = _generate_variant(
            style_key,
            seed_task,
            machine_prompt,
            allow_lists=allow_lists,
            rng=rng,
            max_attempts=max_attempts,
            section_texts=section_texts,
            api_key=api_key,
            endpoint=endpoint,
            model=model,
        )
        result["variants"].append(variant)

    result["used_llm"] = any(v.get("used_llm") for v in result["variants"])
    if any(v.get("reason") == "success" for v in result["variants"]):
        result["reason"] = "success"
    else:
        result["reason"] = "validation_failed"
    return result
