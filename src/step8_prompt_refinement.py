"""
Step 8 - Prompt Refinement

Purpose
- Optionally rewrite the rigid machine_prompt from Step 7 into fluent paragraphs so evaluation prompts resemble natural user requests.

Design guardrails
- Post-process only: never invent, delete, or reorder obligations.
- Validation-first: reject rewrites that collapse too much content or produce empty text.
- Configurable: callers can disable the polish pass (CLI flag or PIPELINE_ENABLE_STEP8 env) when cost, latency, or determinism are priorities.

Workflow overview
1. Run DeepSeek with a detailed editing persona that forbids Markdown lists and preserves all IF/THEN/ELSE logic verbatim.
2. Compare token counts and section coverage to catch degenerate or incomplete rewrites (with a retry).
3. Fall back to the original machine_prompt when disabled, empty, or invalid.
"""

from __future__ import annotations

import re
from typing import Dict, Any, Optional

from .utils.deepseek_client import call_chat_completions, DeepSeekError


REQUIRED_HEADINGS = [
    "SYSTEM INSTRUCTIONS",
    "SURVEY GENERATION INSTRUCTIONS",
    "MISSION BRIEF",
    "SURVEY TASK BRIEF",
    "NON-NEGOTIABLE GLOBAL RULES",
    "SURVEY-LEVEL CONSTRAINTS",
    "CONSTRAINT SUMMARY (BY TYPE)",
    "CONSTRAINT SUMMARY (BY PRIORITY)",
    "CONSTRAINT SUMMARY BY TYPE",
    "CONSTRAINT SUMMARY BY PRIORITY",
    "STRUCTURED RESPONSE BLUEPRINT",
    "SECTION-BY-SECTION SYNTHESIS PLAN",
    "RESPONSE BLUEPRINT",
    "CURRENT CONDITION ASSUMPTION",
    "DEFAULT BRANCH ASSUMPTIONS",
    "DEFAULT CONDITION ASSUMPTION",
    "EVALUATION NOTICE",
]

SECTION_HEADINGS = [
    ("mission", [
        "1. MISSION BRIEF",
        "1. MISSION BRIEF & DELIVERABLE",
        "1. TASK BRIEF",
        "1. SURVEY TASK BRIEF",
        "SURVEY TASK BRIEF",
    ]),
    ("global", [
        "2. NON-NEGOTIABLE GLOBAL RULES",
        "2. SURVEY-LEVEL CONSTRAINTS",
        "SURVEY-LEVEL CONSTRAINTS",
        "2. CONSTRAINT SUMMARY (BY TYPE)",
        "2. CONSTRAINT SUMMARY (BY PRIORITY)",
        "2. CONSTRAINT SUMMARY BY TYPE",
        "2. CONSTRAINT SUMMARY BY PRIORITY",
    ]),
    ("blueprint", [
        "3. STRUCTURED RESPONSE BLUEPRINT",
        "3. SECTION-BY-SECTION SYNTHESIS PLAN",
        "SECTION-BY-SECTION SYNTHESIS PLAN",
        "3. RESPONSE BLUEPRINT",
        "3. RESPONSE BLUEPRINT & REMAINING RULES",
    ]),
    ("conditions", [
        "4. CURRENT CONDITION ASSUMPTION",
        "4. DEFAULT CONDITION ASSUMPTION",
        "4. DEFAULT BRANCH ASSUMPTIONS",
        "DEFAULT BRANCH ASSUMPTIONS",
    ]),
]

SECTION_COVERAGE_THRESHOLDS = {
    # Each tuple uses a lenient overlap ratio to avoid false negatives while still catching drops.
    "mission": {"min_ratio": 0.12, "min_hits": 3},
    "global": {"min_ratio": 0.10, "min_hits": 4},
    "blueprint": {"min_ratio": 0.04, "min_hits": 6},
    "conditions": {"min_ratio": 0.07, "min_hits": 2},
}


def _tokenize_lower(text: str) -> set[str]:
    """Lowercase alphanumeric token set."""
    return set(re.findall(r"[A-Za-z0-9']+", text.lower()))


def _section_overlap(source: str, target: str) -> tuple[float, int]:
    """Jaccard-style overlap ratio between two strings' token sets."""
    src_tokens = _tokenize_lower(source)
    tgt_tokens = _tokenize_lower(target)
    if not src_tokens:
        return 1.0, 0
    hits = len(src_tokens & tgt_tokens)
    ratio = hits / len(src_tokens)
    return ratio, hits


def _extract_sections(machine_prompt: str) -> Dict[str, str]:
    """
    Slice the machine prompt into the major sections we expect to preserve.
    Returns a dict keyed by section name with the text span for that section.
    """
    positions = []
    prompt_lower = machine_prompt.lower()
    for name, headings in SECTION_HEADINGS:
        candidates = headings if isinstance(headings, list) else [headings]
        best_idx = -1
        best_heading = ""
        for heading in candidates:
            idx = prompt_lower.find(heading.lower())
            if idx != -1 and (best_idx == -1 or idx < best_idx):
                best_idx = idx
                best_heading = heading
        if best_idx != -1:
            positions.append((best_idx, name, best_heading))
    positions.sort(key=lambda x: x[0])

    # Fallback: if heading aliases did not match, infer sections from numbered heading lines.
    if not positions:
        matches = list(re.finditer(r"(?m)^\s*\d+\.\s+[^\n]+", machine_prompt))
        inferred_names = ["mission", "global", "blueprint", "conditions"]
        for i, m in enumerate(matches[: len(inferred_names)]):
            positions.append((m.start(), inferred_names[i], m.group(0).strip()))
        positions.sort(key=lambda x: x[0])

    sections: Dict[str, str] = {}
    for i, (start_idx, name, heading) in enumerate(positions):
        end_idx = positions[i + 1][0] if i + 1 < len(positions) else len(machine_prompt)
        sections[name] = machine_prompt[start_idx:end_idx].strip()
    return sections


def _validate_polish(original: str,
                     polished: str,
                     *,
                     section_texts: Optional[Dict[str, str]] = None) -> tuple[bool, str]:
    """
    Run lightweight validations on the polished prompt to ensure it remains faithful
    and follows the natural-language formatting rules.
    Returns (is_valid, failure_reason_code). failure_reason_code is empty when valid.
    """
    polished_stripped = (polished or "").strip()
    if not polished_stripped:
        return False, "empty_output"

    orig_len = len(original.split())
    new_len = len(polished_stripped.split())
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

    # Coverage heuristic: ensure each major section from the machine prompt is still represented.
    missing_sections = []
    if section_texts:
        for section_name, section_text in section_texts.items():
            rule = SECTION_COVERAGE_THRESHOLDS.get(
                section_name,
                {"min_ratio": 0.08, "min_hits": 3},
            )
            ratio, hits = _section_overlap(section_text, polished_stripped)
            if hits < rule["min_hits"] or ratio < rule["min_ratio"]:
                missing_sections.append(f"{section_name}:{ratio:.2f}|{hits}")

    if missing_sections:
        return False, f"coverage_missing({'/'.join(missing_sections)})"

    return True, ""


def refine_instruction_prompt(machine_prompt: str,
                              seed_task: str = "",
                              *,
                              enable: bool = True,
                              api_key: Optional[str] = None,
                              endpoint: Optional[str] = None,
                              model: Optional[str] = None,
                              max_attempts: int = 2) -> Dict[str, Any]:
    """
    Optionally rewrite machine_prompt for better readability via deepseek.

    Returns dict:
        {
            "text": <final prompt string>,
            "used_llm": bool,
            "reason": "success" | "disabled" | "empty_prompt" | "validation_failed" | "llm_error: ...",
            "attempts": <int>,
        }
    """
    cleaned = (machine_prompt or "").strip()
    result = {
        "text": machine_prompt,
        "used_llm": False,
        "reason": "disabled" if not enable else "not_run",
        "attempts": 0,
    }

    if not cleaned:
        result["reason"] = "empty_prompt"
        return result

    if not enable:
        return result

    sections_for_validation = _extract_sections(machine_prompt)
    has_conditions_section = bool((sections_for_validation.get("conditions") or "").strip())

    condition_coverage_rule = (
        "- Preserve the default assumption about which conditions are active when nothing is specified."
        if has_conditions_section
        else "- Do not add new condition-assumption statements if the source does not include them."
    )
    condition_checklist_rule = (
        "every stage/branch obligation, and the default condition assumptions. If anything is missing,"
        if has_conditions_section
        else "every stage/branch obligation. If anything is missing,"
    )
    paragraph_order_hint = (
        "(4) the default assumption about which conditions are active when the evaluator is silent."
        if has_conditions_section
        else "(4) a concise closing paragraph that preserves source constraints without adding new assumptions."
    )

    system_prompt = (
        f"""
        You are a senior technical editor who specializes in transforming rigid specification
        documents into natural instructions, the way a human user would describe a task to a model.

        Rewrite the provided document so it reads like cohesive prose, not a template. Preserve
        every rule, numeric threshold, IF / THEN / ELSE branch, and the original stage order.
        Make the obligations sound like narrative requirements embedded in sentences rather than
        explicit checklists. Keep the semantics identical: do not add, remove, weaken, or strengthen
        any condition, and do not move content between branches or stages. All numbers, keywords,
        and decision logic must stay exactly as given.

        Non-negotiable coverage:
        - Restate the mission/deliverable clearly.
        - Include every global rule.
        - Walk through every stage duty and IF/THEN/ELSE branch, with both triggers and default paths.
        {condition_coverage_rule}

        Formatting expectations:
        - Avoid bullet lists, numbered outlines, or Stage labels such as “Stage 1 - …”.
        - Do not use Markdown features (no **bold**, no headings, no code blocks).
        - Express structure through paragraphs and connective sentences.
        - You may refer to stages or scenarios in natural sentences (e.g., “Begin by …”, “Next,
          examine …”), but do not write literal list markers.

        Before finalizing, run a quick checklist to confirm you kept the mission, every global rule,
        {condition_checklist_rule}
        rewrite until all required parts are present.
        """
    )

    user_prompt = (
        f"""
        Rewrite the following instruction so it sounds like a natural request from a human requester.
        Replace the mechanical numbering and bullet lists with flowing paragraphs. Do not introduce
        Markdown formatting or literal list markers. Integrate each stage’s duties and branch logic
        into sentences that still make the conditional structure obvious. When the source contains
        a conditional branch, clearly describe both outcomes in prose (e.g., “If X occurs, then …;
        otherwise …”) so evaluators can still see the selection logic. Keep every rule,
        numerical value, keyword requirement, and branch condition intact, just expressed more
        conversationally and coherently. Write 4-6 paragraphs in this order: (1) mission/deliverable;
        (2) non-negotiable global rules; (3) stage-by-stage duties including every IF/THEN/ELSE
        branch with both triggers and default paths; {paragraph_order_hint} If any of these sections are empty in
        the source, explicitly say so instead of omitting them.

        SEED TASK CONTEXT:
        """
        f"{seed_task or 'N/A'}\n\n"
        "ORIGINAL DOCUMENT:\n<<<\n"
        f"{machine_prompt.strip()}\n"
        ">>>\n\n"
        "Return ONLY the rewritten document, as plain text paragraphs."
    )

    attempts = max(1, int(max_attempts))
    last_failure = ""
    polished_final = ""

    for attempt_idx in range(attempts):
        prompt_body = user_prompt
        if attempt_idx > 0:
            prompt_body = (
                user_prompt
                + "\n\nREVISION NOTE: The prior rewrite failed validation because it "
                f"missed required content ({last_failure or 'unspecified'}). Ensure the mission, "
                + (
                    "all global rules, every stage/branch duty, and the default condition assumptions are explicitly present in prose."
                    if has_conditions_section
                    else "all global rules and every stage/branch duty are explicitly present in prose; do not add new condition assumptions."
                )
            )
        try:
            polished = call_chat_completions(
                messages=[{"role": "user", "content": prompt_body}],
                system_prompt=system_prompt,
                api_key=api_key,
                endpoint=endpoint,
                model=model,
                temperature=0.6,
                max_tokens=4096,
                timeout=180,
                retries=3,
                retry_backoff_sec=1.5,
            ).strip()
        except DeepSeekError as err:
            result["reason"] = f"llm_error: {err}"
            return result

        result["attempts"] = attempt_idx + 1
        is_valid, failure_code = _validate_polish(
            machine_prompt,
            polished,
            section_texts=sections_for_validation,
        )
        if is_valid:
            polished_final = polished
            break

        last_failure = failure_code

    if not polished_final:
        result["reason"] = f"validation_failed:{last_failure or 'unknown'}"
        return result

    result["text"] = polished_final
    result["used_llm"] = True
    result["reason"] = "success"
    return result
