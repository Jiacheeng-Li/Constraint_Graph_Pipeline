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
    "MISSION BRIEF",
    "NON-NEGOTIABLE GLOBAL RULES",
    "STRUCTURED RESPONSE BLUEPRINT",
    "CURRENT CONDITION ASSUMPTION",
    "EVALUATION NOTICE",
]

SECTION_HEADINGS = [
    ("mission", "1. MISSION BRIEF"),
    ("global", "2. NON-NEGOTIABLE GLOBAL RULES"),
    ("blueprint", "3. STRUCTURED RESPONSE BLUEPRINT"),
    ("conditions", "4. CURRENT CONDITION ASSUMPTION"),
]

SECTION_COVERAGE_THRESHOLDS = {
    # Each tuple uses a lenient overlap ratio to avoid false negatives while still catching drops.
    "mission": {"min_ratio": 0.18, "min_hits": 4},
    "global": {"min_ratio": 0.14, "min_hits": 5},
    "blueprint": {"min_ratio": 0.05, "min_hits": 8},
    "conditions": {"min_ratio": 0.10, "min_hits": 3},
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
    for name, heading in SECTION_HEADINGS:
        idx = machine_prompt.find(heading)
        if idx != -1:
            positions.append((idx, name, heading))
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

    system_prompt = (
        """
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
        - Preserve the default assumption about which conditions are active when nothing is specified.

        Formatting expectations:
        - Avoid bullet lists, numbered outlines, or Stage labels such as “Stage 1 - …”.
        - Do not use Markdown features (no **bold**, no headings, no code blocks).
        - Express structure through paragraphs and connective sentences.
        - You may refer to stages or scenarios in natural sentences (e.g., “Begin by …”, “Next,
          examine …”), but do not write literal list markers.

        Before finalizing, run a quick checklist to confirm you kept the mission, every global rule,
        every stage/branch obligation, and the default condition assumptions. If anything is missing,
        rewrite until all four parts are present.
        """
    )

    user_prompt = (
        """
        Rewrite the following instruction so it sounds like a natural request from a human requester.
        Replace the mechanical numbering and bullet lists with flowing paragraphs. Do not introduce
        Markdown formatting or literal list markers. Integrate each stage’s duties and branch logic
        into sentences that still make the conditional structure obvious. When the source contains
        a conditional branch, clearly describe both outcomes in prose (e.g., “If X occurs, then …;
        otherwise …”) so evaluators can still see the selection logic. Keep every rule,
        numerical value, keyword requirement, and branch condition intact, just expressed more
        conversationally and coherently. Write 4-6 paragraphs in this order: (1) mission/deliverable;
        (2) non-negotiable global rules; (3) stage-by-stage duties including every IF/THEN/ELSE
        branch with both triggers and default paths; (4) the default assumption about which
        conditions are active when the evaluator is silent. If any of these sections are empty in
        the source, explicitly say so instead of omitting them.

        SEED TASK CONTEXT:
        """
        f"{seed_task or 'N/A'}\n\n"
        "ORIGINAL DOCUMENT:\n<<<\n"
        f"{machine_prompt.strip()}\n"
        ">>>\n\n"
        "Return ONLY the rewritten document, as plain text paragraphs."
    )

    sections_for_validation = _extract_sections(machine_prompt)
    attempts = max(1, int(max_attempts))
    last_failure = ""
    polished_final = ""

    for attempt_idx in range(attempts):
        prompt_body = user_prompt
        if attempt_idx > 0:
            prompt_body = (
                user_prompt
                + "\n\nREVISION NOTE: The prior rewrite failed validation because it "
                f"missed required content ({last_failure or 'unspecified'}). Ensure the mission,"
                " all global rules, every stage/branch duty, and the default condition assumptions"
                " are explicitly present in prose."
            )
        try:
            polished = call_chat_completions(
                messages=[{"role": "user", "content": prompt_body}],
                system_prompt=system_prompt,
                api_key=api_key,
                endpoint=endpoint,
                model=model,
                temperature=0.15,
                max_tokens=min(4096, int(len(machine_prompt) * 1.2) + 512),
                timeout=120,
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
