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
2. Compare token counts to catch degenerate rewrites.
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


def _validate_polish(original: str, polished: str) -> tuple[bool, str]:
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

    return True, ""


def refine_instruction_prompt(machine_prompt: str,
                              seed_task: str = "",
                              *,
                              enable: bool = True,
                              api_key: Optional[str] = None,
                              endpoint: Optional[str] = None,
                              model: Optional[str] = None) -> Dict[str, Any]:
    """
    Optionally rewrite machine_prompt for better readability via deepseek.

    Returns dict:
        {
            "text": <final prompt string>,
            "used_llm": bool,
            "reason": "success" | "disabled" | "empty_prompt" | "validation_failed" | "llm_error: ...",
        }
    """
    cleaned = (machine_prompt or "").strip()
    result = {
        "text": machine_prompt,
        "used_llm": False,
        "reason": "disabled" if not enable else "not_run",
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

        Conditional structure requirements:
        - For each selection or branch in the source, explicitly restate the trigger and the
          obligations for both the triggered path and the default/otherwise path.
        - Use natural sentences such as “If the reviewer demands formal citations, then …,
          otherwise …” so the branching logic remains obvious.
        - Never delete a branch, merge distinct paths, or blur their responsibilities.

        Formatting expectations:
        - Avoid bullet lists, numbered outlines, or Stage labels such as “Stage 1 - …”.
        - Do not use Markdown features (no **bold**, no headings, no code blocks).
        - Express structure through paragraphs and connective sentences.
        - You may refer to stages or scenarios in natural sentences (e.g., “Begin by …”, “Next,
          examine …”), but do not write literal list markers.

        The final document should feel like a detailed brief written by a human expert while
        retaining the full logical skeleton and constraints from the source text.
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
        conversationally and coherently.

        SEED TASK CONTEXT:
        """
        f"{seed_task or 'N/A'}\n\n"
        "ORIGINAL DOCUMENT:\n<<<\n"
        f"{machine_prompt.strip()}\n"
        ">>>\n\n"
        "Return ONLY the rewritten document, as plain text paragraphs."
    )

    try:
        polished = call_chat_completions(
            messages=[{"role": "user", "content": user_prompt}],
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

    is_valid, failure_code = _validate_polish(machine_prompt, polished)
    if not is_valid:
        result["reason"] = f"validation_failed:{failure_code}"
        return result

    result["text"] = polished
    result["used_llm"] = True
    result["reason"] = "success"
    return result
