"""
Step 3.5 - Survey Knowledge Constraints (optional, survey-mode only)

Purpose
- Derive knowledge-grounding constraints from SurveyGen metadata/evidence.
- Keep backward compatibility by returning empty outputs unless survey_mode=True.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from .graph_schema import ConstraintNode


_BRACKET_CITE_RE = re.compile(r"\[\s*([A-Za-z0-9][A-Za-z0-9_.:-]*)\s*\]")
_GENERIC_INTENT_RE = re.compile(r"^(paragraph|section)\s*\d+$", re.I)
_VISUAL_INTENT_RE = re.compile(
    r"^\s*(figure|fig\.?|table|supplementary\s+figure|supplementary\s+table)\b",
    re.I,
)
_STOPWORDS = {
    "the", "and", "for", "with", "from", "into", "that", "this", "these", "those",
    "overview", "introduction", "background", "discussion", "conclusion", "figure", "table",
    "results", "methods", "analysis", "survey", "systematic", "review", "meta", "study",
}


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _extract_evidence_ref_map(evidence: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    refs = evidence.get("references")
    if isinstance(refs, dict):
        return {str(k): v for k, v in refs.items() if isinstance(v, dict)}
    if isinstance(evidence, dict):
        if all(isinstance(v, dict) for v in evidence.values()):
            return {str(k): v for k, v in evidence.items() if isinstance(v, dict)}
    return {}


def _title_keywords(title: str, limit: int = 3) -> List[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", (title or "").lower())
    out: List[str] = []
    for t in tokens:
        if t in _STOPWORDS:
            continue
        if t in out:
            continue
        out.append(t)
        if len(out) >= limit:
            break
    return out


def _extract_core_keywords(evidence: Dict[str, Any], limit: int = 4) -> List[str]:
    survey_meta = evidence.get("survey", {}) if isinstance(evidence, dict) else {}
    candidates: List[str] = []
    for key in ("keywords", "topics", "fieldsOfStudy"):
        candidates.extend(_as_list(survey_meta.get(key)))

    title = str(survey_meta.get("title") or "").strip()
    if not candidates and title:
        candidates.extend(_title_keywords(title, limit=limit))

    out: List[str] = []
    for c in candidates:
        c_norm = c.strip()
        if not c_norm or c_norm.lower() in _STOPWORDS:
            continue
        if c_norm in out:
            continue
        out.append(c_norm)
        if len(out) >= limit:
            break
    return out


def _top_high_impact_ref_ids(ref_map: Dict[str, Dict[str, Any]], limit: int = 12) -> List[str]:
    scored: List[tuple[int, str]] = []
    for rid, payload in ref_map.items():
        raw = payload.get("citationCount", 0)
        try:
            score = int(raw) if raw is not None else 0
        except Exception:
            score = 0
        scored.append((score, rid))
    scored.sort(reverse=True)
    return [rid for _, rid in scored[:limit] if rid]


def _refs_from_text(text: str, ref_id_lookup: Dict[str, str]) -> List[str]:
    seen: List[str] = []
    for m in _BRACKET_CITE_RE.finditer(text or ""):
        raw_rid = str(m.group(1) or "").strip()
        if not raw_rid:
            continue
        rid = ref_id_lookup.get(raw_rid.lower())
        if not rid:
            continue
        if rid not in seen:
            seen.append(rid)
    return seen


def _is_visual_intent(intent: str) -> bool:
    return bool(_VISUAL_INTENT_RE.match((intent or "").strip()))


def extract_knowledge_constraints(
    *,
    segmentation: Dict[str, Any],
    evidence: Dict[str, Any],
    survey_mode: bool = False,
    knowledge_style: str = "abstract",
    generation_granularity: str = "section",
) -> Dict[str, Any]:
    """
    Returns:
    {
      "global_constraints": List[ConstraintNode],
      "block_constraints": Dict[str, List[ConstraintNode]],
      "meta": Dict[str, Any],
    }
    """
    if not survey_mode or not evidence:
        return {"global_constraints": [], "block_constraints": {}, "meta": {"enabled": False}}

    style = (knowledge_style or "abstract").strip().lower()
    if style not in {"abstract", "explicit"}:
        style = "abstract"
    granularity = (generation_granularity or "section").strip().lower()
    if granularity not in {"section", "whole"}:
        granularity = "section"
    global_only = granularity == "whole"

    ref_map = _extract_evidence_ref_map(evidence)
    ref_id_lookup = {str(rid).lower(): str(rid) for rid in ref_map.keys()}
    core_keywords = _extract_core_keywords(evidence)
    high_impact = _top_high_impact_ref_ids(ref_map)

    global_nodes: List[ConstraintNode] = []
    block_nodes: Dict[str, List[ConstraintNode]] = {}

    # Knowledge global: ensure topic anchoring.
    if core_keywords:
        selected = core_keywords[: min(3, len(core_keywords))]
        global_nodes.append(
            ConstraintNode(
                cid="KG1",
                desc=(
                    "Anchor the survey to core domain concepts and explicitly cover: "
                    + ", ".join(selected)
                    + "."
                ),
                scope="global",
                verifier_spec={"check": "must_include_keywords", "args": {"keywords": selected}},
                trace_to=None,
                derived_from="step3_5_knowledge",
            )
        )

    # Knowledge global: prioritize stronger references from evidence.
    if len(high_impact) >= 3:
        global_nodes.append(
            ConstraintNode(
                cid="KG2",
                desc=(
                    "Ground key claims using high-impact sources from the evidence pool "
                    "(prioritize seminal or highly cited works)."
                ),
                scope="global",
                verifier_spec={
                    "check": "citation_refs_from_allowed_set",
                    "args": {
                        "allowed_ref_ids": high_impact,
                        "min_unique": min(3, len(high_impact)),
                        "max_out_of_set": 999,
                    },
                },
                trace_to=None,
                derived_from="step3_5_knowledge",
            )
        )

    cited_ref_ids_global: List[str] = []
    ref_first_block: Dict[str, str] = {}

    # Section-level knowledge constraints (survey headings + section-relevant citations).
    for blk in segmentation.get("blocks", []):
        bid = str(blk.get("block_id") or "").strip()
        intent = str(blk.get("intent") or "").strip()
        span = str(blk.get("text_span") or "")
        if not bid:
            continue
        if _is_visual_intent(intent):
            continue

        locals_for_block: List[ConstraintNode] = []

        sec_ref_ids = _refs_from_text(span, ref_id_lookup)
        if sec_ref_ids:
            for rid in sec_ref_ids:
                if rid not in cited_ref_ids_global:
                    cited_ref_ids_global.append(rid)
                if rid not in ref_first_block:
                    ref_first_block[rid] = bid
        if global_only:
            continue

        if intent and not _GENERIC_INTENT_RE.match(intent):
            title_kws = _title_keywords(intent, limit=1)
            if title_kws:
                locals_for_block.append(
                    ConstraintNode(
                        cid=f"{bid}_K1",
                        desc=f"Keep this section focused on its heading theme: {intent}.",
                        scope="local",
                        verifier_spec={"check": "must_include_keywords", "args": {"keywords": title_kws}},
                        trace_to=bid,
                        derived_from="step3_5_knowledge",
                    )
                )

        if sec_ref_ids:
            locals_for_block.append(
                ConstraintNode(
                    cid=f"{bid}_K2",
                    desc="Ground this section with citations from its section-relevant evidence subset.",
                    scope="local",
                    verifier_spec={
                        "check": "citation_refs_from_allowed_set",
                        "args": {
                            "allowed_ref_ids": sec_ref_ids,
                            "min_unique": 1,
                            "max_out_of_set": 999,
                        },
                    },
                    trace_to=bid,
                    derived_from="step3_5_knowledge",
                )
            )

        if locals_for_block:
            block_nodes[bid] = locals_for_block

    # Ensure every cited ref_id appears as an explicit knowledge constraint with article metadata.
    if not global_only:
        for idx, rid in enumerate(cited_ref_ids_global, start=1):
            payload = ref_map.get(rid, {}) if isinstance(ref_map.get(rid), dict) else {}
            title = str(payload.get("title") or "").strip()
            doi = str(payload.get("doi") or "").strip()
            url = str(payload.get("url") or payload.get("pdfUrl") or "").strip()
            block_id = ref_first_block.get(rid, "")
            if not block_id:
                continue

            article_tag_parts = []
            if title:
                article_tag_parts.append(f"title: {title}")
            if doi:
                article_tag_parts.append(f"doi: {doi}")
            if url:
                article_tag_parts.append(f"url: {url}")
            article_tag = "; ".join(article_tag_parts) if article_tag_parts else "metadata unavailable"

            block_nodes.setdefault(block_id, []).append(
                ConstraintNode(
                    cid=f"{block_id}_KR{idx}",
                    desc=f"Include citation [{rid}] and align it with evidence metadata ({article_tag}).",
                    scope="local",
                    verifier_spec={
                        "check": "citation_refs_from_allowed_set",
                        "args": {
                            "allowed_ref_ids": [rid],
                            "min_unique": 1,
                            "max_out_of_set": 999,
                        },
                    },
                    trace_to=block_id,
                    derived_from="step3_5_knowledge",
                )
            )

    explicit_reference_list: List[Dict[str, str]] = []
    if style == "explicit" and cited_ref_ids_global:
        rendered_rows: List[str] = []
        for rid in cited_ref_ids_global:
            payload = ref_map.get(rid, {}) if isinstance(ref_map.get(rid), dict) else {}
            title = str(payload.get("title") or "").strip() or "(untitled)"
            url = str(payload.get("url") or payload.get("pdfUrl") or "").strip()
            doi = str(payload.get("doi") or "").strip()
            if not url and doi:
                url = f"https://doi.org/{doi}"
            explicit_reference_list.append(
                {
                    "ref_id": rid,
                    "title": title,
                    "url": url,
                }
            )
            if url:
                rendered_rows.append(f"[{rid}] {title} ({url})")
            else:
                rendered_rows.append(f"[{rid}] {title}")

        if rendered_rows:
            global_nodes.append(
                ConstraintNode(
                    cid=f"KG{len(global_nodes)+1}",
                    desc=(
                        "Use the following evidence references explicitly when writing the introduction: "
                        + "; ".join(rendered_rows)
                        + "."
                    ),
                    scope="global",
                    verifier_spec={
                        "check": "citation_refs_from_allowed_set",
                        "args": {
                            "allowed_ref_ids": cited_ref_ids_global,
                            "min_unique": min(len(cited_ref_ids_global), max(3, min(10, len(cited_ref_ids_global)))),
                            "max_out_of_set": 999,
                        },
                    },
                    trace_to=None,
                    derived_from="step3_5_knowledge",
                )
            )

    if cited_ref_ids_global:
        global_nodes.append(
            ConstraintNode(
                cid=f"KG{len(global_nodes)+1}",
                desc=(
                    "Cover all cited evidence ids observed in the reference answer; "
                    f"expected distinct citation ids: {len(cited_ref_ids_global)}."
                ),
                scope="global",
                verifier_spec={
                    "check": "citation_refs_from_allowed_set",
                    "args": {
                        "allowed_ref_ids": cited_ref_ids_global,
                        "min_unique": len(cited_ref_ids_global),
                        "max_out_of_set": 999,
                    },
                },
                trace_to=None,
                derived_from="step3_5_knowledge",
            )
        )

    return {
        "global_constraints": global_nodes,
        "block_constraints": block_nodes,
        "meta": {
            "enabled": True,
            "instruction_knowledge_style": style,
            "generation_granularity": granularity,
            "core_keywords": core_keywords,
            "high_impact_ref_count": len(high_impact),
            "blocks_with_knowledge_constraints": len(block_nodes),
            "cited_ref_count_from_answer": len(cited_ref_ids_global),
            "cited_ref_ids_from_answer": cited_ref_ids_global,
            "explicit_reference_list": explicit_reference_list,
            "cited_reference_catalog": {
                rid: {
                    "title": (ref_map.get(rid) or {}).get("title", ""),
                    "doi": (ref_map.get(rid) or {}).get("doi", ""),
                    "url": (ref_map.get(rid) or {}).get("url", ""),
                    "pdfUrl": (ref_map.get(rid) or {}).get("pdfUrl", ""),
                    "paperId": (ref_map.get(rid) or {}).get("paperId", ""),
                    "corpusId": (ref_map.get(rid) or {}).get("corpusId", ""),
                    "citationCount": (ref_map.get(rid) or {}).get("citationCount"),
                    "referenceCount": (ref_map.get(rid) or {}).get("referenceCount"),
                    "fieldsOfStudy": (ref_map.get(rid) or {}).get("fieldsOfStudy", []),
                }
                for rid in cited_ref_ids_global
            },
        },
    }
