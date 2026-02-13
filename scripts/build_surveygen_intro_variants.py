#!/usr/bin/env python
"""
Build Intro-only A*B variants for SurveyGen:

A (knowledge strategy)
  - explicit: list concrete references (title + url) in instruction constraints.
  - abstract: topic-level guidance without fixed reference list.

B (generation granularity)
  - whole: global constraints only (no block constraints).
  - section: global + block constraints.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import sys
from typing import Any, Dict, Iterable, List, Set, Tuple


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _iter_jsonl(path: pathlib.Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _resolve_pdf_url(meta: Dict[str, Any]) -> str:
    if not isinstance(meta, dict):
        return ""
    for key in ("pdfUrl", "pdf_url", "pdf"):
        val = _safe_text(meta.get(key))
        if val:
            return val
    open_access = meta.get("openAccessPdf")
    if isinstance(open_access, dict):
        val = _safe_text(open_access.get("url"))
        if val:
            return val
    doi = _safe_text(meta.get("doi"))
    if doi:
        return f"https://doi.org/{doi}"
    return _safe_text(meta.get("url"))


def _collect_citation_usage(sections: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    usage: Dict[str, Dict[str, Any]] = {}
    for sec in sections:
        sec_idx = sec.get("index")
        for para in sec.get("paragraphs", []) or []:
            seen_in_para: Set[str] = set()
            for cit in para.get("citations", []) or []:
                ref_id = _safe_text(cit.get("ref_id"))
                if not ref_id or ref_id in seen_in_para:
                    continue
                seen_in_para.add(ref_id)
                if ref_id not in usage:
                    usage[ref_id] = {"count": 0, "section_indices": set()}
                usage[ref_id]["count"] += 1
                if sec_idx is not None:
                    usage[ref_id]["section_indices"].add(sec_idx)
    return usage


def _build_intro_evidence(survey: Dict[str, Any], intro_sections: List[Dict[str, Any]]) -> Dict[str, Any]:
    metadata = survey.get("metadata", {}) or {}
    references = survey.get("references", []) or []
    corpus_id = _safe_text(survey.get("corpusId"))
    usage = _collect_citation_usage(intro_sections)

    evidence_by_ref: Dict[str, Dict[str, Any]] = {}
    for ref in references:
        ref_id = _safe_text(ref.get("ref_id"))
        if not ref_id:
            continue
        meta = ref.get("metadata", {}) or {}
        evidence_by_ref[ref_id] = {
            "title": _safe_text(ref.get("title")) or _safe_text(meta.get("title")),
            "doi": _safe_text(meta.get("doi")),
            "pdfUrl": _resolve_pdf_url(meta),
            "url": _safe_text(meta.get("url")),
            "paperId": _safe_text(meta.get("paperId")),
            "corpusId": _safe_text(meta.get("corpusId")) or _safe_text(ref.get("matched_paper_id")),
            "citationCount": meta.get("citationCount"),
            "referenceCount": meta.get("referenceCount"),
            "fieldsOfStudy": meta.get("fieldsOfStudy") or [],
            "second_level_reference_ids": ref.get("referenced_works") or [],
            "citation_count_in_intro": int((usage.get(ref_id) or {}).get("count", 0)),
            "cited_in_intro_sections": sorted((usage.get(ref_id) or {}).get("section_indices", set())),
        }

    return {
        "survey": {
            "sample_id": "",
            "corpusId": corpus_id,
            "paperId": _safe_text(metadata.get("paperId")),
            "title": _safe_text(metadata.get("title")),
            "doi": _safe_text(metadata.get("doi")),
            "url": _safe_text(metadata.get("url")),
            "year": metadata.get("year"),
            "fieldsOfStudy": metadata.get("fieldsOfStudy") or [],
            "keywords": metadata.get("keywords") or [],
            "topics": metadata.get("topics") or [],
        },
        "meta": {
            "focus": "introduction_only",
            "total_references": len(evidence_by_ref),
            "cited_references_in_intro": sum(1 for v in evidence_by_ref.values() if int(v.get("citation_count_in_intro", 0)) > 0),
            "citation_style_hint": "ref_id",
        },
        "references": evidence_by_ref,
    }


def _paragraph_with_inline_refs(paragraph: Dict[str, Any]) -> str:
    base = _normalize_whitespace(_safe_text(paragraph.get("text")))
    ref_ids: List[str] = []
    seen: Set[str] = set()
    for cit in paragraph.get("citations", []) or []:
        rid = _safe_text(cit.get("ref_id"))
        if not rid or rid in seen:
            continue
        seen.add(rid)
        ref_ids.append(rid)
    if ref_ids:
        return f"{base} {' '.join(f'[{rid}]' for rid in ref_ids)}"
    return base


def _extract_intro_sections(survey: Dict[str, Any]) -> List[Dict[str, Any]]:
    sections = survey.get("sections", []) or []
    out = []
    for sec in sections:
        title = _safe_text(sec.get("title")).lower()
        if title in {"introduction", "intro"} or title.startswith("introduction"):
            out.append(sec)
    if out:
        return out
    if sections:
        return [sections[0]]
    return []


def _build_intro_answer(intro_sections: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for sec in intro_sections:
        title = _safe_text(sec.get("title")) or "Introduction"
        lines.append(f"## {title}")
        lines.append("")
        for para in sec.get("paragraphs", []) or []:
            text = _paragraph_with_inline_refs(para)
            if text:
                lines.append(text)
                lines.append("")
    return _normalize_whitespace("\n".join(lines)) + "\n"


def _build_seed_instruction(metadata: Dict[str, Any]) -> str:
    title = _safe_text(metadata.get("title")) or "Untitled Topic"
    return (
        f"Please write a rigorous Introduction section for a survey on \"{title}\". "
        "The introduction should establish background, explain motivation, "
        "summarize key technical directions, and ground claims with literature evidence.\n"
    )


def _find_survey(path: pathlib.Path, corpus_id: str) -> Dict[str, Any]:
    for obj in _iter_jsonl(path):
        if _safe_text(obj.get("corpusId")) == corpus_id:
            return obj
    return {}


def _write_text(path: pathlib.Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: pathlib.Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    project_root = pathlib.Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.pipeline_runner import run_pipeline_once  # pylint: disable=import-outside-toplevel

    surveys_path = pathlib.Path(args.surveys_file)
    out_root = pathlib.Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    survey = _find_survey(surveys_path, str(args.corpus_id))
    if not survey:
        raise ValueError(f"CorpusId {args.corpus_id} not found in {surveys_path}")

    intro_sections = _extract_intro_sections(survey)
    if not intro_sections:
        raise ValueError(f"No sections found for corpusId={args.corpus_id}")

    sample_id = f"surveygen_{args.corpus_id}"
    sample_root = out_root / sample_id
    sample_root.mkdir(parents=True, exist_ok=True)

    answer_text = _build_intro_answer(intro_sections)
    evidence = _build_intro_evidence(survey, intro_sections)
    evidence["survey"]["sample_id"] = sample_id
    seed_instruction = _build_seed_instruction(survey.get("metadata", {}) or {})

    _write_text(sample_root / "answer.intro.txt", answer_text)
    _write_json(sample_root / "evidence.intro.json", evidence)
    _write_text(sample_root / "seed_instruction.txt", seed_instruction)

    combos: List[Tuple[str, str]] = [
        ("explicit", "whole"),
        ("explicit", "section"),
        ("abstract", "whole"),
        ("abstract", "section"),
    ]

    manifest_path = sample_root / "variants_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for knowledge_style, granularity in combos:
            variant_key = f"{knowledge_style}_{granularity}"
            run_id = f"{sample_id}__intro_{variant_key}"
            pipeline_data_dir = pathlib.Path(args.pipeline_data_dir) / run_id

            result = run_pipeline_once(
                sample_id=run_id,
                original_instruction=seed_instruction,
                model_answer=answer_text,
                base_data_dir=str(pipeline_data_dir),
                evidence=evidence,
                survey_mode=True,
                knowledge_style=knowledge_style,
                explicit_reference_strategy=(
                    "postfix_after_step8" if knowledge_style == "explicit" else "in_prompt_guarded"
                ),
                generation_granularity=granularity,
                survey_block_mode=("llm" if granularity == "section" else "heading"),
                enable_step8_polish=args.enable_step8_polish,
                enable_step7_5=False,
                enable_augment=False,
                enable_augment_diversity=False,
                enable_step8_5=False,
            )

            variant_dir = sample_root / variant_key
            variant_dir.mkdir(parents=True, exist_ok=True)
            _write_text(variant_dir / "answer.txt", answer_text)
            _write_json(variant_dir / "evidence.json", evidence)
            _write_text(variant_dir / "instruction.machine.txt", pathlib.Path(result["prompt_path_machine"]).read_text(encoding="utf-8"))
            _write_text(variant_dir / "instruction.txt", pathlib.Path(result["prompt_path"]).read_text(encoding="utf-8"))

            graph_path = pathlib.Path((result.get("graph_paths") or {}).get("graph_json", ""))
            if graph_path.exists():
                shutil.copyfile(graph_path, variant_dir / "graph.json")

            run_meta = {
                "sample_id": sample_id,
                "variant": variant_key,
                "knowledge_style": knowledge_style,
                "explicit_reference_strategy": result.get("explicit_reference_strategy", ""),
                "generation_granularity": granularity,
                "pipeline_run_id": run_id,
                "pipeline_data_dir": str(pipeline_data_dir),
                "artifacts": {
                    "machine_prompt": result.get("prompt_path_machine", ""),
                    "prompt": result.get("prompt_path", ""),
                    "graph_json": (result.get("graph_paths") or {}).get("graph_json", ""),
                    "eval": result.get("eval_path", ""),
                    "bundle": result.get("bundle_path", ""),
                    "evidence_copy": result.get("evidence_path", ""),
                },
                "llm_statuses": result.get("llm_statuses", []),
                "knowledge_constraints_meta": result.get("knowledge_constraints", {}),
            }
            _write_json(variant_dir / "run_meta.json", run_meta)
            manifest.write(json.dumps(run_meta, ensure_ascii=False) + "\n")

    print("===== INTRO VARIANTS DONE =====")
    print(f"corpus_id       : {args.corpus_id}")
    print(f"output_root     : {sample_root}")
    print(f"variants        : 4")
    print(f"manifest        : {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SurveyGen Intro-only A*B variants.")
    parser.add_argument(
        "--surveys-file",
        type=str,
        default="experiments/SurveyGen/Surveys_Full_Text.jsonl",
        help="Path to Surveys_Full_Text.jsonl",
    )
    parser.add_argument(
        "--corpus-id",
        type=str,
        default="233650333",
        help="Survey corpusId to build Intro variants for.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/SurveyGen/intro_variants",
        help="Output root directory.",
    )
    parser.add_argument(
        "--pipeline-data-dir",
        type=str,
        default="experiments/SurveyGen/intro_variants/_pipeline_artifacts",
        help="Data dir root for per-variant pipeline artifacts.",
    )
    parser.add_argument(
        "--enable-step8-polish",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable step8 polish (default on).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
