#!/usr/bin/env python
"""
Build SurveyGen training samples for the pipeline.

Per survey, emit:
  - instruction.txt
  - answer.txt
  - evidence.json
"""

import argparse
import json
import pathlib
import re
import sys
from typing import Any, Dict, Iterable, List, Set, Tuple

_VISUAL_SECTION_RE = re.compile(
    r"^\s*(?:figure|fig\.?|table|image|chart|supplementary\s+(?:figure|fig\.?|table|image)|appendix\s+(?:figure|table))\b",
    re.I,
)


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _clip(text: str, max_len: int) -> str:
    t = text.strip()
    if max_len <= 0 or len(t) <= max_len:
        return t
    return t[: max_len - 3].rstrip() + "..."


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_visual_section_title(title: str) -> bool:
    clean = _safe_text(title)
    if not clean:
        return False
    if _VISUAL_SECTION_RE.match(clean):
        return True
    return not bool(re.search(r"[A-Za-z0-9]", clean))


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
                    usage[ref_id] = {
                        "count": 0,
                        "section_indices": set(),
                        "citation_titles": set(),
                    }
                usage[ref_id]["count"] += 1
                if sec_idx is not None:
                    usage[ref_id]["section_indices"].add(sec_idx)
                cit_title = _safe_text(cit.get("title"))
                if cit_title:
                    usage[ref_id]["citation_titles"].add(cit_title)

    return usage


def _build_evidence(survey: Dict[str, Any]) -> Dict[str, Any]:
    metadata = survey.get("metadata", {}) or {}
    sections = survey.get("sections", []) or []
    references = survey.get("references", []) or []
    corpus_id = _safe_text(survey.get("corpusId"))

    usage = _collect_citation_usage(sections)
    evidence_by_ref: Dict[str, Dict[str, Any]] = {}

    for ref in references:
        ref_id = _safe_text(ref.get("ref_id"))
        if not ref_id:
            continue

        meta = ref.get("metadata", {}) or {}
        title = _safe_text(ref.get("title")) or _safe_text(meta.get("title"))
        doi = _safe_text(meta.get("doi"))
        url = _safe_text(meta.get("url"))
        pdf_url = _resolve_pdf_url(meta)

        ref_usage = usage.get(ref_id, {})
        evidence_by_ref[ref_id] = {
            "title": title,
            "doi": doi,
            "pdfUrl": pdf_url,
            "url": url,
            "paperId": _safe_text(meta.get("paperId")),
            "corpusId": _safe_text(meta.get("corpusId")) or _safe_text(ref.get("matched_paper_id")),
            "citationCount": meta.get("citationCount"),
            "referenceCount": meta.get("referenceCount"),
            "fieldsOfStudy": meta.get("fieldsOfStudy") or [],
            "second_level_reference_ids": ref.get("referenced_works") or [],
            "citation_count_in_survey": int(ref_usage.get("count", 0)),
            "cited_in_sections": sorted(ref_usage.get("section_indices", set())),
        }

    for ref_id, u in usage.items():
        if ref_id in evidence_by_ref:
            continue
        fallback_title = sorted(u.get("citation_titles", set()))
        evidence_by_ref[ref_id] = {
            "title": fallback_title[0] if fallback_title else "",
            "doi": "",
            "pdfUrl": "",
            "url": "",
            "paperId": "",
            "corpusId": "",
            "citationCount": None,
            "referenceCount": None,
            "fieldsOfStudy": [],
            "second_level_reference_ids": [],
            "citation_count_in_survey": int(u.get("count", 0)),
            "cited_in_sections": sorted(u.get("section_indices", set())),
        }

    cited_ref_count = sum(1 for v in evidence_by_ref.values() if int(v.get("citation_count_in_survey", 0)) > 0)
    total_ref_count = len(evidence_by_ref)
    style_hint = "ref_id"

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
            "total_references": total_ref_count,
            "cited_references": cited_ref_count,
            "citation_style_hint": style_hint,
        },
        "references": evidence_by_ref,
    }


def _paragraph_with_inline_refs(
    paragraph: Dict[str, Any],
    inline_ref_ids: bool,
    max_refs_per_paragraph: int,
) -> str:
    base = _normalize_whitespace(_safe_text(paragraph.get("text")))
    if not inline_ref_ids:
        return base

    ref_ids: List[str] = []
    seen: Set[str] = set()
    for cit in paragraph.get("citations", []) or []:
        rid = _safe_text(cit.get("ref_id"))
        if not rid or rid in seen:
            continue
        seen.add(rid)
        ref_ids.append(rid)

    if max_refs_per_paragraph > 0:
        ref_ids = ref_ids[:max_refs_per_paragraph]

    if not ref_ids:
        return base
    return f"{base} {' '.join(f'[{rid}]' for rid in ref_ids)}"


def _build_answer(
    survey: Dict[str, Any],
    inline_ref_ids: bool,
    max_refs_per_paragraph: int,
) -> str:
    sections = survey.get("sections", []) or []
    sorted_sections = sorted(sections, key=lambda s: (s.get("index", 0), _safe_text(s.get("title"))))

    lines: List[str] = []
    for sec in sorted_sections:
        title = _safe_text(sec.get("title")) or f"Section {sec.get('index', '')}".strip()
        if _is_visual_section_title(title):
            continue

        paragraphs = sec.get("paragraphs", []) or []
        if not paragraphs:
            continue
        section_texts: List[str] = []
        for para in paragraphs:
            text = _paragraph_with_inline_refs(
                para,
                inline_ref_ids=inline_ref_ids,
                max_refs_per_paragraph=max_refs_per_paragraph,
            )
            if text:
                section_texts.append(text)

        if not section_texts:
            continue

        lines.append(f"## {title}")
        lines.append("")
        for text in section_texts:
            lines.append(text)
            lines.append("")

    return _normalize_whitespace("\n".join(lines)) + "\n"


def _choose_keywords(metadata: Dict[str, Any], max_keywords: int = 8) -> List[str]:
    kws = metadata.get("keywords") or []
    topics = metadata.get("topics") or []
    out: List[str] = []

    def _push(value: Any) -> None:
        text = _safe_text(value)
        if not text or text in out:
            return
        out.append(text)

    for item in kws:
        _push(item)
        if len(out) >= max_keywords:
            return out

    for item in topics:
        _push(item)
        if len(out) >= max_keywords:
            return out
    return out


def _build_instruction_zh(
    metadata: Dict[str, Any],
    *,
    section_count: int,
    min_unique_refs: int,
    inline_ref_ids: bool,
) -> str:
    title = _safe_text(metadata.get("title")) or "未命名主题"
    abstract = _clip(_safe_text(metadata.get("abstract")), 700)
    fields = ", ".join(_safe_text(x) for x in (metadata.get("fieldsOfStudy") or []) if _safe_text(x))
    keywords = _choose_keywords(metadata, max_keywords=8)
    kw_str = "、".join(keywords) if keywords else "无"

    cite_hint = (
        "引用请使用 evidence.json 里的 ref_id（如 [b12]），并尽量在关键结论后给出引用。"
        if inline_ref_ids
        else "请为关键结论附上文内引用（如 [1] 或 (Author, Year)），并确保可在 evidence.json 对齐来源。"
    )

    lines = [
        f"请围绕“{title}”完成一份多角度研究综述报告。",
        "",
        "任务要求：",
        f"1. 结合背景、核心问题、方法路线、关键发现、争议与局限、未来方向进行系统性调研。",
        f"2. 至少使用 {max(4, min(section_count, 12))} 个结构化小节组织内容，并给出清晰标题。",
        f"3. {cite_hint}",
        f"4. 至少引用 {min_unique_refs} 条不同文献来源。",
        "5. 结尾单独给出“研究空白与后续议题”。",
        "",
        "题目信息：",
        f"- 标题：{title}",
        f"- 学科：{fields or '未知'}",
        f"- 关键词：{kw_str}",
        f"- 摘要：{abstract or '无'}",
    ]
    return "\n".join(lines).strip() + "\n"


def _build_instruction_en(
    metadata: Dict[str, Any],
    *,
    section_count: int,
    min_unique_refs: int,
    inline_ref_ids: bool,
) -> str:
    title = _safe_text(metadata.get("title")) or "Untitled topic"
    abstract = _clip(_safe_text(metadata.get("abstract")), 900)
    fields = ", ".join(_safe_text(x) for x in (metadata.get("fieldsOfStudy") or []) if _safe_text(x))
    keywords = _choose_keywords(metadata, max_keywords=8)
    kw_str = ", ".join(keywords) if keywords else "N/A"

    cite_hint = (
        "Use citation ref_ids from evidence.json (for example [b12]) for evidence-backed claims."
        if inline_ref_ids
        else "Use inline citations (for example [1] or (Author, Year)) and keep them aligned with evidence.json."
    )

    lines = [
        f"Please produce a multi-angle research survey report on: \"{title}\".",
        "",
        "Requirements:",
        "1. Cover background, core problems, methods, key findings, controversies/limitations, and future directions.",
        f"2. Organize the report into at least {max(4, min(section_count, 12))} titled sections.",
        f"3. {cite_hint}",
        f"4. Cite at least {min_unique_refs} distinct sources.",
        "5. End with a dedicated section: \"Research Gaps and Next Questions\".",
        "",
        "Topic context:",
        f"- Title: {title}",
        f"- Fields: {fields or 'Unknown'}",
        f"- Keywords: {kw_str}",
        f"- Abstract: {abstract or 'N/A'}",
    ]
    return "\n".join(lines).strip() + "\n"


def _build_instruction_via_pipeline(
    *,
    sample_id: str,
    seed_instruction: str,
    answer_text: str,
    evidence: Dict[str, Any],
    pipeline_data_dir: str,
    use_step8_polish: bool,
) -> Tuple[str, Dict[str, Any]]:
    """
    Reuse the existing pipeline to reverse-engineer constraints from answer text,
    then return either Step7 machine prompt or Step8 polished prompt.
    """
    project_root = pathlib.Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.pipeline_runner import run_pipeline_once  # pylint: disable=import-outside-toplevel

    run_id = f"{sample_id}__instgen"
    result = run_pipeline_once(
        sample_id=run_id,
        original_instruction=seed_instruction,
        model_answer=answer_text,
        base_data_dir=pipeline_data_dir,
        evidence=evidence,
        survey_mode=True,
        enable_step8_polish=use_step8_polish,
        enable_step7_5=False,
        enable_augment=False,
        enable_augment_diversity=False,
        enable_step8_5=False,
    )

    prompt_path = result.get("prompt_path") if use_step8_polish else result.get("prompt_path_machine")
    if not prompt_path:
        raise RuntimeError("Pipeline returned empty prompt path.")
    prompt_text = pathlib.Path(prompt_path).read_text(encoding="utf-8")
    return prompt_text, result


def _iter_jsonl(path: pathlib.Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _sample_id(prefix: str, survey: Dict[str, Any], idx: int) -> str:
    corpus_id = _safe_text(survey.get("corpusId"))
    if corpus_id:
        return f"{prefix}{corpus_id}"
    return f"{prefix}{idx:05d}"


def _write_text(path: pathlib.Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: pathlib.Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def build_samples(args: argparse.Namespace) -> Tuple[int, int]:
    surveys_path = pathlib.Path(args.surveys_file)
    references_path = pathlib.Path(args.references_file) if args.references_file else None
    second_level_path = pathlib.Path(args.second_level_file) if args.second_level_file else None
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    references_available = bool(references_path and references_path.exists())
    second_level_available = bool(second_level_path and second_level_path.exists())
    pipeline_data_dir = args.pipeline_data_dir or str(output_dir / "_pipeline_artifacts")

    produced = 0
    skipped = 0
    end_index = None if args.limit < 0 else args.start + args.limit

    with manifest_path.open("a", encoding="utf-8") as manifest:
        for idx, survey in enumerate(_iter_jsonl(surveys_path)):
            if idx < args.start:
                continue
            if end_index is not None and idx >= end_index:
                break

            metadata = survey.get("metadata", {}) or {}
            sections = survey.get("sections", []) or []
            references = survey.get("references", []) or []
            if not sections or not references:
                skipped += 1
                continue

            sample_id = _sample_id(args.sample_prefix, survey, idx)
            sample_dir = output_dir / sample_id
            if args.skip_existing and sample_dir.exists():
                skipped += 1
                continue

            answer_text = _build_answer(
                survey,
                inline_ref_ids=args.inline_ref_ids,
                max_refs_per_paragraph=args.max_refs_per_paragraph,
            )
            if _as_int(len(answer_text), 0) < args.min_answer_chars:
                skipped += 1
                continue

            evidence = _build_evidence(survey)
            evidence["survey"]["sample_id"] = sample_id
            evidence["meta"]["source_files"] = {
                "survey_full_text": str(surveys_path),
                "references_for_surveys": str(references_path) if references_available else "",
                "second_level_references": str(second_level_path) if second_level_available else "",
            }
            evidence["meta"]["second_level_available"] = second_level_available
            ref_count = len(evidence.get("references", {}))
            min_unique_refs = max(3, min(12, max(1, int(round(ref_count * 0.12)))))

            if args.instruction_lang == "en":
                seed_instruction = _build_instruction_en(
                    metadata,
                    section_count=len(sections),
                    min_unique_refs=min_unique_refs,
                    inline_ref_ids=args.inline_ref_ids,
                )
            else:
                seed_instruction = _build_instruction_zh(
                    metadata,
                    section_count=len(sections),
                    min_unique_refs=min_unique_refs,
                    inline_ref_ids=args.inline_ref_ids,
                )

            instruction = seed_instruction
            instruction_source_actual = "metadata"
            pipeline_meta: Dict[str, Any] = {}

            if args.instruction_source in {"pipeline-machine", "pipeline-polished"}:
                try:
                    instruction, pipe_result = _build_instruction_via_pipeline(
                        sample_id=sample_id,
                        seed_instruction=seed_instruction,
                        answer_text=answer_text,
                        evidence=evidence,
                        pipeline_data_dir=pipeline_data_dir,
                        use_step8_polish=(args.instruction_source == "pipeline-polished"),
                    )
                    instruction_source_actual = args.instruction_source
                    pipeline_meta = {
                        "graph_json_path": (pipe_result.get("graph_paths") or {}).get("graph_json", ""),
                        "prompt_path_machine": pipe_result.get("prompt_path_machine", ""),
                        "prompt_path": pipe_result.get("prompt_path", ""),
                        "evidence_path": pipe_result.get("evidence_path", ""),
                        "global_constraints_count": pipe_result.get("global_constraints_count", 0),
                        "constraint_total_count": pipe_result.get("constraint_total_count", 0),
                        "selection_count": pipe_result.get("selection_count", 0),
                        "evidence_loaded": pipe_result.get("evidence_loaded", False),
                    }
                except Exception as exc:  # noqa: BLE001
                    if args.pipeline_strict:
                        raise
                    instruction_source_actual = "metadata_fallback"
                    pipeline_meta = {"error": str(exc)}

            _write_text(sample_dir / "instruction.txt", instruction)
            _write_text(sample_dir / "answer.txt", answer_text)
            _write_json(sample_dir / "evidence.json", evidence)

            manifest.write(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "corpusId": evidence["survey"].get("corpusId"),
                        "title": evidence["survey"].get("title"),
                        "section_count": len(sections),
                        "reference_count": ref_count,
                        "instruction_source_requested": args.instruction_source,
                        "instruction_source_actual": instruction_source_actual,
                        "pipeline_meta": pipeline_meta,
                        "instruction_file": str(sample_dir / "instruction.txt"),
                        "answer_file": str(sample_dir / "answer.txt"),
                        "evidence_file": str(sample_dir / "evidence.json"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            produced += 1

    return produced, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SurveyGen samples for Constraint Graph pipeline.")
    parser.add_argument(
        "--surveys-file",
        type=str,
        default="experiments/SurveyGen/Surveys_Full_Text.jsonl",
        help="Path to survey_full_text JSONL.",
    )
    parser.add_argument(
        "--references-file",
        type=str,
        default="experiments/SurveyGen/References_for_Surveys.json",
        help="Path to references_for_surveys file (tracked in evidence meta).",
    )
    parser.add_argument(
        "--second-level-file",
        type=str,
        default="",
        help="Optional path to second_level_references file (tracked in evidence meta).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/SurveyGen/samples",
        help="Directory where per-sample folders are written.",
    )
    parser.add_argument(
        "--sample-prefix",
        type=str,
        default="surveygen_",
        help="Prefix used for generated sample IDs.",
    )
    parser.add_argument("--start", type=int, default=0, help="Start index in surveys file.")
    parser.add_argument("--limit", type=int, default=10, help="Number of surveys to process; -1 for all.")
    parser.add_argument(
        "--instruction-lang",
        type=str,
        choices=["zh", "en"],
        default="zh",
        help="Language for generated instruction prompts.",
    )
    parser.add_argument(
        "--instruction-source",
        type=str,
        choices=["pipeline-machine", "pipeline-polished", "metadata"],
        default="pipeline-machine",
        help=(
            "Source of instruction.txt. "
            "pipeline-machine: Step7 machine prompt; "
            "pipeline-polished: Step8 prompt; "
            "metadata: direct metadata template."
        ),
    )
    parser.add_argument(
        "--pipeline-data-dir",
        type=str,
        default="",
        help="Output dir for pipeline artifacts when instruction-source uses pipeline.",
    )
    parser.add_argument(
        "--pipeline-strict",
        action="store_true",
        help="Fail the run if pipeline-based instruction generation fails.",
    )
    parser.add_argument(
        "--inline-ref-ids",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to append [ref_id] markers into answer paragraphs.",
    )
    parser.add_argument(
        "--max-refs-per-paragraph",
        type=int,
        default=-1,
        help="Cap appended ref_id markers per paragraph when --inline-ref-ids is on; <=0 means no cap.",
    )
    parser.add_argument(
        "--min-answer-chars",
        type=int,
        default=500,
        help="Skip surveys whose constructed answer is shorter than this threshold.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip sample directories that already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    produced, skipped = build_samples(args)
    print("===== SURVEYGEN BUILD DONE =====")
    print(f"surveys_file   : {args.surveys_file}")
    print(f"references_file: {args.references_file}")
    print(f"second_level   : {args.second_level_file or '(not provided)'}")
    print(f"output_dir     : {args.output_dir}")
    print(f"instruction_src: {args.instruction_source}")
    print(f"produced       : {produced}")
    print(f"skipped        : {skipped}")


if __name__ == "__main__":
    main()
