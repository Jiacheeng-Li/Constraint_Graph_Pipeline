#!/usr/bin/env python
"""
Run Step6.5 -> Step7.5 -> Step8(/8.5) for mother graphs under experiments/alpaca.

Rules implemented:
- Build augmented graphs from mother graphs directly.
- Use one shared template for g1-g5, m1_t1, m2_t1 of the same parent sample.
- Do not apply diversity strategy to follow-up turns (m1_t2/t3, m2_t2); force stage_blueprint.
- Optionally apply Step8.5 on a random subset (default 20%) of diversity-target variants.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import random
import re
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.graph_schema import ConstraintGraph
from src.graph_augmenter import _graph_from_serialized
from src.step6_5_graph_augment import augment_graphs_only
from src.step7_5_prompt_renderer import render_prompt_variant
from src.step8_prompt_refinement import refine_instruction_prompt
from src.step8_5_prompt_diversification import diversify_instruction_prompt
from src.utils.export_utils import write_json, write_text


TEMPLATE_NAMES = ("stage_blueprint", "branch_first", "grouped_by_check", "priority_layered")
DEFAULT_TEMPLATE_WEIGHTS = {
    "stage_blueprint": 4,
    "branch_first": 2,
    "grouped_by_check": 2,
    "priority_layered": 2,
}


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _stable_seed(base_seed: int, key: str) -> int:
    payload = f"{base_seed}:{key}".encode("utf-8")
    return int(hashlib.md5(payload).hexdigest(), 16) % (2**31)


def _strip_think_prefix(text: str) -> Tuple[str, bool]:
    if not text:
        return text, False
    marker = "</think>"
    if marker in text:
        _, tail = text.split(marker, 1)
        return tail.lstrip(), True
    return text, False


def _parse_template_weights(raw: str) -> Dict[str, int]:
    if not raw.strip():
        return dict(DEFAULT_TEMPLATE_WEIGHTS)
    parsed: Dict[str, int] = {}
    for part in raw.split(","):
        token = part.strip()
        if not token or ":" not in token:
            continue
        name, value = token.split(":", 1)
        key = name.strip().lower().replace("-", "_")
        if key not in TEMPLATE_NAMES:
            continue
        try:
            weight = int(value.strip())
        except ValueError:
            continue
        if weight > 0:
            parsed[key] = weight
    if not parsed:
        return dict(DEFAULT_TEMPLATE_WEIGHTS)
    for key in TEMPLATE_NAMES:
        parsed.setdefault(key, 1)
    return parsed


def _has_priority_one(graph: ConstraintGraph) -> bool:
    for node in graph.global_constraints:
        if node.priority_level == 1:
            return True
    for bcs in graph.block_constraint_sets:
        for node in bcs.constraints:
            if node.priority_level == 1:
                return True
    return False


def _check_type_count(graph: ConstraintGraph) -> int:
    checks: set[str] = set()
    for node in graph.global_constraints:
        check = str((node.verifier_spec or {}).get("check", "")).strip()
        if check:
            checks.add(check)
    for bcs in graph.block_constraint_sets:
        for node in bcs.constraints:
            check = str((node.verifier_spec or {}).get("check", "")).strip()
            if check:
                checks.add(check)
    return len(checks)


def _heuristic_template(graph: ConstraintGraph) -> str:
    if graph.selections:
        return "branch_first"
    if _has_priority_one(graph):
        return "priority_layered"
    if _check_type_count(graph) >= 3:
        return "grouped_by_check"
    return "stage_blueprint"


def _weighted_template_choice(rng: random.Random, weights: Dict[str, int]) -> str:
    keys = list(TEMPLATE_NAMES)
    vals = [max(1, int(weights.get(k, 1))) for k in keys]
    return rng.choices(keys, weights=vals, k=1)[0]


def _choose_parent_template(graph: ConstraintGraph,
                            sample_id: str,
                            seed: int,
                            *,
                            weights: Dict[str, int],
                            heuristic_ratio: float) -> Tuple[str, str]:
    rng = random.Random(_stable_seed(seed, f"{sample_id}:template"))
    heuristic = _heuristic_template(graph)
    if rng.random() < max(0.0, min(1.0, heuristic_ratio)):
        return heuristic, "heuristic"
    return _weighted_template_choice(rng, weights), "weighted_random"


def _is_diversity_target(sample_id: str) -> bool:
    if re.search(r"__g[1-5]$", sample_id):
        return True
    if sample_id.endswith("__m1_t1"):
        return True
    if sample_id.endswith("__m2_t1"):
        return True
    return False


def _list_mother_graphs(graphs_dir: str) -> List[str]:
    paths: List[str] = []
    for name in os.listdir(graphs_dir):
        if not name.endswith(".graph.json"):
            continue
        if "__" in name:
            continue
        paths.append(os.path.join(graphs_dir, name))
    paths.sort()
    return paths


def _read_sample_id_list(path: str) -> set[str]:
    out: set[str] = set()
    if not path:
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            token = line.strip()
            if not token or token.startswith("#"):
                continue
            out.add(token)
    return out


def _filter_graph_paths(paths: List[str],
                        *,
                        sample_id_regex: str,
                        sample_id_prefix: str,
                        sample_id_list: set[str],
                        start_index: int,
                        limit: int) -> List[str]:
    regex = re.compile(sample_id_regex) if sample_id_regex else None
    filtered: List[str] = []
    for path in paths:
        name = os.path.basename(path)
        sample_id = name[:-len(".graph.json")]
        if sample_id_prefix and not sample_id.startswith(sample_id_prefix):
            continue
        if regex and not regex.search(sample_id):
            continue
        if sample_id_list and sample_id not in sample_id_list:
            continue
        filtered.append(path)
    filtered.sort()

    start = max(0, int(start_index))
    if start >= len(filtered):
        return []
    sliced = filtered[start:]
    if limit >= 0:
        sliced = sliced[: int(limit)]
    return sliced


def _load_existing_summary(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return {}
    return {}


def run_once(graph_json_path: str,
             *,
             output_dir: str,
             seed: int,
             priority_ratio: float,
             heuristic_ratio: float,
             template_weights: Dict[str, int],
             enable_curriculum: bool,
             enable_m1: bool,
             enable_m2: bool,
             enable_step8: bool,
             enable_step8_5: bool,
             step8_5_ratio: float,
             step8_5_variants: int,
             step8_5_styles: Optional[List[str]]) -> Dict[str, Any]:
    base_name = os.path.basename(graph_json_path)
    sample_id = base_name[:-len(".graph.json")]
    graph = _graph_from_serialized(_load_json(graph_json_path))
    rng = random.Random(_stable_seed(seed, f"{sample_id}:augment"))
    augmented_entries = augment_graphs_only(
        graph=graph,
        sample_id=sample_id,
        base_dir=output_dir,
        rng=rng,
        priority_ratio=priority_ratio,
        enable_curriculum=enable_curriculum,
        enable_m1=enable_m1,
        enable_m2=enable_m2,
    )
    shared_template, template_source = _choose_parent_template(
        graph,
        sample_id=sample_id,
        seed=seed,
        weights=template_weights,
        heuristic_ratio=heuristic_ratio,
    )

    instructions_dir = os.path.join(output_dir, "instructions")
    os.makedirs(instructions_dir, exist_ok=True)

    rendered: List[Dict[str, Any]] = []
    for entry in sorted(augmented_entries, key=lambda x: x.get("sample_id", "")):
        variant_id = str(entry.get("sample_id") or "")
        graph_path = str((entry.get("paths") or {}).get("graph_json") or "")
        if not graph_path:
            continue
        aug_graph = _graph_from_serialized(_load_json(graph_path))
        selected_template = shared_template if _is_diversity_target(variant_id) else "stage_blueprint"
        render_out = render_prompt_variant(
            aug_graph,
            template_pool=[selected_template],
            template_seed=_stable_seed(seed, f"{variant_id}:render"),
            heuristic_ratio=0.0,
            selection_key=variant_id,
            template_limit=1,
        )
        variant = render_out.get("variant", {})
        machine_prompt = str(variant.get("machine_prompt") or "").strip()
        if not machine_prompt:
            continue
        template_key = str(variant.get("template") or selected_template)
        machine_path = os.path.join(instructions_dir, f"{variant_id}.machine.tmpl_{template_key}.txt")
        write_text(machine_path, machine_prompt)

        if enable_step8:
            polish = refine_instruction_prompt(
                machine_prompt=machine_prompt,
                seed_task=aug_graph.seed_task,
                enable=True,
            )
            prompt_text = str(polish.get("text") or machine_prompt)
        else:
            polish = {"used_llm": False, "reason": "disabled", "attempts": 0}
            prompt_text = machine_prompt
        prompt_text, stripped_think = _strip_think_prefix(prompt_text)
        polish["stripped_think"] = stripped_think
        prompt_path = os.path.join(instructions_dir, f"{variant_id}.prompt.tmpl_{template_key}.txt")
        write_text(prompt_path, prompt_text)

        step8_5_files: List[str] = []
        if enable_step8_5 and _is_diversity_target(variant_id):
            p_rng = random.Random(_stable_seed(seed, f"{variant_id}:step8_5_ratio"))
            if p_rng.random() < max(0.0, min(1.0, step8_5_ratio)):
                diversify = diversify_instruction_prompt(
                    machine_prompt=machine_prompt,
                    seed_task=aug_graph.seed_task,
                    enable=True,
                    num_variants=max(1, int(step8_5_variants)),
                    style_seed=_stable_seed(seed, f"{variant_id}:step8_5_styles"),
                    style_pool=step8_5_styles,
                )
                style_seen: Dict[str, int] = {}
                for style_variant in diversify.get("variants", []):
                    text = str(style_variant.get("text") or "").strip()
                    if not text:
                        continue
                    text, _ = _strip_think_prefix(text)
                    style_slug = str(style_variant.get("style_slug") or style_variant.get("style") or "variant")
                    style_seen[style_slug] = style_seen.get(style_slug, 0) + 1
                    suffix = f"_{style_seen[style_slug]}" if style_seen[style_slug] > 1 else ""
                    out_name = (
                        f"{variant_id}.prompt.tmpl_{template_key}.{style_slug}{suffix}.txt"
                    )
                    out_path = os.path.join(instructions_dir, out_name)
                    write_text(out_path, text)
                    step8_5_files.append(out_path)

        rendered.append({
            "sample_id": variant_id,
            "template": template_key,
            "template_source": template_source if _is_diversity_target(variant_id) else "fixed_stage_blueprint",
            "machine_path": machine_path,
            "prompt_path": prompt_path,
            "step8_reason": polish.get("reason"),
            "step8_5_files": step8_5_files,
        })

    return {
        "parent_sample_id": sample_id,
        "graph_json": graph_json_path,
        "shared_template": shared_template,
        "shared_template_source": template_source,
        "rendered_count": len(rendered),
        "rendered": rendered,
    }


def _run_once_safe(graph_json_path: str,
                   *,
                   output_dir: str,
                   seed: int,
                   priority_ratio: float,
                   heuristic_ratio: float,
                   template_weights: Dict[str, int],
                   enable_curriculum: bool,
                   enable_m1: bool,
                   enable_m2: bool,
                   enable_step8: bool,
                   enable_step8_5: bool,
                   step8_5_ratio: float,
                   step8_5_variants: int,
                   step8_5_styles: Optional[List[str]]) -> Dict[str, Any]:
    sample_id = os.path.basename(graph_json_path)[:-len(".graph.json")]
    try:
        summary = run_once(
            graph_json_path,
            output_dir=output_dir,
            seed=seed,
            priority_ratio=priority_ratio,
            heuristic_ratio=heuristic_ratio,
            template_weights=template_weights,
            enable_curriculum=enable_curriculum,
            enable_m1=enable_m1,
            enable_m2=enable_m2,
            enable_step8=enable_step8,
            enable_step8_5=enable_step8_5,
            step8_5_ratio=step8_5_ratio,
            step8_5_variants=step8_5_variants,
            step8_5_styles=step8_5_styles,
        )
        return {"ok": True, "sample_id": sample_id, "summary": summary}
    except Exception as exc:
        return {"ok": False, "sample_id": sample_id, "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch Step6.5->7.5->8(/8.5) for experiments/alpaca mother graphs.")
    parser.add_argument("--graphs-dir", type=str, default="experiments/alpaca/graphs")
    parser.add_argument("--output-dir", type=str, default="experiments/alpaca")
    parser.add_argument("--sample-id-prefix", type=str, default="", help="Only process sample IDs with this prefix.")
    parser.add_argument("--sample-id-regex", type=str, default="", help="Only process sample IDs matching this regex.")
    parser.add_argument("--sample-id-list", type=str, default="", help="Path to a newline-separated sample_id allowlist file.")
    parser.add_argument("--start-index", type=int, default=0, help="Start index after filtering.")
    parser.add_argument("--limit", type=int, default=-1, help="Max mother graphs to process (-1 means all).")
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker threads.")
    parser.add_argument("--resume", action="store_true", help="Skip parents already present in existing summary.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first error.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--priority-ratio", type=float, default=0.5)
    parser.add_argument("--heuristic-ratio", type=float, default=0.6)
    parser.add_argument(
        "--template-weights",
        type=str,
        default="stage_blueprint:4,branch_first:2,grouped_by_check:2,priority_layered:2",
        help="Weighted random template fallback, e.g. stage_blueprint:4,branch_first:2,...",
    )
    parser.add_argument("--disable-curriculum", action="store_true")
    parser.add_argument("--disable-m1", action="store_true")
    parser.add_argument("--disable-m2", action="store_true")
    parser.add_argument("--skip-step8", action="store_true")
    parser.add_argument("--enable-step8-5", action="store_true")
    parser.add_argument("--step8-5-ratio", type=float, default=0.2)
    parser.add_argument("--step8-5-variants", type=int, default=1)
    parser.add_argument("--step8-5-styles", type=str, default="")
    parser.add_argument(
        "--summary-path",
        type=str,
        default="",
        help="Optional summary json output path. Default: <output-dir>/reports/alpaca_step6_5_7_5_summary.json",
    )
    args = parser.parse_args()

    all_graph_paths = _list_mother_graphs(args.graphs_dir)
    if not all_graph_paths:
        raise SystemExit(f"No mother graphs found under {args.graphs_dir}")
    sample_id_allow = _read_sample_id_list(args.sample_id_list)
    graph_paths = _filter_graph_paths(
        all_graph_paths,
        sample_id_regex=args.sample_id_regex,
        sample_id_prefix=args.sample_id_prefix,
        sample_id_list=sample_id_allow,
        start_index=args.start_index,
        limit=args.limit,
    )
    if not graph_paths:
        raise SystemExit("No mother graphs left after filtering.")

    styles = [s.strip() for s in args.step8_5_styles.split(",") if s.strip()]
    template_weights = _parse_template_weights(args.template_weights)
    reports_dir = os.path.join(args.output_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    summary_path = args.summary_path or os.path.join(reports_dir, "alpaca_step6_5_7_5_summary.json")
    existing_payload = _load_existing_summary(summary_path) if args.resume else {}
    existing_parents = existing_payload.get("parents", []) if isinstance(existing_payload, dict) else []
    done_sample_ids = {
        item.get("parent_sample_id")
        for item in existing_parents
        if isinstance(item, dict) and item.get("parent_sample_id")
    }
    if args.resume and done_sample_ids:
        graph_paths = [
            p for p in graph_paths
            if os.path.basename(p)[:-len(".graph.json")] not in done_sample_ids
        ]
    if not graph_paths:
        print("Nothing to run after resume filtering.")
        return

    summaries: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    total = len(graph_paths)
    workers = max(1, int(args.workers))
    print(f"[run] total_parents={total} workers={workers} resume={args.resume}")
    if workers == 1:
        for idx, path in enumerate(graph_paths, start=1):
            result = _run_once_safe(
                path,
                output_dir=args.output_dir,
                seed=args.seed,
                priority_ratio=args.priority_ratio,
                heuristic_ratio=args.heuristic_ratio,
                template_weights=template_weights,
                enable_curriculum=not args.disable_curriculum,
                enable_m1=not args.disable_m1,
                enable_m2=not args.disable_m2,
                enable_step8=not args.skip_step8,
                enable_step8_5=args.enable_step8_5,
                step8_5_ratio=args.step8_5_ratio,
                step8_5_variants=args.step8_5_variants,
                step8_5_styles=styles or None,
            )
            if result.get("ok"):
                summary = result["summary"]
                summaries.append(summary)
                print(
                    f"[{idx}/{total}] [done] {summary['parent_sample_id']} "
                    f"template={summary['shared_template']} rendered={summary['rendered_count']}"
                )
            else:
                errors.append({"sample_id": result["sample_id"], "error": result["error"]})
                print(f"[{idx}/{total}] [error] {result['sample_id']} {result['error']}")
                if args.fail_fast:
                    break
    else:
        with cf.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _run_once_safe,
                    path,
                    output_dir=args.output_dir,
                    seed=args.seed,
                    priority_ratio=args.priority_ratio,
                    heuristic_ratio=args.heuristic_ratio,
                    template_weights=template_weights,
                    enable_curriculum=not args.disable_curriculum,
                    enable_m1=not args.disable_m1,
                    enable_m2=not args.disable_m2,
                    enable_step8=not args.skip_step8,
                    enable_step8_5=args.enable_step8_5,
                    step8_5_ratio=args.step8_5_ratio,
                    step8_5_variants=args.step8_5_variants,
                    step8_5_styles=styles or None,
                ): path
                for path in graph_paths
            }
            completed = 0
            for fut in cf.as_completed(futures):
                completed += 1
                result = fut.result()
                if result.get("ok"):
                    summary = result["summary"]
                    summaries.append(summary)
                    print(
                        f"[{completed}/{total}] [done] {summary['parent_sample_id']} "
                        f"template={summary['shared_template']} rendered={summary['rendered_count']}"
                    )
                else:
                    errors.append({"sample_id": result["sample_id"], "error": result["error"]})
                    print(f"[{completed}/{total}] [error] {result['sample_id']} {result['error']}")
                    if args.fail_fast:
                        for pending in futures:
                            pending.cancel()
                        break

    merged_parents = list(existing_parents) if args.resume and existing_parents else []
    merged_ids = {p.get("parent_sample_id") for p in merged_parents if isinstance(p, dict)}
    for item in summaries:
        sid = item.get("parent_sample_id")
        if sid in merged_ids:
            continue
        merged_parents.append(item)
        merged_ids.add(sid)
    merged_parents.sort(key=lambda x: str(x.get("parent_sample_id", "")))

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "graphs_dir": args.graphs_dir,
        "output_dir": args.output_dir,
        "filters": {
            "sample_id_prefix": args.sample_id_prefix,
            "sample_id_regex": args.sample_id_regex,
            "sample_id_list": args.sample_id_list,
            "start_index": args.start_index,
            "limit": args.limit,
            "workers": workers,
            "resume": args.resume,
        },
        "template_weights": template_weights,
        "heuristic_ratio": args.heuristic_ratio,
        "step8_enabled": not args.skip_step8,
        "step8_5_enabled": args.enable_step8_5,
        "step8_5_ratio": args.step8_5_ratio,
        "processed_in_this_run": len(summaries),
        "errors_in_this_run": len(errors),
        "errors": errors,
        "parents": merged_parents,
    }
    write_json(summary_path, payload)
    print(f"summary_written: {summary_path}")
    if errors:
        print(f"errors: {len(errors)}")


if __name__ == "__main__":
    main()
