"""
Step 6.5 - Graph-only Augmentation

Purpose
- Reuse the Step 6 graph augmentation logic (curriculum, M1, M2),
  but only write graph artifacts (.graph.json/.graph.mmd).
- Intended as a lightweight pre-step before Step 7.5 rendering.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from .graph_schema import ConstraintGraph
from .graph_augmenter import (
    _graph_from_serialized,
    _build_curriculum_chain,
    _build_m1_turns,
    _build_m2_turns,
    _apply_priority_injection,
    _ensure_priority_levels,
    _default_sample_id_from_path,
)
from .utils.export_utils import save_graph_outputs


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_graph_only(graph: ConstraintGraph,
                     sample_id: str,
                     base_dir: str) -> Dict[str, str]:
    graphs_dir = os.path.join(base_dir, "graphs")
    return save_graph_outputs(graph, sample_id=sample_id, base_dir=graphs_dir)


def augment_graphs_only(graph: ConstraintGraph,
                        sample_id: str,
                        base_dir: str,
                        rng: random.Random,
                        *,
                        priority_ratio: float = 0.5,
                        enable_curriculum: bool = True,
                        enable_m1: bool = True,
                        enable_m2: bool = True) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []

    if enable_curriculum:
        chain = _build_curriculum_chain(graph, rng)
        if chain:
            for level in [1, 2, 3, 4, 5]:
                g = copy.deepcopy(chain[level])
                meta = dict(g.meta or {})
                meta.update({
                    "parent_sample_id": sample_id,
                    "curriculum_level": level,
                })
                g.meta = meta
                if rng.random() < priority_ratio:
                    info = _apply_priority_injection(g, rng)
                    g.meta.update(info)
                _ensure_priority_levels(g)
                paths = _save_graph_only(g, f"{sample_id}__g{level}", base_dir)
                outputs.append({
                    "sample_id": f"{sample_id}__g{level}",
                    "paths": paths,
                    "meta": g.meta,
                })

    if enable_m1:
        turns = _build_m1_turns(graph, rng)
        if turns:
            for idx, g in enumerate(turns, start=1):
                meta = dict(g.meta or {})
                meta.update({
                    "parent_sample_id": sample_id,
                    "mode": "m1",
                    "sequence_id": f"{sample_id}__m1",
                    "turn_index": idx,
                    "turn_total": len(turns),
                    "turn_notice": True,
                    "delta_only_instruction": idx > 1,
                    "delta_reference": "previous_turn",
                })
                g.meta = meta
                _ensure_priority_levels(g)
                paths = _save_graph_only(g, f"{sample_id}__m1_t{idx}", base_dir)
                outputs.append({
                    "sample_id": f"{sample_id}__m1_t{idx}",
                    "paths": paths,
                    "meta": g.meta,
                })

    if enable_m2:
        turns = _build_m2_turns(graph, rng)
        if turns:
            for idx, g in enumerate(turns, start=1):
                meta = dict(g.meta or {})
                meta.update({
                    "parent_sample_id": sample_id,
                    "mode": "m2",
                    "sequence_id": f"{sample_id}__m2",
                    "turn_index": idx,
                    "turn_total": len(turns),
                    "turn_notice": True,
                    "delta_only_instruction": idx > 1,
                    "delta_reference": "previous_turn",
                })
                g.meta = meta
                _ensure_priority_levels(g)
                paths = _save_graph_only(g, f"{sample_id}__m2_t{idx}", base_dir)
                outputs.append({
                    "sample_id": f"{sample_id}__m2_t{idx}",
                    "paths": paths,
                    "meta": g.meta,
                })

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step6.5: augment graphs and write .graph.json/.graph.mmd outputs.",
    )
    parser.add_argument("--graph-json", type=str, help="Path to a .graph.json file.")
    parser.add_argument("--graphs-dir", type=str, help="Directory containing .graph.json files.")
    parser.add_argument("--sample-id", type=str, default="", help="Override sample_id for output naming.")
    parser.add_argument("--output-dir", type=str, default="", help="Base output dir (same layout as pipeline).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--priority-ratio", type=float, default=0.5, help="Probability of priority injection per sample.")
    parser.add_argument("--disable-curriculum", action="store_true", help="Disable curriculum generation.")
    parser.add_argument("--disable-m1", action="store_true", help="Disable M1 multi-turn generation.")
    parser.add_argument("--disable-m2", action="store_true", help="Disable M2 multi-turn generation.")
    parser.add_argument("--include-augmented", action="store_true", help="Also process graphs with '__' in the sample_id.")
    args = parser.parse_args()

    if not args.graph_json and not args.graphs_dir:
        raise SystemExit("Provide --graph-json or --graphs-dir.")

    rng = random.Random(args.seed)

    def _run_one(path: str) -> List[Dict[str, Any]]:
        sample_id = args.sample_id or _default_sample_id_from_path(path)
        if not args.include_augmented and ("__" in os.path.basename(path) or "__" in sample_id):
            return []
        base_dir = args.output_dir or os.path.dirname(os.path.dirname(path))
        graph_data = _load_json(path)
        graph = _graph_from_serialized(graph_data)
        return augment_graphs_only(
            graph=graph,
            sample_id=sample_id,
            base_dir=base_dir,
            rng=rng,
            priority_ratio=args.priority_ratio,
            enable_curriculum=not args.disable_curriculum,
            enable_m1=not args.disable_m1,
            enable_m2=not args.disable_m2,
        )

    all_outputs: List[Dict[str, Any]] = []
    if args.graph_json:
        all_outputs.extend(_run_one(args.graph_json))
    else:
        for name in os.listdir(args.graphs_dir):
            if not name.endswith(".graph.json"):
                continue
            if not args.include_augmented and "__" in name:
                continue
            all_outputs.extend(_run_one(os.path.join(args.graphs_dir, name)))

    timestamp = datetime.now(timezone.utc).isoformat()
    print(f"step6_5_completed_at_utc: {timestamp}")
    print(f"graphs_written: {len(all_outputs)}")


if __name__ == "__main__":
    main()
