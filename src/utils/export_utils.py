"""
utils/export_utils.py

统一的落盘/导出工具：
- write_text: 写入纯文本
- write_json: 写入 JSON
- save_graph_outputs: 将 ConstraintGraph 导出为 .graph.json 和 .graph.mmd

注意：
- save_graph_outputs 之前定义在 step6_graph_assembly.py；现在搬到这里，
  供 pipeline_runner 作为唯一的正式写盘调用点。
- step6_graph_assembly.py 的 __main__ 只做可视化预览，不再直接写盘。
"""

import os
import json
from typing import Dict, Any

from ..step6_graph_assembly import serialize_graph, make_mermaid
from ..graph_schema import ConstraintGraph


def write_json(path: str, obj: Any) -> None:
    """写 JSON 文件（UTF-8，带缩进）。会自动创建父目录。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_text(path: str, text: str) -> None:
    """写纯文本文件（UTF-8）。会自动创建父目录。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def save_graph_outputs(graph: ConstraintGraph, sample_id: str, base_dir: str = "data/graphs") -> Dict[str, str]:
    """
    将 Step6 的主要图产物各自落盘：
    - 结构化图快照 serialize_graph(graph)  ->  <base_dir>/<sample_id>.graph.json
    - Mermaid 可视化 make_mermaid(graph)   ->  <base_dir>/<sample_id>.graph.mmd

    返回一个 dict，包含两个最终写入的路径。
    """
    os.makedirs(base_dir, exist_ok=True)

    graph_json_path = os.path.join(base_dir, f"{sample_id}.graph.json")
    mermaid_path = os.path.join(base_dir, f"{sample_id}.graph.mmd")

    # 写 graph.json
    with open(graph_json_path, "w", encoding="utf-8") as f_json:
        json.dump(serialize_graph(graph), f_json, ensure_ascii=False, indent=2)

    # 写 .mmd
    with open(mermaid_path, "w", encoding="utf-8") as f_mmd:
        f_mmd.write("%% Mermaid flowchart for sample `" + sample_id + "`\n")
        f_mmd.write("%% Paste this into any Mermaid renderer to view the graph.\n\n")
        f_mmd.write("flowchart LR\n")

        mermaid_full = make_mermaid(graph).splitlines()
        # 避免重复写 'flowchart LR' 这一行
        if mermaid_full and mermaid_full[0].strip().startswith("flowchart"):
            mermaid_full = mermaid_full[1:]
        for line in mermaid_full:
            f_mmd.write(line + "\n")

    return {
        "graph_json": graph_json_path,
        "mermaid_mmd": mermaid_path,
    }
