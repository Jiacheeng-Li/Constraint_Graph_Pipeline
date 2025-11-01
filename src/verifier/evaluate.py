

"""
verifier/evaluate.py

该模块负责对候选回答(candidate_answer)进行自动评估。

输入核心：
    1. eval_protocol: 来自 Step7 的评估协议（通常从 data/reports/<id>.eval.json 里读出来的 "eval_protocol" 字段）。
       结构示例（简化）：
       {
         "seed_task": "...",
         "global_scoring": [
            {
              "cid": "G1",
              "desc": "The answer must be written primarily in English.",
              "verifier": {"check": "require_language", "args": {"lang": "en"}},
              "logic": "MANDATORY_GLOBAL"
            },
            ...
         ],
         "block_scoring": [
            {
              "block_id": "B1",
              "role": "Opening / Context setup",
              "logic_type": "AND" | "sub-chain",
              "requirements": [
                {
                  "cid": "B1_C1",
                  "desc": "Explain why the modern space race matters geopolitically.",
                  "verifier": {"check": "tone_neutral_llm_judge", "args": {}}
                },
                ...
              ]
            },
            ...
         ],
         "conditional_scoring": [
            {
              "sid": "SEL_B3",
              "where": "B3 Conclusion / Outlook / Recommendation",
              "condition": "If the stance is critical/negative",
              "branch_real": [ {"cid": ..., "desc": ..., "verifier": ...}, ... ],
              "branch_alt":  [ {"cid": ..., "desc": ..., "verifier": ...}, ... ],
              "scoring_rule": {
                "must_choose_one_branch": True,
                "branch_real_logic": "AND",
                "branch_alt_logic": "AND"
              },
            }
         ],
         "meta": { ... }
       }

    2. candidate_answer: 要被打分的模型输出（字符串）。

输出：
    run_evaluation(...) 返回一个 dict，包含：
        {
          "per_constraint": [  # 每个约束的逐项打分
             {
               "cid": "B2_C1",
               "desc": "Give at least two concrete real-world examples.",
               "passed": True/False/None,
               "details": <verifier返回的细节> | None,
               "scope": "global" | "block" | "branch",
               "block_id": "B2" | None,
               "selection_id": "SEL_B3" | None,
             },
             ...
          ],
          "branch_choice": {
             "SEL_B3": {
                "chosen": "branch_real" | "branch_alt" | "undecided",
                "score_branch_real": <float or None>,
                "score_branch_alt": <float or None>
             },
             ...
          },
          "summary": {
             "global_pass_rate": 0.8,
             "block_pass_rate": 0.7,
             "branch_pass_rate": 0.6,
             "overall_pass_rate": 0.72,
          }
        }

思想：
    - global_scoring: 这些是整篇必须满足的要求。我们直接逐项调用对应的 verifier。
    - block_scoring: 每个 block 下有若干 requirements，逻辑大多是 AND 或 sub-chain。
         我们这里逐项跑 verifier，暂时不做复杂顺序评分；调用方可用 pass 率来做决策。
    - conditional_scoring: 这是 if/then/else 分支。
         我们会：
            1. 对 branch_real 的所有 requirement 评分，得到一个均值/通过率 score_real
            2. 对 branch_alt  的所有 requirement 评分，得到一个均值/通过率 score_alt
            3. 选择更高分的分支，视为模型“更像遵守的那条分支”。
         这样我们不需要显式解析模型有没有声明“我走哪条分支”。

    - 每个约束项调用的 verifier 来自 verifier_registry。
      verifier_spec 统一格式：{"check": <name>, "args": {...}}

    - 我们不在这里做任何磁盘读写；由上游负责把 eval_protocol 读进来、把结果写出去。

后续扩展点：
    - 可以给 run_evaluation 增加内容分块（针对 block_id 的文本片段，而不是整篇），
      以便更细化地检查局部约束。现阶段我们直接对整篇 candidate_answer 运行所有检查，
      因为我们还没把回答分块的文本回传进 eval_protocol。后面可以在 eval_protocol 里
      加入 block-level span 片段并使用它。
"""

from typing import Dict, Any, List, Optional, Tuple

from ..utils.verifier_registry import run_verifier  # 我们需要确保 verifier_registry 暴露 run_verifier


# ------------------------------------------------------------
# 内部小工具：统计通过率 / 评分
# ------------------------------------------------------------

def _avg_bool(values: List[Optional[bool]]) -> Optional[float]:
    """
    计算布尔列表的平均值（True=1, False=0, None=忽略）。
    如果全是 None 则返回 None。
    """
    numeric = [1.0 if v is True else 0.0 for v in values if v is not None]
    if not numeric:
        return None
    return sum(numeric) / float(len(numeric))


def _score_branch(requirements: List[Dict[str, Any]], candidate_answer: str,
                  selection_id: str) -> Tuple[List[Dict[str, Any]], Optional[float]]:
    """
    对一条分支(branch_real 或 branch_alt)的所有 requirement 执行校验。

    返回：
        (per_req_results, avg_score)

    per_req_results 是一个 list[dict]，每一项结构为：
        {
          "cid": ..., "desc": ..., "passed": bool|None, "details": any,
          "scope": "branch", "block_id": None, "selection_id": <selection_id>
        }
    avg_score 是通过率均值（忽略 None）。
    """
    per_req_results: List[Dict[str, Any]] = []
    pass_flags: List[Optional[bool]] = []

    for req in requirements:
        cid = req.get("cid")
        desc = req.get("desc")
        verifier_spec = req.get("verifier") or {}
        check_name = verifier_spec.get("check")
        check_args = verifier_spec.get("args", {})

        if not check_name:
            passed = None
            details = {"note": "no verifier provided"}
        else:
            passed, details = run_verifier(
                check_name=check_name,
                check_args=check_args,
                answer_text=candidate_answer,
                context={
                    "cid": cid,
                    "desc": desc,
                    "selection_id": selection_id,
                },
            )

        per_req_results.append({
            "cid": cid,
            "desc": desc,
            "passed": passed,
            "details": details,
            "scope": "branch",
            "block_id": None,
            "selection_id": selection_id,
        })
        pass_flags.append(passed)

    avg_score = _avg_bool(pass_flags)
    return per_req_results, avg_score


# ------------------------------------------------------------
# 核心评测主函数
# ------------------------------------------------------------

def run_evaluation(eval_protocol: Dict[str, Any], candidate_answer: str) -> Dict[str, Any]:
    """
    根据 eval_protocol（Step7 生成）对 candidate_answer 做自动评估。

    返回 dict，包含：
        - per_constraint: 所有逐项约束的检查结果列表
        - branch_choice: 针对每个 conditional 选择了哪条分支
        - summary: 全局/分段/分支层面的通过率统计

    目前我们对 block_scoring 和 conditional_scoring 的检查都直接用整篇 candidate_answer，
    而没有按 block 的文本片段精细切。后续可以在 eval_protocol 里加 block-level span 后改进这里。
    """

    per_constraint_results: List[Dict[str, Any]] = []
    branch_choice_summary: Dict[str, Any] = {}

    global_flags: List[Optional[bool]] = []
    block_flags: List[Optional[bool]] = []
    branch_flags: List[Optional[bool]] = []

    # 1. 全局约束 (global_scoring)
    for g in eval_protocol.get("global_scoring", []):
        cid = g.get("cid")
        desc = g.get("desc")
        verifier_spec = g.get("verifier") or {}
        check_name = verifier_spec.get("check")
        check_args = verifier_spec.get("args", {})

        if not check_name:
            passed = None
            details = {"note": "no verifier provided"}
        else:
            passed, details = run_verifier(
                check_name=check_name,
                check_args=check_args,
                answer_text=candidate_answer,
                context={
                    "cid": cid,
                    "desc": desc,
                    "scope": "global",
                },
            )

        per_constraint_results.append({
            "cid": cid,
            "desc": desc,
            "passed": passed,
            "details": details,
            "scope": "global",
            "block_id": None,
            "selection_id": None,
        })
        global_flags.append(passed)

    # 2. 分阶段约束 (block_scoring)
    for blk in eval_protocol.get("block_scoring", []):
        block_id = blk.get("block_id")
        requirements = blk.get("requirements", [])

        for req in requirements:
            cid = req.get("cid")
            desc = req.get("desc")
            verifier_spec = req.get("verifier") or {}
            check_name = verifier_spec.get("check")
            check_args = verifier_spec.get("args", {})

            if not check_name:
                passed = None
                details = {"note": "no verifier provided"}
            else:
                passed, details = run_verifier(
                    check_name=check_name,
                    check_args=check_args,
                    answer_text=candidate_answer,
                    context={
                        "cid": cid,
                        "desc": desc,
                        "block_id": block_id,
                    },
                )

            per_constraint_results.append({
                "cid": cid,
                "desc": desc,
                "passed": passed,
                "details": details,
                "scope": "block",
                "block_id": block_id,
                "selection_id": None,
            })
            block_flags.append(passed)

    # 3. 条件化分支约束 (conditional_scoring)
    for sel in eval_protocol.get("conditional_scoring", []):
        sid = sel.get("sid")
        real_reqs = sel.get("branch_real", [])
        alt_reqs = sel.get("branch_alt", [])

        # 分别对两条分支打分
        real_results, real_score = _score_branch(real_reqs, candidate_answer, sid)
        alt_results, alt_score   = _score_branch(alt_reqs, candidate_answer, sid)

        # 把两边的逐项结果都加入总 per_constraint_results
        per_constraint_results.extend(real_results)
        per_constraint_results.extend(alt_results)

        # 选择“更像它遵守的分支”
        # 规则：哪个 avg_score 高就选哪个；都 None 时就是 "undecided"
        if real_score is None and alt_score is None:
            chosen = "undecided"
        elif alt_score is None:
            chosen = "branch_real"
        elif real_score is None:
            chosen = "branch_alt"
        else:
            chosen = "branch_real" if (real_score >= alt_score) else "branch_alt"

        branch_choice_summary[sid] = {
            "chosen": chosen,
            "score_branch_real": real_score,
            "score_branch_alt": alt_score,
        }

        # 将该选择的分支的 per-req 通过情况汇入 branch_flags，便于整体统计
        if chosen == "branch_real":
            branch_flags.extend([r.get("passed") for r in real_results])
        elif chosen == "branch_alt":
            branch_flags.extend([r.get("passed") for r in alt_results])
        else:
            # undecided: 两边都不计入 or 计入两边都行？这里我们计入两边的通过率一起参考
            branch_flags.extend([r.get("passed") for r in real_results])
            branch_flags.extend([r.get("passed") for r in alt_results])

    # 汇总通过率
    global_rate = _avg_bool(global_flags)
    block_rate = _avg_bool(block_flags)
    branch_rate = _avg_bool(branch_flags)

    # overall_pass_rate: 粗暴平均（忽略 None）
    overall_components = []
    for v in [global_rate, block_rate, branch_rate]:
        if v is not None:
            overall_components.append(v)
    overall_rate = sum(overall_components) / len(overall_components) if overall_components else None

    summary = {
        "global_pass_rate": global_rate,
        "block_pass_rate": block_rate,
        "branch_pass_rate": branch_rate,
        "overall_pass_rate": overall_rate,
    }

    return {
        "per_constraint": per_constraint_results,
        "branch_choice": branch_choice_summary,
        "summary": summary,
    }
