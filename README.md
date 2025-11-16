# Constraint Graph Pipeline v1.3

## 0. Overview

This pipeline formalizes **complex instruction-following** into a structured process of **constraint extraction**, **graph assembly**, and **instruction synthesis**.

It enables automated transformation from a high-quality example response into a **Constraint Graph** (nodes = constraints, edges = logical relations), and finally into a **complex instruction** suitable for training or evaluation.

---

## 1. Pipeline Overview

### **Step 1. Seed Task Extraction**
- Extract the minimal, atomic core task (e.g., “Analyze the geopolitical implications of the modern space race.”)
- Output format:
```json
{"seed_task": "Analyze the geopolitical implications of the modern space race in a neutral tone."}
```

---

### **Step 2. Response Segmentation (Chain Prototype)**
- Divide the high-quality response into logical blocks (`B1`, `B2`, `B3`...) representing distinct reasoning or narrative stages.
- Each block includes:
  - `intent`: semantic purpose (e.g., "background", "analysis", "summary")
  - `text_span`: raw text from response
  - `order`: sequence position in the overall flow

Example output:
```json
{
  "blocks": [
    {"block_id": "B1", "intent": "Opening / Context setup", "text_span": "..."},
    {"block_id": "B2", "intent": "Main analysis", "text_span": "..."},
    {"block_id": "B3", "intent": "Conclusion / Outlook", "text_span": "..."}
  ],
  "order": ["B1", "B2", "B3"]
}
```

---

### **Step 3. Global Constraint Extraction**
- Identify constraints applying to the **entire text** (language, tone, total length, required keywords).
- Integrate **structural constraints** (required stages/modules and logical order).
- Each constraint node includes a **verifier_spec** for automatic evaluation.

Example output:
```json
{
  "global_constraints": [
    {"cid": "G1", "type": "global.style", "desc": "Must be written in English.",
     "verifier_spec": {"check": "is_english"}},
    {"cid": "G2", "type": "global.structure", "desc": "Include Opening, Body, Conclusion in logical order.",
     "verifier_spec": {"check": "has_sections", "args": {"sections":["Opening","Body","Conclusion"]}}}
  ]
}
```

---

### **Step 4. Block-level Back-translation (Constraint Extraction)**
- For each block, back-translate its implicit constraints.
- Label each constraint as **AND** or **sub-chain**.
- Add metadata:
  - `scope`: local / transitional / global
  - `verifier_spec`: rule-based or LLM-evaluable check

Example:
```json
{
  "block_id": "B2",
  "constraints": [
    {"cid": "C2_1", "desc": "Explain the key players in the space race.",
     "scope": "local",
     "verifier_spec": {"check": "must_include_keywords", "args": {"keywords":["space race","NASA","China"]}}},
    {"cid": "C2_2", "desc": "Maintain neutral and analytical tone.",
     "scope": "local",
     "verifier_spec": {"check": "tone_neutral_llm_judge"}}
  ],
  "local_logic": "AND"
}
```

---

### **Step 5. Selection Augmentation (Conditional Branching)**
- At selected chain points, synthesize **alternative conditional paths** (Selection).
- Each alternative branch includes:
  - `condition`: explicit, observable trigger (e.g., “If tone is negative…”)
  - `trace_to`: reference to the original block ID (for traceability)
  - `constraints`: generated constraints for this branch
  - `verifier_spec` for each constraint (enabling evaluation)

Example:
```json
{
  "selection": {
    "trace_to": "B2",
    "condition": "If the user sentiment is negative",
    "branch_real": {"constraints": ["C2_1","C2_2"]},
    "branch_alt": {
      "constraints": [
        {"cid": "C2_alt_1", "desc": "List two major issues with negative tone.",
         "verifier_spec": {"check": "must_list_n_subpoints", "args": {"n":2}}},
        {"cid": "C2_alt_2", "desc": "Use emotional adjectives expressing frustration.",
         "verifier_spec": {"check": "tone_negative_llm_judge"}}
      ]
    }
  }
}
```

---

### **Step 6. Constraint Graph Assembly**
- Combine all extracted constraints and relations into a unified **Constraint Graph**.
- Node types: global constraints, block constraints, selection branches.
- Edge types: Chain / AND / Selection.
- Attach metadata for provenance:
  - `trace_to` (source block)
  - `derived_from` (step or source origin)

Example output:
```json
{
  "nodes": [...],
  "edges": [
    {"from": "G1", "to": "B1", "type": "AND", "derived_from": "step3"},
    {"from": "B1", "to": "B2", "type": "CHAIN", "derived_from": "step2"},
    {"from": "B2", "to": "Selection_1", "type": "CHAIN", "trace_to": "B2"}
  ],
  "cnode": ["C2_1","C2_2"]
}
```

---

### **Step 7. Graph-to-Instruction Synthesis**
- Translate the full constraint graph back into a natural, human-readable **complex instruction**.
- Include:
  - Seed task statement
  - Global constraints summary
  - Stage-level structure description (non-rigid phrasing)
  - Conditional branches (if/else)
  - Fine-grained verifiable clauses

Example output:
```json
{
  "instruction_text": "Write an analytical essay about the modern space race... If the user's sentiment is negative, describe two main problems and propose improvements.",
  "constraint_summary": {
    "global": ["G1","G2"],
    "blocks": ["C2_1","C2_2"],
    "selections": ["C2_alt_1","C2_alt_2"]
  }
}
```

---

### **Step 8. Scoring Runner (Evaluation Execution)**
- Evaluate candidate model answers automatically based on the generated `eval_protocol`.
- Read from:
  - `data/reports/<sample_id>.eval.json` (protocol)
  - `data/reports/<sample_id>.candidate.txt` (model answer)
- Write to:
  - `data/reports/<sample_id>.score.json` (scoring results)

Example usage:
```bash
python -m src.scoring_runner --sample-id sample_0001
```

Example output:
```json
{
  "sample_id": "sample_0001",
  "summary": {
    "global_pass_rate": 0.8,
    "block_pass_rate": 0.7,
    "branch_pass_rate": 0.6,
    "overall_pass_rate": 0.72
  },
  "branch_choice": {
    "SEL_B3": {"chosen": "branch_alt", "score_branch_real": 0.5, "score_branch_alt": 0.9}
  }
}
```

---

### **Evaluation Workflow Summary**
1. Run `pipeline_runner.py` to generate complex instruction and eval protocol.
2. Let a target model produce an answer for the generated instruction.
3. Save the model’s output to `data/reports/<sample_id>.candidate.txt`.
4. Run `scoring_runner.py` to automatically evaluate constraint satisfaction.

---

## 2. Metadata Enhancements

| Field | Description | Added in Step |
|--------|--------------|---------------|
| `intent` | Functional role of each block | 2 |
| `scope` | Constraint influence scope: local / transitional / global | 4 |
| `trace_to` | Maps generated branch to source block | 5 |
| `derived_from` | Records source step for each edge | 6 |
| `verifier_spec` | Attached to every constraint node for auto-checking | 3–6 |

These metadata fields ensure **traceability**, **evaluation consistency**, and **training usability**.

---

## 3. Output Summary

| Output Type | Description |
|--------------|-------------|
| `graph.json` | Full structured constraint graph with nodes, edges, metadata |
| `instruction_text` | Natural-language complex instruction derived from the graph |
| `constraint_summary.json` | Structured summary of all constraint types and verifiers |

---

## 4. Applications

- **Data Generation** — Synthesize diverse, verifiable complex instructions for LLM training.  
- **Evaluation** — Automatically score responses based on constraint satisfaction and correct branch selection.  
- **Visualization** — Render constraint graphs for interpretability and debugging.  

- **Automated Scoring** — Use the scoring runner module to execute evaluation protocols against candidate answers, producing detailed scoring reports for model performance analysis.

---

## 5. Index
```
Pipeline_10.25/
├─ README.md                        # 流程说明文档
├─ data/
│  ├─ raw_examples/                 # 高质量指令与响应样本
│  ├─ prompts/                      # 各阶段使用的Prompt模板
│  ├─ graphs/                       # 生成的Constraint Graph文件
│  ├─ instructions/                 # 反向生成的复杂指令
│  └─ reports/                      # 评测与验证结果
│
├─ src/
│  ├─ step1_seed_task.py            # Step 1: 种子任务抽取
│  ├─ step2_segmentation.py         # Step 2: 回答分块与Chain原型生成
│  ├─ step3_global_constraints.py   # Step 3: 全局约束抽取（含结构性约束）
│  ├─ step4_back_translation.py     # Step 4: 各块反向约束抽取
│  ├─ step5_selection_augment.py    # Step 5: 伪分支生成与条件约束合成
│  ├─ step6_graph_assembly.py       # Step 6: 约束图组装与元数据整合
│  ├─ step7_instruction_synthesis.py# Step 7: 复杂指令生成
│  ├─ scoring_runner.py             # Step 8: 评分执行与自动评估结果生成
│  │
│  ├─ utils/
│  │   ├─ templates.py              # 硬性/格式化模板池
│  │   ├─ prompts.py                # 软性/LLM 提示词池
│  │   ├─ parsing.py                # 解析LLM输出，提取block与约束信息
│  │   ├─ verifier_registry.py      # 注册并管理verifier_spec对应的检测函数
│  │   ├─ text_clean.py             # 文本清理与规范化工具
│  │   └─ export_utils.py           # 导出graph与instruction文件
│  │
│  ├─ graph_schema.py               # 统一的节点、边、元数据数据结构定义
│  ├─ verifier/
│  │   ├─ hard_checks.py            # 硬性验证（长度、语言、关键词）
│  │   ├─ soft_checks.py            # 软性验证（语气、风格、一致性）
│  │   └─ evaluate.py               # 约束满足度评测与C-node检查
│  │
│  ├─ pipeline_runner.py            # 主调度脚本，顺序执行1~7步 
│  └─ scoring_runner.py             # 测试评分脚本
│
└─ examples/
   ├─ example_instruction.txt
   ├─ example_response.txt
   ├─ example_graph.json
   └─ example_instruction_generated.txt
```