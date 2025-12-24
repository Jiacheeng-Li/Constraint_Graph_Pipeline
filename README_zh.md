# Constraint Graph Pipeline 实现说明（中文）

> 本文档面向希望了解 **Constraint Graph Pipeline** 实现细节的工程师，全面介绍组件职责、数据流、LLM 交互策略与运行方式。

## 1. 系统概览
- **目标**：将一对高质量的「指令 + 模型回答」自动转化为可验证的复杂指令、约束图以及评分协议。
- **核心产物**：
  - `ConstraintGraph`：包含全局/局部约束、分支逻辑、元数据。
  - `machine_prompt`：用于下游模型评测的结构化提示词（Step 7/8 输出）。
  - `eval_protocol`：记录每条约束的 `verifier_spec`，供自动评分脚本调用。
- **执行路径**：Step 1–8 由 `pipeline_runner.py` 串联；候选模型答案通过 `scoring_runner.py` 打分。

## 2. 架构与依赖
- **LLM**：DeepSeek Chat Completion API (`src/utils/deepseek_client.py`)；支持重试、事件日志。
- **核心模块**：`src/step[1-8]_*.py`，各司其职；`src/graph_schema.py` 统一定义数据结构。
- **工具库**：
  - `utils/parsing.py`：宽松 JSON 解析。
  - `utils/text_clean.py`：只做空白规范/截断。
  - `utils/export_utils.py`：写入 JSON、Mermaid、文本文件。
  - `verifier/*`：硬性/软性检查函数，供 `verifier_spec` 调用。
- **运行脚本**：
  - `pipeline_runner.py`：读入指令+回答，依次执行各 Step，产出 graph/prompt/eval。
  - `scoring_runner.py`：加载 eval 协议，对候选答案运行自动评分。

## 3. 数据形态与文件输出
| 位置 | 内容 | 来源 |
|------|------|------|
| `data/raw_examples/` | 输入指令与示例回答 | 手动准备 |
| `data/graphs/<id>.graph.json` | ConstraintGraph 序列化 | Step 6 |
| `data/graphs/<id>.graph.mmd` | Mermaid 可视化 | Step 6 |
| `data/instructions/<id>.prompt.txt` | 最终提示词（Step 7/8） | Step 8（或 Step 7 fallback） |
| `data/reports/<id>.eval.json` | 评分协议 + 元数据 | Step 7 |
| `data/reports/<id>.bundle.json` | 完整 Step 7 bundle | Step 7 |
| `data/reports/<id>.candidate.txt` | 候选模型回答 | 用户写入 |
| `data/reports/<id>.score.json` | 自动评分结果 | scoring runner |

## 4. 工作流程（Step 1–8）
以下 Step 均在 `pipeline_runner.run_pipeline_once` 中按序执行，每一步的输入/输出与失败兜底策略如下。

### Step 1：种子任务抽取 (`step1_seed_task.extract_seed_task`)
- **输入**：`instruction_text` 原文。
- **策略**：
  1. 将全文喂给 DeepSeek，强制输出「单句英文祈使式」任务描述（带语气/风格要求）。
  2. 若调用失败或输出非法，则：抓首句 → 去掉礼貌前缀 → 加 `Analyze ...` 形成兜底指令。
- **输出**：`seed_task` 字符串；作为图的根节点。

### Step 2：回答分块 (`step2_segmentation.segment_response`)
- **输入**：参考答案。
- **策略**：
  1. LLM 根据 JSON 模板返回 `blocks`（含 `block_id`/`intent`/`text_span`/`order_index`）。
  2. `utils.parsing.extract_blocks` 做容错解析。
  3. 若解析失败，启用 `_split_paragraphs_with_bullet_groups` 纯规则分段。
- **输出**：`{"blocks": [...], "order": [...]}`；为 Step4/5/6 提供顺序与语义。

### Step 3：全局约束 (`step3_global_constraints.extract_global_constraints`)
- **输入**：`model_answer` + Step 2 segmentation。
- **策略**：
  - **硬性约束**（无需 LLM）：字数、语言、段落结构、格式风格、禁止符号、关键词等，由启发式检测并附带 `verifier_spec`，然后在所有硬候选中随机抽取 3–5 条（按可用数截断）。
  - **软性约束**：调用 DeepSeek，基于回答文本生成语气/质量偏好，解析后随机抽取 3–5 条（按可用数截断）。
- **输出**：`List[ConstraintNode]`（`scope="global"`，含 `trace_to`/`derived_from`）。

### Step 4：局部约束回译 (`step4_back_translation.extract_block_constraints`)
- **输入**：Step 2 segmentation、`seed_task`。
- **策略**：
  1. 对每个 block：提供结构 outline + 原文 snippet，请 LLM 产出 JSON（包含 `logic` 与多条局部约束）。
  2. 使用白名单、同义映射、去重策略；再按块独立随机抽取硬/软混合约束，总数 1–5 条（受可用候选数量限制）。
  3. LLM 失败时兜底：中立语气 + 最少字数/编号列表约束，仍参与随机抽取。
- **输出**：
  - `block_constraints`: `block_id -> [ConstraintNode...]`（`scope="local"`, `trace_to=block_id`）。
  - `block_logic`: `block_id -> "AND" / "sub-chain"`。

### Step 5：条件化分支 (`step5_selection_augment.generate_selection_branches`)
- **输入**：Step 2 segmentation、`seed_task`、Step 4 输出。
- **策略**：
  1. 根据 `SELECTION_CONFIG` 挑选部分 block 生成 selection（local/global）。
  2. LLM 读取真实分支约束，生成互补的 alternate branch（新 `ConstraintNode`）。
  3. 无 LLM 输出时，使用预设模板：负面语气 + 行动建议等。
- **输出**：
  - 更新后的 `block_constraints` / `block_logic`（含新增约束）。
   - `selections`: `List[SelectionNode]`（含条件说明、真实/备选分支引用的 CID）。

### Step 6：约束图组装 (`step6_graph_assembly.assemble_constraint_graph`)
- **输入**：`seed_task`、Step 2 segmentation、Step 3 全局约束、Step 5 结果。
- **策略**：
  - 将 block/约束/selection 封装成 `ConstraintGraph` dataclass。
  - `serialize_graph` 提供 JSON 视图给 Step 7 与落盘。
- **输出**：`ConstraintGraph` 对象；同时调用 `save_graph_outputs` 写 `.graph.json` + `.graph.mmd`。

### Step 7：指令与评分协议 (`step7_instruction_synthesis.synthesize_instruction_bundle`)
- **输入**：Step 6 的 `ConstraintGraph`。
- **策略**：纯模板逻辑（无 LLM），渲染：
  - `machine_prompt`：包含系统指令、种子任务、全局规则、块级计划、IF/ELSE 分支（落盘 `instructions/<id>.machine.txt` 便于审计/对比）。
  - `eval_protocol`：逐条列出 `verifier_spec`，供评分调用。
  - `meta`：透传图的组装信息。
- **输出**：bundle dict（后续写入 `reports/<id>.bundle.json`）。

### Step 8：提示词润色 (`step8_prompt_refinement.refine_instruction_prompt`)
- **输入**：Step 7 `machine_prompt`、`seed_task`、是否启用。
- **策略**：
  1. 如果禁用 / 输入为空 → 直接返回原文。
  2. 调用 DeepSeek，将模板式提示转为自然段落；禁止新增/删除约束。
  3. `_validate_polish` 检查字数比例/空输出，失败则 fallback。
- **输出**：`{"text": polished_prompt, "used_llm": bool, "reason": ...}`；落盘：
  - `instructions/<id>.machine.txt`：Step 7 生成的原始机器提示词（便于审计/对比）。
  - `instructions/<id>.prompt.txt`：Step 8 润色后的最终提示词（或 fallback 原文）。

## 5. 运行脚本详解
### pipeline_runner.py
1. 解析 CLI 参数与 Step8 开关。
2. 读取指令/回答文本，按 Step 1–8 顺序执行。
3. 调 `save_graph_outputs`、`write_text`、`write_json` 保存所有制品。
4. 从 `LLM_CALL_EVENTS` 抽样，记录各 Step 是否调用 LLM。
5. 最后打印汇总（seed task、块数、全局约束数、文件路径、Step8 状态）。

### scoring_runner.py
1. 读取 `<id>.eval.json`，抽出 `eval_protocol`。
2. 加载候选答案文本（默认 `reports/<id>.candidate.txt`）。
3. 调 `verifier.evaluate.run_evaluation` 输出 `summary`、`per_constraint`、`branch_choice`。
4. 写入 `reports/<id>.score.json` 并打印通过率摘要。

## 6. 关键数据结构
详见 `src/graph_schema.py`：
- `ConstraintNode`: `cid`、`desc`、`scope`、`verifier_spec`、`trace_to`、`derived_from`。
- `BlockSpec`: 记录 `block_id`、`intent`、`text_span`、`order_index`（以及替代块标记）。
- `BlockConstraintSet`: 每块的逻辑类型 + 约束列表。
- `SelectionNode` / `SelectionBranch`: IF/ELSE 分支的条件、路径、合流点信息。
- `ConstraintGraph`: 聚合所有节点并提供 `to_json()` 序列化。

## 7. 容错与工程策略
- **LLM 调用**：统一在 `deepseek_client.call_chat_completions`，含超时、重试、事件记录。
- **JSON 解析**：`utils/parsing.safe_json_load` + `_light_json_fix` 以最大化成功率。
- **长度控制**：`text_clean.clip` 仅在调用方明确要求时才截断；默认保留全部语义。
- **随机性控制**：Step 4 支持 `STEP4_RAND_SEED`，Step 5 的 `random.Random` 可设置种子，便于复现。
- **验证器扩展**：新增检查函数后需在 `verifier_registry` 注册，才能在 `verifier_spec` 中引用。

## 8. 快速上手
```bash
# 1. 准备虚拟环境并安装依赖
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt  # 至少包含 requests 与 verifier 依赖

# 2. 运行整条管线
python -m src.pipeline_runner \
    --sample-id sample_213 \
    --instruction-file data/raw_examples/example_003_instruction.txt \
    --answer-file data/raw_examples/example_003_answer.txt

# 3. 让目标模型回答生成的 prompt，并写入 data/reports/sample_0001.candidate.txt

# 4. 执行评分
python -m src.scoring_runner --sample-id sample_0001
```

## 9. 常见问题
1. **Step 8 输出为空/过短**：检查 `_validate_polish` 的长度比例说明；可手动禁用 Step 8。
2. **verifier 无法识别**：确认在 `verifier_registry` 注册并与 `verifier_spec` 的 `check` 名称一致。
3. **LLM 报错**：查看 `LLM_CALL_EVENTS` 中的 `error` 字段，或打开 `DeepSeekError` 异常堆栈定位。

---
