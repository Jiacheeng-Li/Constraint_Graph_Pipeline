# Constraint Graph Pipeline

## Overview
- Converts a reference instruction/answer pair into a constraint graph, a deterministic machine prompt, and an automated evaluation protocol.
- Enforces verifiable rules by attaching `verifier_spec` metadata to every constraint so downstream scoring is scriptable.
- Supports branching (IF/ELSE) logic via synthetic selections, enabling realistic multi-path evaluation tasks.
- Optional graph augmenter produces curriculum chains (G1–G5), multi-turn variants (M1/M2), and priority-aware constraints.
- Optional Step 6.5/7.5 stages can augment graphs and render multi-template machine prompts directly from graphs.
- Optional Step 8.5 diversification creates multiple natural-language prompt styles from a single machine prompt.
- Ships with runner scripts for generation (`pipeline_runner.py`) and scoring (`scoring_runner.py`).
- **Output paths**:
  - Core path: Steps 1–8.
  - Augmented path (curriculum/multi-turn): Steps 1–6 → graph_augmenter → Step 8.
  - Instruction diversity path: Steps 1–6 → Step 6.5 → Step 7.5 (single template) → Step 8 or Step 8.5.

**中文文档**：需要中文实现说明可参阅 [`README_zh.md`](README_zh.md)。

## Repository Layout
```
Pipeline_10.25/
├─ README.md
├─ data/
│  ├─ raw_examples/              # Sample instruction/answer pairs for quick experiments
│  ├─ graphs/                    # Step 6 outputs (.graph.json + .graph.mmd)
│  ├─ instructions/              # Step 8 machine prompts ready for model evals
│  └─ reports/                   # Step 7 eval specs, candidate answers, and scores
├─ src/
│  ├─ step1_seed_task.py         # Step 1 - seed task extraction
│  ├─ step2_segmentation.py      # Step 2 - answer segmentation
│  ├─ step3_global_constraints.py# Step 3 - global constraint mining
│  ├─ step4_back_translation.py  # Step 4 - block-level reverse engineering
│  ├─ step5_selection_augment.py # Step 5 - conditional branch synthesis
│  ├─ step6_graph_assembly.py    # Step 6 - graph assembly & serialization
│  ├─ step7_instruction_synthesis.py # Step 7 - machine prompt + eval protocol
│  ├─ step8_prompt_refinement.py # Step 8 - optional prompt polish
│  ├─ step6_5_graph_augment.py   # Step 6.5 - graph-only augmentation (G/M variants)
│  ├─ step7_5_prompt_renderer.py # Step 7.5 - multi-template prompt rendering
│  ├─ step8_5_prompt_diversification.py # Step 8.5 - prompt diversification
│  ├─ pipeline_runner.py         # CLI orchestrator for Steps 1->8
│  ├─ scoring_runner.py          # CLI scorer for candidate answers
│  ├─ graph_augmenter.py          # Curriculum + multi-turn augmentation
│  ├─ graph_schema.py            # Dataclasses used across steps
│  ├─ utils/                     # DeepSeek client, parsing, text cleaning, exports, etc.
│  └─ verifier/                  # Hard/soft checks + evaluation entrypoints
└─ examples/                     # Optional showcase artifacts
```

## Quick Start
1. **Install Python deps** - create a virtualenv and `pip install -r requirements.txt` (or install `requests` plus your verifier deps manually).
2. **Prepare input text files** - place an instruction and its high-quality answer under `data/raw_examples` (sample files are provided).
3. **Run the pipeline**
   ```bash
   python -m src.pipeline_runner \
       --sample-id sample_0001 \
       --instruction-file data/raw_examples/example_003_instruction.txt \
       --answer-file data/raw_examples/example_003_answer.txt
   ```
   Inspect the generated artifacts under `data/graphs`, `data/instructions`, and `data/reports`.
4. **Score a candidate model answer**
   - Ask your target model to answer `data/instructions/sample_0001.prompt.txt` and save the output to `data/reports/sample_0001.candidate.txt`.
   - Run the scorer:
     ```bash
     python -m src.scoring_runner --sample-id sample_0001
     ```
   - Review `data/reports/sample_0001.score.json` for per-constraint verdicts and branch choices.
5. **Run the augmenter (optional)**
   ```bash
   python -m src.graph_augmenter \
       --graphs-dir experiments/alpaca/graphs \
       --output-dir experiments/alpaca \
       --seed 13
   ```
   Outputs sample IDs with suffixes like `__g1..__g5`, `__m1_t1..__m1_t3`, `__m2_t1..__m2_t2`.
6. **Render a template prompt from a graph (optional, Step 7.5)**
   ```bash
   python -m src.step7_5_prompt_renderer \
       --graph-json data/graphs/sample_0001.graph.json \
       --template-seed 13 \
       --heuristic-ratio 0.6
   ```

## Pipeline at a Glance
| Step | Module | Purpose | Key Outputs |
|------|--------|---------|-------------|
| 1 | `step1_seed_task.extract_seed_task` | Distill the final deliverable into a single imperative English sentence. | `seed_task` |
| 2 | `step2_segmentation.segment_response` | Split the exemplar answer into ordered semantic blocks; ensure unique block labels. | `segmentation` dict |
| 3 | `step3_global_constraints.extract_global_constraints` | Mine document-wide rules (language, structure, tone) using heuristics + LLM. | `List[ConstraintNode]` |
| 4 | `step4_back_translation.extract_block_constraints` | Reverse-engineer local obligations per block with verifiers. | `block_constraints`, `block_logic` |
| 5 | `step5_selection_augment.generate_selection_branches` | Create conditional branches with alternate requirements. | Updated block data + `selections` |
| 6 | `step6_graph_assembly.assemble_constraint_graph` | Compose everything into a `ConstraintGraph`. | Graph dataclass + `.graph.json/.mmd` |
| 6.5 (optional) | `step6_5_graph_augment.augment_graphs_only` | Graph-only augmentation (curriculum + multi-turn variants). | `.graph.json/.mmd` variants |
| 7 | `step7_instruction_synthesis.synthesize_instruction_bundle` | Emit machine prompt + evaluation manifest. | `machine_prompt`, `eval_protocol`, bundle JSON |
| 7.5 (optional) | `step7_5_prompt_renderer.render_prompt_variant` | Render a single template machine prompt from a graph. | `*.machine.tmpl_<template>.txt` |
| 8 | `step8_prompt_refinement.refine_instruction_prompt` | Optionally rewrite the prompt into natural prose. | Polished prompt + validation metadata |
| 8.5 (optional) | `step8_5_prompt_diversification.diversify_instruction_prompt` | Generate multiple natural-language prompt styles. | `*.prompt.<style>.txt` |
| 9 | `scoring_runner.run_scoring_once` | Score a candidate answer against the stored eval protocol. | `<sample_id>.score.json` |

## Data Artifacts
| Path | Produced by | Description |
|------|-------------|-------------|
| `data/graphs/<id>.graph.json` | Step 6 | Serialized constraint graph with all nodes and selections. |
| `data/graphs/<id>.graph.mmd` | Step 6 | Mermaid diagram for quick visualization. |
| `data/instructions/<id>.machine.txt` | Step 7 | Raw machine prompt (pre-Stage 8 polish) for audit/diff. |
| `data/instructions/<id>.prompt.txt` | Step 8 (or Step 7 fallback) | Final machine prompt fed to target models. |
| `data/instructions/<id>.machine.tmpl_<template>.txt` | Step 7.5 | Template-rendered machine prompt from a graph. |
| `data/instructions/<id>.prompt.tmpl_<template>.txt` | Step 7.5 + Step 8 | Template prompt after optional polish. |
| `data/instructions/<id>.prompt.<style>.txt` | Step 8.5 | Diversified natural-language prompt variants. |
| `data/reports/<id>.eval.json` | Step 7 | Eval protocol + metadata bundle required for scoring. |
| `data/reports/<id>.bundle.json` | Step 7 | Full instruction bundle (prompt, eval protocol, serialized graph). |
| `data/reports/<id>.candidate.txt` | User-provided | Candidate model answer awaiting scoring. |
| `data/reports/<id>.score.json` | Scoring runner | Structured evaluation report with per-constraint verdicts + best-effort rate. |

## Evaluation Workflow
1. Generate a task via `pipeline_runner.py` (Steps 1->8).
2. Deliver the resulting prompt to a target model and save its answer.
3. Execute `scoring_runner.py` to check every constraint and branch.
4. Aggregate results across samples or feed them into dashboards as needed.

## Configuration & Environment
- **DeepSeek settings**: set `DEEPSEEK_API_KEY`, `DEEPSEEK_MODEL`, and `DEEPSEEK_ENDPOINT` (defaults live in `utils/deepseek_client.py`).
- **Step 8 toggle**: set `PIPELINE_ENABLE_STEP8=0` or pass `--skip-step8-polish` to disable the polish pass.
- **Augment toggle**: set `PIPELINE_ENABLE_AUGMENT=1` or pass `--enable-augment` to run graph augmentation in the pipeline.
- **Augment + diversity toggle**: set `PIPELINE_ENABLE_AUGMENT_DIVERSITY=1` or pass `--enable-augment-diversity` to run Step 6.5 -> 7.5 -> 8.
- **Step 7.5 toggle**: set `PIPELINE_ENABLE_STEP7_5=1` or pass `--enable-step7-5` to enable template rendering.
- **Step 8.5 toggle**: set `PIPELINE_ENABLE_STEP8_5=1` or pass `--enable-step8-5` to enable prompt diversification.
- **Augment mode override**: pass `--augment-mode {none,augment,augment_diversity,both}` to force a specific path.
- **Determinism knobs**: `STEP4_RAND_SEED` stabilizes block-level sampling; `SELECTION_CONFIG` in Step 5 caps how many selections to attempt.
- **Verifiers**: extend `src/verifier/hard_checks.py` and `src/verifier/soft_checks.py`, then register names inside `verifier_registry` so `verifier_spec` lookups succeed.
- **Dependencies**: Python 3.10+ with `requests` for API calls plus any libraries required by your verifier implementations.

## Graph Augmentation
- **Curriculum chain**: G5 → G1 strict subgraphs, ensures monotonic difficulty.
- **Priority injection**: sets `priority_level=1` on a subset of block-level constraints; remaining constraints stay `priority_level=2`.
- **M1 (progressive refinement)**: emits turn-by-turn graphs; each prompt includes “Only constraints in the current graph are active.”
- **M2 (mutation/rollback)**: tightens numeric hard constraints into A′ (priority=1), then rolls back to A (priority=2).
- The augmenter skips already-augmented graphs by default; use `--include-augmented` to override.
- **Step 6.5 graph-only path**: run `step6_5_graph_augment.py` if you want augmented graphs without prompts.

## Development Notes
- Run individual steps directly (e.g., `python -m src.step2_segmentation`) to debug prompts before using the orchestrator.
- `data/raw_examples` contains reference pairs that map to the sample IDs shown in the CLI examples—handy for regression testing.
- Mermaid graphs are useful when auditing constraint coverage; open them in any Markdown/diagram viewer.
- Keep an eye on `LLM_CALL_EVENTS` (captured inside the pipeline result dict) to monitor API usage per step.
