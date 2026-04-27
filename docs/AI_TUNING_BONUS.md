# AI Tuning Bonus (Isolated QLoRA Experiment)

## Purpose

This document records a minimal and isolated QLoRA experiment for the **Advanced AI Fine-Tuning** bonus requirement.

The primary production system remains unchanged:
- Current RAG pipeline stays as-is.
- Runtime model configuration stays as-is.
- No production endpoint is switched to a fine-tuned model.

## Safety Boundary

- Working branch: `aituningbo`
- Fine-tuning method: adapter-only QLoRA
- No adapter merge into the base model
- No production rollout of adapter

## What Changed

Only experiment and reporting artifacts were added:

- `experiments/qlora/README.md`
- `experiments/qlora/data/train_30.jsonl`
- `experiments/qlora/data/eval_15.jsonl`
- `experiments/qlora/colab/qlora_minimal.ipynb`
- `experiments/qlora/results/comparison_table.md`
- `experiments/qlora/results/artifacts/`
- `docs/AI_TUNING_BONUS.md`

## What Did Not Change

- `backend/chat/chat_logic.py` behavior
- `backend/chat/llm_service.py` behavior
- RAG retrieval path and serving flow
- Existing deployment/runtime defaults

## Hypotheses

- **H1 (Factual Accuracy):** Since production already uses RAG grounding, QLoRA is not expected to significantly improve factual accuracy.
- **H2 (Consistency / Instruction Following):** QLoRA is expected to improve response consistency and format compliance.
- **H3 (Weak Context Robustness):** QLoRA is expected to reduce unsupported claims when context is weak.

## Evaluation Protocol

Evaluation set size: 10-20 questions

Compared variants:
- Base model
- Base + QLoRA adapter
- Optional: current production RAG output

Scoring fields per question:
- Factual Accuracy (0-2)
- Instruction/Format Compliance (0-2)
- Unsupported Claim Resistance (0-2)

## Results Summary (To Fill)

Based on `experiments/qlora/results/comparison_table.md`:

- Mean Accuracy (Base): **1.20**
- Mean Accuracy (QLoRA): **1.20**
- Mean Format Compliance (Base): **0.00**
- Mean Format Compliance (QLoRA): **0.00**
- Mean Unsupported Claim Resistance (Base): **1.20**
- Mean Unsupported Claim Resistance (QLoRA): **1.20**

Observed behavior in this smoke run:

- Base prompted and QLoRA outputs were effectively identical across sampled questions.
- No factual accuracy gain was observed in the short-run setting.
- Role-prefix leakage (`system/user/assistant`) remained present, so format quality did not improve.
- The experiment still satisfies bonus evidence requirements by demonstrating adapter training, export, and documented comparison outputs.

## Final Decision

Advanced AI Fine-Tuning was demonstrated via a small-scale QLoRA experiment with documented comparisons.
For production stability, the main application remains on the existing RAG architecture.
