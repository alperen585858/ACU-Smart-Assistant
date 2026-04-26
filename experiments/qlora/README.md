# QLoRA Experiment Folder

This directory is intentionally isolated from production code paths.

## Structure

- `data/train_30.jsonl`: 30-sample training dataset
- `data/eval_15.jsonl`: evaluation prompts (10-20 questions)
- `colab/qlora_minimal.ipynb`: Colab notebook for adapter-only QLoRA
- `results/comparison_table.md`: side-by-side comparison table
- `results/artifacts/`: screenshots, logs, and run evidence

## Rules

1. Do not modify production runtime settings for this experiment.
2. Do not merge adapters into the base model.
3. Use the fine-tuned adapter for evaluation/demo only.

## Expected Outcome

The experiment should provide evidence for:
- QLoRA process execution
- Measured comparison against base model
- Clear statement on why production remains on existing RAG flow

## Smoke Run Checklist (Colab)

Use this exact order in `colab/qlora_minimal.ipynb`.

1. Confirm Colab runtime is GPU-enabled.
2. Upload `train_30.jsonl` and `eval_15.jsonl` into `/content`.
3. Run install + imports + seed cells.
4. Verify dataset counts:
   - `train rows: 30`
   - `eval rows: 15`
5. Verify formatted training sample includes Qwen chat markers:
   - `<|im_start|>system`
   - `<|im_start|>user`
   - `<|im_start|>assistant`
6. Run 4-bit model loading cell.
7. Run LoRA trainer setup and smoke training.
8. Confirm adapter export folder exists: `/content/qlora_adapter_out`
9. Run 3-mode eval (base, base+prompt, base+adapter).
10. Confirm output file exists: `/content/results/eval_outputs_3mode.jsonl`

Smoke run is successful if the pipeline finishes end-to-end without crash and produces both adapter and eval JSONL outputs.

## Fallback Rules (if Colab fails)

- **OOM (CUDA out of memory)**
  - Reduce `max_seq_length` from `512` to `384`
  - Reduce `per_device_train_batch_size` from `2` to `1`
  - Increase `gradient_accumulation_steps` from `4` to `8`
- **Quantization/package errors**
  - Restart runtime and rerun install cell
  - Re-run from imports cell onward
- **Trainer crash**
  - Run a mini debug with first 10 training rows
  - If successful, return to full 30-row train set
- **Eval export issues**
  - Check `len(results)` before file write
  - Ensure `/content/results` directory is created

## Artifact Collection After Smoke Run

Collect and store the following under `results/artifacts/`:

- Screenshot of completed train log
- Screenshot showing `Saved 15 rows` message
- Downloaded `eval_outputs_3mode.jsonl`
- Downloaded adapter folder (`qlora_adapter_out`, zipped)
