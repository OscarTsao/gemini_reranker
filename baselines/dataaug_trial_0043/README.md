# DataAug Trial 0043 Baseline

This snapshot ports the best-performing criteria matching run (Optuna trial `0043`) from `DataAugmentation_Evaluation` into the Gemini reranker workspace. It achieved macro F1 **0.8535** on the held-out test split and serves as the default baseline when comparing Gemini-tuned rerankers.

## Contents

- `model/best/` – frozen DeBERTa classifier checkpoint plus the Optuna-derived `config.yaml` and validation metrics.
- `model/checkpoints/last.pt` – final training state in case you want to resume.
- `metrics/test_metrics.json` – aggregate test-set metrics.
- `data/augmentation/` – hybrid + auxiliary augmentation CSVs that fed this trial.
- `data/groundtruth/Final_Ground_Truth.json` – canonical labeled pairs.

## Using Inside `gemini_reranker`

1. Point configs at the copied data:
   - `data/groundtruth/Final_Ground_Truth.json`
   - `data/augmentation/hybrid_dataset_*.csv` (plus optional `nlpaug`/`textattack` variants)
2. Warm-start Gemini-tuned models from `model/best/model.pt` or convert to Hugging Face format with `torch.load`.
3. Log comparisons against `metrics/test_metrics.json` to track deltas from the baseline trial.

All files are local to this repository; no external downloads are required for reproduction.
