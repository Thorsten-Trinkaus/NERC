# SpanBERT UltraFine Results

This directory contains the evaluation artifacts for the UltraFine SpanBERT experiments.

## Final Setup

- Base model: `SpanBERT/spanbert-base-cased`
- Fine-tuned model used for the final run:
  `results/spanbert_finetuned/ultrafine_pos_weighted/final_model`
- Training change compared with the initial UltraFine run:
  weighted multi-label BCE loss with `pos_weight`
- Training config for the improved run:
  `--use-pos-weight --pos-weight-max 50 --metric-for-best-model loss`
- Validation-selected threshold for final evaluation:
  `-0.045`

## Final Test Metrics

Source: `test/metrics.json`

| Metric | Value |
|---|---:|
| precision_micro | 0.0574 |
| recall_micro | 0.3847 |
| f1_micro | 0.0999 |
| precision_macro | 0.000821 |
| recall_macro | 0.01429 |
| f1_macro | 0.001478 |
| avg_gold_labels_per_example | 5.3719 |
| avg_predicted_labels_per_example | 35.9965 |
| threshold | -0.045 |

## Main Findings

1. The original unweighted UltraFine training collapsed.
   With the default threshold `0.0`, the model predicted no labels at all. Even after threshold tuning, it effectively degenerated to a trivial high-frequency label behavior.

2. `pos_weight` fixed the collapse and produced usable signal.
   After re-training with weighted BCE loss, the model started predicting a broad set of labels and reached a non-zero, stable micro-F1 on validation and test.

3. Threshold tuning was essential.
   The best validation threshold was slightly negative (`-0.045`), not `0.0`. The model logits stay in a narrow range, so coarse threshold sweeps can miss the best point.

4. The model is still strongly recall-heavy.
   The final model predicts about `36` labels per example on average, while the gold data contains about `5.37`. This leads to improved recall, but low precision.

5. Macro performance remains very weak.
   The macro scores are close to zero, which suggests that rare labels are still not modeled well even after reweighting.

## Interpretation

The weighted loss was a meaningful improvement for UltraFine entity typing. It moved the model away from trivial predictions and raised test performance to `f1_micro = 0.0999`. At the same time, the model is still badly calibrated and overpredicts labels, so the current result should be treated as a stronger baseline rather than a fully satisfactory final system.

## Important Files

- Validation threshold search:
  `validation/threshold_search.json`
- Validation predictions:
  `validation/predictions.json`
- Validation metrics:
  `validation/metrics.json`
- Final test predictions:
  `test/predictions.json`
- Final test metrics:
  `test/metrics.json`

## Commands Used

Weighted fine-tuning:

```bash
sbatch job_spanbert_dataset.sh ultrafine \
  --use-pos-weight \
  --pos-weight-max 50 \
  --metric-for-best-model loss \
  --output-dir results/spanbert_finetuned/ultrafine_pos_weighted
```

Validation threshold search:

```bash
sbatch job_spanbert_probe.sh ultrafine validation \
  --model-dir results/spanbert_finetuned/ultrafine_pos_weighted/final_model \
  --search-thresholds \
  --threshold-start -0.10 \
  --threshold-end -0.01 \
  --threshold-step 0.005 \
  --selection-metric f1_micro
```

Final test run:

```bash
sbatch job_spanbert_probe.sh ultrafine test \
  --model-dir results/spanbert_finetuned/ultrafine_pos_weighted/final_model \
  --threshold -0.045
```

## Suggested Next Steps

- Reduce overprediction, for example with a smaller `pos_weight_max` such as `10` or `20`
- Try label-wise or frequency-aware thresholds instead of one global threshold
- Inspect frequent false positives in `test/predictions.json`
- Compare the weighted UltraFine run against the unweighted run in a short error analysis section
