# SpanBERT FIGER Results

This directory contains the evaluation artifacts for the FIGER SpanBERT experiments.

## Final Setup

- Base model: `SpanBERT/spanbert-base-cased`
- Fine-tuned model used for the final run:
  `results/spanbert_finetuned/figer/final_model`
- Validation-selected threshold for final evaluation:
  `0.3`

## Final Test Metrics

Source: `test/metrics.json`

| Metric | Value |
|---|---:|
| precision_micro | 0.9544 |
| recall_micro | 0.9380 |
| f1_micro | 0.9462 |
| precision_macro | 0.9042 |
| recall_macro | 0.8640 |
| f1_macro | 0.8816 |
| avg_gold_labels_per_example | 2.3877 |
| avg_predicted_labels_per_example | 2.3466 |
| threshold | 0.3 |

## Main Findings

1. SpanBERT performs very strongly on FIGER.
   The final test score reaches `f1_micro = 0.9462`, which is only slightly below OntoNotes despite FIGER being much more fine-grained and multi-label.

2. The model is well calibrated for multi-label prediction.
   The average number of predicted labels per example (`2.3466`) is very close to the gold average (`2.3877`), which suggests that global thresholding works well here.

3. Threshold tuning helped, but the optimum is broad.
   Validation selected `0.3`, yet nearby thresholds such as `0.2` and `0.25` performed almost identically. This means the model is fairly robust to small threshold changes.

4. Rare labels remain harder than frequent ones.
   Macro performance is clearly lower than micro performance (`0.8816` vs. `0.9462`), which points to reduced quality on less common FIGER types even though the overall result is strong.

## Interpretation

FIGER appears to be a very good match for the current SpanBERT setup. The model handles multi-label prediction much better here than on UltraFine and retains strong precision and recall at the same time. This makes FIGER an important intermediate point between the simple OntoNotes setup and the much more difficult UltraFine setting.

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

Validation threshold search:

```bash
sbatch job_spanbert_probe.sh figer validation \
  --search-thresholds \
  --threshold-start -0.5 \
  --threshold-end 1.0 \
  --threshold-step 0.05 \
  --selection-metric f1_micro
```

Final test run:

```bash
sbatch job_spanbert_probe.sh figer test \
  --threshold 0.3
```

## Suggested Next Steps

- Compare FIGER directly against UltraFine to isolate what breaks on very large label spaces
- Inspect per-label errors for low-frequency FIGER types
- Add a short cross-dataset summary covering calibration, label density, and threshold sensitivity
