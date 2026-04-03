# SpanBERT OntoNotes Results

This directory contains the evaluation artifacts for the OntoNotes SpanBERT experiments.

## Final Setup

- Base model: `SpanBERT/spanbert-base-cased`
- Fine-tuned model used for the final run:
  `results/spanbert_finetuned/ontonotes/final_model`
- Validation-selected threshold for final evaluation:
  `0.2`

## Final Test Metrics

Source: `test/metrics.json`

| Metric | Value |
|---|---:|
| precision_micro | 0.9732 |
| recall_micro | 0.9726 |
| f1_micro | 0.9729 |
| precision_macro | 0.9721 |
| recall_macro | 0.9734 |
| f1_macro | 0.9727 |
| avg_gold_labels_per_example | 1.0 |
| avg_predicted_labels_per_example | 0.9994 |
| threshold | 0.2 |

## Main Findings

1. SpanBERT works extremely well on OntoNotes.
   The final test score reaches `f1_micro = 0.9729`, which is by far the strongest result among the three current SpanBERT runs.

2. The model is very well calibrated.
   OntoNotes is effectively a single-label setup, and the model predicts almost exactly one label per example on average (`0.9994` vs. `1.0` gold).

3. Threshold tuning only made a very small difference.
   The validation search selected `0.2`, but neighboring thresholds performed almost identically. This suggests that the decision boundary is already stable.

4. Micro and macro scores are almost identical.
   This indicates that performance is strong and balanced across the small OntoNotes label set instead of being carried only by a few dominant classes.

## Interpretation

OntoNotes is the most straightforward dataset for the current SpanBERT setup. The model is both accurate and stable, and threshold calibration is almost trivial compared with the more fine-grained multi-label datasets. This makes OntoNotes a strong baseline and a useful upper-bound reference for the rest of the project.

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
sbatch job_spanbert_probe.sh ontonotes validation \
  --search-thresholds \
  --threshold-start -0.5 \
  --threshold-end 1.0 \
  --threshold-step 0.05 \
  --selection-metric f1_micro
```

Final test run:

```bash
sbatch job_spanbert_probe.sh ontonotes test \
  --threshold 0.2
```

## Suggested Next Steps

- Use OntoNotes as the stable reference point when comparing harder datasets
- Add a short cross-dataset comparison section against FIGER and UltraFine
- Inspect a few rare failure cases in `test/predictions.json` for qualitative examples
