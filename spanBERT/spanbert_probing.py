import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding


BASE_DIR = Path(__file__).resolve().parent.parent
SPANBERT_DATA_DIR = BASE_DIR / "datasets" / "spanbert_data"
SPANBERT_RESULTS_DIR = BASE_DIR / "results" / "spanbert_finetuned"
SPANBERT_PROBING_DIR = BASE_DIR / "results" / "spanbert_probing"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["ontonotes", "figer", "ultrafine"])
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--search-thresholds", action="store_true")
    parser.add_argument("--threshold-start", type=float, default=-10.0)
    parser.add_argument("--threshold-end", type=float, default=2.0)
    parser.add_argument("--threshold-step", type=float, default=0.25)
    parser.add_argument("--selection-metric", choices=["f1_micro", "f1_macro"], default="f1_micro")
    return parser.parse_args()


def load_spanbert_dataset(dataset_name):
    split_dir = SPANBERT_DATA_DIR / dataset_name
    train_jsonl = split_dir / "train.jsonl"

    if train_jsonl.exists():
        data_files = {}
        for split in ("train", "validation", "test"):
            path = split_dir / f"{split}.jsonl"
            if path.exists() and path.stat().st_size > 0:
                data_files[split] = str(path)
        dataset = load_dataset("json", data_files=data_files)
    else:
        legacy_file = SPANBERT_DATA_DIR / f"spanbert_{dataset_name}.json"
        if not legacy_file.exists():
            raise FileNotFoundError(
                f"Keine vorbereiteten Daten für {dataset_name} gefunden unter {split_dir} oder {legacy_file}"
            )
        dataset = load_dataset("json", data_files={"train": str(legacy_file)})

    if "validation" not in dataset or len(dataset["validation"]) == 0:
        split_once = dataset["train"].train_test_split(test_size=0.2, seed=13)
        validation_test = split_once["test"].train_test_split(test_size=0.5, seed=13)
        dataset = DatasetDict({
            "train": split_once["train"],
            "validation": validation_test["train"],
            "test": validation_test["test"],
        })
    elif "test" not in dataset or len(dataset["test"]) == 0:
        validation_test = dataset["validation"].train_test_split(test_size=0.5, seed=13)
        dataset = DatasetDict({
            "train": dataset["train"],
            "validation": validation_test["train"],
            "test": validation_test["test"],
        })

    return dataset


def build_preprocess_fn(tokenizer, label2id, max_length):
    def preprocess(batch):
        entities = []
        for sentence, start, end in zip(batch["sentence"], batch["start"], batch["end"]):
            tokens = sentence.split()
            entities.append(" ".join(tokens[start:end]))

        encoding = tokenizer(
            batch["sentence"],
            entities,
            truncation=True,
            padding=False,
            max_length=max_length,
        )

        label_vectors = []
        for labels in batch["labels"]:
            label_vec = np.zeros(len(label2id), dtype=np.float32)
            for label in labels:
                label_vec[label2id[label]] = 1.0
            label_vectors.append(label_vec.tolist())

        encoding["labels"] = label_vectors
        return encoding

    return preprocess


def compute_multilabel_metrics(gold, preds):
    gold = gold.astype(np.int32)
    preds = preds.astype(np.int32)

    tp_micro = int((preds & gold).sum())
    fp_micro = int((preds & (1 - gold)).sum())
    fn_micro = int(((1 - preds) & gold).sum())

    precision_micro = tp_micro / (tp_micro + fp_micro) if (tp_micro + fp_micro) > 0 else 0.0
    recall_micro = tp_micro / (tp_micro + fn_micro) if (tp_micro + fn_micro) > 0 else 0.0
    f1_micro = (
        2 * precision_micro * recall_micro / (precision_micro + recall_micro)
        if (precision_micro + recall_micro) > 0
        else 0.0
    )

    tp_per_label = (preds & gold).sum(axis=0).astype(np.float64)
    fp_per_label = (preds & (1 - gold)).sum(axis=0).astype(np.float64)
    fn_per_label = ((1 - preds) & gold).sum(axis=0).astype(np.float64)

    precision_per_label = np.divide(
        tp_per_label,
        tp_per_label + fp_per_label,
        out=np.zeros_like(tp_per_label),
        where=(tp_per_label + fp_per_label) > 0,
    )
    recall_per_label = np.divide(
        tp_per_label,
        tp_per_label + fn_per_label,
        out=np.zeros_like(tp_per_label),
        where=(tp_per_label + fn_per_label) > 0,
    )
    f1_per_label = np.divide(
        2 * precision_per_label * recall_per_label,
        precision_per_label + recall_per_label,
        out=np.zeros_like(tp_per_label),
        where=(precision_per_label + recall_per_label) > 0,
    )

    return {
        "precision_micro": float(precision_micro),
        "recall_micro": float(recall_micro),
        "f1_micro": float(f1_micro),
        "precision_macro": float(precision_per_label.mean()),
        "recall_macro": float(recall_per_label.mean()),
        "f1_macro": float(f1_per_label.mean()),
    }


def build_threshold_grid(start, end, step):
    if step <= 0:
        raise ValueError("--threshold-step muss > 0 sein.")
    if end < start:
        raise ValueError("--threshold-end muss >= --threshold-start sein.")
    num_steps = int(round((end - start) / step))
    grid = [start + idx * step for idx in range(num_steps + 1)]
    if not grid or grid[-1] < end:
        grid.append(end)
    return [round(value, 10) for value in grid]


if __name__ == "__main__":
    args = parse_args()

    model_dir = Path(args.model_dir) if args.model_dir else SPANBERT_RESULTS_DIR / args.dataset / "final_model"
    if not model_dir.exists():
        raise FileNotFoundError(f"Kein SpanBERT-Modell gefunden unter {model_dir}")

    output_dir = SPANBERT_PROBING_DIR / args.dataset / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Dataset:", args.dataset)
    print("Split:", args.split)
    print("Model dir:", model_dir)

    dataset = load_spanbert_dataset(args.dataset)
    eval_split = dataset[args.split]

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()

    label2id = dict(model.config.label2id)
    id2label = {int(idx): label for idx, label in model.config.id2label.items()}

    preprocess_fn = build_preprocess_fn(tokenizer, label2id, args.max_length)
    encoded_eval = eval_split.map(
        preprocess_fn,
        batched=True,
        remove_columns=eval_split.column_names,
        desc=f"Tokenizing {args.dataset} {args.split} split",
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(encoded_eval, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    all_logits = []
    all_gold = []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch.pop("labels")
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            all_logits.append(outputs.logits.cpu().numpy())
            if isinstance(labels, torch.Tensor):
                all_gold.append(labels.cpu().numpy())
            else:
                all_gold.append(np.asarray(labels))

    logits = np.concatenate(all_logits, axis=0)
    gold = np.concatenate(all_gold, axis=0)
    threshold = args.threshold
    threshold_search = None

    if args.search_thresholds:
        threshold_search = []
        for candidate_threshold in build_threshold_grid(
            args.threshold_start,
            args.threshold_end,
            args.threshold_step,
        ):
            candidate_preds = (logits > candidate_threshold).astype(np.int32)
            candidate_metrics = compute_multilabel_metrics(gold, candidate_preds)
            candidate_metrics["threshold"] = float(candidate_threshold)
            candidate_metrics["avg_predicted_labels_per_example"] = float(candidate_preds.sum(axis=1).mean())
            threshold_search.append(candidate_metrics)

        threshold_search.sort(
            key=lambda row: (
                row[args.selection_metric],
                row["f1_micro"],
                row["precision_micro"],
                -abs(row["avg_predicted_labels_per_example"] - float(gold.sum(axis=1).mean())),
                -row["threshold"],
            ),
            reverse=True,
        )
        threshold = float(threshold_search[0]["threshold"])

    preds = (logits > threshold).astype(np.int32)

    metrics = compute_multilabel_metrics(gold, preds)
    metrics["split"] = args.split
    metrics["num_examples"] = int(len(eval_split))
    metrics["threshold"] = float(threshold)
    metrics["selection_metric"] = args.selection_metric
    metrics["threshold_search_enabled"] = bool(args.search_thresholds)
    metrics["avg_gold_labels_per_example"] = float(gold.sum(axis=1).mean())
    metrics["avg_predicted_labels_per_example"] = float(preds.sum(axis=1).mean())
    metrics["logit_min"] = float(logits.min())
    metrics["logit_max"] = float(logits.max())
    metrics["avg_max_logit_per_example"] = float(logits.max(axis=1).mean())

    if threshold_search is not None:
        metrics["best_threshold_by_search"] = float(threshold)
        metrics["threshold_search_top5"] = threshold_search[:5]

    predictions = []
    for row, pred_vec, gold_vec in zip(eval_split, preds, gold):
        predicted_labels = [id2label[idx] for idx, value in enumerate(pred_vec) if value == 1]
        gold_labels = [id2label[idx] for idx, value in enumerate(gold_vec) if value == 1]
        predictions.append({
            "sentence": row["sentence"],
            "start": row["start"],
            "end": row["end"],
            "gold_labels": gold_labels,
            "predicted_labels": predicted_labels,
        })

    metrics_path = output_dir / "metrics.json"
    predictions_path = output_dir / "predictions.json"
    threshold_search_path = output_dir / "threshold_search.json"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    with open(predictions_path, "w", encoding="utf-8") as file:
        json.dump(predictions, file, indent=2, ensure_ascii=False)

    if threshold_search is not None:
        with open(threshold_search_path, "w", encoding="utf-8") as file:
            json.dump(threshold_search, file, indent=2, ensure_ascii=False)

    print("Saved metrics to", metrics_path)
    print("Saved predictions to", predictions_path)
    if threshold_search is not None:
        print("Saved threshold search to", threshold_search_path)
    print(json.dumps(metrics, indent=2))
