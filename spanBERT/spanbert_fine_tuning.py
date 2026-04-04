import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import DatasetDict, load_dataset
from torch.optim import AdamW
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


BASE_DIR = Path(__file__).resolve().parent.parent
SPANBERT_DATA_DIR = BASE_DIR / "datasets" / "spanbert_data"
MODEL_NAME = "SpanBERT/spanbert-base-cased"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["ontonotes", "figer", "ultrafine"])
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--use-pos-weight", action="store_true")
    parser.add_argument("--pos-weight-max", type=float, default=100.0)
    parser.add_argument("--eval-threshold", type=float, default=0.0)
    parser.add_argument("--metric-for-best-model", choices=["f1", "loss"], default=None)
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
                f"Keine vorbereiteten Daten fuer {dataset_name} gefunden unter {split_dir} oder {legacy_file}"
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


def maybe_truncate(dataset, split_name, max_samples):
    if max_samples is None:
        return dataset
    return dataset.select(range(min(len(dataset), max_samples)))


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


def build_compute_metrics(eval_threshold):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = (logits > eval_threshold).astype(np.int32)
        labels = labels.astype(np.int32)

        true_positives = int((preds & labels).sum())
        false_positives = int((preds & (1 - labels)).sum())
        false_negatives = int(((1 - preds) & labels).sum())

        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (true_positives + false_negatives + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "avg_predicted_labels": float(preds.sum(axis=1).mean()),
        }

    return compute_metrics


def tensor_stats(name, tensor):
    tensor = tensor.detach()
    stats_tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    if not stats_tensor.is_floating_point():
        stats_tensor = stats_tensor.to(torch.float32)
    return {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "nan": bool(torch.isnan(tensor).any().item()),
        "inf": bool(torch.isinf(tensor).any().item()),
        "min": float(stats_tensor.min().item()),
        "max": float(stats_tensor.max().item()),
        "mean": float(stats_tensor.mean().item()),
    }


def compute_pos_weight(train_dataset, label2id, pos_weight_max):
    num_examples = len(train_dataset)
    pos_counts = np.zeros(len(label2id), dtype=np.float64)

    for labels in train_dataset["labels"]:
        for label in labels:
            pos_counts[label2id[label]] += 1.0

    neg_counts = num_examples - pos_counts
    safe_pos_counts = np.where(pos_counts > 0, pos_counts, 1.0)
    pos_weight = neg_counts / safe_pos_counts
    pos_weight = np.clip(pos_weight, 1.0, pos_weight_max)
    pos_weight = np.where(pos_counts > 0, pos_weight, 1.0)

    stats = {
        "num_examples": int(num_examples),
        "num_labels": int(len(label2id)),
        "labels_with_zero_positives": int((pos_counts == 0).sum()),
        "labels_with_single_positive": int((pos_counts == 1).sum()),
        "min_pos_count": float(pos_counts.min()) if len(pos_counts) > 0 else 0.0,
        "max_pos_count": float(pos_counts.max()) if len(pos_counts) > 0 else 0.0,
        "mean_pos_count": float(pos_counts.mean()) if len(pos_counts) > 0 else 0.0,
        "min_pos_weight": float(pos_weight.min()) if len(pos_weight) > 0 else 0.0,
        "median_pos_weight": float(np.median(pos_weight)) if len(pos_weight) > 0 else 0.0,
        "max_pos_weight": float(pos_weight.max()) if len(pos_weight) > 0 else 0.0,
        "mean_pos_weight": float(pos_weight.mean()) if len(pos_weight) > 0 else 0.0,
    }
    return torch.tensor(pos_weight, dtype=torch.float32), stats


def compute_bce_loss(logits, labels, pos_weight=None):
    labels = labels.to(device=logits.device, dtype=logits.dtype)
    if pos_weight is not None:
        pos_weight = pos_weight.to(device=logits.device, dtype=logits.dtype)
    loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return loss_fct(logits, labels)


class WeightedMultiLabelTrainer(Trainer):
    def __init__(self, *args, pos_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = compute_bce_loss(outputs.logits, labels, pos_weight=self.pos_weight)
        return (loss, outputs) if return_outputs else loss


def run_sanity_check(model, dataset, tokenizer, batch_size, learning_rate, device, pos_weight=None):
    print("[SANITY] Running one-batch numerical sanity check...")

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    batch_examples = [dataset["train"][i] for i in range(min(batch_size, len(dataset["train"])))]
    batch = collator(batch_examples)

    print("[SANITY]", tensor_stats("input_ids", batch["input_ids"]))
    print("[SANITY]", tensor_stats("attention_mask", batch["attention_mask"]))
    print("[SANITY]", tensor_stats("labels", batch["labels"]))
    if pos_weight is not None:
        print("[SANITY]", tensor_stats("pos_weight", pos_weight))

    labels = batch["labels"]
    batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
        if key != "labels"
    }
    labels = labels.to(device)

    model.train()
    model.zero_grad(set_to_none=True)

    outputs = model(**batch)
    loss = compute_bce_loss(outputs.logits, labels, pos_weight=pos_weight)
    print("[SANITY]", tensor_stats("logits_before_backward", outputs.logits))
    print("[SANITY] loss_before_backward", float(loss.detach().cpu().item()))

    if torch.isnan(loss) or torch.isinf(loss):
        raise ValueError("[SANITY] Loss is NaN/Inf before backward.")

    loss.backward()

    grad_has_nan = False
    grad_has_inf = False
    sample_grad_stats = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad_has_nan = grad_has_nan or torch.isnan(param.grad).any().item()
        grad_has_inf = grad_has_inf or torch.isinf(param.grad).any().item()
        if len(sample_grad_stats) < 5:
            sample_grad_stats.append(tensor_stats(f"grad::{name}", param.grad))

    print("[SANITY] grad_has_nan", grad_has_nan)
    print("[SANITY] grad_has_inf", grad_has_inf)
    for stat in sample_grad_stats:
        print("[SANITY]", stat)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    param_has_nan = False
    param_has_inf = False
    sample_param_stats = []
    for name, param in model.named_parameters():
        param_has_nan = param_has_nan or torch.isnan(param).any().item()
        param_has_inf = param_has_inf or torch.isinf(param).any().item()
        if len(sample_param_stats) < 5:
            sample_param_stats.append(tensor_stats(f"param::{name}", param))

    print("[SANITY] param_has_nan_after_step", param_has_nan)
    print("[SANITY] param_has_inf_after_step", param_has_inf)
    for stat in sample_param_stats:
        print("[SANITY]", stat)

    if grad_has_nan or grad_has_inf or param_has_nan or param_has_inf:
        raise ValueError("[SANITY] Numerical issue detected during the one-batch check.")

    model.zero_grad(set_to_none=True)
    model.eval()
    print("[SANITY] One-batch numerical sanity check passed.")


if __name__ == "__main__":
    args = parse_args()

    print("[DEBUG] Current working dir:", os.getcwd())
    print("[DEBUG] PyTorch version:", torch.__version__)
    print("[DEBUG] CUDA available:", torch.cuda.is_available())
    print("[DEBUG] Selected dataset:", args.dataset)
    print("[DEBUG] Use pos_weight:", args.use_pos_weight)
    print("[DEBUG] Eval threshold:", args.eval_threshold)

    dataset = load_spanbert_dataset(args.dataset)
    dataset["train"] = maybe_truncate(dataset["train"], "train", args.max_train_samples)
    dataset["validation"] = maybe_truncate(dataset["validation"], "validation", args.max_eval_samples)
    dataset["test"] = maybe_truncate(dataset["test"], "test", args.max_eval_samples)

    all_labels = sorted({
        label
        for split in dataset.keys()
        for labels in dataset[split]["labels"]
        for label in labels
    })
    label2id = {label: idx for idx, label in enumerate(all_labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    print("[DEBUG] Number of labels:", len(all_labels))
    print("[DEBUG] Split sizes:", {split: len(ds) for split, ds in dataset.items()})

    pos_weight = None
    if args.use_pos_weight:
        pos_weight, pos_weight_stats = compute_pos_weight(
            dataset["train"],
            label2id=label2id,
            pos_weight_max=args.pos_weight_max,
        )
        print("[DEBUG] pos_weight stats:", pos_weight_stats)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    preprocess_fn = build_preprocess_fn(tokenizer, label2id, args.max_length)

    encoded_dataset = dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc=f"Tokenizing {args.dataset}",
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(all_labels),
        problem_type="multi_label_classification",
        label2id=label2id,
        id2label=id2label,
        torch_dtype=torch.float32,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dtype=torch.float32)
    model.to(device)

    run_sanity_check(
        model=model,
        dataset=encoded_dataset,
        tokenizer=tokenizer,
        batch_size=args.train_batch_size,
        learning_rate=args.learning_rate,
        device=device,
        pos_weight=pos_weight,
    )

    metric_for_best_model = args.metric_for_best_model
    if metric_for_best_model is None:
        metric_for_best_model = "loss" if args.use_pos_weight and args.dataset == "ultrafine" else "f1"
    greater_is_better = metric_for_best_model != "loss"
    print("[DEBUG] metric_for_best_model:", metric_for_best_model)

    output_dir = args.output_dir or str(BASE_DIR / "results" / "spanbert_finetuned" / args.dataset)
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        report_to="none",
        fp16=False,
        logging_nan_inf_filter=False,
    )

    trainer = WeightedMultiLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=build_compute_metrics(args.eval_threshold),
        pos_weight=pos_weight,
    )

    trainer.train()
    test_metrics = trainer.evaluate(encoded_dataset["test"], metric_key_prefix="test")
    print("[DEBUG] Test metrics:", test_metrics)

    final_dir = Path(output_dir) / "final_model"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"[DEBUG] Model and tokenizer saved to {final_dir}")
