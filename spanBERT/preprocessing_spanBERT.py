import argparse
import gzip
import json
import random
from collections import defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent / "datasets"
FIGER_PATH = BASE_DIR / "figer"
ULTRA_FINE_CROWD = BASE_DIR / "ultra_fine" / "crowd"
ULTRA_FINE_DS = BASE_DIR / "ultra_fine" / "ds"
ONTO_CANDIDATES = [
    BASE_DIR / "ontonotes",
    BASE_DIR / "onto",
    BASE_DIR / "ontonotes_json",
]

OUTPUT_DIR = BASE_DIR / "spanbert_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ONTO_LEGACY_PATH = OUTPUT_DIR / "spanbert_ontonotes.json"


def load_dataset(dataset_format, data_files):
    if dataset_format != "json":
        raise ValueError("Nur json wird unterstützt")

    loaded = {}
    for split, file_path in data_files.items():
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset-Datei nicht gefunden: {path}")
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            if not text:
                loaded[split] = []
            elif text[0] == "[":
                loaded[split] = json.loads(text)
            else:
                loaded[split] = read_ndjson(path)
    return loaded


def read_ndjson(path):
    data = []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
                continue
            except json.JSONDecodeError:
                pass

            start = 0
            while start < len(line):
                try:
                    obj, next_idx = json.JSONDecoder().raw_decode(line, start)
                    data.append(obj)
                    start = next_idx
                except json.JSONDecodeError:
                    break
                while start < len(line) and line[start].isspace():
                    start += 1

    if not data:
        text = Path(path).read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            return []
        if text.startswith("["):
            return json.loads(text)

        decoder = json.JSONDecoder()
        idx = 0
        length = len(text)
        while idx < length:
            try:
                obj, next_idx = decoder.raw_decode(text, idx)
            except ValueError:
                break
            data.append(obj)
            idx = next_idx
            while idx < length and text[idx].isspace():
                idx += 1

    return data


def extract_entities_from_bio(tokens, tags):
    entities = []
    start = None
    label = None

    for i, tag in enumerate(tags):
        if tag.startswith("B-"):
            if start is not None:
                entities.append((" ".join(tokens[start:i]), label))
            start = i
            label = tag[2:]
        elif tag.startswith("I-") and start is not None:
            continue
        else:
            if start is not None:
                entities.append((" ".join(tokens[start:i]), label))
                start = None
                label = None

    if start is not None:
        entities.append((" ".join(tokens[start:]), label))

    return entities


def find_span(tokens, entity_tokens):
    for i in range(len(tokens)):
        if tokens[i:i + len(entity_tokens)] == entity_tokens:
            return i, i + len(entity_tokens)
    return None, None


def format_for_spanbert(tokens, entity_text, labels):
    entity_tokens = entity_text.split()
    start, end = find_span(tokens, entity_tokens)

    if start is None:
        return None

    return {
        "sentence": " ".join(tokens),
        "start": start,
        "end": end,
        "labels": sorted({label.lower() for label in labels}),
    }


def resolve_validation_file(base_path):
    for candidate in ("validation.json", "dev.json"):
        path = base_path / candidate
        if path.exists():
            return path
    return None


def check_required_files(base_path, required_files):
    missing = [f for f in required_files if not (base_path / f).exists()]
    if missing:
        present = [f for f in required_files if (base_path / f).exists()]
        print(f"Verzeichnis: {base_path}")
        print(f"Anwesende Dateien: {present}")
        print(f"Fehlende Dateien: {missing}")
    return missing


def split_records(records, seed=13, train_ratio=0.9, validation_ratio=0.05):
    shuffled = list(records)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_validation = int(n_total * validation_ratio)

    return {
        "train": shuffled[:n_train],
        "validation": shuffled[n_train:n_train + n_validation],
        "test": shuffled[n_train + n_validation:],
    }


def read_varint(stream):
    shift = 0
    result = 0
    while True:
        byte = stream.read(1)
        if not byte:
            return None
        value = byte[0]
        result |= (value & 0x7F) << shift
        if not (value & 0x80):
            return result
        shift += 7


def write_outputs(dataset_name, split_records_map):
    dataset_dir = OUTPUT_DIR / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    for split in ("train", "validation", "test"):
        records = split_records_map.get(split, [])
        all_records.extend(records)

        jsonl_path = dataset_dir / f"{split}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for row in records:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        if split == "validation":
            dev_jsonl_path = dataset_dir / "dev.jsonl"
            with open(dev_jsonl_path, "w", encoding="utf-8") as f:
                for row in records:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

    legacy_path = OUTPUT_DIR / f"spanbert_{dataset_name}.json"
    with open(legacy_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    print(
        f"Gespeichert für {dataset_name}: "
        f"train={len(split_records_map.get('train', []))}, "
        f"validation={len(split_records_map.get('validation', []))}, "
        f"test={len(split_records_map.get('test', []))}"
    )


def find_ontonotes_path():
    for candidate in ONTO_CANDIDATES:
        validation_path = resolve_validation_file(candidate)
        if (candidate / "train.json").exists() and validation_path and (candidate / "test.json").exists():
            return candidate
    return None


def process_ontonotes():
    print("Processing OntoNotes...")

    onto_path = find_ontonotes_path()
    if onto_path is None:
        if ONTO_LEGACY_PATH.exists():
            print(f"OntoNotes-Rohdaten nicht gefunden, verwende Legacy-Dump: {ONTO_LEGACY_PATH}")
            try:
                records = json.loads(ONTO_LEGACY_PATH.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"Legacy-Dump konnte nicht geladen werden: {e}")
                return False

            split_records_map = split_records(records)
            write_outputs("ontonotes", split_records_map)
            return True

        print(
            "OntoNotes-Quelle nicht gefunden. Erwartet wird z.B. "
            "datasets/ontonotes/{train,validation|dev,test}.json"
        )
        return False

    validation_path = resolve_validation_file(onto_path)
    dataset = load_dataset("json", data_files={
        "train": str(onto_path / "train.json"),
        "validation": str(validation_path),
        "test": str(onto_path / "test.json"),
    })

    id2label = {
        0: "O", 1: "B-PER", 2: "I-PER",
        3: "B-ORG", 4: "I-ORG",
        5: "B-LOC", 6: "I-LOC",
        7: "B-MISC", 8: "I-MISC",
    }

    split_records_map = defaultdict(list)
    for split, rows in dataset.items():
        for row in rows:
            tokens = row["tokens"]
            tags = [id2label[i] for i in row["ner_tags"]]
            entities = extract_entities_from_bio(tokens, tags)

            for entity_text, label in entities:
                sample = format_for_spanbert(tokens, entity_text, [label])
                if sample:
                    split_records_map[split].append(sample)

    write_outputs("ontonotes", split_records_map)
    return True


def process_figer():
    print("Processing FIGER...")

    validation_path = resolve_validation_file(FIGER_PATH)
    json_ready = (
        (FIGER_PATH / "train.json").exists()
        and validation_path is not None
        and (FIGER_PATH / "test.json").exists()
    )

    split_records_map = defaultdict(list)

    if json_ready:
        dataset = load_dataset("json", data_files={
            "train": str(FIGER_PATH / "train.json"),
            "validation": str(validation_path),
            "test": str(FIGER_PATH / "test.json"),
        })

        for split, rows in dataset.items():
            for row in rows:
                tokens = row.get("tokens") or row.get("sentence", "").split()
                labels = row.get("labels", [])
                start = row.get("start")
                end = row.get("end")
                if labels and start is not None and end is not None:
                    split_records_map[split].append({
                        "sentence": " ".join(tokens),
                        "start": start,
                        "end": end,
                        "labels": sorted({label.lower() for label in labels}),
                    })
    else:
        hf_loaded = False
        try:
            from datasets import load_dataset as hf_load_dataset

            print("JSON-Dateien fehlen, versuche FIGER von Hugging Face zu laden ...")
            hf_dataset = hf_load_dataset("DGME/figer")

            for split in hf_dataset.keys():
                for row in hf_dataset[split]:
                    left = row.get("left_context_token", [])
                    mention_tokens = row.get("mention_span", [])
                    right = row.get("right_context_token", [])
                    labels = row.get("y_str", [])

                    if not mention_tokens or not labels:
                        continue

                    if isinstance(mention_tokens, str):
                        mention_tokens = mention_tokens.split()

                    tokens = left + mention_tokens + right
                    sample = {
                        "sentence": " ".join(tokens),
                        "start": len(left),
                        "end": len(left) + len(mention_tokens),
                        "labels": sorted({label.lower() for label in labels}),
                    }
                    split_records_map[split].append(sample)

            hf_loaded = True
            print("FIGER erfolgreich von Hugging Face geladen.")
        except Exception as e:
            print(f"Hugging-Face-Download für FIGER fehlgeschlagen: {e}")

        if hf_loaded:
            write_outputs("figer", split_records_map)
            return True

    if (FIGER_PATH / "train.data.gz").exists():
        print("JSON-Dateien fehlen, lese FIGER aus train.data.gz ...")
        import sys

        sys.path.append(str(FIGER_PATH))
        from entity_pb2 import Mention

        parsed_records = []
        with gzip.open(FIGER_PATH / "train.data.gz", "rb") as f:
            while True:
                size = read_varint(f)
                if size is None:
                    break
                blob = f.read(size)
                if not blob:
                    break

                mention = Mention()
                mention.ParseFromString(blob)

                if mention.start < 0 or mention.end > len(mention.tokens) or mention.start >= mention.end:
                    continue
                if not mention.labels:
                    continue

                parsed_records.append({
                    "sentence": " ".join(mention.tokens),
                    "start": mention.start,
                    "end": mention.end,
                    "labels": sorted({label.lower() for label in mention.labels}),
                })

        split_records_map.update(split_records(parsed_records))
    else:
        print("FIGER-Quelle nicht gefunden. Erwartet wird train.data.gz oder train/dev/test JSON.")
        return False

    write_outputs("figer", split_records_map)
    return True


def process_ultrafine(include_ds=False):
    print("Processing Ultra-Fine...")

    train_path = ULTRA_FINE_CROWD / "train.json"
    validation_path = resolve_validation_file(ULTRA_FINE_CROWD)
    test_path = ULTRA_FINE_CROWD / "test.json"

    if not train_path.exists() or validation_path is None or not test_path.exists():
        print(
            "UltraFine-Crowd-Quelle unvollständig. Erwartet wird "
            "datasets/ultra_fine/crowd/{train,dev|validation,test}.json"
        )
        return False

    dataset = load_dataset("json", data_files={
        "train": str(train_path),
        "validation": str(validation_path),
        "test": str(test_path),
    })

    split_records_map = defaultdict(list)

    for split, rows in dataset.items():
        for row in rows:
            left = row.get("left_context_token", [])
            mention = row.get("mention_span", "")
            right = row.get("right_context_token", [])
            labels = row.get("y_str", [])

            if not mention or not labels:
                continue

            tokens = left + mention.split() + right
            sample = format_for_spanbert(tokens, mention, labels)
            if sample:
                split_records_map[split].append(sample)

    extra_count = 0
    if include_ds and ULTRA_FINE_DS.exists():
        for ds_file in [
            "headword_train.json",
            "headword_dev.json",
            "headword_test.json",
            "el_train.json",
            "el_dev.json",
            "augmented_train.json",
        ]:
            path = ULTRA_FINE_DS / ds_file
            if not path.exists():
                continue
            try:
                rows = read_ndjson(path)
            except Exception as e:
                print(f"Fehler beim Laden von {path}: {e}")
                continue

            target_split = "train"
            if "dev" in ds_file:
                target_split = "validation"
            elif "test" in ds_file:
                target_split = "test"

            for row in rows:
                left = row.get("left_context_token", [])
                mention = row.get("mention_span", "")
                right = row.get("right_context_token", [])
                labels = row.get("y_str", []) or row.get("y_type_str", [])

                if not mention or not labels:
                    continue

                tokens = left + mention.split() + right
                sample = format_for_spanbert(tokens, mention, labels)
                if sample:
                    split_records_map[target_split].append(sample)
                    extra_count += 1

    write_outputs("ultrafine", split_records_map)
    print(f"Ultra-Fine zusätzliche ds-Beispiele: {extra_count}")
    return True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ontonotes", "figer", "ultrafine"],
        choices=["ontonotes", "figer", "ultrafine"],
        help="Welche Datensätze vorbereitet werden sollen.",
    )
    parser.add_argument(
        "--include-ultrafine-ds",
        action="store_true",
        help="Zusätzliche UltraFine-ds-Dateien einmischen. Standard ist nur crowd, um die Größe beherrschbar zu halten.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    processors = {
        "ontonotes": process_ontonotes,
        "figer": process_figer,
        "ultrafine": lambda: process_ultrafine(include_ds=args.include_ultrafine_ds),
    }

    for dataset_name in args.datasets:
        processors[dataset_name]()
