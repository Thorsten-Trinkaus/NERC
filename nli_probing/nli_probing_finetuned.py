from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
import os
import gc
import random
from tqdm import tqdm
from collections import Counter

device = torch.device("cpu")

# -- Config

base = "./onto_Experiments/onto_predictions__none__premise_sep_hypothesis"
test_file = "nli_ontonotes_test.json"
output_file = "ontonotes_test_predictions.json"

#base = "./figer_Experiments/figer_predictions__none__premise_sep_hypothesis"
#test_file = "nli_figer_test.json"
#output_file = "figer_test_predictions.json"

#base = "./ultra_Experiments/figer_predictions__none__premise_sep_hypothesis"
#test_file = "nli_ultra_test.json"
#output_file = "ultra_test_predictions.json"



batch_size = 2
chunk_size = 50000   
num_models = 5

# -- Loading Model Paths
def load_models(base_path):
    models = []
    for d in os.listdir(base_path):
        path = os.path.join(base_path, d)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json")):
            models.append(path)
    return sorted(models)

part_models = load_models(base)
part_models = part_models[-num_models:]

print(f"Using {len(part_models)} models")

# -- Load Test Data
with open(test_file, "r", encoding="utf-8") as f:
    test_data = json.load(f)

random.seed(42)
random.shuffle(test_data)
test_data = test_data[:200000]

print(f"Total test samples: {len(test_data)}")

# -- Results
final_results = []

# -- Chunks
for chunk_start in range(0, len(test_data), chunk_size):
    chunk_end = min(chunk_start + chunk_size, len(test_data))
    print(f"\nProcessing chunk {chunk_start} - {chunk_end}")

    chunk = test_data[chunk_start:chunk_end]

    premises = [x["premise"] for x in chunk]
    hypotheses = [x["hypothesis"] for x in chunk]

    votes_per_example = [Counter() for _ in range(len(chunk))]

    # -- Loop Over All Models
    for part_path in part_models:
        print(f"Model: {part_path}")

        tokenizer = AutoTokenizer.from_pretrained(part_path)
        model = AutoModelForSequenceClassification.from_pretrained(part_path)
        model.to(device)
        model.eval()

        idx_global = 0

        # -- Batching Loop
        for i in tqdm(range(0, len(chunk), batch_size)):
            batch_premises = premises[i:i+batch_size]
            batch_hypotheses = hypotheses[i:i+batch_size]

            inputs = tokenizer(
                batch_premises,
                batch_hypotheses,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )

            with torch.no_grad():
                outputs = model(**inputs)
                preds = torch.argmax(outputs.logits, dim=1).tolist()

            for j, pred in enumerate(preds):
                label = "ENTAILMENT" if pred == 1 else "NOT ENTAILMENT"
                votes_per_example[idx_global + j][label] += 1

            idx_global += len(preds)

            del inputs
            del outputs

        # -- Clear Mem
        del model
        del tokenizer
        gc.collect()

    # -- Merging Votes
    for idx in range(len(chunk)):
        if votes_per_example[idx]:
            final_label = votes_per_example[idx].most_common(1)[0][0]
        else:
            final_label = "NOT ENTAILMENT"

        final_results.append({
            "premise": premises[idx],
            "hypothesis": hypotheses[idx],
            "gold_label": chunk[idx]["label"],
            "prediction": final_label
        })

    gc.collect()

# -- Save Final File
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(final_results, f, ensure_ascii=False, indent=2)
