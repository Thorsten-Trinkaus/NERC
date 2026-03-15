import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# to check which device is being used:
print("Device:", device)

model_name = "textattack/roberta-base-mnli"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

model.to(device)
model.eval()

def run_nli_probing(input_file, output_file, batch_size=32):
    
    # to check which dataset is currently being loaded
    print(f"\nLoading Dataset: {input_file}")

    with open(input_file, encoding="utf-8") as f:
        data = json.load(f)
    
    results = []

    
    premises = [x["premise"] for x in data]
    hypotheses = [x["hypothesis"] for x in data]
    
    for i in tqdm(range(0, len(data), batch_size)):

        batch_premises = premises[i:i+batch_size]
        batch_hypotheses = hypotheses[i:i+batch_size]

        inputs = tokenizer(
            batch_premises,
            batch_hypotheses,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()

        for j, pred in enumerate(preds):
            if pred == 2:
                prediction = "ENTAILMENT"
            else:
                prediction = "NOT ENTAILMENT"

            results.append({
                "premise": batch_premises[j],
                "hypothesis": batch_hypotheses[j],
                "gold_label": data[i+j]["label"],
                "prediction": prediction
            })

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


datasets = {
    #"nli_ontonotes.json": "ontonotes_predictions.json",
    "nli_figer.json": "figer_predictions.json",
    #"nli_ultra.json": "ultrafine_predictions.json"
}

for dataset, output in datasets.items():
    print("Currently running NLI probing for:", dataset)

    run_nli_probing(dataset,output)
