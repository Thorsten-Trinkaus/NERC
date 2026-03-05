import json
from pathlib import Path
from datasets import load_dataset, load_from_disk
import random


# -- Project Paths --

base_dir = Path(__file__).parent

onto_path = base_dir / "Data" / "OntoNotes"
figer_path = base_dir / "Data" / "FIGER"
ultra_fine_crowd = base_dir / "Data" / "Ultra-Fine" / "ultrafine_acl18" / "release" / "crowd"


# -- Helper Functions --

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


def format_for_t5(sentence, entities):
    t5_data = []
    text = " ".join([t for t,_ in sentence])
    for entity, label in entities:
        t5_input = f"sentence: {text} entity: {entity}"
        t5_target = ",".join(label) if isinstance(label, list) else label.lower()
        t5_data.append({"input": t5_input, "target": t5_target})
    return t5_data


def format_for_nli(sentence, entities, all_labels):
    nli_data = []
    text = " ".join([t for t,_ in sentence])
    for entity, true_label in entities:
        if isinstance(true_label, list):
            true_labels = [l.lower() for l in true_label]
        else:
            true_labels = [true_label.lower()]
        for label in all_labels:
            hypothesis = f"{entity} is a {label.lower()}."
            nli_label = "ENTAILMENT" if label.lower() in true_labels else "NOT ENTAILMENT"
            nli_data.append({"premise": text, "hypothesis": hypothesis, "label": nli_label})
    return nli_data


# -- OntoNotes --

#ds_onto = {
    #"train": load_from_disk(str(onto_path / "train")),
    #"validation": load_from_disk(str(onto_path / "validation")),
    #"test": load_from_disk(str(onto_path / "test")),
#}

#t5_onto = []
#nli_onto = []
#labels_onto = ["PER", "LOC", "ORG", "MISC"]

#label_list = ds_onto["train"].features["ner_tags"].feature.names

# format the data
#for split in ["train", "validation", "test"]:

    #for sent in ds_onto[split]:

        #tokens = sent["tokens"]
        #tag_ids = sent["ner_tags"]

        # convert label-IDs to strings
        #tags = [label_list[i] for i in tag_ids]

        #entities = extract_entities_from_bio(tokens, tags)

        #if not entities:
            #continue

        #sentence = list(zip(tokens, tags))

        #t5_onto.extend(format_for_t5(sentence, entities))
        #nli_onto.extend(format_for_nli(sentence, entities, labels_onto))

# create .json for t5 and nli (OntoNotes dataset)
#with open(base_dir / "t5_ontonotes.json","w",encoding="utf-8") as f:
    ##json.dump(t5_onto,f,ensure_ascii=False,indent=2)

#with open(base_dir / "nli_ontonotes.json","w",encoding="utf-8") as f:
    #json.dump(nli_onto,f,ensure_ascii=False,indent=2)


# -- FIGER --

ds_figer = load_dataset(
    "json",
    data_files={
        "train": str(figer_path / "figer_train.jsonl"),
        "dev": str(figer_path / "figer_dev.jsonl"),
        "test": str(figer_path / "figer_test.jsonl"),
    }
)

# we will only sample 100.000 shuffled train examples
train_sample = ds_figer["train"].shuffle(seed=42).select(range(100_000))

ds_figer = {
    "train": train_sample,
    "dev": ds_figer["dev"],
    "test": ds_figer["test"]
}

t5_figer = []
nli_figer = []

# collect each label from our sampled train
all_labels_figer = set()
for row in ds_figer["train"]:
    for l in row["labels"]:
        all_labels_figer.add(l.lower())

all_labels_figer = list(all_labels_figer)

# due to the size of the FIGER dataset we have decided to
# only use 5 negative labels for the NLI (and all positive ones)

def format_for_nli_safe(sentence, entities, all_labels, max_negatives=5):
    nli_data = []
    text = " ".join([t for t,_ in sentence])

    for entity, true_labels in entities:
        true_labels = [l.lower() for l in true_labels]

        # positive examples
        for label in true_labels:
            hypothesis = f"{entity} is a {label}."
            nli_data.append({
                "premise": text,
                "hypothesis": hypothesis,
                "label": "ENTAILMENT"
            })

        # negative sampling
        negative_pool = list(set(all_labels) - set(true_labels))
        sampled_negatives = random.sample(
            negative_pool,
            min(max_negatives, len(negative_pool))
            )

        for label in sampled_negatives:
            hypothesis = f"{entity} is a {label}."
            nli_data.append({
                "premise": text,
                "hypothesis": hypothesis,
                "label": "NOT ENTAILMENT"
            })

    return nli_data

# format the data
for split in ["train", "dev", "test"]:
    for row in ds_figer[split]:

        tokens = row["tokens"]
        mention = row["entityName"]
        labels = row["labels"]

        if not labels:
            continue

        sentence = list(zip(tokens, ["O"] * len(tokens)))
        entities = [(mention, labels)]

        t5_figer.extend(format_for_t5(sentence, entities))
        nli_figer.extend(
            format_for_nli_safe(sentence, entities, all_labels_figer, max_negatives=5)
        )
print("FIGER:", len(t5_figer), len(nli_figer))

# create .json for t5 and nli (FIGER dataset)
with open(base_dir / "t5_figer.json", "w", encoding="utf-8") as f:
    json.dump(t5_figer, f, ensure_ascii=False, indent=2)

with open(base_dir / "nli_figer.json", "w", encoding="utf-8") as f:
    json.dump(nli_figer, f, ensure_ascii=False, indent=2)


# -- Ultra-Fine --

#ultra_fine_ds = load_dataset(
    #"json",
    #data_files={
        #"train": str(ultra_fine_crowd / "train.json"),
        #"validation": str(ultra_fine_crowd / "dev.json"),
        #"test": str(ultra_fine_crowd / "test.json")
    #}
#)

#t5_ultra = []
#nli_ultra = []

# collect each label
#all_labels_ultra = set()
#for split in ["train","validation","test"]:
    #for row in ultra_fine_ds[split]:
        #all_labels_ultra.update([l.lower() for l in row.get("y_str", [])])
#all_labels_ultra = list(all_labels_ultra)

# format the data
#for split in ["train","validation","test"]:
    #for row in ultra_fine_ds[split]:

        #left = " ".join(row.get("left_context_token", []))
        #mention = row.get("mention_span", "")
        #right = " ".join(row.get("right_context_token", []))

        #text = f"{left} {mention} {right}".strip()

        #labels = row.get("y_str", [])
        #if not labels:
            #continue

        #entities = [(mention, labels)]
        #tokens = [(t,"O") for t in text.split()]

        #t5_ultra.extend(format_for_t5(tokens, entities))
        #nli_ultra.extend(format_for_nli(tokens, entities, all_labels_ultra))

# create .json for t5 and nli (ultra-fine dataset)
#with open(base_dir / "t5_ultra.json","w",encoding="utf-8") as f:
    #json.dump(t5_ultra,f,ensure_ascii=False,indent=2)
#with open(base_dir / "nli_ultra.json","w",encoding="utf-8") as f:
    #json.dump(nli_ultra,f,ensure_ascii=False,indent=2)
