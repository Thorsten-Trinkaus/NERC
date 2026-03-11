![GitHub License](https://img.shields.io/github/license/Thorsten-Trinkaus/NERC?style=flat-square)
![GitHub last commit](https://img.shields.io/github/last-commit/Thorsten-Trinkaus/NERC?style=flat-square&color=red)

# NERC of Different Granularities

Course project for **Formale Semantik** (University of Heidelberg).
We investigate **Named Entity Recognition & Classification (NERC)** under **increasing label granularity** (from coarse-grained up to ultra-fine entity typing).

## Datasets

[OntoNotes: The 90% Solution](https://aclanthology.org/N06-2015/) (Hovy et al., NAACL 2006)

[Fine-grained entity recognition (FIGER)](https://ojs.aaai.org/index.php/AAAI/article/view/8122/7980) (Ling and Weld, AAAI 2012)

[Ultra-Fine Entity Typing](https://www.cs.utexas.edu/~eunsol/html_pages/open_entity.html) (Choi et
al., ACL 2018)


# NERC Dataset Analysis – OntoNotes, FIGER, and Ultra-Fine

This subproject focuses on the analysis of datasets used for **Named Entity Recognition and Classification (NERC)** as well as **Fine-Grained Entity Typing**.

The goal is to examine the characteristics and challenges of the following datasets:

- OntoNotes (Hovy et al., NAACL 2006)
- FIGER – Fine-Grained Entity Recognition (Ling & Weld, AAAI 2012)
- Ultra-Fine Entity Typing (Choi et al., ACL 2018)

The analysis focuses on:

- Label granularity  
- Label distribution  
- Ambiguous entities  
- Multi-word entities  
- Challenges for T5-based models  
- Challenges for NLI-based approaches  
- Possible preprocessing strategies  

---

# Overview of the Datasets

| Dataset | Task | Granularity | Multi-Label |
|--------|------|-------------|-------------|
| [OntoNotes](#ontonotes-dataset) | Classical NER | Coarse | No |
| [FIGER](#figer-dataset) | Fine-Grained Entity Typing | Fine | Yes |
| [Ultra-Fine](#ultra-fine-entity-typing-dataset) | Ultra-Fine Entity Typing | Very fine | Yes |

These three datasets represent different levels of complexity in entity typing.

---

# OntoNotes Dataset

The OntoNotes dataset is a well-established benchmark for **classical Named Entity Recognition (NER)**.

### Typical Entity Types

- PERSON
- ORG
- GPE
- LOC
- EVENT
- PRODUCT
- LANGUAGE
- DATE
- MONEY
- WORK_OF_ART

---

## Label Granularity

OntoNotes uses **coarse-grained labels**, meaning relatively general entity categories.

| Entity | Label |
|------|------|
| Barack Obama | PERSON |
| Apple | ORG |
| Berlin | GPE |

These categories are broad and do not distinguish finer subtypes.

Example issue:

Apple → ORG

However, this could also refer to:

- a company  
- a brand  
- a product  

---

## Label Distribution

The label distribution shows a clear **class imbalance**.

Frequent classes:

- PERSON  
- ORG  
- GPE  

Rare classes:

- EVENT  
- LAW  
- LANGUAGE  

This imbalance can affect model performance.

---

## Ambiguous Entities

An example is:

Amazon

Possible meanings:

- ORG (company)  
- LOC (river)  
- GPE (region)

The correct classification strongly depends on the context.

---

## Multi-Word Entities

The dataset contains many **multi-token entities**.

Examples:

- New York City  
- Bank of America  
- United Nations  

Models therefore need to detect **contiguous token spans**.

---

# FIGER Dataset

The FIGER dataset extends NER to **fine-grained entity typing**.

### Number of Types

Approximately **112 different entity types**.

Examples:

- /person/actor  
- /person/politician  
- /location/city  
- /organization/company  
- /organization/sports_team  

---

## Label Granularity

Example:

Entity: Barack Obama

Possible labels:

- person  
- politician  
- president  
- author  

This creates a **hierarchical structure of labels**.

---

## Multi-Label Problem

An entity can have **multiple types simultaneously**.

Example:

Elon Musk

Labels:

- person  
- entrepreneur  
- engineer  
- businessman  

---

## Label Distribution

FIGER also exhibits **strong class imbalance**.

Frequent:

- person  
- organization  
- location  

Rare:

- person/skateboarder  
- person/cartoonist  

---

## Multi-Word Entities

Examples:

- Los Angeles Lakers  
- United States of America  
- New York Stock Exchange  

---

# Ultra-Fine Entity Typing Dataset

This dataset extends entity typing to **extremely fine-grained categories**.

Labels are often **free-form natural language descriptions**.

Examples:

- person  
- father  
- songwriter  
- politician  
- skyscraper  

---

## Label Granularity

An entity can have labels at multiple levels of specificity.

Example:

Trump

Possible labels:

- person  
- businessman  
- president  
- politician  
- celebrity  

---

## Label Distribution

Many labels appear **only a few times in the dataset**, resulting in a **long-tail distribution**.

---

# Challenges for T5

T5 is a **generative sequence-to-sequence model**.

Challenges for NERC tasks include:

- multiple labels per entity  
- structured output requirements  
- open label vocabularies (especially in Ultra-Fine)

Example output:

Barack Obama → person, politician, president

The model must therefore generate **multiple correct labels simultaneously**.

---

# Challenges for NLI-Based Approaches

In NLI-based approaches, entity typing is formulated as a **textual inference problem**.

Example hypothesis:

The entity is a politician.

Problem:

Many labels require many hypotheses.

For example:

100 labels  
→ 100 NLI inferences per entity

This significantly increases **computational cost**.

---

# Preprocessing Strategies

## Entity Span Detection

Example:

[Barack Obama] visited [Berlin]

First, the **entity spans** are identified.

---

## Label Normalization

Example:

sports_team → sports team  
film_actor → actor

This simplifies **generative modeling**.

---

## Splitting Hierarchical Labels

Example:

person/politician

becomes:

person  
politician

---

## Handling Rare Labels

Possible strategies:

- Removing extremely rare labels  
- Merging similar labels  
- Applying few-shot learning techniques  

---

# Summary

The three datasets differ significantly in their complexity:

| Dataset | Granularity | Labels | Difficulty |
|--------|-------------|-------|-------------|
| OntoNotes | Coarse | ~18 | Low |
| FIGER | Fine | ~100 | Medium |
| Ultra-Fine | Very fine | Thousands | High |

As granularity increases, the challenges also grow:

- Multi-label classification  
- Long-tail label distributions  
- Context-dependent interpretation of entities  

These characteristics create different requirements for **T5-based models** and **NLI-based approaches**.
---

> Status: Work in progress (this repo will evolve as experiments and structure solidify!).
