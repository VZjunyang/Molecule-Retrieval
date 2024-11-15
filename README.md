# Identifying Molecules from Natural Language Queries

## Overview

This project focuses on building a pipeline to retrieve corresponding molecular structures based on natural language queries (NLD) and molecular graphs. The primary goal is to assess the benefits of **multimodularity** over **unimodularity** in machine learning models.

---

## Project Scope

**Objective:**  
Develop a system to map natural language queries to molecular structures. The main challenges involve:

1. Integrating natural language descriptions with graph data.
2. Assigning these descriptions to molecular graphs.

---

## Problem Description

This project uses **AI and ML techniques** to:

- Extract and understand natural language descriptions (NLD) of molecular structures.
- Link these descriptions to molecular graphs using multimodal machine learning models.

---

## Dataset Details

The dataset originates from a Kaggle challenge by the **École Normale Supérieure Paris-Saclay** and includes the following subsets:

| Set               | Molecule/NLD Pairs |
|--------------------|--------------------|
| **Training Set**   | 23,408            |
| **Validation Set** | 3,301             |
| **Test Set**       | 3,301             |

Additionally, vector embeddings for **102,981 molecule graphs** are available. These embeddings correspond to molecules in the training, validation, and test sets.

---

## Implementation Plan

### Step 1: Establish a Baseline
- Use **ensemble models** such as XGBoost.
- Generate NLP-based features with pre-trained LLMs (e.g., BERT, SciBERT, DistilBERT).

### Step 2: Incorporate Graph-Based Features
- Utilize **Graph Neural Networks (GNNs)** to capture molecular graph structures.
- Integrate these graph-based features into the model.

### Step 3: Explore Complex Predictive Models
- Implement **Convolutional Neural Networks (CNNs)** to enhance feature extraction.

### Step 4: Fine-Tune and Optimize
- Perform hyperparameter tuning to optimize performance.

### Step 5: Experiment with Neural Network Ensembles
- Combine multiple models for improved accuracy and robustness.

---

## Challenges

### Key Challenges
1. **Graph Structure Complexity:**  
   Incorporating molecular graphs into the learning process is challenging due to their intricate nature. **GNNs** and multimodular approaches will be key solutions.

2. **Metric Selection:**  
   Identifying the appropriate metrics for evaluating model performance.

3. **Computational Efficiency:**  
   The large dataset requires computationally efficient methods to ensure timely execution.

---

## Goals and Significance

This project bridges the gap between **natural language processing (NLP)** and **graph-based machine learning**. It applies concepts explored in coursework to address a complex, real-world problem.

---

## Authors

- **Victor Zhuang** ([vjzhuang@mit.edu](mailto:vjzhuang@mit.edu))
- **Noé Bertramo** ([noe_bert@mit.edu](mailto:noe_bert@mit.edu))
- **Badiss Ben Abdallah** ([badiss26@mit.edu](mailto:badiss26@mit.edu))

---

## References

The project idea is based on a Kaggle challenge by **École Normale Supérieure Paris-Saclay**. For more details, visit the challenge's Kaggle repository.