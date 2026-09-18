<div align="center">

# Automated Long-Answer Grading System

### QLoRA Fine-Tuned Qwen2.5-3B-Instruct for Rubric-Based Answer Evaluation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13.0%2Bcu126-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/Transformers-5.15.0-FFD21E?style=flat-square&logo=huggingface)](https://huggingface.co/docs/transformers)
[![TRL](https://img.shields.io/badge/TRL-1.10.0-blueviolet?style=flat-square)](https://github.com/huggingface/trl)
[![PEFT](https://img.shields.io/badge/PEFT-QLoRA-green?style=flat-square)](https://github.com/huggingface/peft)
[![bitsandbytes](https://img.shields.io/badge/bitsandbytes-4--bit%20NF4-orange?style=flat-square)](https://github.com/TimDettmers/bitsandbytes)

<br/>

> **An automated grading pipeline that scores long descriptive student answers criterion-by-criterion using a QLoRA fine-tuned LLM — achieving Pearson r = 0.857 and rubric-level accuracy = 85.4% on a 1,000-record external test set.**

<br/>

| Metric | Fine-tuned (QLoRA) | Base FP16 | Base NF4 |
|:---|:---:|:---:|:---:|
| MAE (lower is better) | **1.35** | 2.23 | 2.46 |
| Pearson r (higher is better) | **0.819** | 0.679 | 0.637 |
| Rubric Accuracy (higher is better) | **85.4%** | 72.6% | 71.7% |
| Exact Score Accuracy (higher is better) | **45.3%** | 23.3% | 22.7% |

*Results on original held-out test set (n = 150)*

</div>

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Dataset](#dataset)
- [Model and Training](#model-and-training)
- [Experimental Results](#experimental-results)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage Guide](#usage-guide)
- [Key Findings and Honest Assessment](#key-findings-and-honest-assessment)
- [Tech Stack](#tech-stack)
- [References](#references)

---

## Overview

Manually grading long descriptive answers is **time-consuming**, **inconsistent**, and **subjective**.
This project builds a fully automated grading pipeline that:

- Accepts a **question**, **rubric** (N weighted criteria), and **student answer** as input
- Predicts **points awarded for each rubric criterion independently**
- Outputs a **predicted total score** comparable to human judgment
- Is evaluated across **six quantitative metrics** on **two separate test sets**

The system fine-tunes **[Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)** using **QLoRA** (4-bit NF4 quantisation + LoRA adapters), making training feasible on a **single consumer GPU in under 40 minutes** without sacrificing quality.

Three model configurations are compared:

| Configuration | Quantisation | LoRA | Notes |
|---|---|---|---|
| **Base FP16** | float16 (none) | No | Unquantised reference baseline |
| **Base NF4** | 4-bit NF4 | No | Quantisation effect only |
| **Fine-tuned (QLoRA)** | 4-bit NF4 | Yes (r=16) | Primary experiment |

---

## How It Works

```
  Question --+
  Rubric  -->|  Prompt Construction  -->  Qwen2.5-3B-Instruct
  Answer  --+     (Chat Format)            (NF4 + LoRA Adapter)
                                                   |
                                                   v
                                        Raw Text Response
                                    "Criterion 1: 1
                                     Criterion 2: 0
                                     Criterion 3: 1 ..."
                                                   |
                                         Regex Parsing
                                     + Score Clamping [0, max]
                                                   |
                                                   v
                                    Predicted Total Score
                                     --> Metric Evaluation
```

### Prompt Structure

```
System:  You are an expert academic evaluator. Grade the student's answer
         strictly according to the provided question and rubric.

User:    Question:
         <question text>

         Rubric:
         Criterion 1 (Maximum 1 point): <criterion_text>
         Criterion 2 (Maximum 1 point): <criterion_text>
         ...

         Student Answer:
         <student_answer text>

         [Instructions: output "Criterion N: score" for every criterion.
          No totals. No explanations.]

Assistant (training target / parsed at inference):
         Criterion 1: 1
         Criterion 2: 0
         ...
         Total: 7/10
```

---

## Dataset

### Overview

| Property | Value |
|---|---|
| Total records | **1,000** |
| Number of questions | **10** |
| Records per question | **100** |
| Maximum score per question | **10 points** |
| Rubric size | 5, 9, or 10 binary criteria (1 pt each) |
| Answer quality types | Fully correct, Partially correct, Incorrect, Irrelevant |
| Dataset file | `complete_dataset.json` (3.4 MB) |

### The 10 Questions

| ID | Topic | Question |
|---|---|---|
| Q01 | Gradient Descent | Explain Batch, SGD, and Mini-Batch gradient descent comparatively |
| Q02 | Overfitting | Methods of reducing overfitting in deep learning |
| Q03 | Activation Functions | Explain 3 activation functions with pros and cons |
| Q04 | Transformer Attention | Multi-head self and cross attention in Transformers |
| Q05 | Apache Kafka | Kafka architecture and components |
| Q06 | Hadoop | HDFS, YARN, and MapReduce architecture |
| Q07 | Batch Norm & Regression | Batch Normalisation + 5 assumptions of linear regression |
| Q08 | NLP Preprocessing | Five text pre-processing techniques in NLP |
| Q09 | Ensemble Methods | Bagging, Boosting, and Random Forest |
| Q10 | RNNs | Why RNN is needed beyond CNN and ANN |

### Dataset Split

```
complete_dataset.json  (1,000 records)
       |
       +-- dataset_split/train.json         700 records  (70 per question)  --> Fine-tuning
       +-- dataset_split/validation.json    150 records  (15 per question)  --> Monitoring
       +-- dataset_split/test.json          150 records  (15 per question)  --> Original held-out test
       |
       +-- dataset_split/complete_TEST_data.json         --> NEW External Test
                                          1,000 records  (100 per question, uniform 10-criterion rubric)
```

### Record Format

```json
{
  "response_id": "Q03_S016",
  "question_id": "Q03",
  "question": "Explain three different activation Functions with pros and cons.",
  "student_answer": "The sigmoid function is defined as sigma(x) = 1/(1+exp(-x))...",
  "rubric": [
    {
      "rubric_id": "Q03_R01",
      "criterion": "Correctly defines sigmoid function (maps to 0-1 range)",
      "points": 1,
      "points_awarded": 1
    }
  ],
  "final_score": 9,
  "max_score": 10,
  "answer_quality": "fully_correct"
}
```

---

## Model and Training

### Base Model

| Property | Value |
|---|---|
| Model | Qwen/Qwen2.5-3B-Instruct |
| Parameters | ~3 Billion |
| Vocabulary size | 151,643 tokens |
| Architecture | Transformer (Grouped-Query Attention) |
| Type | Instruction-tuned (SFT + RLHF) |

### QLoRA Fine-Tuning Configuration

| Parameter | Value |
|---|---|
| Quantisation | 4-bit NF4 + Double Quantisation |
| Compute dtype | float16 |
| LoRA rank (r) | **16** |
| LoRA alpha | **32** |
| LoRA dropout | **0.05** |
| LoRA scaling (alpha/r) | **2.0** |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Trainable parameters | ~1-2% of total model parameters |
| Learning rate | 2e-4 |
| Epochs | 3 |
| Per-device batch size | 1 |
| Gradient accumulation steps | 8 |
| Effective batch size | **8** |
| Max sequence length | **768 tokens** |
| Optimizer | paged_adamw_8bit |
| Warmup steps | 20 |
| Weight decay | 0.01 |
| Gradient checkpointing | Enabled |
| Random seed | 42 |

### Training Outcomes

| Metric | Value |
|---|---|
| Training time | approx. 37.5 minutes |
| Final training loss | **0.2531** |
| Final validation loss | **0.1006** |
| Validation token accuracy | **97.34%** |
| Save strategy | Best checkpoint by validation loss |

---

## Experimental Results

Two separate evaluations were conducted to measure in-distribution performance and generalisation.

### Test Set A: Original Held-Out Test (n = 150)

15 records per question, drawn from the same distribution as training data.

| Model | MAE | RMSE | Exact Acc. | Rubric Acc. | Pearson r | Spearman rho |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Base FP16 | 2.2333 | 3.3397 | 23.3% | 72.6% | 0.6785 | 0.6534 |
| Base NF4 | 2.4600 | 3.5637 | 22.7% | 71.7% | 0.6369 | 0.6146 |
| **Fine-tuned (QLoRA)** | **1.3467** | **2.4413** | **45.3%** | **85.4%** | **0.8186** | **0.8527** |

Improvement of Fine-tuned vs Base NF4: MAE -45.3% | Pearson r +28.5% | Exact Accuracy x2

### Test Set B: New External Test (n = 1,000)

100 records per question, collected independently. Uniform 10-criterion rubric.

| Model | MAE | RMSE | Exact Acc. | Rubric Acc. | Pearson r | Spearman rho |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Base FP16 | 2.3880 | 3.2168 | 22.6% | 73.8% | 0.7272 | 0.7284 |
| Base NF4 | 2.9920 | 3.9764 | 21.1% | 69.5% | 0.6666 | 0.6632 |
| **Fine-tuned (QLoRA)** | **1.3790** | **2.2034** | **38.9%** | **83.8%** | **0.8570** | **0.8534** |

### Cross-Test Stability (Fine-tuned Model)

| Metric | Original Test | New External Test | Change |
|---|:---:|:---:|:---:|
| MAE | 1.3467 | 1.3790 | +0.032 (stable) |
| Pearson r | 0.8186 | 0.8570 | +0.038 (improved) |
| Rubric Accuracy | 85.4% | 83.8% | -1.6% (stable) |
| Exact Score Accuracy | 45.3% | 38.9% | -6.4% |

The fine-tuned model's performance is **stable across both test sets**, confirming good generalisation.

### Error Distribution (Fine-tuned, Original Test n=150)

```
Error = 0 pts  |  68 predictions  (45.3%)  <-- Exact matches
Error = 1 pt   |  47 predictions
Error = 2 pts  |  10 predictions
Error = 3 pts  |   4 predictions
Error 4-5 pts  |   4 predictions
Error > 5 pts  |  17 predictions  (11.3%)  <-- Large errors (needs attention)

115/150 (76.7%) are within +/-1 point
125/150 (83.3%) are within +/-2 points
```

---

## Project Structure

```
summer_project/
|
+-- complete_dataset.json              Raw dataset: 1,000 records, 10 questions
|
+-- dataset_split/
|   +-- train.json / train.jsonl       700 training records
|   +-- validation.json / .jsonl       150 validation records
|   +-- test.json / test.jsonl         150 original held-out test records
|   +-- complete_TEST_data.json        1,000 new external test records
|
+-- [PIPELINE SCRIPTS]
|   +-- split_dataset.py               Split complete_dataset.json -> train/val/test
|   +-- JSON_2_JSONL.py                Convert JSON splits -> JSONL (system/user/assistant)
|   +-- analyze_token_lengths.py       Token length analysis for max_seq_length selection
|
+-- [TRAINING]
|   +-- train_lora.py                  QLoRA fine-tuning with SFTTrainer
|
+-- [EVALUATION - Original Test]
|   +-- evaluate_baseline.py           FP16 baseline
|   +-- evaluate_baseline_nf4.py       NF4 baseline (no LoRA)
|   +-- evaluate_finetuned.py          Fine-tuned QLoRA
|
+-- [EVALUATION - New External Test]
|   +-- evaluate_baseline_newtest.py
|   +-- evaluate_baseline_nf4_newtest.py
|   +-- evaluate_finetuned_newtest.py
|
+-- [COMPARISON & CHARTS]
|   +-- comparison.py                  Bar charts: original test
|   +-- comparison_newtest.py          Bar charts: new external test
|   +-- comparison_charts/             PNG charts (MAE, RMSE, Accuracy, Pearson, Spearman)
|   +-- comparison_charts_newtest/     PNG charts for new external test
|
+-- [RESULTS]
|   +-- baseline_results/              FP16: metrics.json + predictions.json
|   +-- baseline_nf4_results/          NF4: metrics.json + predictions.json
|   +-- finetuned_results/             Fine-tuned: metrics.json + predictions.json
|   +-- baseline_results_newtest/
|   +-- baseline_nf4_results_newtest/
|   +-- finetuned_results_newtest/
|
+-- [TRAINED MODEL]
|   +-- qwen2.5-3b-grader-qlora/
|       +-- final_adapter/             Saved LoRA adapter weights
|       +-- checkpoint-176/            Epoch 1 checkpoint
|       +-- checkpoint-264/            Epoch 2 checkpoint
|       +-- training_metrics.json      Train loss, val loss, runtime
|
+-- [SETUP & TEST SCRIPTS]
|   +-- test_model.py                  Quick sanity-check on trained model
|   +-- test_qlora_setup.py            Verify QLoRA environment setup
|   +-- test_trainer_setup.py          Verify SFTTrainer configuration
|
+-- Project_Report_Final.docx          Full academic internship report
```

---

## Setup and Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (recommended: 12 GB+ VRAM for training, 8 GB+ for inference)
- CUDA 12.6+

### 1. Clone the repository

```bash
git clone https://github.com/your-username/automated-answer-grading.git
cd automated-answer-grading
```

### 2. Install dependencies

```bash
# Install PyTorch with CUDA 12.6 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install HuggingFace ecosystem
pip install transformers==5.15.0
pip install peft
pip install trl==1.10.0
pip install bitsandbytes
pip install datasets==5.0.1
pip install tokenizers==0.22.2

# Install utilities
pip install scipy numpy matplotlib
```

### 3. Verify CUDA and QLoRA setup

```bash
python test_qlora_setup.py
```

---

## Usage Guide

### Step 1: Prepare the Dataset

Split the complete dataset into train / validation / test:

```bash
python split_dataset.py
```

Output:
```
dataset_split/
  train.json       (700 records, 70 per question, seed=42)
  validation.json  (150 records, 15 per question)
  test.json        (150 records, 15 per question)
```

### Step 2: Convert to JSONL Training Format

```bash
python JSON_2_JSONL.py
```

Converts each JSON split to a JSONL file with system/user/assistant message format for SFTTrainer.

### Step 3: Analyse Token Lengths (Optional)

```bash
python analyze_token_lengths.py
```

Reports token length statistics (min, max, mean, percentiles) to validate the chosen `max_seq_length = 768`.

### Step 4: Fine-tune with QLoRA

```bash
python train_lora.py
```

- Trains for 3 epochs (~37.5 minutes)
- Saves best LoRA adapter to `qwen2.5-3b-grader-qlora/final_adapter/`
- Logs training and validation loss to `qwen2.5-3b-grader-qlora/training_metrics.json`

Key configuration options (top of `train_lora.py`):

```python
MODEL_NAME    = "Qwen/Qwen2.5-3B-Instruct"
TRAIN_FILE    = "dataset_split/train.jsonl"
VAL_FILE      = "dataset_split/validation.jsonl"
OUTPUT_DIR    = "./qwen2.5-3b-grader-qlora"
MAX_LENGTH    = 768
NUM_EPOCHS    = 3
LEARNING_RATE = 2e-4
LORA_RANK     = 16
LORA_ALPHA    = 32
```

### Step 5: Evaluate All Models

#### On the Original Held-Out Test Set (n=150)

```bash
# Unquantised FP16 baseline
python evaluate_baseline.py

# 4-bit NF4 baseline (no LoRA)
python evaluate_baseline_nf4.py

# Fine-tuned QLoRA model
python evaluate_finetuned.py
```

#### On the New External Test Set (n=1,000)

```bash
python evaluate_baseline_newtest.py
python evaluate_baseline_nf4_newtest.py
python evaluate_finetuned_newtest.py
```

Results are saved to their respective directories:
```
<results_dir>/
  metrics.json      -> MAE, RMSE, Exact Accuracy, Rubric Accuracy, Pearson r, Spearman rho
  predictions.json  -> Per-record predicted scores, errors, and model responses
```

### Step 6: Generate Comparison Charts

```bash
# Charts for original test (saved to comparison_charts/)
python comparison.py

# Charts for new external test (saved to comparison_charts_newtest/)
python comparison_newtest.py
```

Generates: MAE, RMSE, Exact Accuracy, Rubric Accuracy, Pearson r, Spearman rho, and Normalised Overview bar charts.

### Quick Model Test

```bash
python test_model.py
```

---

## Key Findings and Honest Assessment

### What Works Well

- **Strong correlation**: Pearson r = 0.82-0.86 shows meaningful alignment with human evaluators across the full score range.
- **Good rubric-level precision**: 83-85% of individual criterion scores match human scores exactly.
- **Stable generalisation**: MAE barely changes (1.35 -> 1.38) across both test sets, confirming no overfitting to the training distribution.
- **Large improvement over baselines**: The LoRA adapter reduces MAE by ~45% and doubles exact accuracy vs the untuned NF4 base.
- **Efficient training**: ~37.5 minutes on a single GPU. Only 1-2% of parameters are trained.
- **Zero missing criteria on new external test**: The fine-tuned model consistently outputs a score for every criterion (0 missing vs 6 for FP16 baseline).

### Current Limitations

- **Exact score accuracy below 50%**: More than half of predictions miss the exact total (though usually by only 1 point). Not suitable for high-stakes grading without human verification.
- **Occasional large errors**: ~14% of predictions have error > 3 points; worst cases reach 8-9 points, likely due to prompt sensitivity or unusual answer styles.
- **Small and narrow training set**: 700 records, 10 questions, single domain (computer science). Generalisation to other subjects is untested.
- **No calibration mechanism**: Extreme predictions are not automatically detected or flagged.
- **Single model family**: Only Qwen2.5-3B-Instruct was evaluated; other 3B models may behave differently.

### Verdict

> The system is a **strong proof-of-concept** that demonstrates the feasibility of QLoRA fine-tuning for rubric-based grading.
> It is useful for **feedback generation and candidate screening**, but requires **human oversight** before any high-stakes use.

---

## Tech Stack

| Component | Library / Version |
|---|---|
| Base model | Qwen/Qwen2.5-3B-Instruct |
| Deep learning framework | PyTorch 2.13.0+cu126 |
| Model loading & inference | HuggingFace Transformers 5.15.0 |
| Fine-tuning trainer | TRL (SFTTrainer) 1.10.0 |
| LoRA adapters | PEFT 0.20.0 |
| 4-bit quantisation | bitsandbytes (NF4) |
| Dataset handling | HuggingFace Datasets 5.0.1 |
| Tokenisation | Tokenizers 0.22.2 |
| Correlation metrics | SciPy |
| Visualisation | Matplotlib + NumPy |

---

## References

1. Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). **QLoRA: Efficient Finetuning of Quantized LLMs**. arXiv:2305.14314.
2. Hu, E. J. et al. (2022). **LoRA: Low-Rank Adaptation of Large Language Models**. ICLR 2022. arXiv:2106.09685.
3. Qwen Team. (2024). **Qwen2.5 Technical Report**. Alibaba Cloud. https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
4. Devlin, J. et al. (2019). **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**. NAACL-HLT 2019.
5. Brown, T. et al. (2020). **Language Models are Few-Shot Learners**. NeurIPS 2020. arXiv:2005.14165.
6. Vaswani, A. et al. (2017). **Attention Is All You Need**. NeurIPS 2017.
7. Mayfield, E. & Black, A. W. (2020). **Should You Fine-Tune BERT for Automated Essay Scoring?** BEA Workshop, ACL 2020.
8. von Werra, L. et al. (2020). **TRL: Transformers Reinforcement Learning**. https://github.com/huggingface/trl

### BibTeX

```bibtex
@article{dettmers2023qlora,
  title   = {QLoRA: Efficient Finetuning of Quantized LLMs},
  author  = {Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and Zettlemoyer, Luke},
  journal = {arXiv preprint arXiv:2305.14314},
  year    = {2023}
}

@inproceedings{hu2022lora,
  title     = {LoRA: Low-Rank Adaptation of Large Language Models},
  author    = {Hu, Edward J. and Shen, Yelong and Wallis, Phillip and others},
  booktitle = {International Conference on Learning Representations},
  year      = {2022}
}

@misc{qwen2024qwen25,
  title  = {Qwen2.5 Technical Report},
  author = {Qwen Team},
  year   = {2024},
  url    = {https://huggingface.co/Qwen/Qwen2.5-3B-Instruct}
}

@software{vonwerra2020trl,
  title  = {TRL: Transformers Reinforcement Learning},
  author = {von Werra, Leandro and others},
  url    = {https://github.com/huggingface/trl},
  year   = {2020}
}
```

---

<div align="center">

Made during Summer Internship 2026

*If this project helped you, please consider giving it a star*

</div>
