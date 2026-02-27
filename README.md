# Prompt Sensitivity Checker for LLMs  
### A Controlled Study on Prompt Framing, Temperature, and Model Choice

---

## Overview

This project investigates how **prompt phrasing**, **decoding temperature**, and **model selection** affect binary decision accuracy in Large Language Models (LLMs).

The goal is to measure how sensitive LLM performance is to seemingly small prompt variations under controlled experimental conditions.

The evaluation is implemented with multiple open-source instruction-tuned models on a subset of the **BoolQ** dataset using deterministic and stochastic decoding.

---

## Research Questions

1. How much does accuracy vary across prompt templates when decoding is deterministic?
2. Does temperature amplify or dampen prompt sensitivity?
3. Do different models exhibit different robustness patterns?
4. Does reasoning-style prompting improve binary decision reliability?

---

## Dataset

### BoolQ (Boolean Questions)

**BoolQ validation split** from the Hugging Face Datasets library.

Each example contains:
- `question` – A yes/no question
- `passage` – A supporting Wikipedia paragraph
- `answer` – Boolean ground truth (True/False)

Example:

```json
{
  "question": "Is ethanol a renewable fuel?",
  "passage": "...",
  "answer": true
}
```
**Important Note:**

This project intentionally uses question-only input (Passage is not included). Therefore, this is not an official BoolQ reading comprehension evaluation. Instead, the measured output:

Question-only binary decision reliability under prompt variation.

Ground-truth labels (gold) are derived as:

`True → "yes"`

`False → "no"`

---

## Prompt Variants

All prompts enforce strict binary output (`yes` or `no`) but differ in framing:

- direct
- role_expert
- constraints
- step_by_step
- minimal

The task remains constant; only phrasing changes.

---

## 4. Experimental Design

The evaluation is across:
- Models (e.g., Qwen2.5-1.5B-Instruct, TinyLlama-1.1B-Chat-v1.0)
- Temperatures (e.g., 0.0, 0.3, 0.7)
- Prompt Variants (5 different framing strategies)
- 200 validation samples

Total generations:

`models × temperatures × samples × variants`

Example: (2 models × 3 temps × 200 × 5 variants) → 6000 model generations

---

## Evaluation Metric

Predictions are evaluated using:

- Regex-based extraction of `yes` or `no`
- Accuracy = correct / total

If the model does not output a valid label, it is counted as incorrect.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/ilameya/prompt-sensitivity-checker.git
cd prompt-sensitivity-checker
```

Create a virtual environment:

```bash
python3 -m venv venv
```

Activate the environment (In MacOS/Linux):
```bash
source venv/bin/activate    
```
In Windows:
```bash
venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt   
```

## Run

```bash
python src/eval.py \
  --n 200 \
  --temperatures 0.0,0.3,0.7 \
  --models "<model1>,<model2>"
```

#### Configuration

The experiment is fully configurable through command-line.

Key parameters:
- `n` [Number of evaluation samples from BoolQ (default: 200)]
- `temperatures`
[Comma-separated decoding temperatures (default: 0.0,0.3,0.7)]
- `models`
[Comma-separated Hugging Face model identifiers]

---
## Outputs

After execution, you will find:

### `results/results.csv`
Full prediction logs (one row per generation).

### `results/summary.csv`
Accuracy aggregated by model, temperature, and prompt variant.

### `plots/`
- Accuracy comparison at T=0 (You can generate for all temperatures)
- Accuracy vs temperature (per model)

---

## Interpretation
- Differences between prompt variants at `T=0.0` indicate prompt sensitivity. 
- Changes across temperature indicate decoding stability.
- Differences between models indicate robustness differences.
---
This setup allows controlled comparison of prompt phrasing while holding task, data, and evaluation constant.