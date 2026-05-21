# Stat298-AI-Interpretability-Final-Project
Final project for Spring 2026 version of Stat 298: AI Interpretability at UC Berkeley.
Abhay Paiddipalli Surya Appana Dhruv Pendharkar
---

## CoT Injection Experiment

### Research Question

Does a language model's Chain-of-Thought (CoT) trace serve as a genuine computational scaffold — a sequence of intermediate computations the model actually depends on — or is it a post-hoc rationalization that the model generates after the answer is already determined internally?

This experiment tests that question by **injecting wrong intermediate answers mid-generation** and measuring whether the model recovers (ignores the error), propagates it (accepts the wrong value and continues from there), or collapses (produces no parseable answer). A key secondary analysis uses **linear probing of hidden states** at the injection point to test for *dissociation*: cases where the model's internal representations already encode the correct answer, yet the surface CoT output still follows the injected wrong token.

---

### Injection Protocol

#### Step 1 — Baseline CoT Generation

For each problem, the model first generates a full chain-of-thought response without any injection. This baseline is used purely to obtain the step structure (it is not shown to the model again). The reasoning trace is segmented into steps:

- **GSM8K**: each non-blank line that does not contain "the answer is" is one step.
- **PrOntoQA**: each sentence (split on `.`, `!`, `?`) that does not contain "the answer is" is one step.

#### Step 2 — Selecting the Injection Point

An injection index is chosen within the baseline step list according to the `injection_point` parameter:

| `injection_point` | Step index (out of `n` total steps) |
|---|---|
| `early` | `max(1, n // 4)` |
| `middle` | `n // 2` |
| `late` | `max(n - 2, n // 2 + 1)` |

All steps *before* the injection index form a prefix that is prepended to the injected prompt verbatim.

#### Step 3 — Generating the Wrong Intermediate Answer

The step at the injection index is the **target step**. Its last numeric value (for GSM8K) or logical claim (for PrOntoQA) is replaced according to the `error_type`:

**GSM8K (numeric replacement):**

| `error_type` | Modification |
|---|---|
| `near` | Multiply the correct intermediate value by a random factor from `{0.80, 0.85, 1.15, 1.20}` |
| `far` | Multiply by `10`, `0.1`, or `100` — an implausible magnitude shift |
| `self_consistent` | Set to `gold_answer × uniform(0.4, 0.6)` — numerically close to the final answer but wrong |
| `constraint_violating` | Set to `-abs(correct_value × 0.5)` — a negative count, violating an implicit domain constraint |
| `correct_rephrased` | Keep the correct value unchanged (control condition) |

**PrOntoQA (logical replacement):**

| `error_type` | Modification |
|---|---|
| `correct_rephrased` | Step unchanged (control) |
| `near`, `self_consistent` | Replace `" is "` with `" is not "` — a minimal logical negation |
| `far`, `constraint_violating` | Replace entire step with `"Therefore, the opposite conclusion must hold here."` |

#### Step 4 — Injected Prompt Construction

The injected prompt is assembled as a raw string:

```
<chat_template_header>
<system_prompt>
<user_question>
<cot_prefix_steps>
<injected_wrong_step>
```

This string is passed directly to the tokenizer (not pre-tokenized) and fed to the model, which is then asked to continue generation for up to `max_answer_tokens` tokens.

#### Step 5 — Outcome Classification

The model's continuation is parsed for a final answer and compared against the gold answer and the injected wrong value:

| Outcome | Condition |
|---|---|
| `full_recovery` | Model answer matches the gold answer (within 1% relative tolerance for GSM8K; exact string match for PrOntoQA) |
| `propagation` | Model answer matches the injected wrong value (within 5% relative tolerance for GSM8K) |
| `collapse` | No parseable answer, or answer matches neither gold nor injected value |
| `control_correct` / `control_incorrect` | Used only for the `correct_rephrased` control condition |

#### Step 6 — Linear Probing (Dissociation Analysis)

When `run_probing=True`, the hidden states at the **first generated token** after the injected prompt are extracted for all transformer layers. For each layer, a logistic regression probe is trained to predict whether the final outcome will be `full_recovery` (label 1) or not (label 0) based solely on that layer's hidden representation.

A **dissociation case** is identified when:
- The probe at the peak-accuracy layer predicts recovery with confidence > 0.6 (i.e., the internal representation encodes the correct answer), and
- The actual output outcome is `propagation` (i.e., the model nevertheless followed the wrong injected token).

These cases are the central finding: the model's internal state "knew" the correct answer but its surface output was captured by the injected error.

---

### Datasets

| Dataset | Source | Task | Difficulty proxy |
|---|---|---|---|
| GSM8K | `gsm8k / main / test` (HuggingFace) | Grade-school math word problems | Number of non-answer lines in the reference solution |
| PrOntoQA | `EleutherAI/prontoqa / test` (HuggingFace) | Propositional logic proofs | Number of sentences in the proof field |

Problems are stratified into 3 difficulty bins (easy / medium / hard) and sampled equally across bins for balance.

---

### Supported Models

- `Qwen/Qwen2.5-3B-Instruct` (~7 GB VRAM)
- `Qwen/Qwen2.5-7B-Instruct` (~16 GB VRAM)

### How to Run

#### Installation

```bash
pip install -r requirements.txt
```

#### Command-Line Script

Run the full experiment (injection + probing + visualization) from the terminal:

```bash
# Default: Qwen2.5-7B on GSM8K, 200 problems, probing enabled
python cot_injection.py

# Smaller model, PrOntoQA dataset, 100 problems
python cot_injection.py --model Qwen/Qwen2.5-3B-Instruct --dataset prontoqa --n_problems 100

# Disable probing (faster, no hidden-state extraction)
python cot_injection.py --no_probing

# All options
python cot_injection.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset gsm8k \
    --n_problems 200 \
    --output_dir ./results \
    --max_cot_tokens 512 \
    --max_answer_tokens 200 \
    --temperature 0.0
```

Results are saved to `./results/<model_tag>_<dataset>/`:
- `results.csv` — per-condition outcome table
- `hidden_states.npy` — per-layer hidden state arrays (if probing enabled)
- `results_figure.pdf` / `results_figure.png` — summary visualization

#### Jupyter Notebook

The full experiment is also available as an interactive notebook which can be run with Google Colab:

```bash
jupyter notebook cot_injection_experiment.ipynb
```

Run all cells top-to-bottom. The main execution cell at the bottom runs the complete pipeline: model loading → dataset loading → injection experiment → probing → dissociation analysis → visualization → qualitative inspection.

#### Multi-Model Comparison

After running experiments for both model sizes, compare results with:

```python
from cot_injection import compare_across_models

compare_across_models(
    result_dir='./results',
    dataset='gsm8k',
    model_tags=['Qwen2.5-3B-Instruct', 'Qwen2.5-7B-Instruct'],
)
```

This produces a two-panel figure comparing recovery rates by difficulty and propagation rates by injection point across model sizes.

All results were obtained through experiments run on a single Nvidia A100 GPU with 80 GB of VRAM
