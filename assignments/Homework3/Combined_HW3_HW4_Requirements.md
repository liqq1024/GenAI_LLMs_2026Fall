# COSC 661 — Combined Homework 3 & 4
## Prompt Engineering, Decoding, and LoRA Adaptation

**Individual assignment · 150 points · Google Colab · Estimated 8–12 hours**

### Research question
Can prompt design and decoding improve professional-tone rewriting, and does LoRA adaptation improve performance beyond prompting alone?

### Provided resources
- `train.csv`: 160 **synthetic** training pairs; `test.csv`: 20 held-out pairs. These are instructional examples, not human-rated ground truth.
- `HW3_4_Colab_Starter.ipynb`: tested-workflow starter (requires a Colab GPU and first-run downloads).
- `evaluation.py`: format-compliance calculation, output comparison and human-scoring template.
- `human_ratings_template.csv`: blank manual-scoring worksheet; `results_template.csv`: output collection schema.
- Recommended model: `Qwen/Qwen2.5-0.5B-Instruct`, Hugging Face open-weight model. Check its license and terms before redistribution. The notebook downloads it from Hugging Face; model weights are not included.

### Dataset limitations
Synthetic examples are deliberately small and formulaic. Training and test **rows** are disjoint, but their linguistic patterns overlap; results should not be interpreted as evidence of production-quality professional writing. Do not treat reference wording as the only correct answer. Students should discuss generalization limits.

### Part A — Prompt engineering and decoding (40 points)
**A1 (15):** Create a basic zero-shot prompt and an improved prompt with constraints and 2–3 few-shot examples. Explain design choices. Do not use held-out test references as examples.

**A2 (15):** On five fixed test inputs, compare the improved prompt under greedy decoding and sampling (`temperature=0.7`, `top_p=0.9`). Use the same base model. Save outputs. Greedy ignores temperature and top-p.

**A3 (10):** Compare meaning preservation, tone, consistency, and variation. Explain the preferred configuration for this task.

### Part B — LoRA fine-tuning (55 points)
**B1 (10):** Load/inspect at least five training rows; tokenize formatted instruction–response examples; do not train on test rows.

**B2 (15):** Configure LoRA (suggested `r=8`, `alpha=16`, `dropout=0.05`), report total and trainable parameters and percentage.

**B3 (20):** Complete one LoRA training run (1–2 epochs) and save adapter files. Report learning rate, batch size, gradient accumulation, max sequence length, runtime, and loss.

**B4 (10):** Interpret parameter savings and training loss; explain why lower loss alone does not establish generalization.

### Part C — Controlled comparison (40 points)
**C1 (15):** On all 20 test inputs, compare: A = base + basic zero-shot + greedy; B = base + improved few-shot + greedy; C = LoRA-adapted + basic zero-shot + greedy. Keep the model family, evaluation data, generation length and settings fixed where applicable.

**C2 (15):** Compute format-compliance rate across all 20 outputs per condition. Independently score the **same 10** test examples per condition for meaning preservation, professional tone and fluency, each 1–5. Report per-condition means. Human scoring is subjective; state rubric and limitations.

**C3 (10):** Discuss three representative cases: a prompt-related improvement, an adaptation-related improvement (if any), and a remaining failure. Do not fabricate improvements.

### Part D — Conclusion and reproducibility (15 points)
**D1 (10):** Which approach performed best? Did LoRA justify its training cost? What limitations arise from the small synthetic dataset? What next experiment would you conduct?

**D2 (5):** Provide model ID, dataset provenance, seed, package versions, LoRA settings, output CSV, and reproducible execution order. Never include access tokens.

### Submission
1. Executed Colab notebook (`.ipynb`) including results.
2. One 4–5-page PDF report with experimental table, representative outputs, error analysis, and conclusions.
3. LoRA adapter directory (or accessible private repository link). Do not submit full model weights or secrets.
4. `results.csv` and completed `human_ratings.csv`.

### Grading rubric
| Item | Points |
|---|---:|
| A1 Prompt design | 15 |
| A2 Decoding comparison | 15 |
| A3 Analysis | 10 |
| B1 Dataset preparation | 10 |
| B2 LoRA configuration | 15 |
| B3 Fine-tuning | 20 |
| B4 Training interpretation | 10 |
| C1 Controlled comparison | 15 |
| C2 Evaluation | 15 |
| C3 Failure analysis | 10 |
| D1 Conclusion | 10 |
| D2 Reproducibility | 5 |
| **Total** | **150** |

### Academic integrity and experimental validity
Write and understand your own code and analysis in accordance with the course AI-use policy. Do not use test references for training, prompt examples or output generation. Do not hand-edit model outputs before evaluation. No improvement is a valid result. Record Colab hardware and library versions. Instructor should review package versions and model availability before release.
