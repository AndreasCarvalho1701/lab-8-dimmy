# Labdummy DPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal Python project that trains a small language model with DPO for the lab, documents the math clearly in the README, and validates safe behavior against a malicious prompt.

**Architecture:** Keep the repository intentionally small: one dataset file, one training script, one validation script, one Colab notebook, and one README. The training path must use `trl.DPOTrainer` with two model instances and `beta = 0.1`; the validation path must show both safe generated output and numeric evidence that the safe answer is preferred over the rejected one.

**Tech Stack:** Python, torch, transformers, datasets, trl, peft, bitsandbytes, Jupyter/Colab, Git

---

## File Map

- Create: `README.md`
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `data/hhh_preferences.jsonl`
- Create: `train_dpo.py`
- Create: `test_model.py`
- Create: `notebooks/run_in_colab.ipynb`

### Task 1: Scaffold Minimal Repository

**Files:**
- Create: `.gitignore`
- Create: `requirements.txt`

- [ ] **Step 1: Write the failing repository sanity check in notes**

Expected check:
- `requirements.txt` does not exist yet
- `.gitignore` does not exist yet

- [ ] **Step 2: Verify the files are absent before creation**

Run: `ls -la`
Expected: project root exists without the target files

- [ ] **Step 3: Create minimal dependency list**

Add `requirements.txt` with:

```text
torch
transformers
datasets
trl
peft
accelerate
bitsandbytes
jupyter
notebook
```

- [ ] **Step 4: Create `.gitignore`**

Add entries for:

```gitignore
__pycache__/
.ipynb_checkpoints/
.venv/
env/
outputs/
checkpoints/
*.pyc
*.pyo
*.pyd
*.log
```

- [ ] **Step 5: Verify files exist**

Run: `ls -la`
Expected: `requirements.txt` and `.gitignore` are present

### Task 2: Build the Preference Dataset

**Files:**
- Create: `data/hhh_preferences.jsonl`

- [ ] **Step 1: Write the failing dataset validation command in notes**

Expected check:
- file missing or fewer than 30 lines should fail the lab requirement

- [ ] **Step 2: Create the dataset file with at least 30 entries**

Each line must match:

```json
{"prompt":"...","chosen":"...","rejected":"..."}
```

Content constraints:
- exactly the keys `prompt`, `chosen`, `rejected`
- no extra keys
- malicious/security misuse examples
- corporate tone/scope examples
- safe refusals in `chosen`
- unsafe or misaligned replies in `rejected`

- [ ] **Step 3: Verify dataset line count**

Run: `python - <<'PY'
from pathlib import Path
path = Path('data/hhh_preferences.jsonl')
print(sum(1 for _ in path.open()))
PY`
Expected: `30` or more

- [ ] **Step 4: Verify schema strictly**

Run: `python - <<'PY'
import json
from pathlib import Path
path = Path('data/hhh_preferences.jsonl')
for i, line in enumerate(path.open(), start=1):
    data = json.loads(line)
    assert set(data.keys()) == {'prompt', 'chosen', 'rejected'}, (i, data.keys())
print('schema ok')
PY`
Expected: `schema ok`

### Task 3: Implement Training Script

**Files:**
- Create: `train_dpo.py`

- [ ] **Step 1: Write the failing smoke command in notes**

Run target: `python train_dpo.py --help`
Expected before implementation: file missing or import failure

- [ ] **Step 2: Implement CLI and config loading**

Include arguments for:
- `--model-name`
- `--dataset-path`
- `--output-dir`
- `--beta`
- `--max-length`
- `--max-prompt-length`
- `--num-train-epochs`

- [ ] **Step 3: Implement dataset loading and schema validation**

Behavior:
- load `.jsonl` through `datasets`
- fail fast if file missing
- fail fast if keys differ from `prompt`, `chosen`, `rejected`

- [ ] **Step 4: Implement model and tokenizer loading**

Behavior:
- load tokenizer once
- load actor model
- load reference model from same checkpoint
- freeze reference model behavior by passing it as DPO reference model

- [ ] **Step 5: Implement `TrainingArguments` and `DPOTrainer`**

Requirements:
- set `optim='paged_adamw_32bit'`
- set `beta=0.1` by default
- conservative defaults for Colab memory
- call `trainer.train()` explicitly

- [ ] **Step 6: Implement model saving**

Behavior:
- save trained model and tokenizer into `outputs/` or provided `--output-dir`

- [ ] **Step 7: Verify CLI help works**

Run: `python train_dpo.py --help`
Expected: argparse help text is printed without syntax errors

### Task 4: Implement Validation Script

**Files:**
- Create: `test_model.py`

- [ ] **Step 1: Pin the validation case in code**

Use a fixed malicious prompt for reproducibility, for example a request for destructive database action or another clearly unsafe operational instruction.

This same fixed malicious prompt must also be the one used for the numeric `chosen` vs `rejected` comparison in the console evidence.

- [ ] **Step 2: Implement model loading for inference**

Behavior:
- load tokenizer and trained model from output directory
- fail clearly if model path is missing

- [ ] **Step 3: Implement response generation**

Behavior:
- print the malicious prompt
- generate and print the model answer

- [ ] **Step 4: Implement score comparison for one chosen/rejected pair**

Use the fixed malicious validation prompt from Step 1 together with its matching safe `chosen` answer and unsafe `rejected` answer.
Metric requirement:
- use one explicit metric consistently: aggregated sequence log probability under the trained model

Console output must include:
- chosen logprob
- rejected logprob
- final line with `chosen_preferred=True/False`

- [ ] **Step 5: Implement acceptance verdict**

Behavior:
- print `validation_passed=True` only if both are true:
  - generated answer is a safe refusal or harmless redirection
  - aggregated sequence log probability for `chosen` is greater than for `rejected`

- [ ] **Step 6: Verify script help works**

Run: `python test_model.py --help`
Expected: argparse help text is printed without syntax errors

### Task 5: Write README with Academic Documentation

**Files:**
- Create: `README.md`

- [ ] **Step 1: Draft README structure**

Sections required:
- project overview
- objective of the lab
- DPO pipeline explanation
- repository structure
- setup instructions
- training command
- validation command
- explanation of dataset format
- mathematical explanation of DPO beta
- Colab usage
- Git/versioning flow
- AI usage declaration

- [ ] **Step 2: Write the beta explanation with formula-level detail**

Must explicitly explain:
- DPO compares `chosen` vs `rejected`
- `beta` scales preference strength
- larger `beta` acts like a stronger tax/penalty against drifting too far from the reference behavior
- smaller `beta` allows more aggressive deviation toward the preference data
- this helps preserve fluency and original language behavior

Include at least one compact mathematical expression or pseudo-formula.

- [ ] **Step 3: Add the mandatory AI declaration exactly**

Include this line in the README:

```text
Partes geradas/complementadas com IA, revisadas por Andreas
```

- [ ] **Step 4: Document version policy**

Include:
- development tags like `v0.1`, `v0.2`, `v0.3`
- final tag `v1.0`
- final release `v1.0`

- [ ] **Step 5: Review README for clarity and directness**

Check that the README is understandable to a grader without extra explanation.

### Task 6: Build Colab Notebook

**Files:**
- Create: `notebooks/run_in_colab.ipynb`

- [ ] **Step 1: Create notebook sections**

Cells should cover:
- dependency installation
- optional drive mounting
- dataset placement
- training command
- validation command

- [ ] **Step 2: Mirror the CLI commands from scripts**

The notebook should call the same scripts rather than duplicating logic.

- [ ] **Step 3: Add short markdown guidance**

Explain that Colab is the recommended runtime because DPO training is not intended for a weak local machine.

### Task 7: End-to-End Verification

**Files:**
- Verify: `requirements.txt`
- Verify: `data/hhh_preferences.jsonl`
- Verify: `train_dpo.py`
- Verify: `test_model.py`
- Verify: `README.md`

- [ ] **Step 1: Verify Python syntax**

Run: `python -m py_compile train_dpo.py test_model.py`
Expected: no output, exit code 0

- [ ] **Step 2: Verify dataset schema again**

Run the JSONL schema check from Task 2.
Expected: `schema ok`

- [ ] **Step 3: Run training help command**

Run: `python train_dpo.py --help`
Expected: help text prints

- [ ] **Step 4: Run validation help command**

Run: `python test_model.py --help`
Expected: help text prints

- [ ] **Step 5: Run final training in Colab**

Run in Colab:

```bash
python train_dpo.py --model-name <small-model> --dataset-path data/hhh_preferences.jsonl --output-dir outputs/dpo-model
```

Expected:
- trainer initializes
- `trainer.train()` runs
- model saves successfully

- [ ] **Step 6: Run final validation in Colab**

Run in Colab:

```bash
python test_model.py --model-dir outputs/dpo-model --dataset-path data/hhh_preferences.jsonl
```

Expected console evidence:
- malicious prompt printed
- safe generated answer printed
- chosen aggregated logprob printed
- rejected aggregated logprob printed
- `chosen_preferred=True`
- `validation_passed=True`

### Task 8: Git Versioning Milestones

**Files:**
- Track: all created project files

- [ ] **Step 1: Create initial repository milestone**

Run:

```bash
git add .
git commit -m "chore: scaffold DPO lab project"
git tag v0.1
```

- [ ] **Step 2: Tag dataset milestone**

Run:

```bash
git add data/hhh_preferences.jsonl
git commit -m "data: add HHH preference dataset"
git tag v0.2
```

- [ ] **Step 3: Tag training milestone**

Run:

```bash
git add train_dpo.py requirements.txt
git commit -m "feat: add DPO training pipeline"
git tag v0.3
```

- [ ] **Step 4: Tag validation milestone**

Run:

```bash
git add test_model.py notebooks/run_in_colab.ipynb README.md
git commit -m "feat: add validation flow and documentation"
git tag v0.4
```

- [ ] **Step 5: Create final delivery version**

Run after verification is complete:

```bash
git add .
git commit -m "docs: finalize DPO lab delivery"
git tag v1.0
```

- [ ] **Step 6: Create final hosted release**

Create a remote release named `v1.0` that points to the `v1.0` tag.

- [ ] **Step 7: Publish final deliverable to remote**

Run:

```bash
git push origin main
git push origin v1.0
```

Expected:
- remote branch updated
- tag `v1.0` published
- hosted release `v1.0` available for submission
