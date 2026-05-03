## How to implement (Stage 4)

You are about to build a small ML project that feels “real” but still stays readable.
The goal is not to memorize code, it is to learn a repeatable *structure* you can reuse.

By the end, you will have:

- a command you can run: `python run.py train`
- saved outputs (artifacts): model, scaler, metrics, report
- a second command: `python run.py predict` that loads those artifacts and predicts

---

## Before you start: what we are building

Think of the project like a kitchen:

- **Data** is your ingredients
- **Preprocessing** is washing/chopping of Alafi etc
- **Training** is cooking (blending and adding things)
- **Evaluation** is tasting + writing the review
- **Artifacts** are the leftovers you store for later (sauce) (model/scaler/metrics)

We separate each responsibility into a module so the code stays calm and predictable.

---

## Step 0: Create the folder structure first (so you feel organized)

Inside `Part4/step4/`, create this layout:

- `run.py` or   `main.py` remember step 3.
- `README.md`
- `HOW_TO_IMPLEMENT.md`
- `_step4.md` not necessary 
- `src/`
  - `__init__.py`
  - `config.py`
  - `logging_utils.py`
  - `data.py`
  - `preprocess.py`
  - `train.py`
  - `evaluate.py`
  - `artifacts.py`

Why start with folders? Because once the structure exists, each file has “a job”.
That prevents the classic beginner problem: *one giant script that keeps growing.*

---

## Step 1:  Start with `src/config.py` (your “knobs” live here)

**Purpose:** keep parameters in one place so you don’t chase numbers across files.

What goes inside:

- `TrainConfig`: ML knobs like `test_size`, `random_state`, `max_iter`
- `PathsConfig`: where artifacts are saved (model/scaler/metrics/report)

**Student check:** if you can read this file and explain each field in one sentence,
you are doing it right.

---

## Step 2:  Add logging (`src/logging_utils.py`)

**Purpose:** make the program speak in a consistent voice.

What goes inside:

- `setup_logging(level="INFO")`: uses `logging.basicConfig(...)`

**Best practice:** logs should answer: *what happened, where, and when*.

---

## Step 3: Data ingestion + splitting (`src/data.py`)

**Purpose:** all data-related actions in one place.

What goes inside:

- `load_dataset()`:
  - loads `sklearn.datasets.load_breast_cancer()`
  - builds a DataFrame
  - adds `target` column
- `split_features_target(df)`:
  - returns `X` and `y`
- `train_test_split_xy(X, y, test_size, random_state)`:
  - returns `X_train, X_test, y_train, y_test`

**Best practice:** don’t scale here, don’t train here. Keep it pure: data only.

---

## Step 4: Preprocessing (`src/preprocess.py`)

**Purpose:** *fit on train, transform everywhere* (avoid data leakage).

What goes inside:

- `ScalerPreprocessor` wrapper:
  - `create()` makes a `StandardScaler`
  - `fit(X_train)` fits scaler **only on training data**
  - `transform(X)` transforms any dataset using the fitted scaler

**Student check:** if you ever find yourself doing `fit_transform(X_test)`, stop.
That is data leakage.

---

## Step 5: Training (`src/train.py`)

**Purpose:** train the model, and only train the model.

What goes inside:

- `train_logistic_regression(X_train_scaled, y_train, max_iter)`
  - creates `LogisticRegression(max_iter=max_iter)`
  - fits and returns the model

**Best practice:** return the trained model object, don’t save it here.
Saving is a different responsibility.

---

## Step 6: Evaluation (`src/evaluate.py`)

**Purpose:** compute metrics + generate a report.

What goes inside:

- `Metrics` dataclass:
  - stores `accuracy` and a timestamp like `created_at_utc`
  - `from_predictions(y_test, y_pred)` builds it
- `make_classification_report_text(y_test, y_pred)` returns the report string
- `pretty_print_evaluation(metrics, report_text)` prints a nice block

**Best practice:** evaluation should *produce* information. Saving happens elsewhere.

---

## Step 7: Artifact saving/loading (`src/artifacts.py`)

**Purpose:** one place for all file I/O for saved outputs.

What goes inside:

- `ensure_dir(path)` creates the output folder
- `save_model(model, path)` + `load_model(path)` using `joblib`
- `save_scaler(scaler, path)` + `load_scaler(path)`
- `save_metrics(metrics, path)` writes JSON
- `save_report(report_text, path)` writes text

**Student check:** after training, you should see:

- `model.joblib`
- `scaler.joblib`
- `metrics.json`
- `classification_report.txt`

If those files are not there, your pipeline is not "production ready" yet.

---

## Step 8: The only file you run: `run.py`

**Purpose:** glue everything together and provide a friendly CLI.

What goes inside:

### 8.1 CLI structure (two subcommands)

- `train`: builds artifacts
- `predict`: loads artifacts and predicts one sample

This is handled by `argparse`:

- `python run.py train --out-dir artifacts`
- `python run.py predict --artifacts-dir artifacts`

### 8.2 What `train` should do (in human order)

Inside `cmd_train`:

1. read CLI args -> build `TrainConfig` and `PathsConfig`
2. create output folder (`ensure_dir`)
3. load data (`load_dataset`)
4. split X/y (`split_features_target`)
5. train/test split (`train_test_split_xy`)
6. create preprocessor and fit on train (`ScalerPreprocessor.create()` -> `.fit(X_train)`)
7. transform train and test (`.transform`)
8. train model (`train_logistic_regression`)
9. predict on test (`model.predict`)
10. build metrics + report (`Metrics.from_predictions`, `make_classification_report_text`)
11. pretty print (`pretty_print_evaluation`)
12. save everything (`save_model`, `save_scaler`, `save_metrics`, `save_report`)

### 8.3 What `predict` should do

Inside `cmd_predict`:

1. (load model + scaler) from the artifacts folder
2. load dataset again (simple for this course)
3. take one sample (`X.iloc[[0]]`)
4. scale it with the loaded scaler
5. run prediction with the loaded model
6. print predicted vs actual label

---

## A “stop and verify” workflow (recommended)

Build and test in small wins:

1) After creating `data.py`, run a tiny snippet (or just run `train` once later) and confirm
the dataset shape prints.

2) After adding preprocessing, confirm scaler is fit only on training.

3) After adding artifacts, confirm files are written.

4) Finally, confirm `predict` works using saved artifacts.

---

## Common mistakes (and how to fix them)

- **Mistake:** “My results change every time.”
  - **Fix:** ensure `random_state` is set and passed into `train_test_split`.

- **Mistake:** “My accuracy seems too good to be true.”
  - **Fix:** check you didn’t fit the scaler on the full dataset or test set.

- **Mistake:** “I trained a model but can’t use it later.”
  - **Fix:** you must save **both** model and scaler. The scaler is part of the pipeline.

- **Mistake:** “My folder is messy again.”
  - **Fix:** if a function doesn’t clearly belong in a module, that’s a design signal.
    Ask: data / preprocess / train / evaluate / artifacts / CLI glue?

---

## What to try next (optional upgrades) - do this as an assignment before the next class.

Once you have implemented the baseline, try one improvement:

- add CLI flag `--model C=...` (regularization strength)
- add `--save-proba` and save predicted probabilities
- save train/test sizes + config into `metrics.json` for full reproducibility


## HAPPY CODING GUYS!!!!!!!