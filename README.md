# News Topic Classifier

This project trains a reproducible four-class news-topic classifier from the competition
datasets and writes a submission CSV. It provides a fast CPU TF-IDF baseline and an optional
DeBERTa-v3 transformer workflow.

## Setup

Install the locked environment with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Remote Windows workflow

The repository is edited locally and should be executed on the remote Windows machine only:

```bash
ssh whl
cd <repository-path>
uv lock
uv sync --extra transformer
uv run news-classify profile
uv run news-classify evaluate --model tfidf
uv run news-classify evaluate --model deberta --epochs 3 --max-length 256
```

Do not compare models by repeatedly changing the validation split. Keep `--seed 42` and the
same validation size while selecting hyperparameters, then use `submit` once with the winner.

The checked-in `.python-version` makes uv provision CPython 3.12 on this machine (where the
`python` command currently points only to the Microsoft Store alias). For its RTX 4060 Laptop
GPU with 8 GB VRAM, the DeBERTa defaults are a batch size of 8, evaluation batch size of 16,
FP16, and four gradient-accumulation steps (effective batch size 32). If CUDA runs out of memory,
lower `--train-batch-size` to 4 before reducing `--max-length`.

## Inspect the data

```bash
uv run news-classify profile
```

This writes `artifacts/data-profile.json`, including schema, missing values, duplicate counts,
class balance, and title/description length percentiles.

## Evaluate models

Run the fast, strong sparse baseline on a fixed stratified holdout:

```bash
uv run news-classify evaluate --model tfidf
```

Run DeBERTa-v3-base (a GPU is recommended):

```bash
uv sync --extra transformer
uv run --extra transformer news-classify evaluate --model deberta --epochs 3 --max-length 256
```

Evaluation writes metrics, a confusion matrix, and validation predictions under `artifacts/`.
The split is made after removing every conflicting-label `Title` + `Description` pair and then
exact deduplication, preventing duplicate-text leakage and contradictory training targets.

## Train and create a submission

```bash
uv run news-classify submit --model tfidf --output submission.csv
```

After selecting a configuration using `evaluate`, fit it on all deduplicated labelled rows and
write a submission:

```bash
uv run --extra transformer news-classify submit --model deberta --output submission.csv
```

Useful options:

```bash
uv run news-classify evaluate --model tfidf --word-max-features 150000 --c 2.0
uv run news-classify submit --model tfidf --train-url path/to/train.csv --test-url path/to/test.csv
uv run --extra transformer news-classify evaluate --model deberta --learning-rate 2e-5
```

## Project layout

- `src/news_classifier/data.py`: schema validation, data loading, exact deduplication, submission creation
- `src/news_classifier/profiling.py`: data-structure report
- `src/news_classifier/model.py`: TF-IDF word/character n-gram + LinearSVC baseline
- `src/news_classifier/transformer.py`: optional DeBERTa training and inference workflow
- `src/news_classifier/evaluation.py`: fixed stratified evaluation artifacts and metrics
- `src/news_classifier/cli.py`: `profile`, `evaluate`, and `submit` commands
- `tests/`: small offline unit tests
- `docs/research-text-classification.md`: research rationale and primary-source links

The former CountVectorizer-only baseline, manual NLTK preprocessing, Gensim experiment, and Notebook
are intentionally removed: modern pretrained models should receive original text, while the
TF-IDF baseline uses title and description together with substantially richer features.
