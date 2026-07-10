"""Optional Hugging Face transformer training for the news classifier."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import TARGET_COLUMN, TEXT_COLUMN, TITLE_COLUMN


@dataclass(frozen=True)
class TransformerConfig:
    """Configurable, reproducible settings for encoder fine-tuning."""

    model_name: str = "microsoft/deberta-v3-base"
    max_length: int = 256
    learning_rate: float = 2e-5
    epochs: float = 3.0
    train_batch_size: int = 8
    eval_batch_size: int = 16
    gradient_accumulation_steps: int = 4
    seed: int = 42
    output_dir: str = "artifacts/deberta-checkpoints"


@dataclass
class TrainedTransformer:
    """The fitted trainer plus its competition-label mapping."""

    trainer: object
    tokenizer: object
    label_values: list[int]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict competition labels for title/description pairs."""
        dependencies = _load_dependencies()
        dataset = _tokenize_frame(frame, dependencies["Dataset"], self.tokenizer)
        logits = self.trainer.predict(dataset).predictions
        indices = np.argmax(logits, axis=1)
        return np.asarray([self.label_values[index] for index in indices])


def train_transformer(
    train: pd.DataFrame, config: TransformerConfig, validation: pd.DataFrame | None = None
) -> TrainedTransformer:
    """Fine-tune a sequence classifier on original title and description text.

    When a validation frame is supplied, the best checkpoint is selected by macro-F1. For a
    final submission pass `validation=None` after selecting settings through the evaluate command.
    """
    dependencies = _load_dependencies()
    tokenizer = dependencies["AutoTokenizer"].from_pretrained(config.model_name)
    label_values = sorted(train[TARGET_COLUMN].unique().tolist())
    label_to_id = {label: index for index, label in enumerate(label_values)}
    id_to_label = {index: str(label) for label, index in label_to_id.items()}
    model = dependencies["AutoModelForSequenceClassification"].from_pretrained(
        config.model_name,
        num_labels=len(label_values),
        label2id={str(label): index for label, index in label_to_id.items()},
        id2label=id_to_label,
    )
    train_dataset = _tokenize_frame(train, dependencies["Dataset"], tokenizer, label_to_id, config.max_length)
    eval_dataset = (
        _tokenize_frame(validation, dependencies["Dataset"], tokenizer, label_to_id, config.max_length)
        if validation is not None
        else None
    )
    args = _training_arguments(config, dependencies["torch"], has_validation=validation is not None)
    trainer = dependencies["Trainer"](
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=dependencies["DataCollatorWithPadding"](tokenizer=tokenizer),
        compute_metrics=_compute_metrics if validation is not None else None,
    )
    trainer.train()
    return TrainedTransformer(trainer=trainer, tokenizer=tokenizer, label_values=label_values)


def _tokenize_frame(
    frame: pd.DataFrame,
    dataset_class,
    tokenizer,
    label_to_id: dict[int, int] | None = None,
    max_length: int = 256,
):
    data = {TITLE_COLUMN: frame[TITLE_COLUMN].fillna("").tolist(), TEXT_COLUMN: frame[TEXT_COLUMN].fillna("").tolist()}
    if label_to_id is not None:
        data["labels"] = [label_to_id[label] for label in frame[TARGET_COLUMN]]
    dataset = dataset_class.from_dict(data)
    return dataset.map(
        lambda batch: tokenizer(batch[TITLE_COLUMN], batch[TEXT_COLUMN], truncation=True, max_length=max_length),
        batched=True,
        remove_columns=[TITLE_COLUMN, TEXT_COLUMN],
    )


def _training_arguments(config: TransformerConfig, torch, has_validation: bool):
    dependencies = _load_dependencies()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return dependencies["TrainingArguments"](
        output_dir=str(output_dir),
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.epochs,
        weight_decay=0.01,
        eval_strategy="epoch" if has_validation else "no",
        save_strategy="epoch" if has_validation else "no",
        load_best_model_at_end=has_validation,
        metric_for_best_model="macro_f1" if has_validation else None,
        greater_is_better=True,
        save_total_limit=1,
        logging_strategy="steps",
        logging_steps=100,
        report_to=[],
        seed=config.seed,
        data_seed=config.seed,
        fp16=torch.cuda.is_available(),
    )


def _compute_metrics(evaluation_prediction) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    predictions = np.argmax(evaluation_prediction.predictions, axis=1)
    return {
        "accuracy": accuracy_score(evaluation_prediction.label_ids, predictions),
        "macro_f1": f1_score(evaluation_prediction.label_ids, predictions, average="macro"),
    }


def _load_dependencies() -> dict:
    """Import optional packages lazily so the TF-IDF workflow stays lightweight."""
    try:
        import torch
        from datasets import Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments
    except ImportError as error:
        raise RuntimeError(
            "Transformer support is not installed. Run `uv sync --extra transformer` first."
        ) from error
    return {
        "torch": torch,
        "Dataset": Dataset,
        "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
        "AutoTokenizer": AutoTokenizer,
        "DataCollatorWithPadding": DataCollatorWithPadding,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
    }
