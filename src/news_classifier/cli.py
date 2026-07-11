"""Command-line entry points for profiling, evaluation, and submissions."""

import argparse
import json
from pathlib import Path

from .constants import TEST_URL, TRAIN_URL, TEXT_COLUMN
from .data import join_text_fields, load_test_data, load_training_data, write_submission
from .evaluation import stratified_split, write_evaluation
from .model import ModelConfig, fit_model
from .profiling import write_profile
from .transformer import TransformerConfig, train_transformer


def _add_data_arguments(parser: argparse.ArgumentParser, include_test: bool = False) -> None:
    parser.add_argument("--train-url", default=TRAIN_URL, help="Training CSV URL or local path.")
    if include_test:
        parser.add_argument("--test-url", default=TEST_URL, help="Test CSV URL or local path.")


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", choices=("tfidf", "deberta"), default="tfidf")
    parser.add_argument("--word-max-features", type=int, default=150_000)
    parser.add_argument("--char-max-features", type=int, default=100_000)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--model-name", default="microsoft/deberta-v3-base")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Compile the training graph with TorchInductor after an initial warm-up.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="News-topic classification experiments.")
    commands = parser.add_subparsers(dest="command", required=True)

    profile = commands.add_parser("profile", help="Write dataset schema and quality statistics.")
    _add_data_arguments(profile, include_test=True)
    profile.add_argument("--output", default="artifacts/data-profile.json")

    evaluate = commands.add_parser("evaluate", help="Evaluate a model on a fixed stratified holdout.")
    _add_data_arguments(evaluate)
    _add_model_arguments(evaluate)
    evaluate.add_argument("--validation-size", type=float, default=0.1)
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--artifacts-dir", default="artifacts")

    submit = commands.add_parser("submit", help="Fit selected settings on all labelled rows.")
    _add_data_arguments(submit, include_test=True)
    _add_model_arguments(submit)
    submit.add_argument("--seed", type=int, default=42)
    submit.add_argument("--output", default="submission.csv")
    return parser.parse_args()


def _transformer_config(args: argparse.Namespace, output_dir: str) -> TransformerConfig:
    return TransformerConfig(
        model_name=args.model_name,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        torch_compile=args.torch_compile,
        seed=args.seed,
        output_dir=output_dir,
    )


def main() -> None:
    args = parse_args()
    if args.command == "profile":
        train = load_training_data(args.train_url, deduplicate=False)
        test = load_test_data(args.test_url)
        output = write_profile(train, test, args.output)
        print(f"Wrote dataset profile to {output}.")
        return

    train = load_training_data(args.train_url)
    if args.command == "evaluate":
        train_split, validation_split = stratified_split(train, args.validation_size, args.seed)
        if args.model == "tfidf":
            config = ModelConfig(args.word_max_features, args.char_max_features, args.c)
            model = fit_model(join_text_fields(train_split), train_split["Class Index"], config)
            predictions = model.predict(join_text_fields(validation_split))
        else:
            trained = train_transformer(
                train_split, _transformer_config(args, str(Path(args.artifacts_dir) / "deberta")), validation_split
            )
            predictions = trained.predict(validation_split)
        metrics = write_evaluation(validation_split, predictions, args.artifacts_dir, args.model)
        print(json.dumps({"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}, indent=2))
        return

    test = load_test_data(args.test_url)
    if args.model == "tfidf":
        config = ModelConfig(args.word_max_features, args.char_max_features, args.c)
        model = fit_model(join_text_fields(train), train["Class Index"], config)
        predictions = model.predict(join_text_fields(test))
    else:
        trained = train_transformer(train, _transformer_config(args, "artifacts/deberta-final"))
        predictions = trained.predict(test)
    output = write_submission(test["id"], predictions, args.output)
    print(f"Trained on {len(train):,} deduplicated rows; wrote {len(test):,} predictions to {output}.")


if __name__ == "__main__":
    main()
