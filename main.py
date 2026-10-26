"""Command-line entry point for the complete project pipeline."""

import argparse
import json

from config import DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE, EARLY_STOPPING_PATIENCE
from src.train import train_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the appliance energy MLP.")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    args = parser.parse_args()
    print("Loading the official dataset and training the full-batch MLP...", flush=True)

    def log_epoch(epoch: int, total: int, loss: float, validation_loss: float) -> None:
        print(
            f"Epoch {epoch}/{total} - training MSE: {loss:.6f}; "
            f"validation MSE: {validation_loss:.6f}",
            flush=True,
        )

    result = train_pipeline(args.learning_rate, args.epochs, args.patience, log_epoch)
    print(json.dumps(result, indent=2))
    print("Training complete. Model, metrics, figures, predictions, and report are saved.", flush=True)


if __name__ == "__main__":
    main()
