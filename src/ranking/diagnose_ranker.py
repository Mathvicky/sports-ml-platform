"""Inspect Ranker V1 predictions without training a model."""

from pathlib import Path

import joblib
import pandas as pd


FEATURE_COLUMNS = ["sport_match", "content_type_match", "popularity_score"]
EVENT_TARGETS = {
    "like": 1,
    "complete": 1,
    "skip": 0,
    "view": 0,
    "click": 0,
}
THRESHOLDS = [0.50, 0.40, 0.30, 0.20]


def main():
    # Build paths from the project root so the script works from any directory.
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "processed_interactions.csv"
    model_path = project_root / "models" / "ranker_v1.joblib"

    # Load the processed rows and recreate the target used when training Ranker V1.
    interactions = pd.read_csv(data_path)
    interactions["positive_engagement"] = (
        interactions["event_type"].map(EVENT_TARGETS).astype(int)
    )

    # Show how many examples belong to each target class.
    total_examples = len(interactions)
    positive_count = int((interactions["positive_engagement"] == 1).sum())
    negative_count = int((interactions["positive_engagement"] == 0).sum())
    positive_percentage = positive_count / total_examples * 100 if total_examples else 0
    negative_percentage = negative_count / total_examples * 100 if total_examples else 0

    print("Class distribution:")
    print(f"Positive examples: {positive_count} ({positive_percentage:.2f}%)")
    print(f"Negative examples: {negative_count} ({negative_percentage:.2f}%)")

    # Load the already-trained model and use only its original input features.
    model = joblib.load(model_path)
    features = interactions[FEATURE_COLUMNS]

    # Column 1 contains the probability for the positive class (label 1).
    positive_probabilities = model.predict_proba(features)[:, 1]
    print("\nPredicted probability summary:")
    print(f"Minimum: {positive_probabilities.min():.4f}")
    print(f"Average: {positive_probabilities.mean():.4f}")
    print(f"Maximum: {positive_probabilities.max():.4f}")
    print(f"Predictions >= 0.50: {(positive_probabilities >= 0.50).sum()}")

    # Count how many rows would be classified as positive at each cutoff.
    print("\nPositive prediction counts by threshold:")
    for threshold in THRESHOLDS:
        positive_predictions = (positive_probabilities >= threshold).sum()
        print(f"Threshold {threshold:.2f}: {positive_predictions}")


if __name__ == "__main__":
    main()