"""Train and evaluate the first PulsePlay AI engagement ranker."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS = ["sport_match", "content_type_match", "popularity_score"]
EVENT_TARGETS = {
    "like": 1,
    "complete": 1,
    "skip": 0,
    "view": 0,
    "click": 0,
}


def main():
    # Find the project files relative to this script's location.
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "processed_interactions.csv"

    # Load the processed interactions and label positive engagement as 1 or 0.
    interactions = pd.read_csv(data_path)
    interactions["positive_engagement"] = (
        interactions["event_type"].map(EVENT_TARGETS).astype(int)
    )

    # X contains only information available before a recommendation is made.
    # Interaction outcomes and watch time are intentionally left out to avoid leakage.
    X = interactions[FEATURE_COLUMNS]
    y = interactions["positive_engagement"]

    # Keep 20% of the rows aside for testing, preserving the positive/negative ratio.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Train a simple classification model to predict positive engagement.
    model = LogisticRegression(max_iter=1_000)
    model.fit(X_train, y_train)

    # Evaluate predictions on data that was not used to train the model.
    predictions = model.predict(X_test)
    positive_probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    roc_auc = roc_auc_score(y_test, positive_probabilities)

    print(f"Training row count: {len(X_train)}")
    print(f"Test row count: {len(X_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    # Save the fitted model so later recommendation code can load it.
    models_directory = project_root / "models"
    models_directory.mkdir(parents=True, exist_ok=True)
    model_path = models_directory / "ranker_v1.joblib"
    joblib.dump(model, model_path)
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()