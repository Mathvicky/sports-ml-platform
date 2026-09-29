"""Train and evaluate PulsePlay AI's third ranking model."""

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


# Include V2 inputs and the new historical user/category features only.
FEATURE_COLUMNS = [
    "sport_match",
    "content_type_match",
    "popularity_score",
    "user_engagement_rate",
    "user_avg_watch_percentage",
    "user_interaction_count",
    "content_engagement_rate",
    "content_avg_watch_percentage",
    "content_interaction_count",
    "user_sport_engagement_rate",
    "user_content_type_engagement_rate",
    "user_sport_interaction_count",
    "user_content_type_interaction_count",
    "user_sport_avg_watch_percentage",
    "user_content_type_avg_watch_percentage",
]


def main():
    # Resolve the data and model paths relative to the project root.
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "ranking_features_v3.csv"

    # Load the data and order interactions from earliest to latest.
    data = pd.read_csv(data_path)
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    # X contains only allowed input features; y is the engagement label.
    X = data[FEATURE_COLUMNS]
    y = data["positive_engagement"]

    # Keep the earliest 80% for training and reserve the latest 20% for testing.
    split_index = int(len(data) * 0.8)
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # Train a Logistic Regression classifier on the earlier interactions.
    model = LogisticRegression(max_iter=1_000)
    model.fit(X_train, y_train)

    # Evaluate predictions on later rows that were not used for training.
    predictions = model.predict(X_test)
    positive_probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    if y_test.nunique() > 1:
        roc_auc = roc_auc_score(y_test, positive_probabilities)
    else:
        # ROC AUC needs both positive and negative examples in the test set.
        roc_auc = float("nan")

    print(f"Training row count: {len(X_train)}")
    print(f"Test row count: {len(X_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("V1 ROC AUC = 0.5943")
    print("V2 ROC AUC = 0.6091")

    # Save the V3 model separately from the V1 and V2 model files.
    models_directory = project_root / "models"
    models_directory.mkdir(parents=True, exist_ok=True)
    model_path = models_directory / "ranker_v3.joblib"
    joblib.dump(model, model_path)
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()