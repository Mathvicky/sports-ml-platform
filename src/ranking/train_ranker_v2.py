"""Train and evaluate PulsePlay AI's second ranking model."""

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


# Use only preference, popularity, and historical features known before a row.
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
]


def main():
    # Resolve input and output paths from this script's location.
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "ranking_features_v2.csv"

    # Load the dataset and sort from oldest to newest before splitting.
    data = pd.read_csv(data_path)
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    # X contains the approved model inputs, while y contains the target label.
    X = data[FEATURE_COLUMNS]
    y = data["positive_engagement"]

    # Train on the earliest 80% of rows and test on the latest 20%.
    split_index = int(len(data) * 0.8)
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # Fit a logistic regression classifier using historical and preference signals.
    model = LogisticRegression(max_iter=1_000)
    model.fit(X_train, y_train)

    # Measure model quality using only the later, held-out test rows.
    predictions = model.predict(X_test)
    positive_probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    if y_test.nunique() > 1:
        roc_auc = roc_auc_score(y_test, positive_probabilities)
    else:
        # ROC AUC needs at least one positive and one negative test example.
        roc_auc = float("nan")

    print(f"Training row count: {len(X_train)}")
    print(f"Test row count: {len(X_test)}")
    print(f"Test positive rate: {y_test.mean():.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("V1 ROC AUC reference: 0.5943")

    # Save the trained V2 model without changing the V1 model.
    models_directory = project_root / "models"
    models_directory.mkdir(parents=True, exist_ok=True)
    model_path = models_directory / "ranker_v2.joblib"
    joblib.dump(model, model_path)
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()