"""Recommend unseen sports content to one user with Ranker V2."""

import argparse
from pathlib import Path

import joblib
import pandas as pd


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


def recommend_for_user(user_id, project_root):
    """Print up to 10 unseen content items ranked for one user."""
    data_directory = project_root / "data"

    # Load the historical feature rows and the full content catalog.
    history = pd.read_csv(data_directory / "ranking_features_v2.csv")
    history["timestamp"] = pd.to_datetime(history["timestamp"])
    content = pd.read_csv(data_directory / "content.csv")

    # Find the user's latest available row, which contains their latest features.
    user_history = history[history["user_id"] == user_id].sort_values(
        "timestamp", kind="mergesort"
    )
    if user_history.empty:
        print(f"No historical interactions found for user_id '{user_id}'.")
        return

    latest_user_row = user_history.iloc[-1]
    recommendation_time = latest_user_row["timestamp"]

    print(f"Recommendations for user: {user_id}")
    print(f"Preferred sport: {latest_user_row['preferred_sport']}")
    print(f"Preferred content type: {latest_user_row['preferred_content_type']}")

    # Exclude content the user has already interacted with.
    seen_content_ids = set(user_history["content_id"])
    candidates = content[~content["content_id"].isin(seen_content_ids)].copy()
    if candidates.empty:
        print("No unseen content is available to recommend.")
        return

    # Use only interactions strictly earlier than this user's latest timestamp.
    # This excludes the current row's outcome and any same-time interactions.
    past_history = history[history["timestamp"] < recommendation_time]

    if past_history.empty:
        global_engagement_rate = 0.0
        global_avg_watch_percentage = 0.0
        content_history = pd.DataFrame(
            columns=[
                "content_id",
                "content_engagement_rate",
                "content_avg_watch_percentage",
                "content_interaction_count",
            ]
        )
    else:
        # These global values are sensible fallbacks for content with no history.
        global_engagement_rate = past_history["positive_engagement"].mean()
        global_avg_watch_percentage = past_history["watch_percentage"].mean()

        # Summarize each item's past engagement and watch behaviour only.
        content_history = past_history.groupby("content_id").agg(
            content_engagement_rate=("positive_engagement", "mean"),
            content_avg_watch_percentage=("watch_percentage", "mean"),
            content_interaction_count=("content_id", "size"),
        ).reset_index()

    candidates = candidates.merge(content_history, on="content_id", how="left")
    candidates["content_engagement_rate"] = candidates[
        "content_engagement_rate"
    ].fillna(global_engagement_rate)
    candidates["content_avg_watch_percentage"] = candidates[
        "content_avg_watch_percentage"
    ].fillna(global_avg_watch_percentage)
    candidates["content_interaction_count"] = candidates[
        "content_interaction_count"
    ].fillna(0)

    # Add the user's latest historical features and preference-match indicators.
    candidates["sport_match"] = (
        candidates["sport"] == latest_user_row["preferred_sport"]
    ).astype(int)
    candidates["content_type_match"] = (
        candidates["content_type"] == latest_user_row["preferred_content_type"]
    ).astype(int)
    candidates["user_engagement_rate"] = latest_user_row["user_engagement_rate"]
    candidates["user_avg_watch_percentage"] = latest_user_row[
        "user_avg_watch_percentage"
    ]
    candidates["user_interaction_count"] = latest_user_row[
        "user_interaction_count"
    ]

    # Score candidates with the already-trained model; do not train a new one.
    model_path = project_root / "models" / "ranker_v2.joblib"
    model = joblib.load(model_path)
    positive_class_index = list(model.classes_).index(1)
    candidates["predicted_probability"] = model.predict_proba(
        candidates[FEATURE_COLUMNS]
    )[:, positive_class_index]

    # Show the highest-probability unseen items first.
    top_ten = candidates.sort_values(
        "predicted_probability", ascending=False
    ).head(10)
    print("\nTop recommendations:")
    for rank, (_, item) in enumerate(top_ten.iterrows(), start=1):
        print(
            f"{rank}. {item['content_id']} | {item['sport']} | "
            f"{item['content_type']} | "
            f"predicted engagement probability: "
            f"{item['predicted_probability']:.4f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Recommend unseen sports content for one user."
    )
    parser.add_argument("user_id", help="User ID, for example USR-0497")
    arguments = parser.parse_args()

    # The script lives under src/recommendation, two levels below the project root.
    project_root = Path(__file__).resolve().parents[2]
    recommend_for_user(arguments.user_id, project_root)


if __name__ == "__main__":
    main()