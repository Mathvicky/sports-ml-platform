"""Evaluate Ranker V2 as a chronological Top-10 recommender."""

import math
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
TOP_K = 10


def calculate_ndcg_at_k(recommended_ids, relevant_ids):
    """Calculate NDCG@K for binary relevant/not-relevant items."""
    # Give a hit more credit when it appears nearer the top of the list.
    dcg = 0.0
    for rank, content_id in enumerate(recommended_ids, start=1):
        if content_id in relevant_ids:
            dcg += 1 / math.log2(rank + 1)

    # The ideal list puts relevant items first; use it to normalize the score.
    ideal_hit_count = min(TOP_K, len(relevant_ids))
    ideal_dcg = sum(
        1 / math.log2(rank + 1) for rank in range(1, ideal_hit_count + 1)
    )
    return dcg / ideal_dcg if ideal_dcg else 0.0


def main():
    # Resolve project inputs from this script's location.
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "ranking_features_v2.csv"
    model_path = project_root / "models" / "ranker_v2.joblib"

    # Sort interactions and choose a time boundary near the earliest 80% of rows.
    data = pd.read_csv(data_path)
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    if len(data) < 2:
        print("Not enough interactions for chronological evaluation.")
        return

    split_index = max(1, min(int(len(data) * 0.8), len(data) - 1))
    cutoff_timestamp = data.iloc[split_index - 1]["timestamp"]

    # All history is at or before the training boundary. Later rows are labels
    # for evaluation only and are never used to calculate candidate features.
    historical = data[data["timestamp"] <= cutoff_timestamp]
    later_interactions = data[data["timestamp"] > cutoff_timestamp]
    if later_interactions.empty:
        print("No later interactions are available for evaluation.")
        return

    # Build user profiles using only interactions known by the cutoff time.
    user_profiles = historical.groupby("user_id").agg(
        preferred_sport=("preferred_sport", "last"),
        preferred_content_type=("preferred_content_type", "last"),
        user_engagement_rate=("positive_engagement", "mean"),
        user_avg_watch_percentage=("watch_percentage", "mean"),
        user_interaction_count=("content_id", "size"),
    )

    # Build content behaviour features using only the same historical period.
    content_profiles = historical.groupby("content_id").agg(
        content_engagement_rate=("positive_engagement", "mean"),
        content_avg_watch_percentage=("watch_percentage", "mean"),
        content_interaction_count=("content_id", "size"),
    )
    global_engagement_rate = historical["positive_engagement"].mean()
    global_watch_percentage = historical["watch_percentage"].mean()

    # Treat item attributes as a static catalog; future outcomes are not features.
    catalog = data[
        ["content_id", "sport", "content_type", "popularity_score"]
    ].drop_duplicates("content_id")

    # Load the existing model only; this evaluator never trains or saves a model.
    model = joblib.load(model_path)
    positive_class_index = list(model.classes_).index(1)
    user_metrics = []

    # Evaluate users who have at least one later positive interaction.
    for user_id in later_interactions["user_id"].unique():
        if user_id not in user_profiles.index:
            continue

        future_user_rows = later_interactions[
            later_interactions["user_id"] == user_id
        ]
        relevant_ids = set(
            future_user_rows.loc[
                future_user_rows["positive_engagement"] == 1, "content_id"
            ]
        )
        if not relevant_ids:
            continue

        # A recommendation list should contain items the user has not seen.
        historical_user_rows = historical[historical["user_id"] == user_id]
        seen_ids = set(historical_user_rows["content_id"])
        relevant_ids -= seen_ids
        if not relevant_ids:
            continue

        candidates = catalog[~catalog["content_id"].isin(seen_ids)].copy()
        if candidates.empty:
            continue

        # Add preference matches and this user's historical profile.
        user_profile = user_profiles.loc[user_id]
        candidates["sport_match"] = (
            candidates["sport"] == user_profile["preferred_sport"]
        ).astype(int)
        candidates["content_type_match"] = (
            candidates["content_type"] == user_profile["preferred_content_type"]
        ).astype(int)
        candidates["user_engagement_rate"] = user_profile["user_engagement_rate"]
        candidates["user_avg_watch_percentage"] = user_profile[
            "user_avg_watch_percentage"
        ]
        candidates["user_interaction_count"] = user_profile[
            "user_interaction_count"
        ]

        # Add historical item features; cold items use past global averages.
        candidates = candidates.join(content_profiles, on="content_id")
        candidates["content_engagement_rate"] = candidates[
            "content_engagement_rate"
        ].fillna(global_engagement_rate)
        candidates["content_avg_watch_percentage"] = candidates[
            "content_avg_watch_percentage"
        ].fillna(global_watch_percentage)
        candidates["content_interaction_count"] = candidates[
            "content_interaction_count"
        ].fillna(0)

        # Score candidates, then sort by predicted positive-engagement probability.
        probabilities = model.predict_proba(candidates[FEATURE_COLUMNS])[
            :, positive_class_index
        ]
        candidates["predicted_probability"] = probabilities
        top_recommendations = candidates.sort_values(
            ["predicted_probability", "content_id"],
            ascending=[False, True],
            kind="mergesort",
        ).head(TOP_K)
        recommended_ids = top_recommendations["content_id"].tolist()

        hits = len(set(recommended_ids) & relevant_ids)
        precision_at_k = hits / TOP_K
        recall_at_k = hits / len(relevant_ids)
        ndcg_at_k = calculate_ndcg_at_k(recommended_ids, relevant_ids)
        user_metrics.append(
            {
                "precision": precision_at_k,
                "recall": recall_at_k,
                "ndcg": ndcg_at_k,
            }
        )

    # Report the average ranking metrics over users with recommendable positives.
    print(f"Number of users evaluated: {len(user_metrics)}")
    if not user_metrics:
        print("No users had a later positive interaction among unseen items.")
        return

    results = pd.DataFrame(user_metrics)
    print(f"Average Precision@10: {results['precision'].mean():.4f}")
    print(f"Average Recall@10: {results['recall'].mean():.4f}")
    print(f"Average NDCG@10: {results['ndcg'].mean():.4f}")


if __name__ == "__main__":
    main()