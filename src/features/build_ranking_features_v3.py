"""Add past-only user-by-sport and user-by-content-type features."""

from pathlib import Path

import pandas as pd


NEW_FEATURES = [
    "user_sport_engagement_rate",
    "user_content_type_engagement_rate",
    "user_sport_interaction_count",
    "user_content_type_interaction_count",
    "user_sport_avg_watch_percentage",
    "user_content_type_avg_watch_percentage",
]


def build_global_history(interactions):
    """Find engagement and watch averages from earlier timestamps only."""
    history = interactions.groupby("timestamp", as_index=False).agg(
        rows_at_timestamp=("positive_engagement", "size"),
        positives_at_timestamp=("positive_engagement", "sum"),
        watch_total_at_timestamp=("watch_percentage", "sum"),
    )
    history = history.sort_values("timestamp", kind="mergesort")

    # Remove each complete timestamp bucket from the cumulative totals so the
    # fallback values contain only interactions that happened strictly earlier.
    history["prior_interaction_count"] = (
        history["rows_at_timestamp"].cumsum() - history["rows_at_timestamp"]
    )
    history["prior_positive_count"] = (
        history["positives_at_timestamp"].cumsum()
        - history["positives_at_timestamp"]
    )
    history["prior_watch_total"] = (
        history["watch_total_at_timestamp"].cumsum()
        - history["watch_total_at_timestamp"]
    )

    history["global_engagement_rate"] = (
        history["prior_positive_count"]
        .div(history["prior_interaction_count"])
        .where(history["prior_interaction_count"] > 0, 0.0)
    )
    history["global_avg_watch_percentage"] = (
        history["prior_watch_total"]
        .div(history["prior_interaction_count"])
        .where(history["prior_interaction_count"] > 0, 0.0)
    )

    return history[
        ["timestamp", "global_engagement_rate", "global_avg_watch_percentage"]
    ]


def add_user_category_features(
    interactions, global_history, category_column, feature_prefix
):
    """Add prior user history for one category, such as sport or content type."""
    user_category_columns = ["user_id", category_column]

    # Combine same-timestamp rows before accumulating so they cannot count as
    # prior history for one another.
    history = interactions.groupby(
        [*user_category_columns, "timestamp"], as_index=False
    ).agg(
        rows_at_timestamp=("positive_engagement", "size"),
        positives_at_timestamp=("positive_engagement", "sum"),
        watch_total_at_timestamp=("watch_percentage", "sum"),
    )
    history = history.sort_values("timestamp", kind="mergesort")

    # Cumulative totals per user/category, minus the current timestamp bucket,
    # give strictly earlier interaction counts and sums.
    user_category_history = history.groupby(
        user_category_columns, sort=False
    )
    history["prior_interaction_count"] = (
        user_category_history["rows_at_timestamp"].cumsum()
        - history["rows_at_timestamp"]
    )
    history["prior_positive_count"] = (
        user_category_history["positives_at_timestamp"].cumsum()
        - history["positives_at_timestamp"]
    )
    history["prior_watch_total"] = (
        user_category_history["watch_total_at_timestamp"].cumsum()
        - history["watch_total_at_timestamp"]
    )

    # Use the earlier global averages if this user has no history for the category.
    history = history.merge(global_history, on="timestamp", how="left", sort=False)
    has_prior_category_history = history["prior_interaction_count"] > 0
    history[f"{feature_prefix}_engagement_rate"] = (
        history["prior_positive_count"]
        .div(history["prior_interaction_count"])
        .where(has_prior_category_history, history["global_engagement_rate"])
    )
    history[f"{feature_prefix}_avg_watch_percentage"] = (
        history["prior_watch_total"]
        .div(history["prior_interaction_count"])
        .where(has_prior_category_history, history["global_avg_watch_percentage"])
    )
    history[f"{feature_prefix}_interaction_count"] = history[
        "prior_interaction_count"
    ]

    feature_columns = [
        f"{feature_prefix}_engagement_rate",
        f"{feature_prefix}_interaction_count",
        f"{feature_prefix}_avg_watch_percentage",
    ]
    feature_history = history[
        [*user_category_columns, "timestamp", *feature_columns]
    ]

    # Attach each user/category/time summary to its original interaction rows.
    return interactions.merge(
        feature_history,
        on=[*user_category_columns, "timestamp"],
        how="left",
        sort=False,
    )


def main():
    # Resolve the input and output files relative to the project root.
    project_root = Path(__file__).resolve().parents[2]
    data_directory = project_root / "data"
    input_path = data_directory / "ranking_features_v2.csv"
    output_path = data_directory / "ranking_features_v3.csv"

    # Load and sort all interactions before calculating any historical features.
    interactions = pd.read_csv(input_path)
    interactions["timestamp"] = pd.to_datetime(interactions["timestamp"])
    interactions["watch_percentage"] = pd.to_numeric(
        interactions["watch_percentage"], errors="coerce"
    ).fillna(0.0)
    interactions["_original_order"] = range(len(interactions))
    interactions = interactions.sort_values(
        "timestamp", kind="mergesort"
    ).reset_index(drop=True)

    # Build global fallbacks, then add the sport-specific and type-specific history.
    global_history = build_global_history(interactions)
    interactions = add_user_category_features(
        interactions, global_history, "sport", "user_sport"
    )
    interactions = add_user_category_features(
        interactions, global_history, "content_type", "user_content_type"
    )

    # Keep all V2 columns and return rows to timestamp order before saving.
    interactions = interactions.sort_values(
        ["timestamp", "_original_order"], kind="mergesort"
    ).drop(columns="_original_order")
    interactions.to_csv(output_path, index=False)

    print(f"Final row count: {len(interactions)}")
    print("New feature names:")
    print(NEW_FEATURES)
    print("First 5 rows of the new features:")
    print(interactions[NEW_FEATURES].head())
    print("Missing-value count for the new features:")
    print(interactions[NEW_FEATURES].isna().sum())


if __name__ == "__main__":
    main()