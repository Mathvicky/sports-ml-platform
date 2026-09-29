"""Build time-aware behavioural features for PulsePlay interactions."""

from pathlib import Path

import pandas as pd


def build_global_history(interactions):
    """Calculate global averages using only rows at earlier timestamps."""
    # First summarize all interactions that share each timestamp.
    history = interactions.groupby("timestamp", as_index=False).agg(
        rows_at_timestamp=("positive_engagement", "size"),
        positives_at_timestamp=("positive_engagement", "sum"),
        watch_total_at_timestamp=("watch_percentage", "sum"),
    )
    history = history.sort_values("timestamp")

    # Cumulative totals minus the current timestamp bucket leave only past rows.
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

    # Use zero when the dataset has no earlier interactions at all.
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


def add_entity_history(interactions, global_history, entity_column, feature_prefix):
    """Add prior-history averages and counts for a user or content item."""
    # Summarize each entity separately at each timestamp. This treats rows with
    # the same timestamp as simultaneous, so none of them can affect the others.
    history = interactions.groupby(
        [entity_column, "timestamp"], as_index=False
    ).agg(
        rows_at_timestamp=("positive_engagement", "size"),
        positives_at_timestamp=("positive_engagement", "sum"),
        watch_total_at_timestamp=("watch_percentage", "sum"),
    )
    history = history.sort_values("timestamp")

    # Keep a running total for each entity, then remove the current-time bucket.
    entity_groups = history.groupby(entity_column, sort=False)
    history["prior_interaction_count"] = (
        entity_groups["rows_at_timestamp"].cumsum() - history["rows_at_timestamp"]
    )
    history["prior_positive_count"] = (
        entity_groups["positives_at_timestamp"].cumsum()
        - history["positives_at_timestamp"]
    )
    history["prior_watch_total"] = (
        entity_groups["watch_total_at_timestamp"].cumsum()
        - history["watch_total_at_timestamp"]
    )

    # Get prior global averages for timestamps where this entity has no history.
    history = history.merge(global_history, on="timestamp", how="left", sort=False)
    has_entity_history = history["prior_interaction_count"] > 0

    history[f"{feature_prefix}_engagement_rate"] = (
        history["prior_positive_count"]
        .div(history["prior_interaction_count"])
        .where(has_entity_history, history["global_engagement_rate"])
    )
    history[f"{feature_prefix}_avg_watch_percentage"] = (
        history["prior_watch_total"]
        .div(history["prior_interaction_count"])
        .where(has_entity_history, history["global_avg_watch_percentage"])
    )
    history[f"{feature_prefix}_interaction_count"] = history[
        "prior_interaction_count"
    ]

    feature_columns = [
        f"{feature_prefix}_engagement_rate",
        f"{feature_prefix}_avg_watch_percentage",
        f"{feature_prefix}_interaction_count",
    ]
    feature_history = history[[entity_column, "timestamp", *feature_columns]]

    # Each input row gets the historical values for its entity and timestamp.
    return interactions.merge(
        feature_history,
        on=[entity_column, "timestamp"],
        how="left",
        sort=False,
    )


def main():
    # Resolve input and output paths from the project root.
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "processed_interactions.csv"
    output_path = project_root / "data" / "ranking_features_v2.csv"

    # Load interactions, parse timestamps, and sort them before feature building.
    interactions = pd.read_csv(data_path)
    interactions["timestamp"] = pd.to_datetime(interactions["timestamp"])
    interactions["positive_engagement"] = interactions["event_type"].isin(
        ["like", "complete"]
    ).astype(int)
    interactions["watch_percentage"] = pd.to_numeric(
        interactions["watch_percentage"], errors="coerce"
    ).fillna(0.0)
    interactions["_original_order"] = range(len(interactions))
    interactions = interactions.sort_values(
        "timestamp", kind="mergesort"
    ).reset_index(drop=True)

    # Build global fallbacks, then add past-only user and content statistics.
    global_history = build_global_history(interactions)
    interactions = add_entity_history(
        interactions, global_history, "user_id", "user"
    )
    interactions = add_entity_history(
        interactions, global_history, "content_id", "content"
    )

    # Keep all existing columns, including the preference and popularity features.
    interactions = interactions.sort_values(
        ["timestamp", "_original_order"], kind="mergesort"
    ).drop(columns="_original_order")

    # Save the expanded dataset and print a short summary.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    interactions.to_csv(output_path, index=False)

    print(f"Final row count: {len(interactions)}")
    print("Column names:")
    print(interactions.columns.tolist())
    print("First 5 rows:")
    print(interactions.head())
    print("Missing-value count per column:")
    print(interactions.isna().sum())


if __name__ == "__main__":
    main()