"""Rank sports content with a simple popularity-based baseline."""

from pathlib import Path

import pandas as pd


EVENT_WEIGHTS = {
    "skip": 0,
    "view": 1,
    "click": 2,
    "like": 3,
    "complete": 4,
}


def main():
    # Find the project data folder relative to this script.
    project_root = Path(__file__).resolve().parents[2]
    input_path = project_root / "data" / "processed_interactions.csv"

    # Load the processed interactions and give each event its engagement weight.
    interactions = pd.read_csv(input_path)
    interactions["event_weight"] = interactions["event_type"].map(EVENT_WEIGHTS)

    # Give more credit to stronger events when the user watched more of the item.
    interactions["interaction_score"] = (
        interactions["event_weight"] * interactions["watch_percentage"]
    )

    # Average interaction scores per item so frequently seen items do not win only
    # because they have more rows. Add the content's popularity as a small bonus.
    recommendations = interactions.groupby(
        ["content_id", "sport", "content_type"], as_index=False
    ).agg(
        average_interaction_score=("interaction_score", "mean"),
        popularity_score=("popularity_score", "first"),
    )
    recommendations["engagement_score"] = (
        recommendations["average_interaction_score"]
        + recommendations["popularity_score"]
    )

    # Sort from highest to lowest and show the top 10 content items.
    top_ten = recommendations.sort_values(
        "engagement_score", ascending=False
    ).head(10)
    top_ten = top_ten[["content_id", "sport", "content_type", "engagement_score"]]
    top_ten["engagement_score"] = top_ten["engagement_score"].round(3)

    print("Top 10 recommended content items:")
    print(top_ten.to_string(index=False))


if __name__ == "__main__":
    main()