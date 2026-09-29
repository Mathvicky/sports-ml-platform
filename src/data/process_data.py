"""Load, check, join, and enrich PulsePlay AI data."""

from pathlib import Path

import pandas as pd


def print_data_checks(dataset_name, dataframe):
    """Print missing-value and duplicate-row counts for a dataset."""
    print(f"\n{dataset_name} missing values:")
    print(dataframe.isna().sum())
    print(f"{dataset_name} duplicate rows: {dataframe.duplicated().sum()}")


def main():
    # Resolve paths from this script so it can run from any working directory.
    project_root = Path(__file__).resolve().parents[2]
    data_directory = project_root / "data"

    # Load the three CSV files into pandas DataFrames.
    users = pd.read_csv(data_directory / "users.csv")
    content = pd.read_csv(data_directory / "content.csv")
    interactions = pd.read_csv(data_directory / "interactions.csv")

    # Report the sizes and basic data-quality checks before joining.
    print(f"Users row count: {len(users)}")
    print(f"Content row count: {len(content)}")
    print(f"Interactions row count: {len(interactions)}")
    print_data_checks("Users", users)
    print_data_checks("Content", content)
    print_data_checks("Interactions", interactions)

    # Keep every interaction while adding its user's and content's details.
    processed = interactions.merge(users, on="user_id", how="left")
    processed = processed.merge(content, on="content_id", how="left")

    # Turn preference matches into simple numeric features: 1 for a match, 0 otherwise.
    processed["sport_match"] = (
        processed["preferred_sport"] == processed["sport"]
    ).astype(int)
    processed["content_type_match"] = (
        processed["preferred_content_type"] == processed["content_type"]
    ).astype(int)

    # Cap the ratio so unusual input values cannot produce a value above 1.0.
    processed["watch_percentage"] = (
        processed["watch_time_seconds"] / processed["duration_seconds"]
    ).clip(upper=1.0)

    # Save the enriched interactions and summarize the result.
    output_path = data_directory / "processed_interactions.csv"
    processed.to_csv(output_path, index=False)

    print(f"\nFinal row count: {len(processed)}")
    print("Column names:")
    print(processed.columns.tolist())
    print("First 5 rows:")
    print(processed.head())
    print(f"\nSaved processed data to: {output_path}")


if __name__ == "__main__":
    main()