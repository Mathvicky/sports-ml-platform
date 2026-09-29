"""Generate reproducible synthetic data for the PulsePlay AI project."""

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path


RANDOM_SEED = 42
USER_COUNT = 1_000
CONTENT_COUNT = 500
INTERACTION_COUNT = 20_000

SPORTS = ["football", "formula1", "basketball", "cricket", "tennis"]
CONTENT_TYPES = ["highlight", "interview", "analysis", "live_replay", "documentary"]
EVENT_TYPES = ["view", "click", "like", "skip", "complete"]


def generate_users(rng):
    """Create users with a sport and content-type preference."""
    users = []
    age_groups = ["18-24", "25-34", "35-44", "45-54", "55+"]

    for number in range(1, USER_COUNT + 1):
        users.append(
            {
                "user_id": f"USR-{number:04d}",
                "age_group": rng.choice(age_groups),
                "preferred_sport": rng.choice(SPORTS),
                "preferred_content_type": rng.choice(CONTENT_TYPES),
            }
        )

    return users


def generate_content(rng):
    """Create sports content with realistic durations and popularity."""
    content_items = []
    duration_ranges = {
        "highlight": (30, 180),
        "interview": (180, 1_200),
        "analysis": (120, 900),
        "live_replay": (600, 7_200),
        "documentary": (900, 5_400),
    }

    for number in range(1, CONTENT_COUNT + 1):
        content_type = rng.choice(CONTENT_TYPES)
        minimum_duration, maximum_duration = duration_ranges[content_type]
        content_items.append(
            {
                "content_id": f"CNT-{number:04d}",
                "sport": rng.choice(SPORTS),
                "content_type": content_type,
                "duration_seconds": rng.randint(minimum_duration, maximum_duration),
                "popularity_score": round(rng.uniform(0.0, 1.0), 3),
            }
        )

    return content_items


def generate_interactions(rng, users, content_items):
    """Create interactions, favoring content that matches user preferences."""
    interactions = []
    start_time = datetime(2025, 1, 1)
    time_window_seconds = 90 * 24 * 60 * 60

    for _ in range(INTERACTION_COUNT):
        user = rng.choice(users)

        # Matching either preference makes an item more likely to be selected.
        content_weights = []
        for item in content_items:
            weight = 1
            if item["sport"] == user["preferred_sport"]:
                weight += 4
            if item["content_type"] == user["preferred_content_type"]:
                weight += 4
            content_weights.append(weight)
        content = rng.choices(content_items, weights=content_weights, k=1)[0]

        # Matched content is more likely to receive a like or complete event.
        sport_match = content["sport"] == user["preferred_sport"]
        type_match = content["content_type"] == user["preferred_content_type"]
        event_weights = [40, 20, 10, 20, 10]
        if sport_match:
            event_weights[2] += 8
            event_weights[3] -= 5
            event_weights[4] += 12
        if type_match:
            event_weights[2] += 8
            event_weights[3] -= 5
            event_weights[4] += 12
        event_type = rng.choices(EVENT_TYPES, weights=event_weights, k=1)[0]

        duration = content["duration_seconds"]
        if event_type == "skip":
            watch_time = rng.randint(0, max(1, duration // 4))
        elif event_type == "complete":
            watch_time = rng.randint(max(1, int(duration * 0.8)), duration)
        else:
            watch_time = rng.randint(0, duration)

        timestamp = start_time + timedelta(seconds=rng.randint(0, time_window_seconds))
        interactions.append(
            {
                "user_id": user["user_id"],
                "content_id": content["content_id"],
                "event_type": event_type,
                "watch_time_seconds": watch_time,
                "timestamp": timestamp.isoformat(timespec="seconds"),
            }
        )

    # Keep the exported interaction history in chronological order.
    interactions.sort(key=lambda interaction: interaction["timestamp"])
    return interactions


def write_csv(output_path, fieldnames, rows):
    """Write a list of dictionaries as a CSV file."""
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    # A local random generator keeps results repeatable on every run.
    rng = random.Random(RANDOM_SEED)
    users = generate_users(rng)
    content_items = generate_content(rng)
    interactions = generate_interactions(rng, users, content_items)

    # Resolve output paths from this script so it works from any directory.
    project_root = Path(__file__).resolve().parents[2]
    output_directory = project_root / "data"
    output_directory.mkdir(parents=True, exist_ok=True)

    write_csv(
        output_directory / "users.csv",
        ["user_id", "age_group", "preferred_sport", "preferred_content_type"],
        users,
    )
    write_csv(
        output_directory / "content.csv",
        ["content_id", "sport", "content_type", "duration_seconds", "popularity_score"],
        content_items,
    )
    write_csv(
        output_directory / "interactions.csv",
        ["user_id", "content_id", "event_type", "watch_time_seconds", "timestamp"],
        interactions,
    )

    print(f"Generated {len(users)} users, {len(content_items)} content items, "
          f"and {len(interactions)} interactions in {output_directory}.")


if __name__ == "__main__":
    main()
